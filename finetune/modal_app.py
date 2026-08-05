"""Modal launcher — runs training/eval as subprocesses on A100-80GB GPUs.

    modal run --detach modal_app.py::train_full    # full FT, 4x A100-80GB ZeRO-3
    modal run --detach modal_app.py::train_lora    # LoRA, 1x A100-80GB
    modal run --detach modal_app.py::train_qlora   # QLoRA, 1x A100-80GB
    modal run modal_app.py::evaluate --model <ckpt-dir|hf-id> --out <dir> [--n N]
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).parent

image = (
    # CUDA *devel* base: DeepSpeed's import-time op check needs CUDA_HOME/nvcc, which
    # debian_slim lacks. 12.1.1 matches torch 2.4's cu121 build.
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libaio-dev")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(str(REPO), "/root/app", ignore=["__pycache__", "runs", "*.pyc"])
)
app = modal.App("qwen-medmcqa-ft", image=image)
vol = modal.Volume.from_name("qwen-medmcqa-ft", create_if_missing=True)
# HF token, injected into the container as HF_TOKEN. Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=...
# (Qwen2.5 is ungated, so this is optional — it avoids download rate limits / enables gated models.)
SECRETS = [modal.Secret.from_name("huggingface-secret")]
VOLUMES = {"/vol": vol}


def _train(config: str, nproc: int):
    # Single GPU runs plain `python` — torchrun --nproc_per_node=1 still triggers a spurious
    # DDP wrap in transformers, which breaks QLoRA's reentrant gradient checkpointing.
    launcher = ["torchrun", f"--nproc_per_node={nproc}"] if nproc > 1 else ["python"]
    cmd = [*launcher, "train.py", "--config", f"configs/{config}.yaml", "--out", "/vol/runs"]
    subprocess.run(cmd, cwd="/root/app", check=True)
    vol.commit()


@app.function(gpu="A100-80GB:4", volumes=VOLUMES, secrets=SECRETS, timeout=4 * 60 * 60)
def train_full():
    """full FT — 4x A100-80GB ."""
    _train("full", nproc=4)


@app.function(gpu="A100-80GB", volumes=VOLUMES, secrets=SECRETS, timeout=4 * 60 * 60)
def train_lora():
    """LoRA — single A100-80GB."""
    _train("lora", nproc=1)


@app.function(gpu="A100-80GB", volumes=VOLUMES, secrets=SECRETS, timeout=4 * 60 * 60)
def train_qlora():
    """QLoRA — single A100-80GB."""
    _train("qlora", nproc=1)


@app.function(gpu="A100", volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def evaluate(model: str, out: str, n: int = 0):
    cmd = ["python", "eval_medmcqa.py", "--model", model, "--out", out]
    if n:
        cmd += ["--n", str(n)]
    subprocess.run(cmd, cwd="/root/app", check=True)
    vol.commit()


@app.local_entrypoint()
def main(method: str = "full"):
    {"full": train_full, "lora": train_lora, "qlora": train_qlora}[method].remote()
