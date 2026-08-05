# Qwen2.5-14B × MedMCQA — full vs LoRA vs QLoRA

Fine-tune Qwen2.5-14B on MedMCQA three ways and compare memory / speed / cost / accuracy.
Method is the only knob; data, sequence length, and global batch (= 64) are held fixed.
Runs on Modal — serverless, billed per second, auto-teardown.

## Layout
- `train.py` — training; `--config configs/<method>.yaml`, launched via `torchrun` (multi-GPU full-FT) or `python` (single-GPU PEFT)
- `data.py` — MedMCQA load + CoT prompt/target formatting (prompt masked; `exp` reasoning + answer letter is the target)
- `bench.py` — instrumentation: trainable params, tokens/s, peak VRAM → `runs/<method>/metrics.json`
  (loss/throughput also print to stdout, which Modal streams live)
- `configs/` — `zero3.json` (+ `zero2.json`) and `full|lora|qlora.yaml`
- `modal_app.py` — Modal launcher (train / evaluate functions)
- `eval_medmcqa.py` — accuracy + weighted F1 on the dev split (single GPU)

## Setup (local)
```
pip install modal && modal setup
modal secret create huggingface-secret HF_TOKEN=...   # your Hugging Face token
```
Training logs stream to your terminal via Modal. (Qwen2.5 is ungated, so the HF token is
optional — it avoids download rate limits — but `modal_app.py` currently expects the secret.)

## Training
```
modal run --detach modal_app.py::train_full      # full FT, 4× A100-80GB ZeRO-3
modal run --detach modal_app.py::train_lora      # LoRA, 1× A100-80GB
modal run --detach modal_app.py::train_qlora     # QLoRA, 1× A100-80GB
```
Run methods **one at a time** — concurrent writers to the same Modal volume can lose commits.

## Eval + results
```
# Zero-shot baseline, then each trained run:
modal run modal_app.py::evaluate --model Qwen/Qwen2.5-14B --out /vol/runs/base --base
modal run modal_app.py::evaluate --model /vol/runs/lora --out /vol/runs/lora
# Pull results locally:
modal volume get qwen-medmcqa-ft /runs ./runs
```
Each run's results land in `runs/<method>/`: `eval.json` (accuracy, weighted F1) and
`metrics.json` (trainable params, tokens/s, peak VRAM, runtime).

## Notes
- Unit tests live in `tests/` (gitignored, local-only): `python -m pytest tests/`.
- **Global batch is fixed at 64** across methods (full: per_device 2 × grad_accum 8 × 4 GPUs; LoRA/QLoRA: 8 × 8 × 1 GPU).
- Only full-FT needs the multi-GPU node (ZeRO-3); LoRA and QLoRA each run on a single A100-80GB.
- Pin `A100-80GB` explicitly — bare `A100` can yield 40 GB cards, which OOM the 14B full fine-tune.
- Model + HF cache persist on the Modal volume at `/vol/hf`; the 14B downloads once.
