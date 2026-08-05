# Full vs. LoRA vs. QLoRA — Fine-Tuning Qwen2.5-14B on MedMCQA

A controlled comparison of three fine-tuning methods for a 14B LLM on a medical
multiple-choice benchmark, measuring the full **memory / speed / cost / accuracy** trade-off.
Only the *method* varies — data, sequence length, and global batch size (= 64) are held fixed.
Full fine-tuning runs under **DeepSpeed ZeRO-3** across 4× A100-80GB (the only way a 14B full
fine-tune fits); LoRA and QLoRA each run on a single A100-80GB. Training is orchestrated on
[Modal](https://modal.com).

## Key results

**Accuracy** (MedMCQA validation, 1,000 examples):

| Method | Accuracy | Δ vs. baseline |
|---|---|---|
| Zero-shot baseline | 55.2% | — |
| Full fine-tuning | 61.2% | +6.0 |
| **LoRA** | **61.7%** | +6.5 |
| QLoRA | 58.3% | +3.1 |

**Training resources** (Qwen2.5-14B, 10k examples × 3 epochs, A100-80GB):

| Method | GPUs | Peak VRAM | Throughput | GPU-hours |
|---|---|---|---|---|
| Full | 4 | 70.5 GB | 1125 tok/s | 6.09 |
| LoRA | 1 | 43.0 GB | 981 tok/s | 1.74 |
| QLoRA | 1 | 30.1 GB | 747 tok/s | 2.29 |

**Takeaway:** LoRA **matches full fine-tuning's accuracy** (within ±3% sampling noise) at
**~1/3.5 the compute on a single GPU instead of four** — full fine-tuning buys no measurable
accuracy for its far larger cost on this task. QLoRA trades ~3 accuracy points and *more*
GPU-hours (4-bit dequantization overhead) for the smallest memory footprint, so it's the right
choice only when memory is the binding constraint.

## Layout

```
finetune/
  train.py           # training entry point (torchrun for full-FT, python for PEFT)
  data.py            # MedMCQA load + chain-of-thought prompt/target formatting
  eval_medmcqa.py    # accuracy + weighted F1 on the validation split
  bench.py           # instrumentation: trainable params, tokens/s, peak VRAM
  modal_app.py       # Modal launcher (train / evaluate functions)
  configs/           # full|lora|qlora.yaml + ZeRO-2/3 DeepSpeed configs
  requirements.txt   # pinned deps for a reproducible Modal image
```

See [`finetune/README.md`](finetune/README.md) for detailed setup, training, and evaluation commands.

## Quickstart

```bash
pip install modal && modal setup
modal secret create huggingface-secret HF_TOKEN=...   # your Hugging Face token

# Train (run one at a time — concurrent writers to the Modal volume can lose commits):
modal run --detach modal_app.py::train_full     # full FT, 4× A100-80GB ZeRO-3
modal run --detach modal_app.py::train_lora      # LoRA, 1× A100-80GB
modal run --detach modal_app.py::train_qlora     # QLoRA, 1× A100-80GB

# Evaluate (writes accuracy + weighted F1 to eval.json per run):
modal run modal_app.py::evaluate --model Qwen/Qwen2.5-14B --out /vol/runs/base
modal run modal_app.py::evaluate --model /vol/runs/lora --out /vol/runs/lora
modal volume get qwen-medmcqa-ft /runs ./runs   # pull results (eval.json / metrics.json) locally
```

Each run's results land in `runs/<method>/`: `eval.json` (accuracy, weighted F1) and
`metrics.json` (trainable params, tokens/s, peak VRAM, runtime).

## Method summary

- **Full fine-tuning** — all ~14.8B parameters trainable in bf16 under DeepSpeed ZeRO-3
  (shards parameters, gradients, and optimizer state across 4 GPUs; the ~225 GB Adam footprint
  exceeds any single 80 GB card).
- **LoRA** — frozen bf16 base; rank-16 (α 32) adapters on the attention projections
  (`q,k,v,o_proj`), ~25.2M trainable parameters (~600× fewer than full).
- **QLoRA** — identical adapters, but the frozen base is loaded in 4-bit NF4 with double
  quantization.

The training target is **chain-of-thought**: the model is trained on MedMCQA's `exp` explanation
followed by a final answer letter, with loss masked to the completion only.
