---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Analyzing cost of fine-tuning performance for MedMCQA'
backgroundColor: '#000000'
color: '#ffffff'
style: |
  section { font-size: 26px; background-color: #000000; color: #ffffff; }
  h1, h2, h3, h4, h5, h6 { color: #ffffff; }
  strong { color: #ffffff; }
  a { color: #ffffff; }
  table { font-size: 22px; margin: 0 auto; color: #ffffff; border-color: #555555; background-color: #000000; }
  table th, table td { border-color: #555555; background-color: #000000; color: #ffffff; }
  table tr { background-color: #000000; }
  table thead th { background-color: #1a1a1a; color: #ffffff; }
  table tbody tr:nth-child(even) td { background-color: #000000; }
  section.lead h1 { color: #ffffff; }
  header, footer { color: #ffffff; }
  footer { font-size: 14px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Comparing metrics across different fine-tuning techniques for MedMCQA

**Pablo Ruiz Fischer Bennetts** — AI in Healthcare, UT Austin


<!--
I fine-tuned a 14-billion-parameter medical QA model three ways and ask a
practical question: does the expensive method actually buy you accuracy?
-->

---

# Motivation

- Full fine-tuning of a 14B model is memory-intensive — optimizer state pushes it to ~225 GB, well beyond a single GPU's memory, forcing a multi-GPU sharded setup.
- LoRA / QLoRA collapse the job back onto **one GPU** and promise near-equal quality at a **fraction of the cost** — but is that true on a medical benchmark?

### How does each method's **resource cost** compare to the **accuracy** it buys?

Measured across three axes — **peak VRAM · GPU-hours · accuracy** — for full FT vs LoRA vs QLoRA, against a zero-shot baseline.

<!--
Frame it as the decision a lab actually faces: rent 4 GPUs or 1. Everything
downstream is one comparison — what does the extra compute actually buy?
-->


---

# Setup: model, data, task

- **Model:** Qwen2.5-14B (instruct base)
- **Data:** MedMCQA — 4-option medical entrance-exam. We use `exp` as chain-of-thought targets
- **Training:** completion-only loss masking 
- **Eval:** 1000-example held-out subset, exact answer-letter match 

<!--
Emphasize the CoT targets plus completion-only masking .
-->

---

# Fine-tuning methods

|            | Trainable        | Base precision  | GPUs           |
|------------|------------------|-----------------|----------------|
| **Full FT**| 14.8B            | bf16 + ZeRO-3   | 4× A100-80GB   |
| **LoRA**   | 25.2M (r16/α32)  | bf16 frozen     | 1× A100-80GB   |
| **QLoRA**  | 25.2M            | 4-bit NF4       | 1× A100-80GB   |


<!--
The 600x gap on trainable parameters is the main number to remember.
-->

---

# How full FT was fit?

- Full-FT footprint ≈ **16 bytes/param** (fp32 master weights + 2 Adam moments) ≈ **225 GB**, exceeds most GPU's memory.
- DeepSpeed **ZeRO-3** partitions optimizer state, gradients, and parameters across 4 GPUs, leading to **70.5–82.6 GB/GPU**, just fits.


<!--
Zero-3 made it possible to have a FT baseline
-->

---

# Results: accuracy

| Method            | Accuracy  | Δ vs baseline |
|-------------------|-----------|---------------|
| Baseline (no FT)  | 55.2%     | —             |
| **Full FT**       | 61.2%     | +6.0          |
| **LoRA**          | **61.7%** | +6.5          |
| QLoRA             | 58.3%     | +3.1          |

- CoT fine-tuning adds considerable improvements on accuracy
- All methods **beat the 55.2% baseline**
- QLoRA trails LoRA by **3.4 pts** 
- LoRA **ties** full FT (within ±3% eval noise)

<!--
The headline: the cheapest full-precision method, LoRA, matched the most
expensive one.
-->

---

# Resource usage against cost

![w:900](money_plot.png)

- Lora matches accuracy at the fraction of a cost.

<!--
As shown on diagram, Lora matches accuracy at the fraction of a cost.
-->

---

# Resource trade-off

|         | GPUs | Peak VRAM      | tok/s | GPU-h |
|---------|------|----------------|-------|-------|
| Full    | 4    | 70.5–82.6 GB   | 1125  | 6.09  |
| LoRA    | 1    | 43.0–63.5 GB   | 981   | 1.74  |
| QLoRA   | 1    | 30.1–51.8 GB   | 747   | 2.29  |

- QLoRA is cheapest on memory (fits a 24 GB card) but slowest (4-bit dequant overhead) and least accurate

<!--
-->

---

# Takeaways

- **LoRA matched full FT at ~1/3.5 the cost.**
- QLoRA trades accuracy for the smallest footprint, good fit when memory-bound.


---

# Future work

- Equal accuracy at 10k examples could mean a shared ceiling or that full's 600× capacity was never exercised
- **Sweep LoRA rank** (r = 8 / 16 / 32 / 64) to find where adapter capacity starts limiting accuracy

<!--
Be honest: this is the limit of the current evidence, and the next
experiment is well-defined.
-->

