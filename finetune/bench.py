"""Instrumentation: trainable params, steady-state tokens/s, peak VRAM. Rank 0 writes metrics.json."""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback


def _numel(p):
    # Under DeepSpeed ZeRO-3 params are partitioned, ds_numel holds the size.
    return getattr(p, "ds_numel", None) or p.numel()


def count_params(model):
    trainable = sum(_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_numel(p) for p in model.parameters())
    return trainable, total


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _reduce(value, op):
    if not _distributed():
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=op)
    return t.item()


class Benchmark:
    """Shared token counter."""

    def __init__(self):
        self.tokens = 0


class BenchTrainer(Trainer):
    def __init__(self, *args, bench: Benchmark, **kwargs):
        super().__init__(*args, **kwargs)
        self._bench = bench

    def training_step(self, model, inputs, *args, **kwargs):
        self._bench.tokens += int(inputs["attention_mask"].sum().item())
        return super().training_step(model, inputs, *args, **kwargs)


class BenchmarkCallback(TrainerCallback):
    def __init__(self, bench: Benchmark, out_dir: str, meta: dict, warmup: int = 8):
        self.bench, self.out_dir, self.meta, self.warmup = bench, out_dir, meta, warmup
        self.window_start = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.trainable, self.total = count_params(model)
        torch.cuda.reset_peak_memory_stats()
        self.t_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.warmup:  # begin warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.window_start = (time.perf_counter(), self.bench.tokens)

    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        tokens_per_s = None
        if self.window_start is not None:
            t0, tok0 = self.window_start
            tokens = _reduce(self.bench.tokens - tok0, dist.ReduceOp.SUM)
            tokens_per_s = round(tokens / (time.perf_counter() - t0), 1)
        peak = _reduce(torch.cuda.max_memory_allocated(), dist.ReduceOp.MAX) / 1e9
        peak_reserved = _reduce(torch.cuda.max_memory_reserved(), dist.ReduceOp.MAX) / 1e9

        if _distributed() and dist.get_rank() != 0:
            return
        record = {
            **self.meta,
            "trainable_params": self.trainable,
            "total_params": self.total,
            "trainable_pct": round(100 * self.trainable / self.total, 3) if self.total else None,
            "tokens_per_s": tokens_per_s,
            "peak_vram_gb": round(peak, 2),
            "peak_vram_reserved_gb": round(peak_reserved, 2),
            "world_size": dist.get_world_size() if _distributed() else 1,
            "runtime_s": round(time.perf_counter() - self.t_start, 1),
        }
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, indent=2)
        Path(self.out_dir, "metrics.json").write_text(payload)
        print("METRICS " + payload, flush=True)  
