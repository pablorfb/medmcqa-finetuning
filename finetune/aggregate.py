"""Collect runs/<method>/{metrics.json,eval.json} into results.csv + a money plot.

  python aggregate.py --runs runs --rate 8.0 --out results
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def collect(runs_dir: str, rate_per_hr: float = 0.0) -> pd.DataFrame:
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        if not run.is_dir():
            continue
        rec = {"run": run.name}
        for name in ("metrics.json", "eval.json"):
            path = run / name
            if path.exists():
                rec.update(json.loads(path.read_text()))
        if rate_per_hr and "runtime_s" in rec:
            # cost must reflect GPU count: full uses 4 GPUs, LoRA/QLoRA 1. rate is per-GPU $/hr.
            gpu_hours = rec["runtime_s"] / 3600 * rec.get("world_size", 1)
            rec["gpu_hours"] = round(gpu_hours, 3)
            rec["cost_usd"] = round(gpu_hours * rate_per_hr, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def money_plot(df: pd.DataFrame, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["accuracy", "peak_vram_gb"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d["peak_vram_gb"], d["accuracy"])
    for _, r in d.iterrows():
        ax.annotate(r["run"], (r["peak_vram_gb"], r["accuracy"]))
    ax.set_xlabel("peak VRAM / GPU (GB)")
    ax.set_ylabel("MedMCQA accuracy")
    ax.set_title("Accuracy vs memory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--rate", type=float, default=0.0, help="per-GPU $/hr; cost = gpu_hours x rate")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = collect(args.runs, args.rate)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out, "results.csv"), index=False)
    print(df.to_string(index=False))
    if {"accuracy", "peak_vram_gb"}.issubset(df.columns):
        money_plot(df, str(Path(args.out, "money_plot.png")))


if __name__ == "__main__":
    main()
