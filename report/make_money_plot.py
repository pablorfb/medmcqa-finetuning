"""Render the money plot (accuracy vs cost / memory) from the final measured results.

Standalone (matplotlib only) so it runs without the training deps. The values are the real
measurements from Table 2/Table 3; aggregate.py produces the same from the run JSONs in-container.

    python make_money_plot.py    # -> money_plot.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method: (accuracy %, GPU-hours, peak VRAM GB, color)
RUNS = {
    "baseline": (55.2, 0.0, 0.0, "#888888"),
    "full":     (61.2, 6.09, 70.5, "#d1495b"),
    "LoRA":     (61.7, 1.74, 43.0, "#2e8b57"),
    "QLoRA":    (58.3, 2.29, 30.1, "#e08a1e"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, xidx, xlabel in ((ax1, 1, "training cost (GPU-hours)"),
                         (ax2, 2, "peak VRAM / GPU (GB)")):
    ax.axhline(RUNS["baseline"][0], ls="--", lw=1, color="#888888", zorder=0)
    for name, vals in RUNS.items():
        acc, x = vals[0], vals[xidx]
        ax.scatter(x, acc, s=90, color=vals[3], zorder=3, edgecolor="white", linewidth=0.8)
        dy = 0.9 if name != "full" else -1.4  # keep full/LoRA labels from overlapping
        ax.annotate(f"{name}\n{acc:.1f}%", (x, acc), textcoords="offset points",
                    xytext=(6, 6 if dy > 0 else -18), fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MedMCQA accuracy (%)")
    ax.set_ylim(53, 64)
    ax.margins(x=0.18)
    ax.grid(True, alpha=0.25)

ax1.set_title("Accuracy vs. cost")
ax2.set_title("Accuracy vs. memory")
fig.suptitle("LoRA matches full fine-tuning at a fraction of the resources", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("money_plot.png", dpi=150, bbox_inches="tight")
print("wrote money_plot.png")
