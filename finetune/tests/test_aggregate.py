import json

from aggregate import collect


def _run(path, metrics=None, ev=None):
    path.mkdir(parents=True)
    if metrics:
        (path / "metrics.json").write_text(json.dumps(metrics))
    if ev:
        (path / "eval.json").write_text(json.dumps(ev))


def test_collect_joins_metrics_eval_and_cost(tmp_path):
    runs = tmp_path / "runs"
    _run(runs / "full", {"method": "full", "runtime_s": 3600, "world_size": 4, "peak_vram_gb": 70.0}, {"accuracy": 0.61})
    _run(runs / "qlora", {"method": "qlora", "runtime_s": 1800, "world_size": 1, "peak_vram_gb": 22.0}, {"accuracy": 0.60})

    df = collect(str(runs), rate_per_hr=8.0)

    assert set(df["run"]) == {"full", "qlora"}
    full = df[df["run"] == "full"].iloc[0]
    assert full["accuracy"] == 0.61
    assert full["gpu_hours"] == 4.0   # 1 h wall-clock x 4 GPUs
    assert full["cost_usd"] == 32.0   # 4 gpu-hours at $8/GPU-hr — the per-GPU cost multiplier
    qlora = df[df["run"] == "qlora"].iloc[0]
    assert qlora["cost_usd"] == 4.0   # 0.5 h x 1 GPU x $8
