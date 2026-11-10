"""Task 2: profile one training epoch with torch.profiler; share of time in data loading, forward, backward, optimizer.

Each phase of the loop is wrapped in record_function, so the profiler reports one row per phase. The epoch is
profiled with num_workers=0 (the Lab 3 setup) and with the worker count chosen in Task 3 (config.yaml).
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from common import RESULTS_DIR, build_loaders, build_model, build_optimizer, load_config, set_seeds

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def profile_epoch(cfg: dict) -> tuple[dict, str]:
    set_seeds(cfg["seed"])
    model = build_model(cfg)
    train_loader, *_ = build_loaders(cfg)
    opt = build_optimizer(cfg, model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        t0 = time.perf_counter()
        it = iter(train_loader)
        while True:
            with record_function("data_loading"):
                batch = next(it, None)
            if batch is None:
                break
            x, y = batch
            with record_function("forward"):
                loss = loss_fn(model(x), y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad()
        wall = time.perf_counter() - t0
    totals = {e.key: e.cpu_time_total / 1e6 for e in prof.key_averages() if e.key in PHASES}
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
    return {"wall_s": wall, **{p: totals.get(p, 0.0) for p in PHASES}}, table


def main() -> None:
    base = load_config()
    torch.set_num_threads(base["threads"])
    rows = []
    for workers in sorted({0, base["data"]["num_workers"]}):
        cfg = load_config(overrides=[f"data.num_workers={workers}"])
        res, table = profile_epoch(cfg)
        (RESULTS_DIR / f"profile_ops_workers{workers}.txt").write_text(table)
        other = res["wall_s"] - sum(res[p] for p in PHASES)
        for phase in PHASES + ["other"]:
            sec = other if phase == "other" else res[phase]
            rows.append({"num_workers": workers, "phase": phase, "seconds": sec, "share": sec / res["wall_s"]})
        rows.append({"num_workers": workers, "phase": "epoch_total", "seconds": res["wall_s"], "share": 1.0})
        print(f"\nnum_workers={workers}: profiled epoch {res['wall_s']:.1f}s")
        for phase in PHASES + ["other"]:
            sec = other if phase == "other" else res[phase]
            print(f"  {phase:<15} {sec:7.2f}s  {sec / res['wall_s']:6.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "profile_breakdown.csv", index=False, float_format="%.4f")

    pivot = df[df.phase != "epoch_total"].pivot(index="num_workers", columns="phase", values="seconds")
    pivot = pivot[PHASES + ["other"]]
    ax = pivot.plot.barh(stacked=True, figsize=(9, 3), color=["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#999999"])
    ax.set_xlabel("seconds in one profiled epoch")
    ax.set_ylabel("num_workers")
    ax.set_title("Where one training epoch goes (torch.profiler, CNN, batch 128)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "profile_breakdown.png", dpi=130)


if __name__ == "__main__":
    main()
