"""Task 3: vary num_workers in [0,1,2,4,8] to find the input-pipeline bottleneck.

For each setting: a loader-only pass (no model: the pipeline's own speed) and two training epochs with
persistent workers. Epoch 1 includes starting the worker processes; epoch 2 is the steady state. On Windows
workers are spawned, not forked, so each one re-imports torch and receives a pickled copy of the dataset.
"""

import time

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from common import RESULTS_DIR, build_model, build_optimizer, flatten, load_config, load_datasets, repo_path, set_seeds

WORKERS = [0, 1, 2, 4, 8]
EPOCHS = 2


def make_loader(train, cfg, workers, persistent=False):
    gen = torch.Generator().manual_seed(cfg["seed"])
    extra = {"persistent_workers": True} if (persistent and workers > 0) else {}
    return DataLoader(train, batch_size=cfg["training"]["batch_size"], num_workers=workers,
                      sampler=RandomSampler(train, generator=gen), **extra)


def loader_only(train, cfg, workers) -> dict:
    """Iterate the loader with no model. Start-up = time to the first batch."""
    t0 = time.perf_counter()
    it = iter(make_loader(train, cfg, workers))
    next(it)
    first = time.perf_counter() - t0
    for _ in it:
        pass
    total = time.perf_counter() - t0
    return {"startup_s": first, "loader_only_s": total - first,
            "loader_only_samples_s": len(train) / (total - first)}


def train_epochs(train, cfg, workers) -> list[dict]:
    set_seeds(cfg["seed"])
    model = build_model(cfg)
    opt = build_optimizer(cfg, model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(train, cfg, workers, persistent=True)
    model.train()
    out = []
    for _ in range(EPOCHS):
        data_s, n = 0.0, 0
        t0 = time.perf_counter()
        t_data = t0
        for x, y in loader:
            data_s += time.perf_counter() - t_data
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            n += y.numel()
            t_data = time.perf_counter()
        total = time.perf_counter() - t0
        out.append({"epoch_s": total, "data_wait_s": data_s, "compute_s": total - data_s, "throughput": n / total})
    del loader
    return out


def main() -> None:
    cfg = load_config()
    torch.set_num_threads(cfg["threads"])
    train, _, _ = load_datasets(cfg)
    mlflow.set_tracking_uri(repo_path(cfg["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment("lab06-task3-workers")
    rows = []
    for w in WORKERS:
        lo = loader_only(train, cfg, w)
        e1, e2 = train_epochs(train, cfg, w)
        row = {"num_workers": w, **lo, "epoch1_s": e1["epoch_s"], "epoch2_s": e2["epoch_s"],
               "epoch2_data_wait_s": e2["data_wait_s"], "epoch2_compute_s": e2["compute_s"],
               "epoch2_data_wait_share": e2["data_wait_s"] / e2["epoch_s"], "epoch2_throughput": e2["throughput"]}
        rows.append(row)
        print(f"workers {w}: start-up {lo['startup_s']:5.1f}s | loader alone {lo['loader_only_s']:5.1f}s "
              f"({lo['loader_only_samples_s']:.0f}/s) | epoch1 {e1['epoch_s']:5.1f}s | epoch2 {e2['epoch_s']:5.1f}s "
              f"= wait {e2['data_wait_s']:4.1f}s + compute {e2['compute_s']:4.1f}s | {e2['throughput']:.0f} samples/s",
              flush=True)
        with mlflow.start_run(run_name=f"workers={w}"):
            mlflow.log_params({**flatten(cfg), "data.num_workers": w})
            mlflow.log_metrics({k: v for k, v in row.items() if k != "num_workers"})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "workers_study.csv", index=False, float_format="%.3f")
    print(df.to_string(index=False, float_format="%.2f"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = list(range(len(WORKERS)))
    a.bar(x, df.epoch2_compute_s, label="compute (fwd + bwd + step)", color="#4C72B0")
    a.bar(x, df.epoch2_data_wait_s, bottom=df.epoch2_compute_s, label="waiting for data", color="#DD8452")
    a.plot(x, df.epoch1_s, "k^", ms=8, label="epoch 1 total (incl. worker start-up)")
    a.plot(x, df.loader_only_s, "s--", color="#55A868", label="loader alone, no model")
    a.set_xticks(x, [str(w) for w in WORKERS])
    a.set_xlabel("num_workers")
    a.set_ylabel("seconds")
    a.set_title("Steady-state training epoch (epoch 2), CNN, batch 128")
    a.legend(fontsize=8)
    b.bar(x, df.startup_s, color="#C44E52")
    b.set_xticks(x, [str(w) for w in WORKERS])
    b.set_xlabel("num_workers")
    b.set_ylabel("seconds to first batch")
    b.set_title("Worker start-up cost (Windows spawn)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "workers_study.png", dpi=130)


if __name__ == "__main__":
    main()
