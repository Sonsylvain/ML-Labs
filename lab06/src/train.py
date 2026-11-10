"""Task 1: YAML-driven training script with seeding, checkpointing, resume and MLflow logging.

    python lab06/src/train.py                                   # everything from lab06/config.yaml
    python lab06/src/train.py --set training.epochs=3 --set optimizer.name=sgd --set optimizer.lr=0.05
    python lab06/src/train.py --set mlflow.run_name=foo --resume  # continue foo from its last checkpoint

All settings live in one config dict. --set only edits keys that already exist in config.yaml.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIG_PATH, PeakRSS, build_loaders, build_model, build_optimizer, build_scheduler,  # noqa: E402
                    evaluate, flatten, load_config, repo_path, set_seeds)


def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temp file then rename, so a crash mid-write never leaves a corrupt last.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, loader, opt, accum: int, n_train: int) -> dict:
    """One pass over the training set. With accum > 1 the optimizer steps every `accum` micro-batches.

    Each micro-batch's summed loss is divided by the number of samples in its accumulation group, so the
    accumulated gradient equals the gradient of the mean loss over the whole group - the same gradient a
    single batch of accum * batch_size samples would give (including the shorter last group).
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    bs = loader.batch_size
    group = bs * accum
    total, n, steps = 0.0, 0, 0
    data_s = 0.0
    n_batches = len(loader)
    opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    t_data = t0
    for i, (x, y) in enumerate(loader):
        data_s += time.perf_counter() - t_data
        group_size = min(group, n_train - (i // accum) * group)
        loss_sum = loss_fn(model(x), y)
        (loss_sum / group_size).backward()
        if (i + 1) % accum == 0 or i == n_batches - 1:
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
        total += loss_sum.item()
        n += y.numel()
        t_data = time.perf_counter()
    seconds = time.perf_counter() - t0
    return {"train_loss": total / n, "epoch_seconds": seconds, "data_wait_seconds": data_s,
            "throughput": n / seconds, "optimizer_steps": steps}


def train(cfg: dict, resume: bool = False) -> dict:
    torch.set_num_threads(cfg["threads"])
    run_name = cfg["mlflow"]["run_name"]
    ckpt_dir = repo_path(cfg["checkpoint"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"

    set_seeds(cfg["seed"])
    model = build_model(cfg)
    train_loader, val_loader, test_loader, shuffle_gen = build_loaders(cfg)
    opt = build_optimizer(cfg, model.parameters())
    sched = build_scheduler(cfg, opt)
    epochs, accum = cfg["training"]["epochs"], cfg["training"]["accumulation_steps"]
    n_train = len(train_loader.dataset)

    start_epoch, history, run_id = 1, [], None
    if resume:
        state = torch.load(last, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if sched is not None:
            sched.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        shuffle_gen.set_state(state["rng"]["shuffle"])
        start_epoch, history, run_id = state["epoch"] + 1, state["history"], state["mlflow_run_id"]
        print(f"[{run_name}] resumed from {last.relative_to(repo_path('.'))} at epoch {state['epoch']}", flush=True)

    mlflow.set_tracking_uri(repo_path(cfg["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)
    if resume:
        mlflow.set_tag("resumed_at_epoch", start_epoch)
    else:
        mlflow.log_params(flatten(cfg))
        mlflow.log_dict(cfg, "config.yaml")
        mlflow.set_tag("n_params", sum(p.numel() for p in model.parameters()))

    with PeakRSS() as mem:
        for epoch in range(start_epoch, epochs + 1):
            lr = opt.param_groups[0]["lr"]
            row = train_one_epoch(model, train_loader, opt, accum, n_train)
            row["val_loss"], row["val_acc"] = evaluate(model, val_loader)
            if sched is not None:
                sched.step(row["val_loss"]) if cfg["scheduler"]["name"] == "plateau" else sched.step()
            row = {"epoch": epoch, "lr": lr, **row}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"[{run_name}] epoch {epoch}/{epochs}  lr {lr:.2e}  train_loss {row['train_loss']:.4f}  "
                  f"val_loss {row['val_loss']:.4f}  val_acc {row['val_acc']:.4f}  "
                  f"{row['epoch_seconds']:.1f}s ({row['throughput']:.0f} samples/s)", flush=True)

            save_checkpoint(last, {
                "epoch": epoch, "config": cfg, "history": history, "mlflow_run_id": run.info.run_id,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "shuffle": shuffle_gen.get_state()},
            })
            if cfg["training"]["interrupt_after_epoch"] == epoch:
                print(f"[{run_name}] simulated crash after epoch {epoch} (checkpoint saved)", flush=True)
                os._exit(3)  # hard exit: no cleanup, MLflow run left open, like a killed job

    test_loss, test_acc = evaluate(model, test_loader)
    peak_mb = mem.peak / 2**20
    peak_delta_mb = (mem.peak - mem.baseline) / 2**20
    mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc,
                        "peak_rss_mb": peak_mb, "peak_rss_increase_mb": peak_delta_mb})
    hist_path = ckpt_dir / "history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    model_path = ckpt_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(str(hist_path))
    mlflow.log_artifact(str(model_path))
    mlflow.end_run()
    print(f"[{run_name}] test_acc {test_acc:.4f}  peak RSS {peak_mb:.0f} MB (+{peak_delta_mb:.0f} MB during training)",
          flush=True)
    return {"run_name": run_name, "run_id": run.info.run_id, "test_loss": test_loss, "test_acc": test_acc,
            "peak_rss_mb": peak_mb, "peak_rss_increase_mb": peak_delta_mb, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-out", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    print(yaml.safe_dump(cfg, sort_keys=False) if not args.summary_out else "", end="")
    summary = train(cfg, resume=args.resume)
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
