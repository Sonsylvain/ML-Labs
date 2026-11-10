"""Task 6: gradient accumulation. Batch 64 x 4 accumulation steps vs a real batch of 256, 5 epochs each.

Two checks:
1. One step: the accumulated gradient over 4 micro-batches equals the gradient of one 256-sample batch.
2. Training: per-epoch curves of the two 5-epoch runs (same seed, same data order, same Adam 1e-3).
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from common import RESULTS_DIR, build_model, load_config, load_datasets, run_training, set_seeds

EPOCHS = 5


def one_step_check(cfg) -> float:
    train, _, _ = load_datasets(cfg)
    x = torch.stack([train[i][0] for i in range(256)])
    y = torch.tensor([train[i][1] for i in range(256)])
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    set_seeds(cfg["seed"])
    big = build_model(cfg)
    (loss_fn(big(x), y) / 256).backward()

    set_seeds(cfg["seed"])
    acc = build_model(cfg)
    for xb, yb in zip(x.split(64), y.split(64)):
        (loss_fn(acc(xb), yb) / 256).backward()  # same as mean over the 64 samples / 4

    return max((a.grad - b.grad).abs().max().item() for a, b in zip(big.parameters(), acc.parameters()))


if __name__ == "__main__":
    cfg = load_config()
    diff = one_step_check(cfg)
    print(f"one step: max |grad(batch 256) - grad(64 x 4 accumulated)| = {diff:.2e}")

    runs = {"batch256": ["training.batch_size=256", "training.accumulation_steps=1"],
            "batch64x4": ["training.batch_size=64", "training.accumulation_steps=4"]}
    curves, summary = [], []
    for name, args in runs.items():
        s = run_training(args + [f"training.epochs={EPOCHS}", "mlflow.experiment=lab06-task6-accumulation",
                                 f"mlflow.run_name={name}"])
        h = pd.DataFrame(s["history"])
        curves.append(h.assign(run=name))
        summary.append({"run": name, "optimizer_steps": int(h.optimizer_steps.sum()),
                        "throughput_samples_s": h.throughput.mean(), "train_seconds": h.epoch_seconds.sum(),
                        "peak_rss_increase_mb": s["peak_rss_increase_mb"],
                        "final_train_loss": h.train_loss.iloc[-1], "final_val_acc": h.val_acc.iloc[-1],
                        "test_acc": s["test_acc"]})
    allc = pd.concat(curves)
    a, b = (allc[allc.run == r].reset_index(drop=True) for r in runs)
    cmp = pd.DataFrame({"epoch": a.epoch, "train_loss_256": a.train_loss, "train_loss_64x4": b.train_loss,
                        "val_acc_256": a.val_acc, "val_acc_64x4": b.val_acc,
                        "abs_diff_train_loss": (a.train_loss - b.train_loss).abs(),
                        "abs_diff_val_acc": (a.val_acc - b.val_acc).abs()})
    cmp.to_csv(RESULTS_DIR / "accumulation_curves.csv", index=False, float_format="%.6f")
    sm = pd.DataFrame(summary)
    sm.to_csv(RESULTS_DIR / "accumulation_summary.csv", index=False, float_format="%.5f")
    (RESULTS_DIR / "accumulation_grad_check.txt").write_text(
        f"max |grad(batch 256) - grad(64 x 4 accumulated)| over all parameters, first 256 samples: {diff:.3e}\n")
    print(cmp.to_string(index=False, float_format="%.5f"))
    print(sm.to_string(index=False, float_format="%.4f"))

    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, g in allc.groupby("run", sort=False):
        style = "o-" if name == "batch256" else "x--"
        l.plot(g.epoch, g.train_loss, style, label=name)
        r.plot(g.epoch, g.val_acc, style, label=name)
    l.set_title("training loss")
    r.set_title("validation accuracy")
    for ax in (l, r):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accumulation.png", dpi=130)
