"""Task 6: mini-batch SGD (shuffle, batch 32, 50 epochs) for lr in [0.001, 0.01, 0.1] x W in [16, 64]."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, load_split, set_seeds
from model import backward, cross_entropy, forward, init_params

LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
BATCH_SIZE = 32
EPOCHS = 50


def accuracy(params, X, y) -> float:
    return float(np.mean(forward(params, X)[0].argmax(axis=1) == y))


def train(width, lr, data) -> tuple[dict, list[dict]]:
    X_train, X_test, Y_train, Y_test, y_train, y_test = data
    rng = np.random.default_rng(SEED)
    params = init_params(width, rng)  # same seed -> same starting weights for every lr
    n = X_train.shape[0]
    log = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)  # shuffle: a new batch composition every epoch
        batch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]  # last batch is smaller (1257 = 39*32 + 9)
            P, cache = forward(params, X_train[idx])
            batch_losses.append(cross_entropy(P, Y_train[idx]))
            grads = backward(params, cache, Y_train[idx])
            for k in params:
                params[k] -= lr * grads[k]  # SGD step
        row = {"width": width, "lr": lr, "epoch": epoch,
               "train_loss": float(np.mean(batch_losses)),
               "test_loss": cross_entropy(forward(params, X_test)[0], Y_test),
               "train_acc": accuracy(params, X_train, y_train),
               "test_acc": accuracy(params, X_test, y_test)}
        log.append(row)
        print(f"  W={width:<3} lr={lr:<6} epoch {epoch:>2}/{EPOCHS}  train_loss {row['train_loss']:.4f}  "
              f"test_loss {row['test_loss']:.4f}  test_acc {row['test_acc']:.4f}", flush=True)
    return params, log


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {0.001: "#3b6fd8", 0.01: "#d8763b", 0.1: "#2a9d6a"}
    for ax, w in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            run = df[(df.width == w) & (df.lr == lr)]
            ax.plot(run.epoch, run.train_loss, color=colors[lr], label=f"lr={lr} train")
            ax.plot(run.epoch, run.test_loss, color=colors[lr], ls="--", alpha=0.7, label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", lw=0.8, ls=":", label="ln(10) = chance")
        ax.set_title(f"W = {w}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("Lab 2: loss curves (SGD, batch 32, 50 epochs)")
    fig.tight_layout()
    out = RESULTS_DIR / "loss_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main() -> None:
    set_seeds()
    data = load_split()
    logs = []
    for w in WIDTHS:
        for lr in LEARNING_RATES:
            print(f"\n=== W = {w}, lr = {lr} ===")
            logs += train(w, lr, data)[1]

    df = pd.DataFrame(logs)
    df.to_csv(RESULTS_DIR / "training_log.csv", index=False)
    final = df[df.epoch == EPOCHS][["width", "lr", "train_loss", "test_loss", "train_acc", "test_acc"]]
    final.to_csv(RESULTS_DIR / "final_accuracy.csv", index=False, float_format="%.4f")
    print("\nFinal (epoch 50):")
    print(final.to_string(index=False, float_format="%.4f"))
    plot(df)


if __name__ == "__main__":
    main()
