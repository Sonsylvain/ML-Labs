"""Task 4: train MLP, CNN and Attention for 5 epochs (Adam, lr 1e-3, batch 128), record test accuracy."""

import time

import pandas as pd
import torch
from torch import nn

from common import RESULTS_DIR, get_loaders, set_seeds
from models import build_models

EPOCHS = 5
LR = 1e-3


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return loss / n, correct / n


def train(name, model, loaders) -> list[dict]:
    train_loader, val_loader, test_loader = loaders
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.numel()
            n += y.numel()
        epoch_s = time.perf_counter() - t0
        val_loss, val_acc = evaluate(model, val_loader)
        row = {"model": name, "epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
               "val_acc": val_acc, "epoch_seconds": epoch_s}
        log.append(row)
        print(f"  {name:<9} epoch {epoch}/{EPOCHS}  train_loss {row['train_loss']:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({epoch_s:.1f}s)", flush=True)
    log[-1]["test_loss"], log[-1]["test_acc"] = evaluate(model, test_loader)
    print(f"  {name:<9} TEST acc {log[-1]['test_acc']:.4f}")
    return log


def main() -> None:
    logs = []
    for name in build_models():
        set_seeds()  # same seed -> same data order and init stream for every model
        model = build_models()[name]
        loaders = get_loaders()
        print(f"\n=== {name} ({sum(p.numel() for p in model.parameters()):,} params) ===")
        logs += train(name, model, loaders)

    df = pd.DataFrame(logs)
    df.drop(columns=["test_loss", "test_acc"]).to_csv(RESULTS_DIR / "training_log.csv", index=False,
                                                       float_format="%.5f")
    final = df[df.epoch == EPOCHS][["model", "train_loss", "val_acc", "test_loss", "test_acc", "epoch_seconds"]]
    final.to_csv(RESULTS_DIR / "test_accuracy.csv", index=False, float_format="%.4f")
    print("\nFinal (epoch 5):")
    print(final.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
