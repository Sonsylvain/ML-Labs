"""Task 5: train 2 -> 8 -> 1 (sigmoid, BCE) on make_moons with the scalar engine and with PyTorch.

Both engines start from the same weights, see the same data and do the same full-batch
gradient-descent updates, so they should end at the same parameters. Only the speed differs.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import RESULTS_DIR, SEED, load_moons
from value import Value

HIDDEN = 8
LR = 0.5
EPOCHS = 100
RUNS = 5


def init_weights() -> dict:
    """Same starting point for both engines: W ~ N(0, 2/fan_in) (He), biases 0."""
    rng = np.random.default_rng(SEED)
    return {"W1": rng.normal(0, np.sqrt(2 / 2), size=(2, HIDDEN)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, np.sqrt(2 / HIDDEN), size=(HIDDEN, 1)), "b2": np.zeros(1)}


# ---------- scalar engine ----------

def train_engine(w0, X, y):
    W1 = [[Value(w) for w in row] for row in w0["W1"]]
    b1 = [Value(b) for b in w0["b1"]]
    W2 = [Value(w) for w in w0["W2"][:, 0]]
    b2 = Value(w0["b2"][0])
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]
    rows = [(list(map(float, x)), float(t)) for x, t in zip(X, y)]
    losses = []
    for _ in range(EPOCHS):
        loss = Value(0.0)
        for (x0, x1), t in rows:
            h = [(W1[0][j] * x0 + W1[1][j] * x1 + b1[j]).relu() for j in range(HIDDEN)]
            z = b2
            for j in range(HIDDEN):
                z = z + W2[j] * h[j]
            p = z.sigmoid()
            loss = loss - (p.log() * t + (1 - p).log() * (1 - t))
        loss = loss / len(rows)
        for w in params:
            w.grad = 0.0
        loss.backward()
        for w in params:
            w.data -= LR * w.grad
        losses.append(loss.data)
    final = {"W1": np.array([[w.data for w in row] for row in W1]), "b1": np.array([b.data for b in b1]),
             "W2": np.array([[w.data] for w in W2]), "b2": np.array([b2.data])}
    return final, losses


# ---------- PyTorch ----------

def train_torch(w0, X, y):
    params = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in w0.items()}
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    losses = []
    for _ in range(EPOCHS):
        p = torch.sigmoid(torch.relu(Xt @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"])
        loss = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()  # same BCE as the engine
        for t in params.values():
            t.grad = None
        loss.backward()
        with torch.no_grad():
            for t in params.values():
                t -= LR * t.grad
        losses.append(loss.item())
    return {k: v.detach().numpy() for k, v in params.items()}, losses


def predict(w, X):
    h = np.maximum(X @ w["W1"] + w["b1"], 0)
    return (1 / (1 + np.exp(-(h @ w["W2"] + w["b2"])))).ravel()


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - t0, result


if __name__ == "__main__":
    torch.set_num_threads(1)
    X_train, X_test, y_train, y_test = load_moons()
    w0 = init_weights()
    train_torch(w0, X_train, y_train)  # warm-up: first PyTorch call pays one-off setup costs

    rows, results = [], {}
    for run in range(1, RUNS + 1):
        for name, fn in (("scalar engine", train_engine), ("pytorch", train_torch)):
            seconds, results[name] = timed(fn, w0, X_train, y_train)
            rows.append({"run": run, "engine": name, "seconds": seconds})
            print(f"run {run}  {name:<13} {seconds:.4f} s")
    times = pd.DataFrame(rows)
    times.to_csv(RESULTS_DIR / "training_times.csv", index=False)

    (w_eng, l_eng), (w_pt, l_pt) = results["scalar engine"], results["pytorch"]
    med = times.groupby("engine").seconds.median()
    max_param_diff = max(np.abs(w_eng[k] - w_pt[k]).max() for k in w_eng)
    summary = pd.DataFrame([
        {"engine": name, "median_seconds": med[name], "per_epoch_ms": med[name] / EPOCHS * 1e3,
         "final_train_loss": losses[-1],
         "train_acc": np.mean((predict(w, X_train) > 0.5) == y_train),
         "test_acc": np.mean((predict(w, X_test) > 0.5) == y_test)}
        for name, (w, losses) in results.items()])
    summary.to_csv(RESULTS_DIR / "training_summary.csv", index=False, float_format="%.6f")
    print(summary.to_string(index=False))
    print(f"speed-up PyTorch vs engine: {med['scalar engine'] / med['pytorch']:.1f}x")
    print(f"max |param engine - param pytorch| after {EPOCHS} epochs: {max_param_diff:.2e}")
    print(f"max |loss engine - loss pytorch| over all epochs: {np.abs(np.array(l_eng) - np.array(l_pt)).max():.2e}")
    pd.DataFrame({"epoch": range(1, EPOCHS + 1), "scalar_engine": l_eng, "pytorch": l_pt}).to_csv(
        RESULTS_DIR / "loss_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, EPOCHS + 1), l_eng, lw=3, alpha=0.5, label="scalar engine")
    ax.plot(range(1, EPOCHS + 1), l_pt, ls="--", label="PyTorch")
    ax.set(xlabel="epoch", ylabel="train BCE", title=f"make_moons, 2-{HIDDEN}-1, lr {LR}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
