"""Task 7: time forward and backward separately (median of 100 runs) and relate it to matmul FLOPs."""

import time

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, load_split
from model import backward, forward, init_params

WIDTHS = [16, 64]
BATCHES = {"mini-batch 32": 32, "full train 1257": None}
RUNS = 100
WARMUP = 10


def median_ms(fn) -> float:
    for _ in range(WARMUP):  # warm caches / BLAS threads before measuring
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def matmul_flops(width: int, B: int) -> tuple[int, int]:
    """Multiply-adds count as 2 FLOPs. An (m,k)·(k,n) product costs 2·m·k·n."""
    fwd = 2 * B * (64 * width + width * width + width * 10)    # X·W1, A1·W2, A2·W3
    bwd = 2 * B * (2 * width * 10 + 2 * width * width + 64 * width)  # dW3,dA2, dW2,dA1, dW1 (no dX)
    return fwd, bwd


def matmul_only_ms(params, cache, X, dZ3) -> tuple[float, float]:
    """Time just the matrix products of each pass, with all other work stripped out."""
    W1, W2, W3, A1, A2 = params["W1"], params["W2"], params["W3"], cache["A1"], cache["A2"]
    dZ2, dZ1 = cache["Z2"], cache["Z1"]  # any arrays of the right shape will do for timing

    def fwd():
        X @ W1; A1 @ W2; A2 @ W3

    def bwd():
        A2.T @ dZ3; dZ3 @ W3.T; A1.T @ dZ2; dZ2 @ W2.T; X.T @ dZ1

    return median_ms(fwd), median_ms(bwd)


def main() -> None:
    X_train, _, Y_train, *_ = load_split()
    rows = []
    for w in WIDTHS:
        params = init_params(w, np.random.default_rng(SEED))
        for label, B in BATCHES.items():
            X, Y = X_train[:B], Y_train[:B]
            B = X.shape[0]
            _, cache = forward(params, X)
            t_fwd = median_ms(lambda: forward(params, X))
            t_bwd = median_ms(lambda: backward(params, cache, Y))
            f_fwd, f_bwd = matmul_flops(w, B)
            mm_fwd, mm_bwd = matmul_only_ms(params, cache, X, cache["P"] - Y)
            rows.append({"width": w, "batch": B, "forward_ms": t_fwd, "backward_ms": t_bwd,
                         "bwd/fwd time": t_bwd / t_fwd,
                         "fwd_matmul_share": mm_fwd / t_fwd, "bwd_matmul_share": mm_bwd / t_bwd, "fwd_MFLOP": f_fwd / 1e6, "bwd_MFLOP": f_bwd / 1e6,
                         "bwd/fwd FLOP": f_bwd / f_fwd,
                         "fwd_GFLOP/s": f_fwd / (t_fwd * 1e-3) / 1e9,
                         "bwd_GFLOP/s": f_bwd / (t_bwd * 1e-3) / 1e9})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    df.to_csv(RESULTS_DIR / "timing.csv", index=False, float_format="%.5f")
    print(f"Saved {RESULTS_DIR / 'timing.csv'}")


if __name__ == "__main__":
    main()
