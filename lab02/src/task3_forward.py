"""Task 3: forward pass layer by layer; check output shape (B,10) and rows summing to 1."""

import numpy as np

from common import SEED, load_split
from model import forward, init_params, softmax

WIDTHS = [16, 64]
TOL = 1e-6


def main() -> None:
    X_train, X_test, *_ = load_split()
    batches = {"1 sample": X_train[:1], "mini-batch 32": X_train[:32], "full test set": X_test}

    all_ok = True
    for w in WIDTHS:
        params = init_params(w, np.random.default_rng(SEED))
        print(f"\n=== W = {w} ===")
        for name, X in batches.items():
            P, cache = forward(params, X)
            shape_ok = P.shape == (X.shape[0], 10)
            max_dev = np.abs(P.sum(axis=1) - 1).max()
            sum_ok = max_dev < TOL
            all_ok &= shape_ok and sum_ok
            print(f"  {name:<14} X {str(X.shape):<10} -> P {str(P.shape):<10} "
                  f"shape {'OK' if shape_ok else 'FAIL'}  max|rowsum-1| = {max_dev:.1e} "
                  f"{'OK' if sum_ok else 'FAIL'}")

        _, cache = forward(params, X_train[:32])
        print("  layer by layer (batch 32):")
        for k in ["X", "Z1", "A1", "Z2", "A2", "Z3", "P"]:
            v = cache[k]
            print(f"    {k:<3} {str(v.shape):<9} mean|.| = {np.abs(v).mean():.2e}")
        print(f"  share of ReLU units that are zero: A1 {np.mean(cache['A1'] == 0):.0%}, "
              f"A2 {np.mean(cache['A2'] == 0):.0%}")
        print(f"  P[0] = {np.round(cache['P'][0], 4)}")

    print("\nstable softmax on huge logits [1000, 1001, 1002]:",
          np.round(softmax(np.array([[1000.0, 1001.0, 1002.0]])), 4))
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp([1000.0, 1001.0, 1002.0]) / np.exp([1000.0, 1001.0, 1002.0]).sum()
    print("naive softmax on the same logits:           ", naive)

    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
