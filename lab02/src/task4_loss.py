"""Task 4: cross-entropy loss; untrained network should score about ln(10) = 2.3026."""

import numpy as np

from common import SEED, load_split
from model import cross_entropy, forward, init_params

WIDTHS = [16, 64]
TARGET = np.log(10)
TOL = 0.01


def main() -> None:
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_split()
    print(f"target ln(10) = {TARGET:.4f}")

    all_ok = True
    for w in WIDTHS:
        params = init_params(w, np.random.default_rng(SEED))
        for split, X, Y, y in [("train", X_train, Y_train, y_train), ("test", X_test, Y_test, y_test)]:
            P, _ = forward(params, X)
            loss = cross_entropy(P, Y)
            acc = np.mean(P.argmax(axis=1) == y)
            ok = abs(loss - TARGET) < TOL
            all_ok &= ok
            print(f"W={w:<3} {split:<5} loss = {loss:.4f}  (diff {loss - TARGET:+.1e}) "
                  f"{'OK' if ok else 'FAIL'}   accuracy = {acc:.3f}")

    print("\nhow the loss rewards confidence in the TRUE class (true class = 3):")
    Y = np.zeros((1, 10)); Y[0, 3] = 1
    for p_true in [0.99, 0.9, 0.5, 0.1, 0.01]:
        P = np.full((1, 10), (1 - p_true) / 9); P[0, 3] = p_true
        print(f"  P(true) = {p_true:<5} -> loss = {cross_entropy(P, Y):.4f}")

    P = np.zeros((1, 10)); P[0, 7] = 1.0
    print(f"  P(true) = 0     -> loss = {cross_entropy(P, Y):.2f}  (finite thanks to EPS; -log(0) = inf)")

    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
