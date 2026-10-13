"""Task 1: load digits, scale by 1/16, one-hot encode, 70/30 stratified split."""

import numpy as np

from common import load_split


def main() -> None:
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_split()

    print(f"X_train {X_train.shape}  X_test {X_test.shape}")
    print(f"Y_train {Y_train.shape}  Y_test {Y_test.shape}")
    print(f"pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"one-hot rows sum to 1: {np.allclose(Y_train.sum(axis=1), 1)}")
    print(f"label 0 of train = {y_train[0]} -> one-hot {Y_train[0].astype(int)}")

    print("\nclass share (train vs test) -- stratification keeps them equal:")
    for c in range(10):
        print(f"  digit {c}: {np.mean(y_train == c):.3f}  {np.mean(y_test == c):.3f}")


if __name__ == "__main__":
    main()
