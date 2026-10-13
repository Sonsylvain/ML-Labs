"""Shared setup for Lab 2: seeds, paths and the digits data split."""

import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
N_CLASSES = 10  
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Integer labels (N,) -> one-hot matrix (N, n_classes)."""
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def load_split():
    """Digits 8x8, pixels scaled to [0, 1], 70/30 stratified split with random_state=42.

    Returns X_train, X_test, Y_train, Y_test (one-hot) and y_train, y_test (integer labels).
    """
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel intensities are 0..16
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test
