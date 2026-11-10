"""Shared setup for Lab 5: seeds, paths and the make_moons split."""

import random
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_moons(n_samples: int = 200, noise: float = 0.15):
    """make_moons, 70/30 stratified split, float64. Returns X_train, X_test, y_train, y_test."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    return train_test_split(X, y.astype(np.float64), test_size=0.30, stratify=y, random_state=SEED)
