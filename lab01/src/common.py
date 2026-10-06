"""Shared setup for Lab 1: seeds, paths, data split and model definitions."""

import random
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Lab 1 does not use torch. Importing it only to seed it would add hundreds of MB
    # to the process RSS and corrupt the memory measurements, so seed it only if loaded.
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split with random_state=42."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_models() -> dict:
    """Fresh, unfitted baseline models keyed by a short name."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
