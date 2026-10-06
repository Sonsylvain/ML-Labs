"""Task 2: train the two baselines and record test accuracy (4 decimals)."""

import pandas as pd
from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name:<20} accuracy = {acc:.4f}")

    out = RESULTS_DIR / "baseline_accuracy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
