"""Task 1: run the YAML-driven pipeline once with config.yaml as is (the Lab 3 CNN recipe, 10 epochs)."""

import pandas as pd

from common import RESULTS_DIR, run_training

if __name__ == "__main__":
    s = run_training([])
    pd.DataFrame(s["history"]).to_csv(RESULTS_DIR / "baseline_history.csv", index=False, float_format="%.5f")
    print(f"baseline: test_acc {s['test_acc']:.4f}, MLflow run {s['run_id']}")
