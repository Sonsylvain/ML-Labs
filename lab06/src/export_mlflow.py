"""Export every Lab 6 MLflow run (the store itself is git-ignored) to results/mlflow_runs.csv.

Browse the full runs with:  mlflow ui --backend-store-uri lab06/mlruns   (from the repository root)
"""

import mlflow
import pandas as pd

from common import RESULTS_DIR, load_config, repo_path

KEEP = ["experiment", "tags.mlflow.runName", "status", "run_id", "params.optimizer.name", "params.optimizer.lr",
        "params.scheduler.name", "params.training.batch_size", "params.training.accumulation_steps",
        "params.training.epochs", "params.data.num_workers", "metrics.val_acc", "metrics.test_acc",
        "metrics.throughput", "metrics.train_epoch_s", "metrics.peak_rss_increase_mb", "tags.resumed_at_epoch"]

if __name__ == "__main__":
    cfg = load_config()
    mlflow.set_tracking_uri(repo_path(cfg["mlflow"]["tracking_uri"]).as_uri())
    exps = {e.experiment_id: e.name for e in mlflow.search_experiments() if e.name.startswith("lab06")}
    df = mlflow.search_runs(experiment_ids=list(exps), order_by=["start_time ASC"])
    df["experiment"] = df.experiment_id.map(exps)
    df = df[[c for c in KEEP if c in df.columns]]
    df.to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False, float_format="%.4f")
    print(df.drop(columns=["run_id"]).to_string(index=False))
