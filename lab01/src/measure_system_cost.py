"""Task 3: measure training time, single-sample latency, model size and peak memory.

Each model is measured in its own fresh Python subprocess so that memory allocated by one
model (or by the allocator's caches) never inflates the numbers of the other.

Protocol (from the lab manual):
  * training time   : 1 warm-up fit, then median of 5 timed fits (fresh clone each time)
  * inference       : single sample, 1 warm-up, then median of 100 timed predict() calls
  * model size      : joblib.dump to results/model_<name>.joblib, bytes and KB
  * peak memory     : psutil RSS sampled every 1 ms by a background thread during
                      training and during inference (plus the OS peak working set)
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone

from common import RESULTS_DIR, load_split, make_models, set_seeds

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
MB = 1024**2


class PeakRSSSampler:
    """Context manager that records the peak RSS of this process while active."""

    def __init__(self, interval_s: float = 0.001):
        self.proc = psutil.Process()
        self.interval_s = interval_s
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval_s)

    def __enter__(self):
        self.start_rss = self.proc.memory_info().rss
        self.peak = self.start_rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def delta(self) -> int:
        return self.peak - self.start_rss


def count_parameters(model) -> int:
    """Number of stored values the model needs at inference time."""
    if hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size)
    if hasattr(model, "estimators_"):
        # Per node: feature, threshold, left, right, and one value per class.
        n_classes = len(model.classes_)
        return int(sum(t.tree_.node_count * (4 + n_classes) for t in model.estimators_))
    return -1


def measure(name: str) -> dict:
    set_seeds()
    proc = psutil.Process()
    rss_runtime = proc.memory_info().rss  # interpreter + numpy/sklearn/pandas, before data
    X_train, X_test, y_train, y_test = load_split()
    template = make_models()[name]

    # Warm-up fit: discarded for timing, but used for the (cold, worst-case) memory peak.
    with PeakRSSSampler() as train_mem:
        clone(template).fit(X_train, y_train)

    train_times = []
    for _ in range(N_TRAIN_RUNS):
        model = clone(template)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

    x_one = X_test[:1]
    model.predict(x_one)  # warm-up
    infer_times = []
    for _ in range(N_INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x_one)
        infer_times.append(time.perf_counter() - t0)

    # Separate pass for memory so the sampler thread cannot perturb the latency numbers.
    with PeakRSSSampler() as infer_mem:
        for _ in range(N_INFER_RUNS):
            model.predict(x_one)

    model_path = RESULTS_DIR / f"model_{name}.joblib"
    joblib.dump(model, model_path)
    size_bytes = model_path.stat().st_size

    infer_ms = np.array(infer_times) * 1e3
    mem = proc.memory_info()
    return {
        "model": name,
        "train_time_median_ms": float(np.median(train_times) * 1e3),
        "train_time_min_ms": float(np.min(train_times) * 1e3),
        "train_time_max_ms": float(np.max(train_times) * 1e3),
        "infer_latency_median_ms": float(np.median(infer_ms)),
        "infer_latency_p95_ms": float(np.percentile(infer_ms, 95)),
        "model_size_bytes": int(size_bytes),
        "model_size_kb": size_bytes / 1024,
        "n_parameters": count_parameters(model),
        "runtime_rss_mb": rss_runtime / MB,
        "train_peak_rss_mb": train_mem.peak / MB,
        "train_delta_rss_mb": train_mem.delta / MB,
        "infer_peak_rss_mb": infer_mem.peak / MB,
        "infer_delta_rss_mb": infer_mem.delta / MB,
        "os_peak_working_set_mb": getattr(mem, "peak_wset", mem.rss) / MB,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="measure a single model in this process")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(measure(args.worker)))
        return

    rows = []
    for name in make_models():
        print(f"Measuring {name} in a fresh process ...")
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name],
            check=True, capture_output=True, text=True,
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))

    df = pd.DataFrame(rows).round(4)
    out_csv = RESULTS_DIR / "system_cost.csv"
    df.to_csv(out_csv, index=False)
    print(df.T.to_string(header=False))
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
