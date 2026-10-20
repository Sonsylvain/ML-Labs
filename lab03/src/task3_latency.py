"""Task 3b: inference latency, batch 1 (median of 100 runs) and batch 128 (median of 20 runs)."""

import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS_DIR, set_seeds
from models import build_models

PROTOCOL = {1: 100, 128: 20}  # batch size -> timed runs
WARMUP = 10


def median_ms(model, x, runs: int) -> float:
    with torch.inference_mode():
        for _ in range(WARMUP):  # allocator, thread pool and caches settle first
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def main() -> None:
    set_seeds()
    print(f"torch {torch.__version__}, CPU, {torch.get_num_threads()} threads")
    rows = []
    for name, model in build_models().items():
        model.eval()
        for batch, runs in PROTOCOL.items():
            ms = median_ms(model, torch.randn(batch, 1, 28, 28), runs)
            rows.append({"model": name, "batch": batch, "runs": runs, "median_ms": ms,
                         "ms_per_sample": ms / batch})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    df.to_csv(RESULTS_DIR / "latency.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
