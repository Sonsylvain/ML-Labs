# Lab 1 — Environment and first system measurements

## 1. Goal

Set up a reproducible, pinned Python environment, train two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, measure their **system cost** (training time,
single-sample inference latency, model size, peak memory), and decide which deployment
tiers (Cloud, Edge, Mobile, TinyML) each model could run on.

## 2. Method

**Environment.** Python 3.11.8 in an isolated `.venv`, all 25 packages installed at the exact
pinned versions (`results/versions.txt`, check: 25/25 match). The pinned set is not
satisfiable as published: `mlflow==2.14.1` requires `pyarrow<16` while the manual pins
`pyarrow==16.1.0`. The environment was installed with a resolver override
(`overrides.txt`) so that every package keeps its pinned version.

**Hardware.** Windows 11 Pro laptop, Intel Core (family 6, model 142, 4 cores / 8 threads),
15.8 GB RAM, CPU only.

**Data and models.** `load_breast_cancer(return_X_y=True)`: 569 samples, 30 features.
Stratified 70/30 split with `random_state=42` → 398 train / 171 test. Seeds 42 for
`random` and `numpy`. Torch is not used in this lab and is not imported, because importing
it would add hundreds of MB to the process memory being measured.

- Linear: `LogisticRegression(max_iter=1000, random_state=42)` on raw (unscaled) features, as specified.
- Ensemble: `RandomForestClassifier(n_estimators=100, random_state=42)`.

**Measurement protocol.** Each model is measured in its own fresh Python process so that
the memory of one model cannot leak into the numbers of the other.

| Quantity | How it was measured |
|---|---|
| Training time | 1 warm-up fit, then 5 timed fits on fresh clones (`time.perf_counter`); median reported |
| Inference latency | `predict()` on one test sample; 1 warm-up, then 100 timed calls; median (and p95) reported |
| Model size | `joblib.dump` to `results/model_<name>.joblib` (one file per model instead of a single `model.joblib`, so both sizes are kept); file size in bytes and KB |
| Peak memory | `psutil.Process().memory_info().rss` sampled every 1 ms by a background thread during training and during 100 inference calls; also the OS peak working set |
| Runtime baseline | RSS of the process after imports and before loading data |

The memory sampling runs in a separate pass from the latency timing so that it cannot
slow down the timed calls. 1 MB = 1024² bytes.

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| Logistic Regression | **0.9474** |
| Random Forest | 0.9357 |

### System cost (`results/system_cost.csv`)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Training time, median of 5 (ms) | 883.1 | 495.0 |
| Training time, min–max (ms) | 826.7–916.2 | 432.6–562.8 |
| Single-sample latency, median of 100 (ms) | **0.237** | 12.259 |
| Single-sample latency, p95 (ms) | 0.328 | 32.457 |
| Model size on disk (bytes) | **1,055** | 290,889 |
| Model size on disk (KB) | **1.03** | 284.07 |
| Stored values (coefficients / tree-node fields) | 31 | 18,828 |
| Runtime RSS before data, Python + libraries (MB) | 142.9 | 143.0 |
| Peak RSS during training (MB) | 144.2 | 144.3 |
| Training memory above runtime baseline (MB) | 0.65 | 0.63 |
| Peak RSS during inference (MB) | 144.3 | 144.7 |
| Inference memory above baseline (MB) | 0.02 | 0.07 |

### Deployment fit (`results/deployment_fit.csv`)

Inputs to the check: peak RSS during inference, median single-sample latency, and model
file size. A tier counts as **fits** only if memory, latency and size are all within its budget.
Memory ranges are checked against the tier's ceiling. The column "smallest device"
repeats the memory check against the lower end of the range.

| Tier (memory / latency / size budget) | LogReg memory | LogReg latency | LogReg size | **LogReg fits** | RF memory | RF latency | RF size | **RF fits** |
|---|---|---|---|---|---|---|---|---|
| Cloud (≥ 1 GB / ≤ 100 ms / ≤ 500 MB) | 144 MB ✓ | 0.24 ms ✓ | 1.0 KB ✓ | **Yes** | 145 MB ✓ | 12.3 ms ✓ | 284 KB ✓ | **Yes** |
| Edge (256–1024 MB / ≤ 50 ms / ≤ 50 MB) | ✓ (also on 256 MB device) | ✓ | ✓ | **Yes** | ✓ (also on 256 MB device) | ✓ | ✓ | **Yes** |
| Mobile (64–256 MB / ≤ 20 ms / ≤ 10 MB) | ✓ ≤ 256 MB, ✗ on 64 MB device | ✓ | ✓ | **Yes** (mid/high-end) | ✓ ≤ 256 MB, ✗ on 64 MB device | ✓ (p95 32 ms ✗) | ✓ | **Yes, with caveats** |
| TinyML (≤ 256 KB / ≤ 10 ms / ≤ 100 KB) | ✗ 144 MB | ✓ | ✓ | **No** (as measured) | ✗ 145 MB | ✗ 12.3 ms | ✗ 284 KB | **No** |

**Where each model could run**

- **Logistic Regression: Cloud, Edge, Mobile.** It meets every latency and size budget
  with a lot of margin. Its latency of 0.24 ms is 40× below even the TinyML limit, and its
  1 KB file is 100× below the TinyML size limit. The only thing that fails TinyML is memory,
  and that 144 MB belongs to the Python/NumPy/scikit-learn runtime, not to the model. The
  model adds about 0.02 MB at inference time and needs only 31 numbers (124 bytes as float32)
  and about 30 multiply-adds per prediction. Exported to plain C, for example as
  hard-coded coefficients on a microcontroller, it would fit TinyML easily. In its current
  Python form it does not.
- **Random Forest: Cloud and Edge; Mobile with caveats; not TinyML.** On Mobile, its median
  latency (12.3 ms) is under the 20 ms limit, but the p95 (32.5 ms) is over it. A phone
  that is busy or thermally throttled would miss the budget for some requests. It fails
  TinyML on all three axes: the 284 KB file is 2.8× too large, the 12.3 ms latency is over
  10 ms, and the runtime memory is far too high. Its 18,828 stored tree-node values would
  not fit in 100 KB without heavy pruning or quantisation.
- **On the smallest Mobile devices (64 MB)** neither model fits as a Python process, because
  the interpreter and libraries alone use about 143 MB.

**Additional observation (not part of the specified baselines).** Logistic regression on
unscaled features hit the iteration limit: lbfgs used all 1000 iterations and raised a
`ConvergenceWarning`. That is why it trains *slower* (883 ms) than a 100-tree forest
(495 ms). A side check that added `StandardScaler` made it converge in 19 iterations,
training in about 18 ms with 0.9883 accuracy. The specified baseline was kept for the
results above.

## 4. Conclusions

> *Draft. Rewrite these in your own words before submitting.*

1. **The runtime, not the model, dominates memory.** Both models add less than 1 MB on
   top of about 143 MB of Python and library overhead. For small classical models, the
   deployment target is decided by the software stack you ship (Python vs. compiled C/ONNX
   runtime), and model memory is almost irrelevant. This is why TinyML needs a different
   toolchain and not just a smaller model.
2. **The simpler model wins on every system metric and on accuracy.** Logistic regression
   is about 50× faster per prediction (0.24 ms vs 12.3 ms) and about 275× smaller (1 KB vs
   284 KB) than the random forest, and it is more accurate (0.9474 vs 0.9357). The extra
   capacity of the forest buys nothing on this dataset, so logistic regression is the
   model to deploy on every tier.
3. **Medians hide tail behaviour, and preprocessing changes cost.** The forest's p95 latency
   is 2.6× its median and crosses the Mobile budget, which a single run or only the median
   would hide. Missing feature scaling made logistic regression fail to converge and train
   about 50× slower than necessary. System cost depends on the whole pipeline, not just the
   choice of algorithm.

## Reproducing

From the repository root, with `.venv` activated:

```
python lab01/src/run_all.py
```

This runs `print_versions.py`, `train_baselines.py`, `measure_system_cost.py` and
`deployment_fit.py`, and overwrites everything in `lab01/results/`. Timing numbers will
vary slightly between runs and machines.
