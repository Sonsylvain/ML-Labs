# Lab 4 — Data pipeline and quality

UCI Adult, scikit-learn 1.5.0, pandas 2.2.2. The whole pipeline runs with one command from the repository root:

```bash
python lab04/src/run_all.py
```

| File | Content |
|---|---|
| `config.yaml` | every pipeline setting (seed, split, cleaning, model, audit size) |
| `src/common.py` | schema, raw loading, split, cleaning transformer, derived features |
| `src/task2_ingest.py` … `src/task7_hash.py` | one script per task |
| `results/` | validation report, cleaning summary, metrics, audits, `preprocessor.joblib`, `run_config.yaml` |
| `data_card.md` | source, license, size, biases, privacy |

## 1. Problem definition

**Task.** The task is binary classification: predict whether a person's annual income is above $50K from 1994 US Census attributes.
The dataset is UCI Adult, loaded with `fetch_openml(name='adult', version=2, as_frame=True)`.

**Inputs.** There are 14 features. Six are numeric: age, fnlwgt, education-num, capital-gain, capital-loss and hours-per-week. Eight are categorical: workclass, education, marital-status, occupation, relationship, race, sex and native-country.
Three categorical features have missing values, written as `?`.

**Output.** `income`, with the positive class `>50K` (23.9% of rows) and the negative class `<=50K`.

**Success metrics.** F1 on the positive class and accuracy, both measured on a held-out test set. Accuracy alone would be misleading because of the 76/24 class imbalance: predicting `<=50K` for everyone already scores 76%.
F1 shows whether the model actually finds the high-income minority.

**Constraints.**
- *No leakage.* The data is split 70/30 (stratified) before any cleaning. The imputers, clipping bounds, encoder and scaler are fitted on train only. Duplicates are removed from train only, and the test set is never inspected while fitting.
- *Reproducible pipeline.* All settings are in `config.yaml` and the seed is fixed (42). The raw data is hashed at ingestion and verified again at the end. The processed dataset gets a SHA256 that is stored with the config, and one command rebuilds everything.

The raw data is never modified. Every step works on copies.

## 2. Ingestion

`task2_ingest.py` reads the OpenML frame as served and writes `results/ingestion_log.json`.

| Rows | Columns | Raw content SHA256 |
|---:|---:|---|
| 48,842 | 15 | `02a7bbc4a27c04e1d48193362caf1e3cecb4b700c4c09fff7ffbd11a70db11a2` |

The script asserts the expected 48,842 × 15. `task7_hash.py` re-hashes the raw frame at the end and fails if it changed.

## 3. Validation

The full report is in `results/validation_report.md` (CSV versions: `validation_report.csv`, `validation_violations.csv`).
The schema is the one given in the manual. The allowed category values come from the UCI `adult.names` data dictionary, with `?` added. The OpenML target column is named `class` and is mapped to the schema name `income`.
The OpenML parser already turns `?` into NaN, so the pipeline converts it back to `?` and applies the null policy itself.

| Check | Result |
|---|---|
| Types (15 columns) | all match: 6 × int64, 9 × category |
| Ranges: age 17–90, hours 1–99, gain 0–99999, loss 0–4356 | 0 violations (min/max sit exactly on the bounds) |
| Category values outside the data dictionary | 0 |
| Missing (`?`) | workclass 2,799 · occupation 2,809 · native-country 857. 3,620 rows have at least one |
| NaN in numeric columns | 0 |
| Exact duplicate rows | 52 (reported here and dropped later from train only) |

No row was dropped by validation. The checks found no violations, so the only data-quality issues are the missing values and the duplicates, and the cleaning step handles both.

## 4. Cleaning and transformation

Order: stratified 70/30 split (seed 42) → drop exact duplicates in train → fit the cleaner on train → transform train and test.
The cleaner (`ColumnTransformer`) applies a median imputer and then 1st/99th-percentile clipping to the numeric columns, and a most-frequent imputer to the categorical columns.
It is saved with `joblib.dump` to `results/preprocessor.joblib`.

| Item | Value |
|---|---:|
| Train rows after split → after dedup | 34,189 → 34,165 (24 dropped) |
| Test rows | 14,653 (no rows dropped) |
| Positive rate train / test | 23.94% / 23.93% |
| Cells imputed train / test | 4,446 / 2,013 |
| Numeric cells clipped train / test | 2,357 / 1,026 |

Bounds learned on train (`results/clip_bounds.csv`). The imputed modes are workclass Private, occupation Prof-specialty and native-country United-States (`results/imputer_modes.csv`).

| Feature | Median | 1st pct | 99th pct |
|---|---:|---:|---:|
| age | 37 | 17 | 74 |
| fnlwgt | 178,272 | 27,035.2 | 509,629 |
| education-num | 10 | 3 | 16 |
| capital-gain | 0 | 0 | 15,024 |
| capital-loss | 0 | 0 | 1,980 |
| hours-per-week | 40 | 8 | 80 |

The test set is transformed with these train statistics but is never deduplicated or used for fitting.
The clipping cuts capital-gain at 15,024, so the 99,999 values (a census top-code) no longer dominate the scale.

## 5. Derived features

- `capital_net = capital-gain − capital-loss`
- `hours_per_age = hours-per-week / (age − 16)` (age ≥ 17, so the denominator is ≥ 1)

Both are computed row by row from the cleaned values. Nothing is learned from the data, so they cannot carry test information into training.
The baseline is `LogisticRegression(max_iter=2000)` on standardised numeric columns plus one-hot categorical columns. The encoder and scaler are fitted on train. Results are in `results/baseline_metrics.csv`.

| Features | Inputs | Test accuracy | Test F1 |
|---|---:|---:|---:|
| Without derived features | 14 | 0.8540 | 0.6601 |
| With `capital_net` + `hours_per_age` | 16 | **0.8578** | **0.6696** |

The two features add 0.38 points of accuracy and 0.95 points of F1. `capital_net` merges the two sparse capital columns into one signed value. `hours_per_age` relates working hours to career stage, which a linear model cannot build from the two separate columns.

## 6. Label-quality check

Adult's labels come from census records, not from human annotators, so the manual's Adult variant is used: a duplicate-conflict audit plus a data-dictionary re-check. Cohen's kappa does not apply.

**Duplicate-conflict audit** (all 48,842 rows, `results/duplicate_conflicts.csv`). 111 rows share their 14 feature values with at least one other row.
Of these, **5 groups (10 rows)** have identical features but **conflicting income labels** (one `<=50K`, one `>50K` in each pair).
These are label noise, or two different people who look identical in the data, and no model can classify both rows of a pair correctly.
They make up 0.02% of the data. They are reported and kept, and the conflicting rows that fall in the test set are not touched.

**50-row data-dictionary check** (`results/label_audit_sample.csv`, `df.sample(50, random_state=42)`). Each row was checked by script and then read by hand against `adult.names`. The rules:
- the ranges from Task 3
- education matches education-num (e.g. HS-grad = 9, Bachelors = 13)
- Husband → Male and Wife → Female
- Husband/Wife → a Married-* marital status
- capital-gain and capital-loss are not both positive

**Result: 0 impossible values in the 50 rows.** Reading the rows by hand turned up some unusual but possible combinations. Row 23881 is married with relationship Own-child (a married adult child living with their parents), and row 3375 works 4 hours per week. Neither breaks the data dictionary, so neither was changed.

## 7. Hash, config and data card

`task7_hash.py` writes the processed dataset to `data/adult/adult_processed.csv`, which is git-ignored and rebuilt by the pipeline. The file holds train and test with a `split` column, the original row index, the 14 cleaned features, the 2 derived features and the 0/1 income, and has 48,818 rows × 19 columns.
The script hashes the file bytes and writes `results/run_config.yaml`, which contains the full config, the raw hash and the processed hash:

```
processed sha256: 35d1279dfc962ed5fb66ca19d7ce298b4bc6271428ba75d5286e9fa2c6f0547f
```

Two runs of `python lab04/src/run_all.py` produced the same hash.

The data card is in [`data_card.md`](data_card.md): source, license (CC BY 4.0), size, known biases (class, sex, race, geography, 1994 dollars, label noise) and privacy concerns.
