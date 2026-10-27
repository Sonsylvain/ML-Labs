"""Task 4: split before cleaning, train-only dedup, train-fitted imputer and clipping, save the transformer."""

import joblib
import pandas as pd

from common import NUMERIC, RESULTS_DIR, TARGET, prepare

if __name__ == "__main__":
    p = prepare()
    cleaner = p["cleaner"]
    joblib.dump(cleaner, RESULTS_DIR / "preprocessor.joblib")

    clip = cleaner.named_transformers_["num"].named_steps["clip"]
    num_imp = cleaner.named_transformers_["num"].named_steps["impute"]
    cat_imp = cleaner.named_transformers_["cat"]
    pd.DataFrame({"feature": NUMERIC, "median": num_imp.statistics_,
                  "p01": clip.lower_, "p99": clip.upper_}).to_csv(RESULTS_DIR / "clip_bounds.csv", index=False)
    pd.DataFrame({"feature": cat_imp.feature_names_in_, "mode": cat_imp.statistics_}).to_csv(
        RESULTS_DIR / "imputer_modes.csv", index=False)

    train_raw, test_raw = p["train_raw"], p["test_raw"]
    summary = pd.DataFrame([
        ("train rows after split", p["n_train_before"]),
        ("train duplicates dropped", p["n_train_before"] - len(train_raw)),
        ("train rows after dedup", len(train_raw)),
        ("test rows (untouched)", len(test_raw)),
        ("train positive rate", round((train_raw[TARGET] == ">50K").mean(), 4)),
        ("test positive rate", round((test_raw[TARGET] == ">50K").mean(), 4)),
        ("train cells imputed", int(train_raw.eq("?").sum().sum())),
        ("test cells imputed", int(test_raw.eq("?").sum().sum())),
        ("train numeric cells clipped", int((p["X_train"][NUMERIC].values != train_raw[NUMERIC].values).sum())),
        ("test numeric cells clipped", int((p["X_test"][NUMERIC].values != test_raw[NUMERIC].values).sum())),
    ], columns=["item", "value"]).astype({"value": object})
    summary.to_csv(RESULTS_DIR / "cleaning_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.read_csv(RESULTS_DIR / "clip_bounds.csv").to_string(index=False))
