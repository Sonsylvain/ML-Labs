"""Task 6: duplicate-conflict audit and a 50-row check against the data dictionary."""

import json

import pandas as pd

from common import CONFIG, EDUCATION_NUM, FEATURES, RANGES, RESULTS_DIR, SEED, TARGET, as_schema, fetch_raw


def dictionary_problems(r) -> list[str]:
    """Values that are impossible under the UCI adult.names data dictionary."""
    problems = [f"{c}={r[c]} outside {lo}-{hi}" for c, (lo, hi) in RANGES.items() if not lo <= r[c] <= hi]
    if EDUCATION_NUM.get(r["education"]) != r["education-num"]:
        problems.append(f"education={r['education']} but education-num={r['education-num']}")
    if r["relationship"] == "Husband" and r["sex"] != "Male":
        problems.append("Husband but sex Female")
    if r["relationship"] == "Wife" and r["sex"] != "Female":
        problems.append("Wife but sex Male")
    if r["relationship"] in ("Husband", "Wife") and not r["marital-status"].startswith("Married"):
        problems.append(f"{r['relationship']} but marital-status {r['marital-status']}")
    if r["capital-gain"] > 0 and r["capital-loss"] > 0:
        problems.append("both capital-gain and capital-loss > 0")
    return problems


if __name__ == "__main__":
    df = as_schema(fetch_raw())

    # Duplicate-conflict audit: identical 14 features, different income label
    labels_per_group = df.groupby(FEATURES, observed=True)[TARGET].transform("nunique")
    conflicts = df[labels_per_group > 1].sort_values(FEATURES)
    n_groups = conflicts.groupby(FEATURES, observed=True).ngroups
    conflicts.to_csv(RESULTS_DIR / "duplicate_conflicts.csv", index_label="row")

    # 50 random rows checked against the data dictionary
    sample = df.sample(n=CONFIG["label_audit"]["sample_size"], random_state=SEED).copy()
    sample["problems"] = ["; ".join(dictionary_problems(r)) for _, r in sample.iterrows()]
    sample.to_csv(RESULTS_DIR / "label_audit_sample.csv", index_label="row")

    summary = {"duplicate_feature_rows": int(df.duplicated(FEATURES, keep=False).sum()),
               "conflicting_groups": int(n_groups), "conflicting_rows": int(len(conflicts)),
               "sample_rows_checked": len(sample),
               "sample_rows_with_impossible_values": int((sample["problems"] != "").sum())}
    (RESULTS_DIR / "label_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    flagged = sample[sample["problems"] != ""]
    print(flagged[["problems"]].to_string() if len(flagged) else "no impossible values in the sample")
