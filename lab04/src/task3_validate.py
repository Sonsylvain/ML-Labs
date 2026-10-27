"""Task 3: schema, range, null-policy and duplicate checks. Violations are reported, no row is dropped."""

import pandas as pd
from pandas.api.types import is_integer_dtype

from common import ALLOWED, CATEGORICAL, MISSING, RANGES, RESULTS_DIR, TARGET, TYPES, as_schema, fetch_raw

if __name__ == "__main__":
    raw = fetch_raw()
    df = as_schema(raw)
    checks, violations = [], []

    # Columns and types (types are checked on the raw frame as served)
    raw_types = raw.rename(columns={"class": TARGET})
    for col, kind in TYPES.items():
        if col not in raw_types:
            checks.append(("type", col, kind, "missing column", 1))
            continue
        ok = is_integer_dtype(raw_types[col]) if kind == "int" else isinstance(raw_types[col].dtype, pd.CategoricalDtype)
        checks.append(("type", col, kind, str(raw_types[col].dtype), int(not ok)))
    extra = sorted(set(raw_types.columns) - set(TYPES))
    checks.append(("columns", "unexpected columns", "none", ", ".join(extra) or "none", len(extra)))

    # Ranges
    for col, (lo, hi) in RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        checks.append(("range", col, f"{lo}-{hi}", f"min {df[col].min()}, max {df[col].max()}", len(bad)))
        violations += [(i, col, v, f"outside {lo}-{hi}") for i, v in bad[col].items()]

    # Allowed categories ("?" allowed) and missing counts
    for col in CATEGORICAL + [TARGET]:
        bad = df[~df[col].isin(ALLOWED[col])]
        checks.append(("category", col, f"{len(ALLOWED[col])} allowed values incl. '?'",
                       f"{df[col].nunique()} distinct", len(bad)))
        violations += [(i, col, v, "value not in data dictionary") for i, v in bad[col].items()]
    missing = {c: int((df[c] == MISSING).sum()) for c in CATEGORICAL + [TARGET]}
    missing |= {c: int(df[c].isna().sum()) for c in TYPES if c not in missing}
    for col, n in missing.items():
        rule = "'?' allowed" if col in CATEGORICAL + [TARGET] else "no NaN"
        checks.append(("missing", col, rule, f"{n} rows ({n / len(df):.2%})", 0 if rule != "no NaN" else n))

    # Duplicates: counted, kept (cleaning decides what to do, on train only)
    n_dup = int(df.duplicated().sum())
    checks.append(("duplicates", "exact duplicate rows", "reported", f"{n_dup} rows", 0))

    report = pd.DataFrame(checks, columns=["check", "column", "rule", "observed", "violations"])
    report.to_csv(RESULTS_DIR / "validation_report.csv", index=False)
    pd.DataFrame(violations, columns=["row", "column", "value", "reason"]).to_csv(
        RESULTS_DIR / "validation_violations.csv", index=False)

    lines = ["# Validation report", "",
             f"Rows checked: {len(df)} (none dropped). Total rule violations: {int(report['violations'].sum())}. "
             f"Rows with missing values ('?'): {int((df == MISSING).any(axis=1).sum())}. "
             f"Exact duplicate rows: {n_dup}.", "",
             "| Check | Column | Rule | Observed | Violations |", "|---|---|---|---|---:|"]
    lines += [f"| {r.check} | {r.column} | {r.rule} | {r.observed} | {r.violations} |" for r in report.itertuples()]
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
