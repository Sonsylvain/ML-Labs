"""Task 5: LogisticRegression baseline with and without the two derived features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATEGORICAL, CONFIG, NUMERIC, RESULTS_DIR, SEED, add_derived, prepare


def evaluate(X_train, y_train, X_test, y_test, numeric):
    model = Pipeline([
        ("encode", ColumnTransformer([("num", StandardScaler(), numeric),
                                      ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("lr", LogisticRegression(max_iter=CONFIG["model"]["max_iter"], random_state=SEED)),
    ]).fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred)


if __name__ == "__main__":
    p = prepare()
    rows = []
    acc, f1 = evaluate(p["X_train"], p["y_train"], p["X_test"], p["y_test"], NUMERIC)
    rows.append(("without derived features", len(NUMERIC) + len(CATEGORICAL), acc, f1))
    acc, f1 = evaluate(add_derived(p["X_train"]), p["y_train"], add_derived(p["X_test"]), p["y_test"],
                       NUMERIC + ["capital_net", "hours_per_age"])
    rows.append(("with capital_net + hours_per_age", len(NUMERIC) + len(CATEGORICAL) + 2, acc, f1))
    out = pd.DataFrame(rows, columns=["features", "n_input_features", "test_accuracy", "test_f1"]).round(4)
    out.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    print(out.to_string(index=False))
