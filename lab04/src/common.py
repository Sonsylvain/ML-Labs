"""Shared setup for Lab 4: config, paths, schema, raw loading, split and the cleaning transformer."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LAB_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_DIR.parent / "data" / "adult"  # git-ignored
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = yaml.safe_load((LAB_DIR / "config.yaml").read_text())
SEED = CONFIG["seed"]
TARGET = CONFIG["dataset"]["target"]
POSITIVE = CONFIG["dataset"]["positive_label"]

# Schema (Task 3). "?" is the missing marker and is allowed in every categorical column.
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
TYPES = {c: "int" for c in NUMERIC} | {c: "category" for c in CATEGORICAL} | {TARGET: "category"}
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
MISSING = "?"
# Allowed values from the UCI adult.names data dictionary, plus "?".
ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate",
                  "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                       "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                       "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                       "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                       "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    TARGET: ["<=50K", ">50K"],
}
ALLOWED = {c: v + [MISSING] for c, v in ALLOWED.items()}
EDUCATION_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
                 "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
                 "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}


def fetch_raw() -> pd.DataFrame:
    """The dataset exactly as OpenML serves it (cached under data/adult/openml)."""
    bunch = fetch_openml(name=CONFIG["dataset"]["name"], version=CONFIG["dataset"]["version"],
                         as_frame=True, parser="auto", data_home=str(DATA_DIR / "openml"))
    return bunch.frame


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (values + index), used to prove the raw data is never modified."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def as_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy with the schema's target name and the "?" missing marker made explicit.

    The OpenML parser already turned "?" into NaN; this puts "?" back so the null policy is
    applied by this pipeline, not hidden inside the loader. The raw frame is not touched."""
    df = raw.rename(columns={"class": TARGET}).copy()
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].astype(object).where(df[c].notna(), MISSING)
    return df


def split(df: pd.DataFrame):
    """70/30 stratified split, done before any cleaning."""
    return train_test_split(df, test_size=CONFIG["split"]["test_size"], random_state=SEED,
                            stratify=df[TARGET] if CONFIG["split"]["stratify"] else None)


def to_xy(df: pd.DataFrame):
    """Features with "?" turned into NaN for the imputers, and the 0/1 target."""
    X = df[FEATURES].copy()
    X[CATEGORICAL] = X[CATEGORICAL].replace(MISSING, np.nan)
    return X, (df[TARGET] == POSITIVE).astype(int)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to bounds learned from the training data only."""

    def __init__(self, quantiles=(0.01, 0.99)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_, self.upper_ = np.quantile(X, self.quantiles, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_cleaner() -> ColumnTransformer:
    """Imputation (median / mode) and percentile clipping; output keeps the feature names."""
    numeric = Pipeline([("impute", SimpleImputer(strategy=CONFIG["cleaning"]["numeric_imputer"])),
                        ("clip", QuantileClipper(tuple(CONFIG["cleaning"]["clip_quantiles"])))])
    categorical = SimpleImputer(strategy=CONFIG["cleaning"]["categorical_imputer"])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
                             verbose_feature_names_out=False).set_output(transform="pandas")


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Task 5 features. Row-wise only, nothing is learned, so they cannot leak test information."""
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def prepare():
    """Raw -> schema -> split -> train-only dedup -> cleaner fitted on train."""
    data = as_schema(fetch_raw())
    train, test = split(data)
    n_train_before = len(train)
    if CONFIG["cleaning"]["drop_train_duplicates"]:
        train = train.drop_duplicates()
    X_train, y_train = to_xy(train)
    X_test, y_test = to_xy(test)
    cleaner = build_cleaner().fit(X_train)
    return {"cleaner": cleaner, "n_train_before": n_train_before,
            "X_train": cleaner.transform(X_train), "y_train": y_train,
            "X_test": cleaner.transform(X_test), "y_test": y_test,
            "train_raw": train, "test_raw": test}
