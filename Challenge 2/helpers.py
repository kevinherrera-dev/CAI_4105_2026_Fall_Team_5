"""Shared setup for Challenge 2 (Team 5). Every teammate's notebook imports from here,
so all models use the same data, preprocessing, folds and scoring.

Usage (from a notebook in the `Challenge 2/` folder):

    from helpers import load_data, evaluate, results_table
    from sklearn.neighbors import KNeighborsClassifier

    X, y = load_data()
    rows = [evaluate("KNN", KNeighborsClassifier(), X, y, scale=True)]
    results_table(rows)

`scale=True` for models that care about feature scale (Logistic Regression, KNN, SVM),
`scale=False` for tree models (Decision Tree, Random Forest, Gradient Boosting).
For Naive Bayes either works; we use scale=True.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent / "Data"
TARGET = "Revenue"
ID_COL = "ID"

NUM_COLS = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
# OperatingSystems, Browser, Region and TrafficType are number codes, not amounts, so they are categories.
CAT_COLS = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]

# Same 5 folds for everyone. Stratified = every fold keeps the same ~15.5% share of buyers.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Primary metric: F1 for Revenue=True (y == 1). zero_division=0 avoids warnings for models that never predict a buyer.
SCORING = {
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "accuracy": "accuracy",
}


def load_data():
    """Training data -> (X, y). X has the 17 feature columns (no ID); y is Revenue as 0/1 (1 = purchase)."""
    df = pd.read_csv(DATA_DIR / "challenge2_train.csv")
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(int)
    return X, y


def load_test():
    """Test data -> (X_test, ids). Use ONLY for the final predictions (Step 6), never for training or tuning."""
    df = pd.read_csv(DATA_DIR / "challenge2_test.csv")
    return df[NUM_COLS + CAT_COLS], df[ID_COL]


def make_preprocessor(scale=True):
    """Numeric: fill blanks with the median (+ standardize if scale=True). Categorical: fill blanks with the
    most common value + one-hot encode; categories not seen in training become all zeros."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", Pipeline(cat_steps), CAT_COLS),
    ])


def make_model_pipeline(model, scale=True):
    """Preprocessing + model in ONE pipeline, so every CV fold learns the scaling/encoding from its own
    training part only (no information leaks from the validation part). Use this inside GridSearchCV:
    parameter names get the prefix "model__", e.g. {"model__C": [0.1, 1, 10]}."""
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("model", model)])


def evaluate(name, model, X, y, scale=True):
    """5-fold stratified CV of preprocessing + model. Returns one results row (a dict):
    train and CV F1 (the gap shows overfitting), CV precision, recall and accuracy."""
    pipe = make_model_pipeline(model, scale=scale)
    s = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, return_train_score=True, n_jobs=-1)
    return {
        "model": name,
        "train_f1": s["train_f1"].mean(),
        "cv_f1": s["test_f1"].mean(),
        "cv_f1_std": s["test_f1"].std(),
        "cv_precision": s["test_precision"].mean(),
        "cv_recall": s["test_recall"].mean(),
        "cv_accuracy": s["test_accuracy"].mean(),
        "fit_time_s": s["fit_time"].mean(),
    }


def results_table(rows):
    """List of evaluate() rows -> table sorted by CV F1 (best first)."""
    return pd.DataFrame(rows).set_index("model").sort_values("cv_f1", ascending=False).round(3)


def cv_confusion_matrix(model, X, y, scale=True):
    """Confusion matrix from out-of-fold predictions (each row predicted by a model that never saw it).
    Rows = actual, columns = predicted."""
    pred = cross_val_predict(make_model_pipeline(model, scale=scale), X, y, cv=CV, n_jobs=-1)
    labels = ["no purchase (0)", "purchase (1)"]
    return pd.DataFrame(confusion_matrix(y, pred), index=[f"actual {l}" for l in labels],
                        columns=[f"predicted {l}" for l in labels])
