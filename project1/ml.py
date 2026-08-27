"""
Machine learning helpers for Project 1.

Everything that is not django lives here: reading the CSV, guessing whether we
are looking at a classification or a regression problem, drawing the figures and
running the hyperparameter sweep. Keeping it separate from views.py means the
pipeline can be tested without a running server.

Only numpy, pandas, matplotlib and scikit-learn are used.
"""

import os
import time
import uuid

import matplotlib

# Must be selected before pyplot is imported: the django dev server has no GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.conf import settings

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFICATION = "classification"
REGRESSION = "regression"

PLOT_SUBDIR = "project1"

# Columns that are row identifiers rather than features.
ID_LIKE_NAMES = {"id", "index", "unnamed: 0", "no", "sample", "row"}

# A one hot encoded feature with more levels than this is dropped instead.
MAX_CATEGORY_LEVELS = 20


# --------------------------------------------------------------------------- #
#  Loading and inspecting the data
# --------------------------------------------------------------------------- #

class DatasetError(Exception):
    """Raised when a CSV cannot be used for supervised learning."""


def load_dataframe(path, drop_ids=True):
    """Read a CSV and return (dataframe, list of dropped id columns).

    The convention from the project description: the first row holds the
    feature names and the last column holds the label.
    """
    try:
        df = pd.read_csv(path, skipinitialspace=True)
    except Exception as exc:
        raise DatasetError("The file could not be read as a CSV: %s" % exc)

    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] < 2:
        raise DatasetError(
            "The file needs at least two columns: one feature and one label."
        )
    if len(df) < 10:
        raise DatasetError("The file needs at least 10 rows to train on.")

    dropped = []
    if drop_ids:
        for column in list(df.columns)[:-1]:
            if _is_id_column(df, column):
                dropped.append(column)
        df = df.drop(columns=dropped)

    return df, dropped


def _is_id_column(df, column):
    """True when a column looks like a row identifier."""
    if str(column).strip().lower() in ID_LIKE_NAMES:
        return True
    series = df[column]
    if not pd.api.types.is_integer_dtype(series):
        return False
    if series.nunique() != len(series):
        return False
    values = np.sort(series.to_numpy())
    return np.array_equal(values, np.arange(values[0], values[0] + len(values)))


def infer_task(series):
    """Guess whether a target column is a classification or a regression target.

    The rule is deliberately simple so it can be explained to the user, who can
    always override it on the training page.
    """
    clean = series.dropna()
    if clean.empty:
        raise DatasetError("The label column is empty.")

    if not pd.api.types.is_numeric_dtype(clean):
        return CLASSIFICATION
    if pd.api.types.is_bool_dtype(clean):
        return CLASSIFICATION

    n_unique = clean.nunique()
    if pd.api.types.is_float_dtype(clean):
        # Floats that only ever take a handful of whole values are still labels.
        integral = np.allclose(clean.to_numpy(), np.round(clean.to_numpy()))
        return CLASSIFICATION if (integral and n_unique <= 10) else REGRESSION

    return CLASSIFICATION if n_unique <= max(20, 0.05 * len(clean)) else REGRESSION


def task_explanation(series, task):
    """A short sentence justifying the automatic detection, shown in the UI."""
    n_unique = series.nunique()
    if task == CLASSIFICATION:
        if not pd.api.types.is_numeric_dtype(series):
            return "the label column holds text, so it was read as %d classes" % n_unique
        return "the label column holds only %d distinct whole numbers" % n_unique
    return "the label column holds %d distinct numeric values spread over a range" % n_unique


def summarise(df, target):
    """Per column summary shown as a table on the explore page."""
    rows = []
    for column in df.columns:
        series = df[column]
        row = {
            "name": column,
            "role": "label" if column == target else "feature",
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                mean=_fmt(series.mean()),
                std=_fmt(series.std()),
                minimum=_fmt(series.min()),
                maximum=_fmt(series.max()),
            )
        else:
            row.update(mean="–", std="–", minimum="–", maximum="–")
        rows.append(row)
    return rows


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return "%.3g" % value


def numeric_columns(df, exclude=()):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #

PALETTE = ["#275CB2", "#E4572E", "#2A9D8F", "#F4A259", "#7B4B94",
           "#0B7A75", "#C1121F", "#3D5A80", "#8FBC94", "#5C4742"]


def _save(fig, prefix):
    """Write a figure into the media directory and return its URL."""
    folder = os.path.join(settings.MEDIA_ROOT, PLOT_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    filename = "%s-%s.png" % (prefix, uuid.uuid4().hex[:10])
    fig.savefig(os.path.join(folder, filename), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "%s%s/%s" % (settings.MEDIA_URL, PLOT_SUBDIR, filename)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scatter(df, x, y, target, task):
    """Two features against each other, coloured by the label."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if task == CLASSIFICATION:
        for i, (label, group) in enumerate(df.groupby(target, observed=True)):
            ax.scatter(group[x], group[y], s=26, alpha=0.85,
                       color=PALETTE[i % len(PALETTE)], label=str(label),
                       edgecolors="white", linewidths=0.4)
        ax.legend(title=target, frameon=False, fontsize=9)
    else:
        points = ax.scatter(df[x], df[y], c=df[target], cmap="viridis",
                            s=26, alpha=0.9, edgecolors="white", linewidths=0.4)
        fig.colorbar(points, ax=ax, label=target)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("%s against %s" % (y, x))
    _style(ax)
    return _save(fig, "scatter")


def plot_feature_vs_target(df, x, target):
    """One feature against a continuous label, for regression problems."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(df[x], df[target], s=26, alpha=0.85, color=PALETTE[0],
               edgecolors="white", linewidths=0.4)
    ax.set_xlabel(x)
    ax.set_ylabel(target)
    ax.set_title("%s against %s" % (target, x))
    _style(ax)
    return _save(fig, "feature-target")


def plot_distribution(df, column, target, task):
    """Histogram of one feature, split by class when there is one."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if task == CLASSIFICATION and column != target:
        for i, (label, group) in enumerate(df.groupby(target, observed=True)):
            ax.hist(group[column].dropna(), bins=20, alpha=0.6,
                    color=PALETTE[i % len(PALETTE)], label=str(label))
        ax.legend(title=target, frameon=False, fontsize=9)
    else:
        ax.hist(df[column].dropna(), bins=25, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title("Distribution of %s" % column)
    _style(ax)
    return _save(fig, "distribution")


def plot_target_balance(df, target, task):
    """Class counts, or the spread of a continuous label."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if task == CLASSIFICATION:
        counts = df[target].value_counts().sort_index()
        colours = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
        ax.bar([str(i) for i in counts.index], counts.to_numpy(), color=colours)
        ax.set_ylabel("count")
        ax.set_title("How many rows per class")
    else:
        ax.hist(df[target].dropna(), bins=25, color=PALETTE[0], alpha=0.85)
        ax.set_ylabel("count")
        ax.set_title("Spread of %s" % target)
    ax.set_xlabel(target)
    _style(ax)
    return _save(fig, "balance")


def plot_correlation(df, target):
    """Correlation heatmap over the numeric columns."""
    columns = numeric_columns(df)
    if len(columns) < 2:
        return None
    matrix = df[columns].corr().to_numpy()
    size = max(4.5, 0.55 * len(columns) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, "%.2f" % matrix[i, j], ha="center", va="center",
                    fontsize=7, color="white" if abs(matrix[i, j]) > 0.55 else "#222")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation between numeric columns")
    return _save(fig, "correlation")


def plot_score_curve(values, train_scores, test_scores, hyperparameter, score_label,
                     best_index, lower_is_better=False):
    """The point of the sweep: score as a function of the hyperparameter.

    The best value is passed in rather than recomputed here, because for scores
    such as RMSE the best model is the one with the lowest number.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(values))
    ax.plot(positions, train_scores, marker="o", color=PALETTE[3],
            linestyle="--", label="training set")
    ax.plot(positions, test_scores, marker="o", color=PALETTE[0], label="test set")
    ax.scatter([positions[best_index]], [test_scores[best_index]], s=160,
               facecolors="none", edgecolors=PALETTE[1], linewidths=2, zorder=5,
               label="best on test set")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(hyperparameter)
    ax.set_ylabel("%s%s" % (score_label, " (lower is better)" if lower_is_better else ""))
    ax.set_title("%s for each value of %s" % (score_label, hyperparameter))
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, "curve")


def plot_confusion(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred)
    size = max(4.2, 0.5 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > threshold else "#222")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Test set predictions of the best model")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _save(fig, "confusion")


def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(y_true, y_pred, s=28, alpha=0.8, color=PALETTE[0],
               edgecolors="white", linewidths=0.4)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], color=PALETTE[1], linestyle="--",
            linewidth=1.4, label="perfect prediction")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Test set predictions of the best model")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, "predictions")


# --------------------------------------------------------------------------- #
#  Algorithms and scores
# --------------------------------------------------------------------------- #

def _int_grid(values):
    return [int(v) for v in values]


ALGORITHMS = {
    "knn_clf": {
        "label": "k nearest neighbours",
        "task": CLASSIFICATION,
        "estimator": KNeighborsClassifier,
        "needs_scaling": True,
        "parameter": "n_neighbors",
        "parameter_label": "Number of neighbours (k)",
        "parameter_type": "int",
        "default_grid": [1, 3, 5, 7, 9, 13, 19, 25],
        "note": "Predicts the majority class among the k closest points. Small k follows the data closely, large k smooths it out.",
    },
    "tree_clf": {
        "label": "Decision tree",
        "task": CLASSIFICATION,
        "estimator": DecisionTreeClassifier,
        "needs_scaling": False,
        "parameter": "max_depth",
        "parameter_label": "Maximum depth",
        "parameter_type": "int",
        "default_grid": [1, 2, 3, 4, 5, 7, 10, 15],
        "note": "Splits the feature space with simple rules. Depth controls how detailed those rules get.",
    },
    "forest_clf": {
        "label": "Random forest",
        "task": CLASSIFICATION,
        "estimator": RandomForestClassifier,
        "needs_scaling": False,
        "parameter": "n_estimators",
        "parameter_label": "Number of trees",
        "parameter_type": "int",
        "default_grid": [5, 10, 25, 50, 100, 200],
        "note": "Averages many trees trained on different samples. More trees rarely hurt, they just cost time.",
    },
    "logreg_clf": {
        "label": "Logistic regression",
        "task": CLASSIFICATION,
        "estimator": LogisticRegression,
        "needs_scaling": True,
        "parameter": "C",
        "parameter_label": "Inverse regularisation strength (C)",
        "parameter_type": "float",
        "default_grid": [0.001, 0.01, 0.1, 1, 10, 100],
        "fixed": {"max_iter": 2000},
        "note": "A linear boundary. Small C keeps the weights small, large C lets the model fit harder.",
    },
    "svc_clf": {
        "label": "Support vector machine",
        "task": CLASSIFICATION,
        "estimator": SVC,
        "needs_scaling": True,
        "parameter": "C",
        "parameter_label": "Penalty on misclassified points (C)",
        "parameter_type": "float",
        "default_grid": [0.01, 0.1, 1, 10, 100],
        "note": "Finds the widest margin between classes. C decides how much the margin may be violated.",
    },
    "knn_reg": {
        "label": "k nearest neighbours",
        "task": REGRESSION,
        "estimator": KNeighborsRegressor,
        "needs_scaling": True,
        "parameter": "n_neighbors",
        "parameter_label": "Number of neighbours (k)",
        "parameter_type": "int",
        "default_grid": [1, 3, 5, 7, 9, 13, 19, 25],
        "note": "Averages the label of the k closest points.",
    },
    "tree_reg": {
        "label": "Decision tree",
        "task": REGRESSION,
        "estimator": DecisionTreeRegressor,
        "needs_scaling": False,
        "parameter": "max_depth",
        "parameter_label": "Maximum depth",
        "parameter_type": "int",
        "default_grid": [1, 2, 3, 4, 5, 7, 10, 15],
        "note": "Piecewise constant predictions. Deeper trees follow the training data more closely.",
    },
    "forest_reg": {
        "label": "Random forest",
        "task": REGRESSION,
        "estimator": RandomForestRegressor,
        "needs_scaling": False,
        "parameter": "n_estimators",
        "parameter_label": "Number of trees",
        "parameter_type": "int",
        "default_grid": [5, 10, 25, 50, 100, 200],
        "note": "Averages many trees. Steadier than a single tree at the cost of interpretability.",
    },
    "ridge_reg": {
        "label": "Ridge regression",
        "task": REGRESSION,
        "estimator": Ridge,
        "needs_scaling": True,
        "parameter": "alpha",
        "parameter_label": "Regularisation strength (alpha)",
        "parameter_type": "float",
        "default_grid": [0.001, 0.01, 0.1, 1, 10, 100],
        "note": "Linear regression that pulls the weights towards zero. Large alpha means a flatter model.",
    },
    "svr_reg": {
        "label": "Support vector regression",
        "task": REGRESSION,
        "estimator": SVR,
        "needs_scaling": True,
        "parameter": "C",
        "parameter_label": "Penalty on large errors (C)",
        "parameter_type": "float",
        "default_grid": [0.01, 0.1, 1, 10, 100],
        "note": "Ignores small errors and penalises the rest. C sets how strong that penalty is.",
    },
}

SCORES = {
    "accuracy": {
        "label": "Accuracy",
        "task": CLASSIFICATION,
        "sklearn": "accuracy",
        "invert": False,
        "note": "Share of correct predictions. Misleading when one class dominates.",
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "task": CLASSIFICATION,
        "sklearn": "balanced_accuracy",
        "invert": False,
        "note": "Accuracy averaged per class, so rare classes count as much as common ones.",
    },
    "f1_macro": {
        "label": "F1 (macro)",
        "task": CLASSIFICATION,
        "sklearn": "f1_macro",
        "invert": False,
        "note": "Balances precision and recall, averaged evenly over the classes.",
    },
    "r2": {
        "label": "R²",
        "task": REGRESSION,
        "sklearn": "r2",
        "invert": False,
        "note": "Share of the variance explained. 1 is perfect, 0 is no better than predicting the mean.",
    },
    "rmse": {
        "label": "RMSE",
        "task": REGRESSION,
        "sklearn": "neg_root_mean_squared_error",
        "invert": True,
        "note": "Typical error in the unit of the label. Lower is better.",
    },
    "mae": {
        "label": "Mean absolute error",
        "task": REGRESSION,
        "sklearn": "neg_mean_absolute_error",
        "invert": True,
        "note": "Average size of the error, less sensitive to outliers than RMSE. Lower is better.",
    },
}


def algorithms_for(task):
    return [(key, spec) for key, spec in ALGORITHMS.items() if spec["task"] == task]


def scores_for(task):
    return [(key, spec) for key, spec in SCORES.items() if spec["task"] == task]


# --------------------------------------------------------------------------- #
#  The training pipeline
# --------------------------------------------------------------------------- #

def build_matrix(df, target):
    """Split the frame into a feature matrix and a label vector.

    Numeric features keep their values, categorical features with few levels are
    one hot encoded, and anything with too many levels is dropped and reported.
    """
    features = [c for c in df.columns if c != target]
    numeric, categorical, ignored = [], [], []
    for column in features:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        elif df[column].nunique() <= MAX_CATEGORY_LEVELS:
            categorical.append(column)
        else:
            ignored.append(column)

    if not numeric and not categorical:
        raise DatasetError("None of the feature columns can be used for training.")

    X = df[numeric + categorical]
    y = df[target]
    if y.isna().any():
        keep = ~y.isna()
        X, y = X[keep], y[keep]
    return X, y, numeric, categorical, ignored


def _one_hot_encoder():
    """OneHotEncoder changed its keyword between sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(spec, value, numeric, categorical):
    steps = []
    if numeric:
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if spec["needs_scaling"]:
            numeric_steps.append(("scale", StandardScaler()))
        steps.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        steps.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _one_hot_encoder()),
        ]), categorical))

    kwargs = dict(spec.get("fixed", {}))
    kwargs[spec["parameter"]] = value
    estimator = spec["estimator"](**kwargs)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=0)

    return Pipeline([
        ("prepare", ColumnTransformer(steps)),
        ("model", estimator),
    ])


def run_sweep(df, target, algorithm, values, test_size, random_state, score_key,
              stratify=True):
    """Train one model per hyperparameter value and score them all.

    Returns a dictionary with the per value results, the best model and the
    figures. This is the whole Task 4 pipeline in one function.
    """
    spec = ALGORITHMS[algorithm]
    score = SCORES[score_key]
    scorer = get_scorer(score["sklearn"])

    X, y, numeric, categorical, ignored = build_matrix(df, target)

    classes = None
    if spec["task"] == CLASSIFICATION:
        y = y.astype(str)
        classes = sorted(y.unique())
        counts = y.value_counts()
        if counts.min() < 2:
            stratify = False

    split_on = y if (stratify and spec["task"] == CLASSIFICATION) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=split_on
    )

    rows, best, best_index = [], None, 0
    started = time.time()
    for index, value in enumerate(values):
        model = build_pipeline(spec, value, numeric, categorical)
        model.fit(X_train, y_train)
        train_raw = float(scorer(model, X_train, y_train))
        test_raw = float(scorer(model, X_test, y_test))
        rows.append({
            "value": value,
            "train": -train_raw if score["invert"] else train_raw,
            "test": -test_raw if score["invert"] else test_raw,
            "train_raw": train_raw,
            "test_raw": test_raw,
        })
        if best is None or test_raw > best["test_raw"]:
            best = {"value": value, "test_raw": test_raw, "model": model}
            best_index = index

    elapsed = time.time() - started

    curve_url = plot_score_curve(
        [r["value"] for r in rows],
        [r["train"] for r in rows],
        [r["test"] for r in rows],
        spec["parameter_label"],
        score["label"],
        best_index,
        lower_is_better=score["invert"],
    )

    predictions = best["model"].predict(X_test)
    if spec["task"] == CLASSIFICATION:
        detail_url = plot_confusion(y_test, predictions, classes)
    else:
        detail_url = plot_predictions(np.asarray(y_test, dtype=float),
                                      np.asarray(predictions, dtype=float))

    best_row = next(r for r in rows if r["value"] == best["value"])
    return {
        "rows": rows,
        "best_value": best["value"],
        "best_score": best_row["test"],
        "best_train_score": best_row["train"],
        "curve_url": curve_url,
        "detail_url": detail_url,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "numeric": numeric,
        "categorical": categorical,
        "ignored": ignored,
        "stratified": bool(split_on is not None),
        "seconds": round(elapsed, 2),
        "overfitting": _overfitting_hint(best_row, score),
    }


def _overfitting_hint(best_row, score):
    """A plain sentence about the gap between training and test performance."""
    train, test = best_row["train"], best_row["test"]
    if score["invert"]:
        if train <= 0:
            return None
        gap = (test - train) / max(abs(train), 1e-9)
    else:
        gap = (train - test) / max(abs(train), 1e-9)
    if gap > 0.15:
        return ("The best model scores clearly better on the data it was trained "
                "on than on the held out data, which is the usual sign of "
                "overfitting.")
    return None
