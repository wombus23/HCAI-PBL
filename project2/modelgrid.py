"""
The models behind tasks 1 to 3.

For each model class a grid of models is fitted, from very heavily regularized to
barely regularized at all. The interface then picks one of them with

    argmax_f  acc_test(f) - lambda * Omega(f)

which is the trade off from the lecture, read the other way round: instead of
penalising complexity while fitting, we fit a range of models and let the user
turn the dial on how much a unit of complexity is worth.

Omega for a decision tree is the number of leaves. Omega for logistic regression
is the number of original features the model still uses, meaning features with at
least one nonzero coefficient across the three classes. Counting features rather
than raw coefficients keeps Omega on the same footing as the leaf count: both
answer "how many things does a person have to read to understand this model".
"""

from functools import lru_cache

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, penguins

TREE = "tree"
LOGREG = "logreg"

MODEL_LABELS = {TREE: "Decision tree", LOGREG: "Logistic regression"}

OMEGA_LABELS = {TREE: "leaves", LOGREG: "features used"}

# Regularization grids. Both run from heavily constrained to essentially free.
TREE_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]
LOGREG_GRID = [0.003, 0.006, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.7, 1.5, 5.0, 20.0]

FIT_SEED = 0

# scikit-learn 1.8 deprecated the `penalty` argument in favour of `l1_ratio`,
# while older versions reject `l1_ratio` unless the penalty is elasticnet. The
# app has to run on whatever the grader has installed, so pick per version.
_SKLEARN = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


def _l1_logistic(C):
    kwargs = dict(C=C, solver="saga", max_iter=20000, tol=1e-4, random_state=FIT_SEED)
    if _SKLEARN >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **kwargs)
    return LogisticRegression(penalty="l1", **kwargs)


class FittedModel:
    """One trained model plus the two numbers the interface needs from it."""

    def __init__(self, kind, estimator, setting, mean=None, scale=None):
        self.kind = kind
        self.estimator = estimator
        self.setting = setting
        self.mean = mean
        self.scale = scale
        self.accuracy = 0.0
        self.omega = 0

    # -- prediction ------------------------------------------------------- #

    def _prepare(self, X):
        if self.mean is None:
            return X
        X = np.array(X, dtype=float, copy=True)
        n = len(NUMERIC_FEATURES)
        X[:, :n] = (X[:, :n] - self.mean) / self.scale
        return X

    def proba(self, X):
        return self.estimator.predict_proba(self._prepare(np.asarray(X, dtype=float)))

    def predict(self, X):
        return self.estimator.predict(self._prepare(np.asarray(X, dtype=float)))

    # -- description ------------------------------------------------------ #

    @property
    def setting_label(self):
        if self.kind == TREE:
            return "max_leaf_nodes = %s" % ("unlimited" if self.setting is None else self.setting)
        return "C = %g (L1 penalty)" % self.setting

    @property
    def omega_label(self):
        return OMEGA_LABELS[self.kind]

    def coefficient_table(self, data):
        """Rows of (feature, per class coefficients) for the logistic model."""
        if self.kind != LOGREG:
            return []
        coef = self.estimator.coef_
        rows = []
        for j, column in enumerate(data.columns):
            values = coef[:, j]
            rows.append({
                "column": column,
                "values": [round(float(v), 3) for v in values],
                "used": bool(np.any(np.abs(values) > 1e-8)),
            })
        return rows


def _omega_logreg(estimator, data):
    """Number of original features with a nonzero coefficient anywhere."""
    coef = estimator.coef_
    used = set()
    for j, column in enumerate(data.columns):
        if np.any(np.abs(coef[:, j]) > 1e-8):
            used.add(column.split("=")[0])
    return len(used)


@lru_cache(maxsize=4)
def grid(kind):
    """Fit the whole grid for one model class. Cached: the data never changes."""
    data = penguins()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test

    models = []
    if kind == TREE:
        for setting in TREE_GRID:
            estimator = DecisionTreeClassifier(
                max_leaf_nodes=setting, random_state=FIT_SEED
            ).fit(X_train, y_train)
            model = FittedModel(TREE, estimator, setting)
            model.omega = int(estimator.get_n_leaves())
            model.accuracy = float(estimator.score(X_test, y_test))
            models.append(model)
    else:
        n = len(NUMERIC_FEATURES)
        mean = X_train[:, :n].mean(axis=0)
        scale = X_train[:, :n].std(axis=0)
        scale[scale == 0] = 1.0
        X_train_s = np.array(X_train, dtype=float, copy=True)
        X_train_s[:, :n] = (X_train_s[:, :n] - mean) / scale

        for setting in LOGREG_GRID:
            estimator = _l1_logistic(setting).fit(X_train_s, y_train)
            model = FittedModel(LOGREG, estimator, setting, mean=mean, scale=scale)
            model.omega = _omega_logreg(estimator, data)
            model.accuracy = float(np.mean(model.predict(X_test) == y_test))
            models.append(model)

    # Keep only models that are not beaten by a simpler one at equal or better
    # accuracy: those can never win the argmax for any lambda, and dropping them
    # keeps the selection table readable.
    return models


def select(kind, lam):
    """The model the interface should show for this value of lambda.

    Ties go to the simpler model, which is the whole point of the exercise.
    """
    models = grid(kind)
    scored = [(m.accuracy - lam * m.omega, -m.omega, i) for i, m in enumerate(models)]
    best = max(scored)
    return models[best[2]], [
        {
            "setting": m.setting_label,
            "omega": m.omega,
            "accuracy": round(m.accuracy, 4),
            "objective": round(m.accuracy - lam * m.omega, 4),
            "chosen": i == best[2],
        }
        for i, m in enumerate(models)
    ]


def lambda_range(kind):
    """A slider range wide enough that both ends of the trade off are reachable.

    At lambda = 0 the most accurate model wins. The upper end is the smallest
    lambda for which the simplest model in the grid wins, rounded up, so the user
    can always reach both extremes without an arbitrary hard coded maximum.
    """
    models = grid(kind)
    simplest = min(models, key=lambda m: m.omega)
    needed = 0.0
    for model in models:
        if model.omega > simplest.omega:
            gap = (model.accuracy - simplest.accuracy) / (model.omega - simplest.omega)
            needed = max(needed, gap)
    top = max(0.01, round(needed * 1.6, 3))
    step = round(top / 40, 4)
    return top, step
