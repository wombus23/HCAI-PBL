"""
Task 3: learning to defer.

With expert labels available for the training set, the system has to decide, per
article, whether to answer itself or hand over. The rule used here compares two
estimates of who is more likely to be right:

    defer(x)  <=>  P(expert correct | x) - P(classifier correct | x) > t

with t = 0 as the natural operating point and t swept to trace the whole
coverage curve.

Two design choices are worth stating.

The classifier's own maximum probability is not used raw as P(classifier
correct). A linear model on TF-IDF is overconfident, so it is calibrated first:
out of fold predictions on the training set give an honest correctness signal,
and a one dimensional logistic regression maps confidence to a probability of
being right. Skipping this step makes the system defer far too little, because
both sides of the comparison are then on different scales.

P(expert correct | x) is a logistic regression on the same TF-IDF features,
trained on the expert's answers. It never sees the region labels, so it has to
recover the competence profile from the text.

Evaluation reports more than team accuracy, as the sheet asks: how often the
system defers, how accurate each side is on the part it kept, and how close the
deferral decisions come to the oracle rule of deferring exactly when the expert
is right and the classifier is wrong.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .data import SEED, classifier, corpus, features
from .expert import expert

REJECTOR = dict(C=1.0, max_iter=2000, random_state=SEED)


def calibrated_confidence():
    """P(classifier correct | x) on the training and test sets.

    The mapping is fitted on out of fold predictions, so the model is never
    judged on articles it was fitted on.
    """
    data, f = corpus(), features()
    model = LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)
    out_of_fold = cross_val_predict(model, f["X_train"], data["y_train"],
                                    cv=3, method="predict_proba")
    train_conf = out_of_fold.max(axis=1)
    train_correct = (out_of_fold.argmax(axis=1) == data["y_train"]).astype(int)

    mapping = LogisticRegression(max_iter=1000)
    mapping.fit(train_conf.reshape(-1, 1), train_correct)

    fitted = classifier()
    test_conf = fitted.predict_proba(f["X_test"]).max(axis=1)
    return {
        "train": mapping.predict_proba(train_conf.reshape(-1, 1))[:, 1],
        "test": mapping.predict_proba(test_conf.reshape(-1, 1))[:, 1],
        "raw_train": train_conf,
        "raw_test": test_conf,
        "mapping": mapping,
        "oof_accuracy": float(train_correct.mean()),
    }


def fit_rejector(rows=None):
    """P(expert correct | x), from the expert's answers on `rows` of the train set."""
    f, e = features(), expert()
    X = f["X_train"] if rows is None else f["X_train"][rows]
    y = e["train_correct"] if rows is None else e["train_correct"][rows]
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(**REJECTOR).fit(X, y)


def _summary(defer, clf_correct, exp_correct):
    """The numbers that describe one deferral policy."""
    n = len(defer)
    system = np.where(defer, exp_correct, clf_correct)
    only_expert = (exp_correct == 1) & (clf_correct == 0)
    only_clf = (clf_correct == 1) & (exp_correct == 0)

    caught = int((defer & only_expert).sum())
    lost = int((defer & only_clf).sum())
    return {
        "accuracy": round(float(system.mean()), 4),
        "deferral_rate": round(float(defer.mean()), 4),
        "n_deferred": int(defer.sum()),
        "accuracy_deferred": round(float(exp_correct[defer].mean()), 4) if defer.any() else None,
        "accuracy_retained": round(float(clf_correct[~defer].mean()), 4) if (~defer).any() else None,
        # of the articles only the expert gets right, how many did we hand over
        "recall_of_gains": round(caught / max(1, int(only_expert.sum())), 4),
        # of the articles we handed over, how many did the expert actually win
        "precision": round(float(exp_correct[defer].mean()), 4) if defer.any() else None,
        "damage": round(lost / max(1, n), 4),
    }


def evaluate(threshold=0.0):
    """Task 3 in one call: the policy, the baselines and the coverage curves."""
    data, f = corpus(), features()
    e, conf = expert(), calibrated_confidence()
    fitted = classifier()

    clf_correct = (fitted.predict(f["X_test"]) == data["y_test"]).astype(int)
    exp_correct = e["test_correct"]

    rejector = fit_rejector()
    p_expert = rejector.predict_proba(f["X_test"])[:, 1]
    p_clf = conf["test"]
    margin = p_expert - p_clf

    learned = _summary(margin > threshold, clf_correct, exp_correct)
    rate = learned["deferral_rate"]

    # Baselines at the same deferral rate, so the comparison is fair.
    rng = np.random.default_rng(SEED)
    k = int(round(rate * len(clf_correct)))
    random_defer = np.zeros(len(clf_correct), dtype=bool)
    random_defer[rng.choice(len(clf_correct), k, replace=False)] = True

    confidence_defer = np.zeros(len(clf_correct), dtype=bool)
    confidence_defer[np.argsort(p_clf)[:k]] = True

    oracle = (exp_correct == 1) & (clf_correct == 0)

    curves = _coverage_curves(margin, p_clf, clf_correct, exp_correct, rng)

    ceiling = float(np.maximum(clf_correct, exp_correct).mean())
    floor = float(clf_correct.mean())
    gain = (learned["accuracy"] - floor) / max(1e-9, ceiling - floor)

    return {
        "threshold": threshold,
        "learned": learned,
        "random": _summary(random_defer, clf_correct, exp_correct),
        "confidence_only": _summary(confidence_defer, clf_correct, exp_correct),
        "oracle": _summary(oracle, clf_correct, exp_correct),
        "classifier_only": round(floor, 4),
        "expert_only": round(float(exp_correct.mean()), 4),
        "ceiling": round(ceiling, 4),
        "normalised_gain": round(float(gain), 4),
        "rejector_auc": round(float(_auc(exp_correct, p_expert)), 4),
        "curves": curves,
        "calibration": _calibration_table(conf, clf_correct),
    }


def _coverage_curves(margin, p_clf, clf_correct, exp_correct, rng):
    """Team accuracy as a function of how much work is handed over."""
    n = len(clf_correct)
    rates = np.linspace(0, 1, 21)
    order_learned = np.argsort(-margin)          # most worth deferring first
    order_confidence = np.argsort(p_clf)         # least confident first
    order_oracle = np.argsort(-((exp_correct == 1) & (clf_correct == 0)).astype(float)
                              - 0.5 * (exp_correct == 1))
    shuffled = rng.permutation(n)

    def walk(order):
        out = []
        for rate in rates:
            defer = np.zeros(n, dtype=bool)
            defer[order[:int(round(rate * n))]] = True
            out.append(round(float(np.where(defer, exp_correct, clf_correct).mean()), 4))
        return out

    return {
        "rates": [round(float(r), 3) for r in rates],
        "learned": walk(order_learned),
        "confidence_only": walk(order_confidence),
        "random": walk(shuffled),
        "oracle": walk(order_oracle),
    }


def _calibration_table(conf, clf_correct, bins=8):
    """Raw confidence against how often the classifier is actually right."""
    raw = conf["raw_test"]
    edges = np.quantile(raw, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for i in range(len(edges) - 1):
        mask = (raw >= edges[i]) & (raw <= edges[i + 1] if i == len(edges) - 2 else raw < edges[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "confidence": round(float(raw[mask].mean()), 3),
            "calibrated": round(float(conf["test"][mask].mean()), 3),
            "actual": round(float(clf_correct[mask].mean()), 3),
            "n": int(mask.sum()),
        })
    return rows


def _auc(y, scores):
    """Rank based AUC, written out to avoid importing yet another helper."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
