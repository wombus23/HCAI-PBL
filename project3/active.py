"""
Task 4: active learning for expert competence discovery.

The setting changes. The classifier can still be trained on the whole labelled
training set, but there are no expert answers at all. Every expert label has to
be paid for by asking, and the question is which articles to ask about so that
the system learns where deferring is worth it as quickly as possible.

Four strategies are compared under an identical budget:

- **random** — the control. Any strategy that cannot beat it is not doing
  anything.
- **classifier uncertainty** — the textbook active learning baseline: ask about
  the articles the classifier is least sure of. It is the obvious choice and it
  is subtly wrong here, because the quantity being learned is not the label but
  the expert's competence, and the two are not the same thing.
- **expert uncertainty** — ask where the current estimate of P(expert correct)
  is closest to 0.5, that is, where the competence model itself is least sure.
- **deferral margin** — ask where P(expert correct) and P(classifier correct)
  are closest together, so the deferral decision is on the point of flipping.
  This is the only one of the four that targets the decision the system actually
  has to make: knowing the expert's competence precisely in a region where the
  classifier is at 99% changes nothing, because the system will keep those
  articles either way.

The first round is random for every strategy, since with no answers in hand
there is no model to be uncertain with. Results are averaged over five seeds.
"""

import numpy as np

from .data import SEED, classifier, corpus, features
from .defer import _auc, calibrated_confidence, fit_rejector
from .expert import expert

STRATEGIES = ["random", "classifier_uncertainty", "expert_uncertainty", "deferral_margin"]
STRATEGY_LABELS = {
    "random": "Random",
    "classifier_uncertainty": "Classifier uncertainty",
    "expert_uncertainty": "Expert uncertainty",
    "deferral_margin": "Deferral margin",
}

ROUNDS = 20
BATCH = 50
SEEDS = [0, 1, 2, 3, 4]


def _choose(strategy, pool, n, p_expert_pool, p_clf_pool, rng):
    """Which articles to ask about next."""
    if strategy == "random" or p_expert_pool is None:
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    if strategy == "classifier_uncertainty":
        score = p_clf_pool                               # least confident first
    elif strategy == "expert_uncertainty":
        score = np.abs(p_expert_pool - 0.5)              # closest to a coin flip
    else:
        score = np.abs(p_expert_pool - p_clf_pool)       # deferral about to flip

    return pool[np.argsort(score)[:n]]


def run(rounds=ROUNDS, batch=BATCH, seeds=SEEDS):
    """One curve per strategy: team accuracy against number of expert queries."""
    data, f = corpus(), features()
    e, conf = expert(), calibrated_confidence()
    fitted = classifier()

    clf_correct_test = (fitted.predict(f["X_test"]) == data["y_test"]).astype(int)
    exp_correct_test = e["test_correct"]
    p_clf_test = conf["test"]
    p_clf_train = conf["train"]

    n_train = f["X_train"].shape[0]
    baseline = float(clf_correct_test.mean())
    ceiling = float(np.maximum(clf_correct_test, exp_correct_test).mean())

    results = {}
    for strategy in STRATEGIES:
        accuracy = np.zeros((len(seeds), rounds))
        auc = np.zeros((len(seeds), rounds))
        deferral = np.zeros((len(seeds), rounds))

        for s, seed in enumerate(seeds):
            rng = np.random.default_rng(1000 + seed)
            labelled = np.zeros(n_train, dtype=bool)
            p_expert_pool = None

            for r in range(rounds):
                pool = np.where(~labelled)[0]
                picked = _choose(strategy, pool, batch, p_expert_pool,
                                 p_clf_train[pool] if p_expert_pool is not None else None,
                                 rng)
                labelled[picked] = True

                rejector = fit_rejector(np.where(labelled)[0])
                if rejector is None:
                    # Every answer so far agrees; nothing to learn yet, so the
                    # system keeps everything and scores the classifier alone.
                    accuracy[s, r] = baseline
                    auc[s, r] = 0.5
                    deferral[s, r] = 0.0
                    continue

                p_expert_test = rejector.predict_proba(f["X_test"])[:, 1]
                defer = p_expert_test > p_clf_test
                accuracy[s, r] = np.where(defer, exp_correct_test, clf_correct_test).mean()
                auc[s, r] = _auc(exp_correct_test, p_expert_test)
                deferral[s, r] = defer.mean()

                remaining = np.where(~labelled)[0]
                p_expert_pool = np.full(n_train, np.nan)
                p_expert_pool[remaining] = rejector.predict_proba(f["X_train"][remaining])[:, 1]
                p_expert_pool = p_expert_pool[remaining]

        # The curve starts at zero queries, where there is no competence model
        # at all and the system can only keep every article for itself.
        results[strategy] = {
            "queries": [0] + [(r + 1) * batch for r in range(rounds)],
            "accuracy": [round(baseline, 4)] + [round(float(v), 4) for v in accuracy.mean(axis=0)],
            "accuracy_std": [0.0] + [round(float(v), 4) for v in accuracy.std(axis=0)],
            "auc": [0.5] + [round(float(v), 4) for v in auc.mean(axis=0)],
            "deferral_rate": [0.0] + [round(float(v), 4) for v in deferral.mean(axis=0)],
        }

    return {
        "rounds": rounds,
        "batch": batch,
        "seeds": list(seeds),
        "baseline": round(baseline, 4),
        "ceiling": round(ceiling, 4),
        "strategies": results,
        "labels": STRATEGY_LABELS,
    }


def summarise(active, full_information):
    """How much of the full information result each strategy buys, and how fast.

    `full_information` is the task 3 team accuracy, which was reached with an
    expert answer for every one of the 24,000 training articles. Here the budget
    is 1,000 answers, so the interesting quantity is what share of that gain a
    strategy recovers, not whether it matches it outright.
    """
    baseline = active["baseline"]
    span = max(1e-9, full_information - baseline)
    three_quarters = baseline + 0.75 * span

    rows = []
    for strategy, curve in active["strategies"].items():
        accuracies = np.array(curve["accuracy"])
        queries = np.array(curve["queries"])
        matched = np.where(accuracies >= full_information)[0]
        near = np.where(accuracies >= three_quarters)[0]
        rows.append({
            "strategy": STRATEGY_LABELS[strategy],
            "final": curve["accuracy"][-1],
            "final_auc": curve["auc"][-1],
            "final_deferral": curve["deferral_rate"][-1],
            "share_of_gain": round(float((accuracies[-1] - baseline) / span), 3),
            "queries_to_match": int(queries[matched[0]]) if len(matched) else None,
            "queries_to_75": int(queries[near[0]]) if len(near) else None,
            "area": round(float(np.trapezoid(accuracies, queries) / (queries[-1] - queries[0])), 4)
            if len(queries) > 1 else None,
        })
    return sorted(rows, key=lambda r: -r["final"])
