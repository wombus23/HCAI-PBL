"""
Task 2: a simulated expert who is good in some regions of the input space and
bad in others.

The project sheet asks for an expert that is not uniformly competent, with
expertise concentrated in specific regions rather than spread evenly. Tying
competence to the true label would be the easy version and would also make task
4 trivial: an active learner could work out the expert's profile from the
classifier's own predictions without ever asking a question. So competence is
attached to the k-means regions of the document space instead. Those regions
lean towards topics without being topics, which means the profile has to be
learned from queries rather than inferred from labels.

Two further details make the expert behave less like a coin flip:

- When the expert is wrong, the wrong answer is not uniform. It is drawn from a
  confusion profile, so a mistake on a Business article usually comes back as
  Sci/Tech rather than Sports. This is what makes deferral quality worth
  measuring: a wrong expert answer is a plausible wrong answer.
- Answers are precomputed once per article with a fixed seed, so querying the
  same article twice gives the same answer. An expert who changes their mind
  when asked again would make the active learning results meaningless.
"""

from functools import lru_cache

import numpy as np

from .data import CLASSES, N_CLUSTERS, SEED, clusters, corpus

# Per region accuracy. Written out rather than sampled, so the profile is a
# design decision that can be read off the page: four strong regions, four
# middling, four weak.
COMPETENCE = [0.95, 0.93, 0.90, 0.88,
              0.70, 0.65, 0.60, 0.55,
              0.35, 0.30, 0.28, 0.25]

# When wrong, which class the expert reaches for. Row = true class, and the
# weights are over the three wrong answers in class order.
CONFUSION = {
    0: {1: 0.10, 2: 0.55, 3: 0.35},   # World mistaken mostly for Business
    1: {0: 0.45, 2: 0.35, 3: 0.20},   # Sports for World
    2: {0: 0.35, 1: 0.10, 3: 0.55},   # Business for Sci/Tech
    3: {0: 0.25, 1: 0.10, 2: 0.65},   # Sci/Tech for Business
}


def _answer(y_true, region_ids, rng):
    """One deterministic answer per article."""
    answers = np.empty(len(y_true), dtype=int)
    correct = rng.random(len(y_true)) < np.array(COMPETENCE)[region_ids]
    for i in range(len(y_true)):
        if correct[i]:
            answers[i] = y_true[i]
        else:
            options = CONFUSION[int(y_true[i])]
            keys = list(options.keys())
            weights = np.array([options[k] for k in keys], dtype=float)
            answers[i] = keys[rng.choice(len(keys), p=weights / weights.sum())]
    return answers


@lru_cache(maxsize=1)
def expert():
    """The expert's answer for every article in the train and test sets."""
    data, regions = corpus(), clusters()
    rng = np.random.default_rng(SEED + 17)
    train = _answer(data["y_train"], regions["train"], rng)
    test = _answer(data["y_test"], regions["test"], rng)
    return {
        "train": train,
        "test": test,
        "train_correct": (train == data["y_train"]).astype(int),
        "test_correct": (test == data["y_test"]).astype(int),
    }


def profile():
    """Strengths and weaknesses, as the numbers that go in the report."""
    data, regions, e = corpus(), clusters(), expert()
    y, ids, correct = data["y_test"], regions["test"], e["test_correct"]

    by_region = []
    for k in range(N_CLUSTERS):
        mask = ids == k
        if mask.sum() == 0:
            continue
        labels, counts = np.unique(y[mask], return_counts=True)
        dominant = int(labels[np.argmax(counts)])
        by_region.append({
            "region": k,
            "size": int(mask.sum()),
            "designed": COMPETENCE[k],
            "accuracy": round(float(correct[mask].mean()), 3),
            "dominant": CLASSES[dominant],
            "purity": round(float(counts.max() / mask.sum()), 2),
        })

    by_class = []
    for c, name in enumerate(CLASSES):
        mask = y == c
        by_class.append({
            "label": name,
            "size": int(mask.sum()),
            "accuracy": round(float(correct[mask].mean()), 3),
        })

    return {
        "accuracy": round(float(correct.mean()), 4),
        "by_region": sorted(by_region, key=lambda r: -r["accuracy"]),
        "by_class": by_class,
        "train_accuracy": round(float(e["train_correct"].mean()), 4),
    }
