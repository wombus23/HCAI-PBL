"""
The study protocol, in one place.

The interface, the report and the analysis all read these numbers from here, so
the document describing the study and the software running it cannot drift apart.

Budgets are matched on participant time rather than on number of tasks. Under the
pilot timing assumptions (8 s per pairwise comparison, 100 s per ranking of ten)
both conditions come to roughly four to five minutes of elicitation, which is the
comparison the primary hypothesis is about. Matching on tasks instead would be
meaningless, and matching on implied pairwise comparisons would quietly favour
the ranking design, since a ranking of ten yields 45 implied pairs but only nine
actual decisions.
"""

import hashlib

import numpy as np

RANKING_SIZE = 10

FULL = {
    "pairwise_tasks": 30,
    "ranking_tasks": 3,
    "holdout_pairs": 10,
}

DEMO = {
    "pairwise_tasks": 4,
    "ranking_tasks": 1,
    "holdout_pairs": 4,
}

SECONDS_PAIRWISE = 8.0
SECONDS_RANKING = 100.0

CONDITION_LABELS = {
    "pairwise": "Design 1 — choose one of two",
    "ranking": "Design 2 — rank ten",
}


def plan(condition, is_demo=False):
    """How many elicitation tasks and held out pairs this participant gets."""
    settings = DEMO if is_demo else FULL
    tasks = settings["pairwise_tasks"] if condition == "pairwise" else settings["ranking_tasks"]
    return tasks, settings["holdout_pairs"]


def task_size(condition):
    return 2 if condition == "pairwise" else RANKING_SIZE


def rng_for(code, phase, position):
    """A deterministic generator per screen.

    The films on a given screen have to survive a page refresh, so they are
    derived from the participant's code and the position in the sequence rather
    than drawn fresh on every render.
    """
    digest = hashlib.sha256(("%s|%s|%d" % (code, phase, position)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


ORDERS = ["pairwise_first", "ranking_first"]


def assign_order(counts):
    """Counterbalanced assignment: whichever order is behind, ties at random.

    Every participant does both designs, so what is randomized is the order.
    Simple randomization can leave the two orders unbalanced by chance at these
    sample sizes, and since order is a factor in the analysis, an unbalanced
    assignment loses power for nothing.
    """
    first = counts.get("pairwise_first", 0)
    second = counts.get("ranking_first", 0)
    if first < second:
        return "pairwise_first"
    if second < first:
        return "ranking_first"
    return ORDERS[int(np.random.default_rng().random() < 0.5)]
