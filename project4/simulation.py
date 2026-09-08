"""
A simulation of the study, used to plan it.

The user study is not going to be run, but the design still has to be defensible:
how many tasks, why that many, and what effect size the analysis would be powered
to detect. Guessing those numbers is unnecessary when the model is fully
specified. This module generates synthetic participants from the Plackett-Luce
model, runs both elicitation designs on them at matched budgets, and measures how
well the fitted preference vector predicts held out choices.

Two ways of matching the budget, because they give opposite answers and the
choice between them is the sharpest design decision in the project:

- **Matched choice events.** A ranking of ten is nine sequential choices, a
  pairwise comparison is one. Matching here asks which design extracts more per
  decision the participant makes.
- **Matched time.** What a participant actually spends is minutes, not decisions.
  Under the timing assumptions below, five minutes buys either about 37 pairwise
  comparisons or about 3 rankings. This is the comparison the study is really
  about, and the one the primary hypothesis is stated in.

The timing constants are assumptions to be replaced by pilot measurements. They
are stated here rather than buried so that a reader can see exactly what the
planned analysis depends on.

The simulated participants are noisy: their held out answers are drawn from the
same Plackett-Luce model rather than taken as the deterministic argmax, so the
achievable accuracy has a ceiling well below 1. Comparing against that ceiling is
the only way the numbers mean anything.
"""

import json
import os

import numpy as np

from .catalogue import catalogue, features_for, sample_ids
from .preference import blocks_from_responses, evaluate_holdout, fit, pair_probability

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "simulation.json")

# Assumed seconds per task, to be replaced by the pilot.
SECONDS_PAIRWISE = 8.0
SECONDS_RANKING = 100.0
RANKING_SIZE = 10

EVENT_BUDGETS = [9, 18, 27, 36, 45, 63, 90]
TIME_BUDGETS = [120, 180, 240, 300, 420, 600]

N_PARTICIPANTS = 150
N_HOLDOUT = 200
TASTE_SIGMA = 0.8
SEED = 0


def _sample_ranking(ids, w, X, rng):
    """A Plackett-Luce draw: pick a favourite, then a favourite of the rest."""
    utilities = X[ids] @ w
    remaining = list(range(len(ids)))
    order = []
    while remaining:
        u = utilities[remaining]
        p = np.exp(u - u.max())
        p /= p.sum()
        choice = int(rng.choice(len(remaining), p=p))
        order.append(int(ids[remaining[choice]]))
        remaining.pop(choice)
    return order


def _holdout(w, X, rng, n=N_HOLDOUT):
    """Held out pairwise answers from the same noisy model, plus its ceiling."""
    pairs, ceiling = [], 0.0
    for _ in range(n):
        a, b = sample_ids(2, rng)
        p = pair_probability(w, X[a], X[b])
        pairs.append([a, b] if rng.random() < p else [b, a])
        ceiling += max(p, 1 - p)
    return pairs, ceiling / n


def _run_design(design, budget, unit, w, X, rng):
    """Elicit under one design at one budget, fit, and score on held out pairs."""
    if design == "pairwise":
        if unit == "events":
            n_tasks = budget
        else:
            n_tasks = int(budget / SECONDS_PAIRWISE)
        size = 2
    else:
        if unit == "events":
            n_tasks = max(1, int(round(budget / (RANKING_SIZE - 1))))
        else:
            n_tasks = max(1, int(budget / SECONDS_RANKING))
        size = RANKING_SIZE

    responses, shown = [], set()
    for _ in range(n_tasks):
        ids = sample_ids(size, rng, exclude=shown)
        shown.update(ids)
        responses.append(_sample_ranking(ids, w, X, rng))

    w_hat = fit(blocks_from_responses(responses, features_for), dimension=X.shape[1])
    return w_hat, n_tasks


def run(n_participants=N_PARTICIPANTS, seed=SEED):
    X = catalogue()["X"]
    dimension = X.shape[1]
    rng = np.random.default_rng(seed)

    grids = {"events": EVENT_BUDGETS, "time": TIME_BUDGETS}
    accumulator = {
        unit: {design: {b: [] for b in budgets} for design in ["pairwise", "ranking"]}
        for unit, budgets in grids.items()
    }
    ceilings = []
    paired = {unit: {b: [] for b in budgets} for unit, budgets in grids.items()}

    for _ in range(n_participants):
        w = rng.normal(0, TASTE_SIGMA, dimension)
        pairs, ceiling = _holdout(w, X, rng)
        ceilings.append(ceiling)

        for unit, budgets in grids.items():
            for budget in budgets:
                scores = {}
                for design in ["pairwise", "ranking"]:
                    w_hat, _ = _run_design(design, budget, unit, w, X, rng)
                    score = evaluate_holdout(w_hat, pairs, features_for)["accuracy"]
                    accumulator[unit][design][budget].append(score)
                    scores[design] = score
                paired[unit][budget].append(scores["ranking"] - scores["pairwise"])

    curves = {}
    for unit, budgets in grids.items():
        curves[unit] = {
            "budgets": budgets,
            "pairwise": [round(float(np.mean(accumulator[unit]["pairwise"][b])), 4) for b in budgets],
            "pairwise_se": [round(float(np.std(accumulator[unit]["pairwise"][b]) / np.sqrt(n_participants)), 4) for b in budgets],
            "ranking": [round(float(np.mean(accumulator[unit]["ranking"][b])), 4) for b in budgets],
            "ranking_se": [round(float(np.std(accumulator[unit]["ranking"][b]) / np.sqrt(n_participants)), 4) for b in budgets],
            "difference": [round(float(np.mean(paired[unit][b])), 4) for b in budgets],
            "difference_sd": [round(float(np.std(paired[unit][b], ddof=1)), 4) for b in budgets],
        }

    primary = curves["time"]["budgets"].index(300)
    effect = curves["time"]["difference"][primary]
    spread = curves["time"]["difference_sd"][primary]
    events_index = curves["events"]["budgets"].index(27)

    return {
        "meta": {
            "participants": n_participants,
            "holdout_pairs": N_HOLDOUT,
            "features": dimension,
            "catalogue": catalogue()["n"],
            "seconds_pairwise": SECONDS_PAIRWISE,
            "seconds_ranking": SECONDS_RANKING,
            "ranking_size": RANKING_SIZE,
            "taste_sigma": TASTE_SIGMA,
            "seed": seed,
        },
        "ceiling": round(float(np.mean(ceilings)), 4),
        "curves": curves,
        "power": {
            "budget_seconds": 300,
            "effect": effect,
            "sd": spread,
            "cohens_d": round(float(effect / spread), 3) if spread else None,
            "between_per_group_80": _sample_size(effect, spread, paired=False),
            "within_total_80": _sample_size(effect, spread, paired=True),
            "events_effect": curves["events"]["difference"][events_index],
            "events_sd": curves["events"]["difference_sd"][events_index],
            "events_within_total_80": _sample_size(
                curves["events"]["difference"][events_index],
                curves["events"]["difference_sd"][events_index], paired=True),
            "note": ("Simulated participants vary only through their taste vector "
                     "and their response noise. Real participants also vary in "
                     "engagement and film knowledge, so the true variance is larger "
                     "and these numbers are a floor. The pilot exists to measure it. "
                     "The gap between the two matched budgets is the reason the study "
                     "is run within participants: the time matched effect is small "
                     "enough that a between participants design would need an "
                     "implausible sample."),
        },
    }


def _sample_size(effect, sd, power_z=0.84, alpha_z=1.96, paired=True):
    """Sample size for 80% power at alpha .05.

    `paired` gives the total number of participants for a within participants
    design, where each person contributes both conditions and the test is on
    their difference. `paired=False` gives the number per group for a between
    participants design, which needs twice the participants for the same variance
    and does not benefit from cancelling out how idiosyncratic a person's taste
    is.
    """
    if not sd or not effect:
        return None
    d = abs(effect) / sd
    n = ((alpha_z + power_z) / d) ** 2
    return int(np.ceil(n if paired else 2 * n))


def save(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as handle:
        json.dump(results, handle, indent=1)
    return RESULTS_PATH


def load():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as handle:
        return json.load(handle)
