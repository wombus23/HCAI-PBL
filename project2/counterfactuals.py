"""
Counterfactual explanations by local sampling.

The method from the project sheet: draw N points around x, keep the ones the
model assigns to the desired class, rank them by MAD weighted L1 distance to x,
show the best k.

Two details the sheet asks about.

Noising non decimal features. The four measurements are continuous and get
gaussian noise scaled by their standard deviation. Island, sex and year are
categorical, so "adding noise" means resampling: with probability p the feature
is replaced by another of its levels, drawn uniformly. Year is included here
rather than treated as a number, because 2008.4 is not a year.

Distance for those same features. The MAD weighted L1 distance is only defined
for continuous features, so categorical ones contribute 1 when they differ and 0
when they do not. That is the usual convention and it makes one changed island
cost about as much as one MAD of bill length, which is a defensible exchange
rate for a user reading the table.

If too few counterfactuals turn up, the search widens: more samples, wider
gaussians, a higher chance of flipping a category. It gives up after a fixed
number of rounds rather than looping forever.
"""

import numpy as np

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, penguins

START_SIGMA = 0.4        # in units of each feature's standard deviation
START_FLIP = 0.15        # probability of resampling a categorical feature
MAX_ROUNDS = 6
ROUNDING = {
    "bill_length_mm": 1,
    "bill_depth_mm": 1,
    "flipper_length_mm": 0,
    "body_mass_g": 0,
}


def _sample(data, x, n, sigma, flip, rng):
    """N noisy copies of x, as a dataframe of raw feature values."""
    columns = {}
    for feature in NUMERIC_FEATURES:
        spread = sigma * data.std[feature]
        values = x[feature] + rng.normal(0, spread, size=n)
        low = float(data.raw[feature].min())
        high = float(data.raw[feature].max())
        values = np.clip(values, low, high)
        columns[feature] = np.round(values, ROUNDING[feature])

    for feature in CATEGORICAL_FEATURES:
        levels = data.categories[feature]
        values = np.array([x[feature]] * n, dtype=object)
        change = rng.random(n) < flip
        if len(levels) > 1:
            for i in np.where(change)[0]:
                alternatives = [l for l in levels if l != x[feature]]
                values[i] = alternatives[rng.integers(len(alternatives))]
        columns[feature] = values

    import pandas as pd
    return pd.DataFrame(columns, columns=FEATURES)


def _distance(data, x, row):
    """MAD weighted L1 over the measurements, plus one per changed category."""
    total = 0.0
    for feature in NUMERIC_FEATURES:
        total += abs(float(row[feature]) - float(x[feature])) / data.mad[feature]
    for feature in CATEGORICAL_FEATURES:
        total += 0.0 if row[feature] == x[feature] else 1.0
    return total


def generate(model, index, target, k=5, n=2000, seed=0):
    """Counterfactuals for penguin `index` towards class name `target`.

    Returns (rows, info). `rows` is ready for the template: the values, which of
    them changed, the distance and the model's confidence in the target class.
    """
    data = penguins()
    x = data.row_dict(index)
    target_index = data.classes.index(target)
    rng = np.random.default_rng(seed)

    original_class = data.classes[int(model.predict(data.X[index:index + 1])[0])]
    if original_class == target:
        return [], {
            "already": True,
            "predicted": original_class,
            "rounds": 0, "sampled": 0, "found": 0,
            "sigma": START_SIGMA, "flip": START_FLIP,
        }

    sigma, flip, size = START_SIGMA, START_FLIP, n
    found, sampled, rounds = None, 0, 0

    for rounds in range(1, MAX_ROUNDS + 1):
        candidates = _sample(data, x, size, sigma, flip, rng)
        sampled += size
        predictions = model.predict(data.encode(candidates))
        hits = candidates[predictions == target_index]

        if len(hits) > 0:
            hits = hits.drop_duplicates()
            found = hits if found is None else _concat(found, hits)
        if found is not None and len(found) >= k:
            break

        # Nothing close enough yet: look further out and harder.
        sigma *= 1.6
        flip = min(0.6, flip * 1.5)
        size = min(size * 2, 40000)

    info = {
        "already": False,
        "predicted": original_class,
        "rounds": rounds,
        "sampled": sampled,
        "found": 0 if found is None else len(found),
        "sigma": round(sigma, 3),
        "flip": round(flip, 3),
    }

    if found is None or len(found) == 0:
        return [], info

    found = found.drop_duplicates().reset_index(drop=True)
    distances = np.array([_distance(data, x, found.iloc[i]) for i in range(len(found))])
    order = np.argsort(distances)[:k]

    probabilities = model.proba(data.encode(found.iloc[order]))[:, target_index]

    rows = []
    for position, i in enumerate(order):
        row = found.iloc[i]
        cells = []
        for feature in FEATURES:
            value = row[feature]
            changed = value != x[feature]
            if feature in NUMERIC_FEATURES:
                value = round(float(value), ROUNDING[feature])
                changed = abs(value - float(x[feature])) > 1e-9
                shown = ("%%.%df" % ROUNDING[feature]) % value
            else:
                shown = str(value)
            cells.append({"feature": feature, "value": shown, "changed": bool(changed)})
        rows.append({
            "cells": cells,
            "distance": round(float(distances[i]), 3),
            "confidence": round(float(probabilities[position]), 3),
            "n_changed": sum(1 for c in cells if c["changed"]),
        })
    return rows, info


def _concat(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


def original_row(index):
    """The penguin being explained, formatted like the counterfactual rows."""
    data = penguins()
    x = data.row_dict(index)
    cells = []
    for feature in FEATURES:
        value = x[feature]
        if feature in NUMERIC_FEATURES:
            value = ("%%.%df" % ROUNDING[feature]) % float(value)
        cells.append({"feature": feature, "value": str(value), "changed": False})
    return {"cells": cells}
