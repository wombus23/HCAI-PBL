"""
Loading and encoding the Palmer Penguins dataset.

The dataset ships with the app as a CSV rather than being pulled in through the
`palmerpenguins` package, so that the project runs anywhere with nothing beyond
numpy, pandas, scikit-learn and matplotlib installed.

Everything downstream (the models, the counterfactuals, the effect plots) works
on the encoding built here, so the column order is fixed in one place.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "sample_data", "penguins.csv")

TARGET = "species"

# The four measurements are the continuous features. Year has three values and
# is treated as categorical, which is also why it is not offered in the feature
# effect plots: a partial dependence curve over three integers is a bar chart
# pretending to be a curve.
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PRETTY = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
}

TEST_SIZE = 0.3
SPLIT_SEED = 0


class Penguins:
    """The prepared dataset: raw rows, the encoded matrix and the split."""

    def __init__(self):
        frame = pd.read_csv(CSV_PATH)
        self.n_raw = len(frame)

        # Rows with missing values are dropped rather than imputed. Imputing a
        # penguin's sex or bill length would put invented data into the
        # explanations, which is the one thing an explainability tool must not
        # do. 11 rows go, all of them missing sex, plus 2 missing everything.
        frame = frame.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
        frame["year"] = frame["year"].astype(int)
        self.raw = frame
        self.n_dropped = self.n_raw - len(frame)

        self.classes = sorted(frame[TARGET].unique())
        self.y = np.array([self.classes.index(v) for v in frame[TARGET]])

        # Fixed category order, so an encoded column always means the same thing.
        self.categories = {c: sorted(frame[c].unique().tolist()) for c in CATEGORICAL_FEATURES}

        self.columns = list(NUMERIC_FEATURES)
        for feature in CATEGORICAL_FEATURES:
            for level in self.categories[feature]:
                self.columns.append("%s=%s" % (feature, level))

        self.X = self.encode(frame)

        # Median absolute deviation per numeric feature, used to weight the
        # distance between counterfactuals. Computed on the whole dataset.
        self.mad = {}
        for feature in NUMERIC_FEATURES:
            values = frame[feature].to_numpy(dtype=float)
            deviation = float(np.median(np.abs(values - np.median(values))))
            self.mad[feature] = deviation if deviation > 0 else float(np.std(values)) or 1.0

        self.std = {f: float(frame[f].std()) for f in NUMERIC_FEATURES}

        from sklearn.model_selection import train_test_split
        indices = np.arange(len(frame))
        self.train_idx, self.test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=self.y
        )

    # -- encoding --------------------------------------------------------- #

    def encode(self, frame):
        """Raw rows to the numeric matrix the models are fitted on."""
        blocks = [frame[NUMERIC_FEATURES].to_numpy(dtype=float)]
        for feature in CATEGORICAL_FEATURES:
            levels = self.categories[feature]
            column = frame[feature].to_numpy()
            block = np.zeros((len(frame), len(levels)))
            for j, level in enumerate(levels):
                block[:, j] = (column == level).astype(float)
            blocks.append(block)
        return np.hstack(blocks)

    def encode_rows(self, rows):
        """Same, for a list of dictionaries of raw feature values."""
        return self.encode(pd.DataFrame(rows, columns=FEATURES))

    # -- convenience ------------------------------------------------------ #

    @property
    def X_train(self):
        return self.X[self.train_idx]

    @property
    def X_test(self):
        return self.X[self.test_idx]

    @property
    def y_train(self):
        return self.y[self.train_idx]

    @property
    def y_test(self):
        return self.y[self.test_idx]

    def numeric_slice(self):
        return slice(0, len(NUMERIC_FEATURES))

    def column_index(self, feature):
        return self.columns.index(feature)

    def row_dict(self, index):
        """One penguin as a plain dictionary of raw values."""
        row = self.raw.iloc[index]
        out = {f: row[f] for f in FEATURES}
        for f in NUMERIC_FEATURES:
            out[f] = float(out[f])
        out["year"] = int(out["year"])
        return out

    def label_of(self, index):
        return self.raw.iloc[index][TARGET]

    def choices(self, limit=None):
        """Dropdown entries: one line per penguin in the dataset."""
        entries = []
        for i in range(len(self.raw)):
            row = self.raw.iloc[i]
            entries.append((i, "#%d — %s, %s, %s, bill %.1f mm, %.0f g"
                            % (i, row[TARGET], row["island"], row["sex"],
                               row["bill_length_mm"], row["body_mass_g"])))
            if limit and len(entries) >= limit:
                break
        return entries


@lru_cache(maxsize=1)
def penguins():
    """The dataset is fixed, so it is read and encoded once per process."""
    return Penguins()
