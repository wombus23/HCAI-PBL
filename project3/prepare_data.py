"""
How `sample_data/` was produced. Kept for provenance; not imported by the app.

The original AG News CSVs (label, title, description; labels 1 to 4) come from
the public mirror of the dataset used by the Hugging Face card:

    https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/

Title and description are concatenated into one text field and the labels are
shifted to 0 to 3, which matches the `fancyzhx/ag_news` layout. The test set is
kept whole; the training set is a stratified 6,000 per class sample of the
120,000, drawn with seed 0.
"""

import numpy as np
import pandas as pd

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
PER_CLASS = 6000
SEED = 0


def load(path):
    frame = pd.read_csv(path, header=None, names=["label", "title", "description"])
    text = (frame["title"].fillna("") + ". " + frame["description"].fillna(""))
    text = text.str.replace("\\", " ", regex=False).str.strip()
    return pd.DataFrame({"label": frame["label"].astype(int) - 1, "text": text})


def main(train_csv, test_csv, out_dir="sample_data"):
    train, test = load(train_csv), load(test_csv)
    rng = np.random.default_rng(SEED)
    parts = []
    for c in range(len(CLASSES)):
        rows = train.index[train["label"] == c].to_numpy()
        parts.append(train.loc[rng.choice(rows, PER_CLASS, replace=False)])
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    sample.to_csv("%s/ag_news_train.csv.gz" % out_dir, index=False, compression="gzip")
    test.to_csv("%s/ag_news_test.csv.gz" % out_dir, index=False, compression="gzip")
    print("train %s, test %s" % (sample.shape, test.shape))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
