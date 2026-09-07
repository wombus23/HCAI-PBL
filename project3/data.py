"""
AG News: loading, features, and the classifier of task 1.

The dataset ships with the app as two gzipped CSVs instead of being pulled in
through the `datasets` package, for the same reason project 2 ships its CSV: the
submission rules ask for no unusual dependencies. The test set is complete
(7,600 articles); the training set is a stratified 24,000 article sample of the
full 120,000, which keeps the repository small and every experiment in this
project runnable in a couple of minutes.

Provenance and the sampling code are in `prepare_data.py`.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sample_data")

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 0

VECTORIZER = dict(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                  strip_accents="unicode", stop_words="english")
LSA_COMPONENTS = 100
N_CLUSTERS = 12


@lru_cache(maxsize=1)
def corpus():
    """The raw texts and labels."""
    train = pd.read_csv(os.path.join(DATA_DIR, "ag_news_train.csv.gz"))
    test = pd.read_csv(os.path.join(DATA_DIR, "ag_news_test.csv.gz"))
    return {
        "train_text": train["text"].astype(str).tolist(),
        "y_train": train["label"].to_numpy(),
        "test_text": test["text"].astype(str).tolist(),
        "y_test": test["label"].to_numpy(),
    }


@lru_cache(maxsize=1)
def features():
    """TF-IDF matrices, plus an LSA projection used for clustering."""
    data = corpus()
    vectorizer = TfidfVectorizer(**VECTORIZER)
    X_train = vectorizer.fit_transform(data["train_text"])
    X_test = vectorizer.transform(data["test_text"])

    svd = TruncatedSVD(n_components=LSA_COMPONENTS, random_state=SEED)
    Z_train = svd.fit_transform(X_train)
    Z_test = svd.transform(X_test)

    return {
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_test": X_test,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "explained": float(svd.explained_variance_ratio_.sum()),
    }


@lru_cache(maxsize=1)
def clusters():
    """K-means regions of the input space, used to give the expert a profile.

    The LSA vectors are normalized first, so k-means works on the angle between
    documents rather than their length. On raw LSA vectors it produces two
    enormous mixed clusters and ten small ones, which would make the expert's
    competence profile almost binary; on normalized vectors the regions come out
    balanced and each one leans towards a topic without being a topic.
    """
    f = features()
    model = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    train_ids = model.fit_predict(normalize(f["Z_train"]))
    test_ids = model.predict(normalize(f["Z_test"]))
    return {"model": model, "train": train_ids, "test": test_ids}


@lru_cache(maxsize=1)
def classifier():
    """Task 1: the baseline the human-AI team has to beat.

    Linear model on TF-IDF. It is the standard strong baseline for AG News, it
    trains in seconds, and it gives calibrated enough probabilities to be used
    as a confidence signal later, which a linear SVM would not.
    """
    data, f = corpus(), features()
    model = LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)
    model.fit(f["X_train"], data["y_train"])
    return model


def top_terms(n=8):
    """The highest weighted features per class, for the report."""
    model, f = classifier(), features()
    names = np.array(f["vectorizer"].get_feature_names_out())
    out = {}
    for c, name in enumerate(CLASSES):
        order = np.argsort(model.coef_[c])[::-1][:n]
        out[name] = names[order].tolist()
    return out
