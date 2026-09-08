"""
Task 1: the feature representation.

A movie is turned into a vector x, and a user's taste is a vector w, with utility
U(x) = w'x. The whole study lives or dies on this choice, because w has to be
estimated from a few dozen answers: every feature added is another number the
participant has to pay for.

What is in the vector, and why:

- **Genres, as 18 binary indicators.** Genre is the strongest, most
  self-explanatory axis of film taste, and a weight on a genre is directly
  readable ("likes horror, dislikes musicals"), which matters because the debrief
  screen shows participants their own profile. Genres are multi-hot, since a film
  is often three of them at once.
- **Release year and duration**, standardized. These capture the era and
  commitment axes of taste ("old films", "nothing over two hours") that genre
  cannot express.
- **IMDb score**, standardized. A proxy for whether someone follows critical
  consensus or ignores it.
- **Popularity**, as log vote count, standardized. Separates the blockbuster
  watcher from the person who has already seen everything famous. The log is
  necessary: raw vote counts span four orders of magnitude, so without it this
  feature would be a single spike on Shawshank.
- **Audience rating**, collapsed to three indicators (family, teen, adult).
  Twenty-odd raw certificate strings would spend degrees of freedom on
  distinctions like Approved against Passed that nobody has an opinion about.

What is deliberately left out:

- **Director and cast identity.** They matter enormously to real taste, but as
  one-hot features they would add thousands of dimensions that no elicitation
  budget can estimate, and they are close to unidentifiable from 30 answers. A
  richer study could use embeddings of the cast; this one cannot afford them.
- **Budget and gross.** Heavily missing, and mostly a proxy for the popularity
  feature already included.
- **Plot keywords.** The same dimensionality problem as cast, with more noise.

That leaves 26 dimensions, which is already ambitious for the budget: it is the
main reason the study measures held out prediction rather than the recovery of w
itself.

The catalogue is also filtered to films most people might plausibly have heard
of (at least 25,000 IMDb votes). This is a study design decision, not a modelling
one. Asking a participant to rank ten films they have never heard of measures
their reading of the poster, not their taste.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "sample_data", "movies.csv.gz")

MIN_VOTES = 25000

GENRES = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
          "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
          "Music", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"]

FAMILY_RATINGS = {"G", "PG", "TV-G", "TV-PG", "TV-Y", "TV-Y7", "Approved"}
TEEN_RATINGS = {"PG-13", "TV-14"}
ADULT_RATINGS = {"R", "NC-17", "X", "TV-MA", "M", "GP"}

CONTINUOUS = ["year", "duration", "imdb_score", "popularity"]
RATING_GROUPS = ["family", "teen", "adult"]

FEATURE_NAMES = (["genre:" + g for g in GENRES]
                 + ["year", "duration", "imdb_score", "popularity"]
                 + ["rating:" + r for r in RATING_GROUPS])

PRETTY = {
    "year": "recent release",
    "duration": "long running time",
    "imdb_score": "critically acclaimed",
    "popularity": "widely seen",
    "rating:family": "family certificate",
    "rating:teen": "teen certificate",
    "rating:adult": "adult certificate",
}


def pretty_feature(name):
    if name.startswith("genre:"):
        return name.split(":", 1)[1]
    return PRETTY.get(name, name)


def _rating_group(value):
    value = str(value).strip()
    if value in FAMILY_RATINGS:
        return "family"
    if value in TEEN_RATINGS:
        return "teen"
    if value in ADULT_RATINGS:
        return "adult"
    return None


@lru_cache(maxsize=1)
def catalogue():
    """The films, their feature matrix, and everything the interface displays."""
    frame = pd.read_csv(CSV_PATH)
    frame = frame[frame["num_voted_users"] >= MIN_VOTES].reset_index(drop=True)
    frame["title_year"] = frame["title_year"].astype(int)

    n = len(frame)
    blocks = []

    genre_lists = frame["genres"].fillna("").str.split("|")
    genre_matrix = np.zeros((n, len(GENRES)))
    for i, entries in enumerate(genre_lists):
        for j, genre in enumerate(GENRES):
            if genre in entries:
                genre_matrix[i, j] = 1.0
    blocks.append(genre_matrix)

    raw = np.column_stack([
        frame["title_year"].to_numpy(dtype=float),
        frame["duration"].to_numpy(dtype=float),
        frame["imdb_score"].to_numpy(dtype=float),
        np.log10(frame["num_voted_users"].to_numpy(dtype=float)),
    ])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.0
    blocks.append((raw - mean) / scale)

    groups = frame["content_rating"].map(_rating_group)
    rating_matrix = np.zeros((n, len(RATING_GROUPS)))
    for j, group in enumerate(RATING_GROUPS):
        rating_matrix[:, j] = (groups == group).astype(float)
    blocks.append(rating_matrix)

    X = np.hstack(blocks)

    cards = []
    for i in range(n):
        row = frame.iloc[i]
        cards.append({
            "id": i,
            "title": row["movie_title"],
            "year": int(row["title_year"]),
            "genres": [g for g in str(row["genres"]).split("|") if g in GENRES][:3],
            "duration": int(row["duration"]),
            "score": float(row["imdb_score"]),
            "director": row["director_name"] if isinstance(row["director_name"], str) else "",
            "stars": ", ".join([s for s in [row.get("actor_1_name"), row.get("actor_2_name")]
                                if isinstance(s, str)]),
        })

    return {
        "frame": frame,
        "X": X,
        "cards": cards,
        "feature_names": FEATURE_NAMES,
        "mean": mean,
        "scale": scale,
        "n": n,
    }


def cards_for(ids):
    """Display data for a list of catalogue ids, in the order given."""
    cards = catalogue()["cards"]
    return [cards[int(i)] for i in ids]


def features_for(ids):
    return catalogue()["X"][np.asarray(ids, dtype=int)]


def sample_ids(count, rng, exclude=()):
    """Films drawn uniformly at random, as the project sheet specifies."""
    total = catalogue()["n"]
    blocked = set(int(i) for i in exclude)
    picked = []
    while len(picked) < count:
        candidate = int(rng.integers(total))
        if candidate not in blocked:
            blocked.add(candidate)
            picked.append(candidate)
    return picked
