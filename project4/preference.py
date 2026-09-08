"""
Task 2: extending Bradley-Terry from a pair to a ranking.

Bradley-Terry says that when a user compares two films,

    P(a > b) = exp(w'x_a) / ( exp(w'x_a) + exp(w'x_b) )

which is a softmax over two utilities. The natural extension to a full ranking
i1 > i2 > ... > in is the Plackett-Luce model, which reads the ranking as a
sequence of choices: the participant picks their favourite out of all n, then
their favourite out of the remaining n-1, and so on.

    P(i1 > i2 > ... > in) = prod_{k=1}^{n-1}  exp(w'x_ik) / sum_{j>=k} exp(w'x_ij)

Why this one, rather than the alternatives:

- **It contains Bradley-Terry exactly.** With n = 2 the product has a single
  factor and reduces to the formula above. That matters here more than usual: the
  whole study compares two elicitation designs, and if each design were fitted
  with a different model, any difference in the results could be the model rather
  than the interface. One likelihood, two ways of feeding it.
- **The log likelihood is concave in w.** Each factor is a log-sum-exp minus a
  linear term. With a Gaussian prior on w the objective is strictly concave, so
  the fit has one optimum and no restarts, seeds or luck are involved.
- **It respects the sequential logic of ranking.** The alternative shortcut is to
  explode a ranking of ten into its 45 implied pairwise comparisons and pour them
  into plain Bradley-Terry. That is tempting and wrong in a specific way: it
  treats 45 comparisons drawn from one ranking as 45 independent observations,
  which overstates the information in a ranking and would bias the study in
  favour of the design being tested. Plackett-Luce counts a ranking of ten as
  nine choice events, which is what it is.

The estimate is the MAP under a Gaussian prior, w ~ N(0, sigma^2 I):

    w_hat = argmax  sum over observations log P(obs | w)  -  ||w||^2 / (2 sigma^2)

The prior is not decoration. With 26 features and 30 answers the unregularized
likelihood is often maximized by sending some coordinates to infinity, since a
participant who never sees a documentary has no finite best estimate for that
weight. The prior keeps unseen directions at zero, which is also the honest
answer.
"""

import numpy as np
from scipy.optimize import minimize

PRIOR_SIGMA = 1.0


def _ranking_log_likelihood(w, blocks):
    """Log likelihood and gradient for a list of rankings.

    Each block is an (n, d) array whose rows are the chosen items in the order
    the participant put them, best first. A pairwise choice is a block with two
    rows.
    """
    total = 0.0
    gradient = np.zeros_like(w)

    for X in blocks:
        utilities = X @ w
        for k in range(len(X) - 1):
            remaining = utilities[k:]
            top = remaining.max()
            weights = np.exp(remaining - top)
            partition = weights.sum()

            total += remaining[0] - (top + np.log(partition))
            probabilities = weights / partition
            gradient += X[k] - probabilities @ X[k:]

    return total, gradient


def fit(blocks, sigma=PRIOR_SIGMA, dimension=None):
    """MAP estimate of the preference vector w."""
    if dimension is None:
        dimension = blocks[0].shape[1]
    if not blocks:
        return np.zeros(dimension)

    penalty = 1.0 / (sigma ** 2)

    def objective(w):
        value, gradient = _ranking_log_likelihood(w, blocks)
        value -= 0.5 * penalty * float(w @ w)
        gradient -= penalty * w
        return -value, -gradient

    result = minimize(objective, np.zeros(dimension), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500})
    return result.x


def blocks_from_responses(responses, features_for):
    """Turn stored answers into likelihood blocks.

    A pairwise answer is stored as [winner, loser]; a ranking as the ids in the
    participant's order. Both become an ordered feature matrix, which is the only
    thing the likelihood needs and the reason one model covers both designs.
    """
    blocks = []
    for order in responses:
        if len(order) >= 2:
            blocks.append(np.asarray(features_for(order), dtype=float))
    return blocks


def pair_probability(w, x_a, x_b):
    """P(a preferred to b) under the fitted model."""
    difference = float((np.asarray(x_a) - np.asarray(x_b)) @ w)
    return 1.0 / (1.0 + np.exp(-difference))


def evaluate_holdout(w, pairs, features_for):
    """Accuracy and log loss of the fitted w on held out pairwise answers.

    `pairs` is a list of [chosen, rejected] pairs the participant answered after
    the elicitation phase. This is the study's primary outcome: it asks whether
    the estimated preferences predict choices the model never saw, which is what
    a recommender actually needs, rather than whether w happens to be close to
    some ground truth that does not exist for a real person.
    """
    if not pairs:
        return {"n": 0, "accuracy": None, "log_loss": None}

    correct, loss = 0, 0.0
    for chosen, rejected in pairs:
        x = features_for([chosen, rejected])
        p = pair_probability(w, x[0], x[1])
        correct += int(p > 0.5)
        loss -= np.log(max(p, 1e-12))
    return {
        "n": len(pairs),
        "accuracy": round(correct / len(pairs), 4),
        "log_loss": round(float(loss / len(pairs)), 4),
    }


def profile(w, feature_names, pretty, top=6):
    """The strongest positive and negative weights, for the debrief screen."""
    order = np.argsort(w)
    likes = [{"feature": pretty(feature_names[i]), "weight": round(float(w[i]), 3)}
             for i in order[::-1][:top] if w[i] > 0.05]
    dislikes = [{"feature": pretty(feature_names[i]), "weight": round(float(w[i]), 3)}
                for i in order[:top] if w[i] < -0.05]
    return {"likes": likes, "dislikes": dislikes}


def recommend(w, X, exclude=(), top=5):
    """Highest utility films the participant has not been shown."""
    utilities = X @ w
    blocked = set(int(i) for i in exclude)
    order = np.argsort(-utilities)
    picked = []
    for index in order:
        if int(index) not in blocked:
            picked.append({"id": int(index), "utility": round(float(utilities[index]), 3)})
        if len(picked) >= top:
            break
    return picked
