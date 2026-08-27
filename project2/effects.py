"""
Global model agnostic explanations: partial dependence and accumulated local
effects, both written out here rather than taken from a library.

On the question the project sheet asks. ALE is the integral of the average
partial derivative of the prediction with respect to the feature:

    ALE_c(v) = integral from z_min to v of E[ d f_c / d x_s | x_s = z ] dz

For logistic regression that derivative is available in closed form. With
softmax probabilities p and coefficient matrix W, and a feature that was
standardized by `scale` before fitting,

    d p_c / d x_s = (1 / scale_s) * p_c * ( W[c, s] - sum_k p_k W[k, s] )

For a decision tree the prediction is piecewise constant, so the derivative is
zero almost everywhere and undefined on the split points: exactly the wrong
object to integrate. There we fall back on the usual discretization, replacing
the derivative inside a bin with the finite difference across that bin's edges.
Both routes are implemented below and the interface says which one it used.
"""

import numpy as np

from .data import NUMERIC_FEATURES, penguins

PDP_POINTS = 40
ALE_BINS = 10


# --------------------------------------------------------------------------- #
#  Partial dependence
# --------------------------------------------------------------------------- #

def partial_dependence(model, feature, n_points=PDP_POINTS):
    """Average predicted probability per class as the feature is forced to v.

    For every value on the grid the whole dataset is copied, the feature is
    overwritten with that value, and the predicted probabilities are averaged.
    """
    data = penguins()
    column = data.column_index(feature)
    values = data.X[:, column]
    grid = np.linspace(values.min(), values.max(), n_points)

    curves = np.zeros((n_points, len(data.classes)))
    X = np.array(data.X, dtype=float, copy=True)
    for i, v in enumerate(grid):
        X[:, column] = v
        curves[i] = model.proba(X).mean(axis=0)

    return grid, curves


# --------------------------------------------------------------------------- #
#  Accumulated local effects
# --------------------------------------------------------------------------- #

def _bin_edges(values, n_bins):
    """Quantile bin edges, with duplicates removed."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    return edges


def _centre(edges, accumulated, values):
    """Shift the curve so that its average over the data is zero."""
    counts, _ = np.histogram(values, bins=edges)
    midpoints = 0.5 * (accumulated[1:] + accumulated[:-1])
    total = counts.sum()
    if total == 0:
        return accumulated
    mean = (counts[:, None] * midpoints).sum(axis=0) / total
    return accumulated - mean


def ale_finite_differences(model, feature, n_bins=ALE_BINS):
    """ALE with the derivative replaced by a difference across each bin.

    Works for any model, and is the only option for a decision tree.
    """
    data = penguins()
    column = data.column_index(feature)
    values = data.X[:, column]
    edges = _bin_edges(values, n_bins)
    n_classes = len(data.classes)

    # Bin index per point, clipped so the lowest value lands in the first bin.
    which = np.clip(np.digitize(values, edges, right=True) - 1, 0, len(edges) - 2)

    local = np.zeros((len(edges) - 1, n_classes))
    for k in range(len(edges) - 1):
        rows = np.where(which == k)[0]
        if len(rows) == 0:
            continue
        lower = np.array(data.X[rows], dtype=float, copy=True)
        upper = np.array(data.X[rows], dtype=float, copy=True)
        lower[:, column] = edges[k]
        upper[:, column] = edges[k + 1]
        local[k] = (model.proba(upper) - model.proba(lower)).mean(axis=0)

    accumulated = np.vstack([np.zeros(n_classes), np.cumsum(local, axis=0)])
    return edges, _centre(edges, accumulated, values)


def ale_exact_derivative(model, feature, n_bins=ALE_BINS):
    """ALE using the analytic derivative of the softmax probabilities.

    Only defined for the logistic model. Inside each bin the average derivative
    is evaluated at the bin midpoint and multiplied by the bin width, which is
    the rectangle rule applied to the integral in the docstring above.
    """
    data = penguins()
    column = data.column_index(feature)
    values = data.X[:, column]
    edges = _bin_edges(values, n_bins)
    n_classes = len(data.classes)

    which = np.clip(np.digitize(values, edges, right=True) - 1, 0, len(edges) - 2)
    scale = model.scale[NUMERIC_FEATURES.index(feature)]
    W = model.estimator.coef_[:, column]

    local = np.zeros((len(edges) - 1, n_classes))
    for k in range(len(edges) - 1):
        rows = np.where(which == k)[0]
        if len(rows) == 0:
            continue
        midpoint = 0.5 * (edges[k] + edges[k + 1])
        block = np.array(data.X[rows], dtype=float, copy=True)
        block[:, column] = midpoint
        p = model.proba(block)                       # rows x classes
        weighted = p @ W                             # sum_k p_k W[k, s]
        derivative = p * (W[None, :] - weighted[:, None]) / scale
        local[k] = derivative.mean(axis=0) * (edges[k + 1] - edges[k])

    accumulated = np.vstack([np.zeros(n_classes), np.cumsum(local, axis=0)])
    return edges, _centre(edges, accumulated, values)


def accumulated_local_effects(model, feature, n_bins=ALE_BINS):
    """Pick the right route for the model, and report which one was used.

    Returns (edges, curves, method, discrepancy). For the logistic model the
    discretized version is computed as well and the largest gap between the two
    is reported, as a check that the exact derivative is doing the right thing.
    """
    if model.kind == "logreg":
        edges, exact = ale_exact_derivative(model, feature, n_bins)
        _, approximate = ale_finite_differences(model, feature, n_bins)
        gap = float(np.max(np.abs(exact - approximate)))
        return edges, exact, "exact derivative", gap
    edges, curves = ale_finite_differences(model, feature, n_bins)
    return edges, curves, "finite differences", None
