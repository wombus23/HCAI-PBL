"""Figures for project 2. Same approach as project 1: matplotlib into media."""

import os
import uuid

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from sklearn.tree import plot_tree

from .data import PRETTY, penguins

PLOT_SUBDIR = "project2"
PALETTE = ["#275CB2", "#E4572E", "#2A9D8F"]


def _save(fig, prefix):
    folder = os.path.join(settings.MEDIA_ROOT, PLOT_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    filename = "%s-%s.png" % (prefix, uuid.uuid4().hex[:10])
    fig.savefig(os.path.join(folder, filename), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "%s%s/%s" % (settings.MEDIA_URL, PLOT_SUBDIR, filename)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def tree_figure(model):
    data = penguins()
    width = max(8, min(20, 1.6 * model.omega))
    height = max(4.5, 1.1 * (model.estimator.get_depth() + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(
        model.estimator,
        feature_names=data.columns,
        class_names=data.classes,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision tree with %d leaves, test accuracy %.3f"
                 % (model.omega, model.accuracy), fontsize=12)
    return _save(fig, "tree")


def coefficient_figure(model):
    """Nonzero coefficients per class, in the standardized feature space."""
    data = penguins()
    coef = model.estimator.coef_
    used = [j for j in range(coef.shape[1]) if np.any(np.abs(coef[:, j]) > 1e-8)]
    if not used:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "Every coefficient was driven to zero:\n"
                          "the model always predicts the majority class.",
                ha="center", va="center", fontsize=11, color="#5a6473")
        ax.axis("off")
        return _save(fig, "coefficients")

    labels = [data.columns[j] for j in used]
    positions = np.arange(len(used))
    height = 0.8 / len(data.classes)

    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.55 * len(used) + 1.5)))
    for c, name in enumerate(data.classes):
        ax.barh(positions + c * height, coef[c, used], height=height,
                color=PALETTE[c], label=name, alpha=0.9)
    ax.axvline(0, color="#8a94a3", linewidth=0.9)
    ax.set_yticks(positions + height)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient (standardized features)")
    ax.set_title("What the model uses: %d features, test accuracy %.3f"
                 % (model.omega, model.accuracy), fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, "coefficients")


def selection_figure(models, chosen_index, lam, omega_label):
    """Accuracy against complexity, with the line the trade off actually draws.

    Every model in the grid is a point. The dashed line has slope lambda and
    passes through the winner, so anything above it would have scored higher and
    anything below it scored lower: the picture of the argmax.
    """
    omegas = np.array([m.omega for m in models], dtype=float)
    accuracies = np.array([m.accuracy for m in models])
    chosen = models[chosen_index]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(omegas, accuracies, s=55, color=PALETTE[0], alpha=0.85,
               edgecolors="white", linewidths=0.6, label="models in the grid", zorder=3)
    ax.scatter([chosen.omega], [chosen.accuracy], s=210, facecolors="none",
               edgecolors=PALETTE[1], linewidths=2.2, zorder=4, label="selected")

    span = np.linspace(omegas.min() - 0.5, omegas.max() + 0.5, 20)
    intercept = chosen.accuracy - lam * chosen.omega
    ax.plot(span, intercept + lam * span, linestyle="--", color=PALETTE[1],
            linewidth=1.3, alpha=0.8,
            label="accuracy − λ·Ω = %.3f" % intercept)

    # The trade off line runs off the top of the plot at larger lambda, so the
    # y range stays pinned to the accuracies the models actually reached.
    margin = max(0.02, 0.08 * (accuracies.max() - accuracies.min()))
    ax.set_ylim(accuracies.min() - margin, accuracies.max() + margin)

    ax.set_xlabel("Ω(f): %s" % omega_label)
    ax.set_ylabel("test accuracy")
    ax.set_title("Trade off at λ = %.4f" % lam, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="best")
    _style(ax)
    return _save(fig, "selection")


def effect_figure(kind, x_values, curves, feature, method=None):
    """PDP or ALE: one curve per species, plus a rug of the observed values."""
    data = penguins()
    fig, ax = plt.subplots(figsize=(7, 4.4))

    if kind == "ale":
        # Curves are defined on the bin edges.
        for c, name in enumerate(data.classes):
            ax.step(x_values, curves[:, c], where="post", color=PALETTE[c],
                    linewidth=1.9, label=name)
        ax.axhline(0, color="#8a94a3", linewidth=0.8)
        ax.set_ylabel("accumulated local effect on P(species)")
        title = "ALE of %s (%s)" % (PRETTY[feature].lower(), method)
    else:
        for c, name in enumerate(data.classes):
            ax.plot(x_values, curves[:, c], color=PALETTE[c], linewidth=1.9, label=name)
        ax.set_ylabel("average P(species)")
        ax.set_ylim(-0.02, 1.02)
        title = "Partial dependence of %s" % PRETTY[feature].lower()

    observed = data.raw[feature].to_numpy(dtype=float)
    low, high = ax.get_ylim()
    ax.plot(observed, np.full_like(observed, low + 0.01 * (high - low)), "|",
            color="#8a94a3", alpha=0.5, markersize=6)

    ax.set_xlabel(PRETTY[feature])
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, kind)
