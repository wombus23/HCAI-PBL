"""Figures for project 4, drawn from the cached simulation results."""

import os
import uuid

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings

PLOT_SUBDIR = "project4"
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


def budget_figure(results, unit):
    """Held out accuracy against budget, one line per elicitation design."""
    curve = results["curves"][unit]
    budgets = np.array(curve["budgets"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (key, label) in enumerate([("pairwise", "Design 1 — pairwise"),
                                      ("ranking", "Design 2 — ranking of ten")]):
        values = np.array(curve[key])
        errors = np.array(curve[key + "_se"])
        ax.plot(budgets, values, color=PALETTE[i], linewidth=2, marker="o",
                markersize=4, label=label)
        ax.fill_between(budgets, values - errors, values + errors,
                        color=PALETTE[i], alpha=0.15, linewidth=0)

    ax.axhline(results["ceiling"], color="#8a94a3", linestyle="--", linewidth=1.2,
               label="ceiling (the simulated user's own noise)")
    ax.axhline(0.5, color="#8a94a3", linestyle=":", linewidth=1, label="chance")

    if unit == "time":
        ax.set_xlabel("elicitation time budget (seconds)")
        ax.set_title("Matched on participant time", fontsize=12)
    else:
        ax.set_xlabel("choice events (a ranking of ten is nine)")
        ax.set_title("Matched on decisions made", fontsize=12)

    ax.set_ylabel("held out pairwise accuracy")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    return _save(fig, "budget-" + unit)


def difference_figure(results):
    """The paired difference, which is what the study would actually test."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, unit in enumerate(["time", "events"]):
        curve = results["curves"][unit]
        budgets = np.array(curve["budgets"], dtype=float)
        # Put the two units on a shared 0 to 1 axis so they can share a panel.
        x = (budgets - budgets.min()) / (budgets.max() - budgets.min())
        differences = np.array(curve["difference"])
        spread = np.array(curve["difference_sd"]) / np.sqrt(results["meta"]["participants"])
        ax.plot(x, differences, color=PALETTE[i], linewidth=2, marker="o", markersize=4,
                label="matched on %s" % unit)
        ax.fill_between(x, differences - spread, differences + spread,
                        color=PALETTE[i], alpha=0.15, linewidth=0)

    ax.axhline(0, color="#8a94a3", linewidth=1)
    ax.set_xlabel("budget, from the smallest to the largest tested")
    ax.set_ylabel("ranking advantage in accuracy")
    ax.set_title("Which design wins depends on what you hold fixed", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, "difference")


def all_figures(results):
    return {
        "time": budget_figure(results, "time"),
        "events": budget_figure(results, "events"),
        "difference": difference_figure(results),
    }
