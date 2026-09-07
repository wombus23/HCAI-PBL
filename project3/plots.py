"""Figures for project 3, redrawn from the cached numbers on each request."""

import os
import uuid

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings

PLOT_SUBDIR = "project3"
PALETTE = ["#275CB2", "#E4572E", "#2A9D8F", "#F4A259", "#7B4B94"]


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


def expert_profile_figure(results):
    """Designed competence against what the expert actually achieved per region."""
    rows = sorted(results["task2"]["by_region"], key=lambda r: r["region"])
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(positions - 0.2, [r["designed"] for r in rows], width=0.4,
           color="#c9d4e8", label="designed competence")
    ax.bar(positions + 0.2, [r["accuracy"] for r in rows], width=0.4,
           color=PALETTE[0], label="measured on the test set")
    ax.axhline(results["task1"]["accuracy"], color=PALETTE[1], linestyle="--",
               linewidth=1.3, label="classifier accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(["%d\n%s" % (r["region"], r["dominant"]) for r in rows], fontsize=8)
    ax.set_xlabel("region of the input space (k-means on the document vectors)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where the expert is worth asking", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    return _save(fig, "expert")


def coverage_figure(results):
    """Team accuracy as a function of how much is handed to the expert."""
    curves = results["task3"]["curves"]
    rates = np.array(curves["rates"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (key, label) in enumerate([
        ("oracle", "oracle deferral (upper bound)"),
        ("learned", "learned rejector"),
        ("confidence_only", "classifier confidence only"),
        ("random", "random deferral"),
    ]):
        ax.plot(rates, curves[key], color=PALETTE[i], linewidth=1.9, label=label,
                linestyle="--" if key == "oracle" else "-")

    ax.axhline(results["task3"]["classifier_only"], color="#8a94a3",
               linewidth=1.0, linestyle=":", label="classifier alone")
    operating = results["task3"]["learned"]["deferral_rate"]
    ax.axvline(operating, color=PALETTE[1], linewidth=1.0, alpha=0.5)
    ax.annotate("operating point\n%.1f%% deferred" % (100 * operating),
                xy=(operating, results["task3"]["learned"]["accuracy"]),
                xytext=(operating + 0.08, results["task3"]["learned"]["accuracy"] - 0.06),
                fontsize=8, color=PALETTE[1])
    ax.set_xlabel("share of articles handed to the expert")
    ax.set_ylabel("team accuracy")
    ax.set_title("Who should answer, and how often", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax)
    return _save(fig, "coverage")


def calibration_figure(results):
    rows = results["task3"]["calibration"]
    raw = [r["confidence"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0.4, 1.0], [0.4, 1.0], color="#8a94a3", linestyle=":", linewidth=1)
    ax.plot(raw, [r["actual"] for r in rows], marker="o", color=PALETTE[0],
            linewidth=1.8, label="how often it is actually right")
    ax.plot(raw, [r["calibrated"] for r in rows], marker="s", color=PALETTE[1],
            linewidth=1.8, label="after calibration")
    ax.set_xlabel("classifier's raw maximum probability")
    ax.set_ylabel("probability of being correct")
    ax.set_title("Why the confidence needs calibrating first", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, "calibration")


def active_figure(results, key="accuracy"):
    """Learning curves for the four query strategies."""
    task4 = results["task4"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for i, (strategy, curve) in enumerate(task4["strategies"].items()):
        queries = np.array(curve["queries"])
        values = np.array(curve[key])
        ax.plot(queries, values, color=PALETTE[i], linewidth=1.9,
                marker="o", markersize=3.2, label=task4["labels"][strategy])
        if key == "accuracy":
            spread = np.array(curve["accuracy_std"])
            ax.fill_between(queries, values - spread, values + spread,
                            color=PALETTE[i], alpha=0.12, linewidth=0)

    if key == "accuracy":
        ax.axhline(results["task3"]["learned"]["accuracy"], color="#8a94a3",
                   linestyle="--", linewidth=1.1,
                   label="task 3, all %s expert labels" % results["meta"]["n_train"])
        ax.axhline(task4["baseline"], color="#8a94a3", linestyle=":", linewidth=1.1,
                   label="classifier alone")
        ax.set_ylabel("team accuracy on the test set")
        ax.set_title("Learning where the expert helps, one question at a time", fontsize=12)
    else:
        ax.set_ylabel("AUC of the expert competence model")
        ax.set_title("How well the competence profile is recovered", fontsize=12)

    ax.set_xlabel("expert labels queried")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    return _save(fig, "active-" + key)


def confusion_figure(results):
    matrix = np.array(results["task1"]["confusion"])
    labels = results["meta"]["classes"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    threshold = matrix.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > threshold else "#222")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Baseline classifier on the test set", fontsize=12)
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _save(fig, "confusion")


def all_figures(results):
    return {
        "confusion": confusion_figure(results),
        "expert": expert_profile_figure(results),
        "calibration": calibration_figure(results),
        "coverage": coverage_figure(results),
        "active": active_figure(results, "accuracy"),
        "active_auc": active_figure(results, "auc"),
    }
