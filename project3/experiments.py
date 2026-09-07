"""
Runs everything and writes the results to `results/results.json`.

The interface reads that file rather than refitting anything, because vectorizing
24,000 documents and running four active learning strategies over five seeds does
not belong inside an HTTP request. The file is committed so the app works
straight after a clone, and the command is re-runnable so the numbers can be
checked rather than taken on trust:

    python manage.py run_experiments

Everything is seeded, so a rerun reproduces the committed file.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from . import active, defer
from .data import CLASSES, HERE, N_CLUSTERS, VECTORIZER, classifier, corpus, features, top_terms
from .expert import COMPETENCE, expert, profile

RESULTS_DIR = os.path.join(HERE, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")


def task1():
    """Baseline classifier trained on every available label."""
    data, f = corpus(), features()
    model = classifier()
    predicted = model.predict(f["X_test"])
    report = classification_report(data["y_test"], predicted,
                                   target_names=CLASSES, output_dict=True)
    return {
        "accuracy": round(float((predicted == data["y_test"]).mean()), 4),
        "per_class": [
            {
                "label": name,
                "precision": round(report[name]["precision"], 3),
                "recall": round(report[name]["recall"], 3),
                "f1": round(report[name]["f1-score"], 3),
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        ],
        "confusion": confusion_matrix(data["y_test"], predicted).tolist(),
        "top_terms": top_terms(),
        "n_features": int(f["X_train"].shape[1]),
    }


def run_all():
    started = time.time()
    data = corpus()

    print("  task 1: baseline classifier")
    one = task1()

    print("  task 2: simulated expert")
    two = profile()

    print("  task 3: learning to defer")
    three = defer.evaluate()

    print("  task 4: active learning (this is the slow one)")
    four = active.run()
    four["summary"] = active.summarise(four, three["learned"]["accuracy"])

    results = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "seconds": round(time.time() - started, 1),
            "n_train": len(data["y_train"]),
            "n_test": len(data["y_test"]),
            "classes": CLASSES,
            "n_regions": N_CLUSTERS,
            "competence": COMPETENCE,
            "vectorizer": {k: str(v) for k, v in VECTORIZER.items()},
        },
        "task1": one,
        "task2": two,
        "task3": three,
        "task4": four,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as handle:
        json.dump(results, handle, indent=1, sort_keys=False)
    return results


def load():
    """What the views read. Returns None when the experiments have not been run."""
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as handle:
        return json.load(handle)
