"""
Project 3 interface.

Everything on the page comes from the cached experiment results. The figures are
redrawn from those numbers on request, which keeps binary images out of the
repository while still letting the page be live.
"""

import os

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render

from . import plots
from .experiments import RESULTS_PATH, load

REPORT_NAME = "project3_report.pdf"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "static", "project3", REPORT_NAME)


def index(request):
    results = load()
    if results is None:
        return render(request, "project3/missing.html", {"path": RESULTS_PATH})

    figures = plots.all_figures(results)
    return render(request, "project3/index.html", {
        "r": results,
        "meta": results["meta"],
        "t1": results["task1"],
        "t2": results["task2"],
        "t3": results["task3"],
        "t4": results["task4"],
        "figures": figures,
        "has_report": os.path.exists(REPORT_PATH),
        "confusion_rows": zip(results["meta"]["classes"], results["task1"]["confusion"]),
    })


def report(request):
    """The PDF report, as the project sheet requires it to be reachable here."""
    if not os.path.exists(REPORT_PATH):
        raise Http404("The report has not been generated yet.")
    return FileResponse(open(REPORT_PATH, "rb"), as_attachment=True,
                        filename=REPORT_NAME)
