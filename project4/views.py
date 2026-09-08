"""
Project 4 interface.

The landing page explains the study and offers the report and the study itself,
as the project sheet asks. Everything under /study/ is the instrument a
participant would actually see.

The study is within participants: consent, then two blocks in a counterbalanced
order, each block being a run of one elicitation design followed by a held out
block and a short workload questionnaire, then a final comparison question and a
debrief showing what the model learned.

Session state is one key: the participant's code. Everything else is in the
database, so a refresh, a back button or a closed tab does not lose or duplicate
an answer.
"""

import csv
import os

import numpy as np
from django.db.models import Count
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone

from . import catalogue, plots, preference, protocol, simulation
from .models import Block, Participant, Response

SESSION_KEY = "project4_participant"
REPORT_NAME = "project4_report.pdf"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "static", "project4", REPORT_NAME)


# --------------------------------------------------------------------------- #
#  Landing page
# --------------------------------------------------------------------------- #

def index(request):
    results = simulation.load()
    return render(request, "project4/index.html", {
        "has_report": os.path.exists(REPORT_PATH),
        "sim": results,
        "figures": plots.all_figures(results) if results else {},
        "full": protocol.FULL,
        "demo": protocol.DEMO,
        "n_participants": Participant.objects.filter(is_demo=False).count(),
        "n_finished": Participant.objects.filter(is_demo=False,
                                                 finished_at__isnull=False).count(),
        "catalogue_size": catalogue.catalogue()["n"],
        "n_features": len(catalogue.FEATURE_NAMES),
        "seconds_pairwise": protocol.SECONDS_PAIRWISE,
        "seconds_ranking": protocol.SECONDS_RANKING,
    })


def report(request):
    if not os.path.exists(REPORT_PATH):
        raise Http404("The report has not been generated yet.")
    return FileResponse(open(REPORT_PATH, "rb"), as_attachment=True,
                        filename=REPORT_NAME)


# --------------------------------------------------------------------------- #
#  The study
# --------------------------------------------------------------------------- #

def consent(request):
    """Information sheet and consent. Nothing is recorded before this is signed."""
    is_demo = (request.POST.get("demo") if request.method == "POST"
               else request.GET.get("demo")) == "1"

    if request.method == "POST":
        if not request.POST.get("agree"):
            return render(request, "project4/consent.html", {
                "error": "The study cannot start without your consent.",
                "is_demo": is_demo, "full": protocol.FULL, "demo": protocol.DEMO,
            })
        return redirect("%s?demo=%s" % (reverse("project4:start"), "1" if is_demo else "0"))

    return render(request, "project4/consent.html",
                  {"is_demo": is_demo, "full": protocol.FULL, "demo": protocol.DEMO})


def start(request):
    """Create the participant, counterbalance the order, build both blocks."""
    is_demo = request.GET.get("demo") == "1"
    counts = dict(Participant.objects.filter(is_demo=is_demo)
                  .values("order").annotate(n=Count("id"))
                  .values_list("order", "n"))
    order = protocol.assign_order(counts)

    participant = Participant.objects.create(order=order, is_demo=is_demo)
    for index, condition in enumerate(participant.conditions):
        tasks, holdout = protocol.plan(condition, is_demo)
        Block.objects.create(participant=participant, index=index, condition=condition,
                             tasks_planned=tasks, holdout_planned=holdout)

    request.session[SESSION_KEY] = str(participant.code)
    return redirect("project4:task")


def _participant(request):
    code = request.session.get(SESSION_KEY)
    if not code:
        return None
    return Participant.objects.filter(code=code).first()


def task(request):
    participant = _participant(request)
    if participant is None:
        return redirect("project4:consent")

    if request.method == "POST":
        return _record(request, participant)

    block = participant.current_block()
    if block is None:
        return redirect("project4:final")
    if block.phase == "ratings":
        return redirect("project4:ratings")

    phase = block.phase
    position = block.tasks_done if phase == "elicitation" else block.holdout_done
    size = protocol.task_size(block.condition) if phase == "elicitation" else 2

    rng = protocol.rng_for(participant.code, "%d-%s" % (block.index, phase), position)
    shown = catalogue.sample_ids(size, rng, exclude=_already_shown(participant))

    return render(request, "project4/task.html", {
        "participant": participant,
        "block": block,
        "phase": phase,
        "position": position + 1,
        "total": block.tasks_planned if phase == "elicitation" else block.holdout_planned,
        "cards": catalogue.cards_for(shown),
        "shown": ",".join(str(i) for i in shown),
        "is_ranking": phase == "elicitation" and block.condition == "ranking",
        "progress": participant.progress,
        "block_number": block.index + 1,
        "condition_label": protocol.CONDITION_LABELS[block.condition],
    })


def _already_shown(participant):
    """Every film this participant has seen, so nothing is ever repeated.

    Two reasons this matters. The held out block must use films that were not
    part of the elicitation, or it measures memory rather than prediction. And
    the second block must not reuse the first block's films, or the second design
    would be scored on material the participant has already thought about.
    """
    seen = set()
    for response in Response.objects.filter(block__participant=participant):
        seen.update(int(i) for i in response.shown)
    return seen


def _record(request, participant):
    block = participant.current_block()
    if block is None:
        return redirect("project4:final")

    phase = request.POST.get("phase")
    position = int(request.POST.get("position", 0))
    shown = [int(i) for i in request.POST.get("shown", "").split(",") if i != ""]
    order = [int(i) for i in request.POST.get("order", "").split(",") if i != ""]
    seconds = float(request.POST.get("seconds") or 0) / 1000.0

    if phase not in (Response.ELICITATION, Response.HOLDOUT) \
            or sorted(order) != sorted(shown) or len(order) < 2:
        return redirect("project4:task")          # malformed, show the screen again

    _, created = Response.objects.get_or_create(
        block=block, phase=phase, position=position,
        defaults={"shown": shown, "order": order, "seconds": round(seconds, 2)},
    )
    if created:
        if phase == Response.ELICITATION:
            block.tasks_done += 1
            block.elicitation_seconds += round(seconds, 2)
        else:
            block.holdout_done += 1
        block.save()

    if block.phase == "ratings":
        _score(block)
        return redirect("project4:ratings")
    return redirect("project4:task")


def _score(block):
    """Fit this block's preference vector and score it on the held out answers."""
    elicitation = [r.order for r in block.responses.filter(phase=Response.ELICITATION)]
    holdout = [r.order for r in block.responses.filter(phase=Response.HOLDOUT)]

    blocks = preference.blocks_from_responses(elicitation, catalogue.features_for)
    dimension = catalogue.catalogue()["X"].shape[1]
    w = preference.fit(blocks, dimension=dimension)

    scores = preference.evaluate_holdout(w, holdout, catalogue.features_for)
    block.weights = [round(float(v), 4) for v in w]
    block.holdout_accuracy = scores["accuracy"]
    block.holdout_log_loss = scores["log_loss"]
    block.save()


RATINGS = [
    ("effort", "How much mental effort did that take?", "None at all", "A great deal"),
    ("frustration", "How frustrating was it?", "Not at all", "Very"),
    ("expressive", "Could you express your taste in films?", "Not at all", "Completely"),
    ("confidence", "How well do you think it now knows your taste?", "Not at all", "Very well"),
]


def ratings(request):
    """The workload questionnaire, asked once after each block."""
    participant = _participant(request)
    if participant is None:
        return redirect("project4:consent")

    block = participant.current_block()
    if block is None:
        return redirect("project4:final")
    if block.phase != "ratings":
        return redirect("project4:task")

    if request.method == "POST":
        answers = {}
        for key, _, _, _ in RATINGS:
            value = request.POST.get(key)
            if value:
                answers[key] = int(value)
        if len(answers) < len(RATINGS):
            return render(request, "project4/ratings.html", {
                "participant": participant, "block": block, "questions": RATINGS,
                "scale": [1, 2, 3, 4, 5, 6, 7],
                "condition_label": protocol.CONDITION_LABELS[block.condition],
                "block_number": block.index + 1,
                "error": "Please answer every question.",
            })
        block.ratings = answers
        block.save()
        return redirect("project4:task" if participant.current_block() else "project4:final")

    return render(request, "project4/ratings.html", {
        "participant": participant,
        "block": block,
        "questions": RATINGS,
        "scale": [1, 2, 3, 4, 5, 6, 7],
        "condition_label": protocol.CONDITION_LABELS[block.condition],
        "block_number": block.index + 1,
    })


def final(request):
    """The one question that needs both blocks to have happened."""
    participant = _participant(request)
    if participant is None:
        return redirect("project4:consent")
    if participant.current_block() is not None:
        return redirect("project4:task")

    if request.method == "POST":
        choice = request.POST.get("preferred")
        if choice in ("pairwise", "ranking", "no_preference"):
            participant.preferred_design = choice
            participant.exit_answers = {
                "comment_length": len(request.POST.get("comment", "")),
                "films_known": request.POST.get("films_known", ""),
            }
            participant.finished_at = timezone.now()
            participant.save()
            return redirect("project4:debrief")

    return render(request, "project4/final.html", {
        "participant": participant,
        "blocks": list(participant.blocks.all()),
    })


def debrief(request):
    participant = _participant(request)
    if participant is None:
        return redirect("project4:consent")

    blocks = list(participant.blocks.all())
    if not all(b.weights for b in blocks):
        return redirect("project4:task")

    data = catalogue.catalogue()
    pooled = np.mean([np.array(b.weights, dtype=float) for b in blocks], axis=0)
    picks = preference.recommend(pooled, data["X"],
                                 exclude=_already_shown(participant), top=5)
    cards = catalogue.cards_for([p["id"] for p in picks])

    return render(request, "project4/debrief.html", {
        "participant": participant,
        "blocks": [{
            "number": b.index + 1,
            "label": protocol.CONDITION_LABELS[b.condition],
            "accuracy": b.holdout_accuracy,
            "minutes": b.minutes,
            "tasks": b.tasks_done,
        } for b in blocks],
        "profile": preference.profile(pooled, data["feature_names"],
                                      catalogue.pretty_feature),
        "recommendations": [dict(card, utility=pick["utility"])
                            for card, pick in zip(cards, picks)],
    })


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

def export(request):
    """One row per answered screen, which is what the analysis starts from."""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="project4_responses.csv"'
    writer = csv.writer(response)
    writer.writerow(["participant", "order", "is_demo", "block", "condition",
                     "phase", "position", "shown", "chosen_order", "seconds",
                     "block_holdout_accuracy", "block_holdout_log_loss",
                     "block_elicitation_seconds", "block_ratings",
                     "preferred_design"])
    rows = Response.objects.select_related("block", "block__participant")
    for row in rows:
        block = row.block
        participant = block.participant
        writer.writerow([
            str(participant.code), participant.order, int(participant.is_demo),
            block.index, block.condition, row.phase, row.position,
            " ".join(str(i) for i in row.shown),
            " ".join(str(i) for i in row.order), row.seconds,
            block.holdout_accuracy, block.holdout_log_loss,
            block.elicitation_seconds, block.ratings, participant.preferred_design,
        ])
    return response
