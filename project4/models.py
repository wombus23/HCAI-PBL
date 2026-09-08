"""
What the study records.

The design is within participants: everyone does both elicitation designs, in a
counterbalanced order, on disjoint sets of films. So the unit of analysis is a
block rather than a person, and the tables mirror that: a participant has two
blocks, a block has its own elicitation answers, its own held out answers, its
own fitted preference vector and its own workload ratings.

Everything needed to refit the model is stored raw: which films were shown, the
order the participant put them in, and how long it took. The fitted vector and
the held out score are stored as a convenience and can be recomputed.

No names, no email addresses, no free text is kept — the comment box records only
its length, as an engagement signal. A participant is a random identifier and
nothing else, which is what makes the consent form's promise of anonymity true
rather than aspirational.
"""

import uuid

from django.db import models

PAIRWISE = "pairwise"
RANKING = "ranking"
CONDITIONS = [(PAIRWISE, "Design 1: pairwise comparisons"),
              (RANKING, "Design 2: ranking of ten")]


class Participant(models.Model):
    ORDERS = [("pairwise_first", "Design 1 then Design 2"),
              ("ranking_first", "Design 2 then Design 1")]

    code = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    order = models.CharField(max_length=20, choices=ORDERS)
    is_demo = models.BooleanField(default=False)

    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    preferred_design = models.CharField(max_length=20, blank=True)
    exit_answers = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return "%s (%s)" % (str(self.code)[:8], self.order)

    @property
    def conditions(self):
        return ([PAIRWISE, RANKING] if self.order == "pairwise_first"
                else [RANKING, PAIRWISE])

    def current_block(self):
        for block in self.blocks.all():
            if block.phase != "done":
                return block
        return None

    @property
    def progress(self):
        blocks = list(self.blocks.all())
        total = sum(b.tasks_planned + b.holdout_planned for b in blocks)
        done = sum(b.tasks_done + b.holdout_done for b in blocks)
        return int(round(100 * done / total)) if total else 0


class Block(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE,
                                    related_name="blocks")
    index = models.PositiveIntegerField()
    condition = models.CharField(max_length=20, choices=CONDITIONS)

    tasks_planned = models.PositiveIntegerField(default=0)
    tasks_done = models.PositiveIntegerField(default=0)
    holdout_planned = models.PositiveIntegerField(default=0)
    holdout_done = models.PositiveIntegerField(default=0)

    weights = models.JSONField(default=list, blank=True)
    holdout_accuracy = models.FloatField(null=True, blank=True)
    holdout_log_loss = models.FloatField(null=True, blank=True)
    elicitation_seconds = models.FloatField(default=0)
    ratings = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["participant", "index"]
        unique_together = [("participant", "index")]

    def __str__(self):
        return "block %d (%s)" % (self.index, self.condition)

    @property
    def phase(self):
        if self.tasks_done < self.tasks_planned:
            return "elicitation"
        if self.holdout_done < self.holdout_planned:
            return "holdout"
        if not self.ratings:
            return "ratings"
        return "done"

    @property
    def minutes(self):
        return round(self.elicitation_seconds / 60.0, 1)


class Response(models.Model):
    ELICITATION = "elicitation"
    HOLDOUT = "holdout"
    PHASES = [(ELICITATION, "Elicitation"), (HOLDOUT, "Held out")]

    block = models.ForeignKey(Block, on_delete=models.CASCADE, related_name="responses")
    phase = models.CharField(max_length=20, choices=PHASES)
    position = models.PositiveIntegerField()

    shown = models.JSONField(default=list)     # catalogue ids, as presented
    order = models.JSONField(default=list)     # catalogue ids, best first
    seconds = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["block", "phase", "position"]
        unique_together = [("block", "phase", "position")]

    def __str__(self):
        return "%s #%d" % (self.phase, self.position)
