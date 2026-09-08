from django.contrib import admin

from .models import Block, Participant, Response


@admin.register(Participant)
class ParticipantAdmin(admin.ModelAdmin):
    list_display = ("code", "order", "is_demo", "preferred_design", "started_at",
                    "finished_at")
    list_filter = ("order", "is_demo")


@admin.register(Block)
class BlockAdmin(admin.ModelAdmin):
    list_display = ("participant", "index", "condition", "tasks_done",
                    "holdout_accuracy", "elicitation_seconds")
    list_filter = ("condition",)


@admin.register(Response)
class ResponseAdmin(admin.ModelAdmin):
    list_display = ("block", "phase", "position", "seconds", "created_at")
    list_filter = ("phase",)
