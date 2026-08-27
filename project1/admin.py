from django.contrib import admin

from .models import Dataset, TrainingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "n_rows", "n_features", "target_name",
                    "detected_task", "uploaded_at")
    list_filter = ("detected_task",)


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("dataset", "algorithm", "score_name", "best_value",
                    "best_score", "created_at")
    list_filter = ("algorithm", "score_name")
