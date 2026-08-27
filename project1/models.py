import os

from django.db import models

from . import ml


class Dataset(models.Model):
    """A CSV the user uploaded, plus what we worked out about it on the way in.

    The metadata is stored once at upload time so the index page can list the
    datasets without reopening every file.
    """

    TASK_CHOICES = [
        (ml.CLASSIFICATION, "Classification"),
        (ml.REGRESSION, "Regression"),
    ]

    name = models.CharField(max_length=120)
    file = models.FileField(upload_to="project1/datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    n_rows = models.PositiveIntegerField(default=0)
    n_features = models.PositiveIntegerField(default=0)
    target_name = models.CharField(max_length=120, blank=True)
    feature_names = models.JSONField(default=list, blank=True)
    dropped_columns = models.JSONField(default=list, blank=True)
    detected_task = models.CharField(max_length=20, choices=TASK_CHOICES,
                                     default=ml.CLASSIFICATION)
    detection_reason = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name

    def load(self):
        """Read the stored CSV back into a dataframe."""
        df, _ = ml.load_dataframe(self.file.path, drop_ids=bool(self.dropped_columns))
        return df

    def delete(self, *args, **kwargs):
        path = self.file.path if self.file else None
        super().delete(*args, **kwargs)
        if path and os.path.exists(path):
            os.remove(path)


class TrainingRun(models.Model):
    """One hyperparameter sweep: the settings the user chose and what came out.

    Keeping runs in the database is what makes the interface useful rather than
    just interactive: the user can compare what they tried an hour ago.
    """

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE,
                                related_name="runs")
    algorithm = models.CharField(max_length=40)
    task_type = models.CharField(max_length=20, choices=Dataset.TASK_CHOICES)
    hyperparameter = models.CharField(max_length=60)
    values = models.JSONField(default=list)
    test_size = models.FloatField(default=0.25)
    random_state = models.IntegerField(default=0)
    stratified = models.BooleanField(default=True)
    score_name = models.CharField(max_length=40)

    results = models.JSONField(default=list)
    best_value = models.CharField(max_length=40, blank=True)
    best_score = models.FloatField(null=True, blank=True)
    seconds = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return "%s on %s" % (self.algorithm_label, self.dataset.name)

    @property
    def algorithm_label(self):
        spec = ml.ALGORITHMS.get(self.algorithm)
        return spec["label"] if spec else self.algorithm

    @property
    def score_label(self):
        spec = ml.SCORES.get(self.score_name)
        return spec["label"] if spec else self.score_name
