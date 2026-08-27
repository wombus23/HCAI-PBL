from django import forms

from . import ml

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SWEEP_VALUES = 15


class UploadForm(forms.Form):
    file = forms.FileField(
        label="CSV file",
        help_text="First row: column names. Last column: the label to predict.",
    )
    name = forms.CharField(
        label="Name for this dataset",
        max_length=120,
        required=False,
        help_text="Leave empty to use the file name.",
    )
    drop_ids = forms.BooleanField(
        label="Drop identifier columns automatically",
        required=False,
        initial=True,
        help_text="Removes columns such as Id that number the rows instead of describing them.",
    )

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        if not uploaded.name.lower().endswith(".csv"):
            raise forms.ValidationError("Only .csv files can be read here.")
        if uploaded.size > MAX_UPLOAD_BYTES:
            raise forms.ValidationError("The file is larger than 10 MB.")
        return uploaded


class ExploreForm(forms.Form):
    """Which figure to draw. The choices depend on the uploaded file, so they
    are filled in at runtime rather than declared on the class."""

    PLOT_CHOICES = [
        ("scatter", "Two features, coloured by the label"),
        ("feature_target", "One feature against the label"),
        ("distribution", "Distribution of one feature"),
        ("balance", "Distribution of the label"),
        ("correlation", "Correlation between all numeric columns"),
    ]

    plot = forms.ChoiceField(label="Figure", choices=PLOT_CHOICES)
    x = forms.ChoiceField(label="First feature", required=False)
    y = forms.ChoiceField(label="Second feature", required=False)

    def __init__(self, columns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [(c, c) for c in columns]
        self.fields["x"].choices = choices
        self.fields["y"].choices = choices


class TrainForm(forms.Form):
    """Everything the user is in control of for one sweep."""

    algorithm = forms.ChoiceField(label="Learning algorithm")
    values = forms.CharField(
        label="Hyperparameter values to try",
        help_text="Separate the values with commas. One model is trained per value.",
    )
    test_size = forms.IntegerField(
        label="Share of the data held back for testing (%)",
        min_value=10, max_value=50, initial=25,
    )
    random_state = forms.IntegerField(
        label="Random seed for the split",
        initial=0,
        help_text="Keep the same seed to compare two runs on the same split.",
    )
    stratify = forms.BooleanField(
        label="Keep the class balance in both halves",
        required=False, initial=True,
    )
    score = forms.ChoiceField(label="Score")

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.fields["algorithm"].choices = [
            (key, spec["label"]) for key, spec in ml.algorithms_for(task)
        ]
        self.fields["score"].choices = [
            (key, spec["label"]) for key, spec in ml.scores_for(task)
        ]
        if task == ml.REGRESSION:
            self.fields["stratify"].widget = forms.HiddenInput()
            self.fields["stratify"].initial = False

    def clean(self):
        cleaned = super().clean()
        algorithm = cleaned.get("algorithm")
        raw = cleaned.get("values")
        if not algorithm or not raw:
            return cleaned

        spec = ml.ALGORITHMS[algorithm]
        parsed = []
        for chunk in raw.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = int(chunk) if spec["parameter_type"] == "int" else float(chunk)
            except ValueError:
                raise forms.ValidationError(
                    "%s is not a valid value for %s." % (chunk, spec["parameter_label"])
                )
            if value <= 0:
                raise forms.ValidationError(
                    "%s must be greater than zero." % spec["parameter_label"]
                )
            if value not in parsed:
                parsed.append(value)

        if not parsed:
            raise forms.ValidationError("Give at least one hyperparameter value.")
        if len(parsed) > MAX_SWEEP_VALUES:
            raise forms.ValidationError(
                "Try at most %d values at a time." % MAX_SWEEP_VALUES
            )

        parsed.sort()
        cleaned["parsed_values"] = parsed
        return cleaned
