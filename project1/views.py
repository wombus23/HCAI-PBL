import os

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from . import ml
from .forms import ExploreForm, TrainForm, UploadForm
from .models import Dataset, TrainingRun


def index(request):
    """Upload a CSV, or pick one that was uploaded earlier."""
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = _create_dataset(request, form)
            if dataset is not None:
                return redirect("project1:explore", pk=dataset.pk)
    else:
        form = UploadForm()

    return render(request, "project1/index.html", {
        "form": form,
        "datasets": Dataset.objects.all(),
        "step": 1,
    })


def _create_dataset(request, form):
    """Save the upload, read it once and remember what we found."""
    uploaded = form.cleaned_data["file"]
    name = form.cleaned_data["name"] or os.path.splitext(uploaded.name)[0]
    drop_ids = form.cleaned_data["drop_ids"]

    dataset = Dataset(name=name[:120], file=uploaded)
    dataset.save()

    try:
        df, dropped = ml.load_dataframe(dataset.file.path, drop_ids=drop_ids)
        target = df.columns[-1]
        task = ml.infer_task(df[target])
    except ml.DatasetError as error:
        dataset.delete()
        messages.error(request, str(error))
        return None

    dataset.n_rows = len(df)
    dataset.n_features = df.shape[1] - 1
    dataset.target_name = target
    dataset.feature_names = list(df.columns[:-1])
    dataset.dropped_columns = dropped
    dataset.detected_task = task
    dataset.detection_reason = ml.task_explanation(df[target], task)
    dataset.save()

    if dropped:
        messages.info(request, "Ignored %s as an identifier column." % ", ".join(dropped))
    return dataset


def explore(request, pk):
    """Look at the data before training anything."""
    dataset = get_object_or_404(Dataset, pk=pk)
    try:
        df = dataset.load()
    except ml.DatasetError as error:
        messages.error(request, str(error))
        return redirect("project1:index")

    target = dataset.target_name
    task = dataset.detected_task
    plottable = ml.numeric_columns(df, exclude=[target])
    if not plottable:
        plottable = [c for c in df.columns if c != target]

    form = ExploreForm(plottable, request.GET or None)
    plot_url, plot_error = None, None

    if form.is_valid():
        kind = form.cleaned_data["plot"]
        x = form.cleaned_data.get("x") or plottable[0]
        y = form.cleaned_data.get("y") or plottable[min(1, len(plottable) - 1)]
    else:
        form = ExploreForm(plottable, initial={
            "plot": "scatter" if len(plottable) > 1 else "distribution",
            "x": plottable[0],
            "y": plottable[min(1, len(plottable) - 1)],
        })
        kind = form.initial["plot"]
        x, y = form.initial["x"], form.initial["y"]

    try:
        plot_url = _draw(df, kind, x, y, target, task)
    except Exception as error:  # a bad column combination should not 500
        plot_error = "That figure could not be drawn: %s" % error

    return render(request, "project1/explore.html", {
        "dataset": dataset,
        "form": form,
        "plot_url": plot_url,
        "plot_error": plot_error,
        "preview_columns": list(df.columns),
        "preview_rows": df.head(8).values.tolist(),
        "summary": ml.summarise(df, target),
        "task_label": dict(Dataset.TASK_CHOICES)[task],
        "other_task": ml.REGRESSION if task == ml.CLASSIFICATION else ml.CLASSIFICATION,
        "other_task_label": dict(Dataset.TASK_CHOICES)[
            ml.REGRESSION if task == ml.CLASSIFICATION else ml.CLASSIFICATION],
        "step": 2,
    })


def _draw(df, kind, x, y, target, task):
    if kind == "scatter":
        return ml.plot_scatter(df, x, y, target, task)
    if kind == "feature_target":
        return ml.plot_feature_vs_target(df, x, target)
    if kind == "distribution":
        return ml.plot_distribution(df, x, target, task)
    if kind == "balance":
        return ml.plot_target_balance(df, target, task)
    if kind == "correlation":
        return ml.plot_correlation(df, target)
    return None


def set_task(request, pk):
    """Override the automatic classification or regression detection."""
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        task = request.POST.get("task")
        if task in dict(Dataset.TASK_CHOICES):
            dataset.detected_task = task
            dataset.detection_reason = "set by hand"
            dataset.save(update_fields=["detected_task", "detection_reason"])
            messages.info(request, "Now treating this dataset as a %s problem." % task)
    return redirect(request.POST.get("next") or "project1:explore", pk=dataset.pk)


def train(request, pk):
    """Choose a model, sweep one hyperparameter, score every fit."""
    dataset = get_object_or_404(Dataset, pk=pk)
    task = dataset.detected_task

    if request.method == "POST":
        form = TrainForm(task, request.POST)
        if form.is_valid():
            run = _run_sweep(request, dataset, task, form)
            if run is not None:
                return redirect("project1:run", pk=run.pk)
    else:
        first_algorithm = ml.algorithms_for(task)[0]
        form = TrainForm(task, initial={
            "algorithm": first_algorithm[0],
            "values": ", ".join(str(v) for v in first_algorithm[1]["default_grid"]),
            "score": ml.scores_for(task)[0][0],
        })

    defaults = {
        key: {"grid": ", ".join(str(v) for v in spec["default_grid"]),
              "label": spec["parameter_label"]}
        for key, spec in ml.algorithms_for(task)
    }

    return render(request, "project1/train.html", {
        "dataset": dataset,
        "form": form,
        "defaults": defaults,
        "task_label": dict(Dataset.TASK_CHOICES)[task],
        "algorithms": ml.algorithms_for(task),
        "scores": ml.scores_for(task),
        "runs": dataset.runs.all()[:10],
        "step": 3,
    })


def _run_sweep(request, dataset, task, form):
    try:
        df = dataset.load()
        result = ml.run_sweep(
            df,
            dataset.target_name,
            form.cleaned_data["algorithm"],
            form.cleaned_data["parsed_values"],
            form.cleaned_data["test_size"] / 100.0,
            form.cleaned_data["random_state"],
            form.cleaned_data["score"],
            stratify=form.cleaned_data.get("stratify", False),
        )
    except ml.DatasetError as error:
        messages.error(request, str(error))
        return None
    except Exception as error:
        messages.error(request, "Training stopped: %s" % error)
        return None

    spec = ml.ALGORITHMS[form.cleaned_data["algorithm"]]
    return TrainingRun.objects.create(
        dataset=dataset,
        algorithm=form.cleaned_data["algorithm"],
        task_type=task,
        hyperparameter=spec["parameter_label"],
        values=form.cleaned_data["parsed_values"],
        test_size=form.cleaned_data["test_size"] / 100.0,
        random_state=form.cleaned_data["random_state"],
        stratified=result["stratified"],
        score_name=form.cleaned_data["score"],
        results=result,
        best_value=str(result["best_value"]),
        best_score=result["best_score"],
        seconds=result["seconds"],
    )


def run(request, pk):
    """The result of one sweep, kept so runs can be compared later."""
    training_run = get_object_or_404(TrainingRun, pk=pk)
    result = training_run.results
    best_value = result.get("best_value")
    rows = [dict(row, is_best=row["value"] == best_value)
            for row in result.get("rows", [])]

    return render(request, "project1/run.html", {
        "run": training_run,
        "dataset": training_run.dataset,
        "result": result,
        "rows": rows,
        "score_note": ml.SCORES[training_run.score_name]["note"],
        "algorithm_note": ml.ALGORITHMS[training_run.algorithm]["note"],
        "test_percent": round(training_run.test_size * 100),
        "step": 3,
    })


def delete_dataset(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        name = dataset.name
        dataset.delete()
        messages.info(request, "Removed %s." % name)
    return redirect("project1:index")
