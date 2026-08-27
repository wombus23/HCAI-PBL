"""
One page, four regions, two shared controls.

The project sheet asks for the counterfactual and feature effect regions to be
linked to the model class and to lambda. The simplest way to guarantee that is to
have a single form holding every control and to keep the whole state in the query
string: whatever the URL says, all four regions are showing the same model.
"""

from django.shortcuts import render

from . import counterfactuals, effects, plots
from .data import NUMERIC_FEATURES, PRETTY, penguins
from .modelgrid import LOGREG, MODEL_LABELS, TREE, grid, lambda_range, select

MAX_K = 10
MAX_N = 20000


def _clamp(value, low, high, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def index(request):
    data = penguins()
    params = request.GET

    kind = params.get("model")
    if kind not in (TREE, LOGREG):
        kind = TREE

    lam_max, lam_step = lambda_range(kind)
    try:
        lam = float(params.get("lam", 0))
    except (TypeError, ValueError):
        lam = 0.0
    lam = max(0.0, min(lam_max, lam))

    model, table = select(kind, lam)
    models = grid(kind)
    chosen_index = next(i for i, row in enumerate(table) if row["chosen"])

    # --- tasks 1 to 3: the model itself -------------------------------------
    if kind == TREE:
        model_figure = plots.tree_figure(model)
        coefficient_rows = []
    else:
        model_figure = plots.coefficient_figure(model)
        coefficient_rows = model.coefficient_table(data)

    selection_figure = plots.selection_figure(models, chosen_index, lam, model.omega_label)

    # --- task 4: counterfactuals --------------------------------------------
    cf_index = _clamp(params.get("cf_index", 0), 0, len(data.raw) - 1, 0)
    cf_k = _clamp(params.get("cf_k", 5), 1, MAX_K, 5)
    cf_n = _clamp(params.get("cf_n", 2000), 200, MAX_N, 2000)

    predicted = data.classes[int(model.predict(data.X[cf_index:cf_index + 1])[0])]
    cf_target = params.get("cf_target")
    if cf_target not in data.classes:
        alternatives = [c for c in data.classes if c != predicted]
        cf_target = alternatives[0]

    cf_rows, cf_info = counterfactuals.generate(model, cf_index, cf_target, k=cf_k, n=cf_n)

    # --- task 5: feature effects --------------------------------------------
    feature = params.get("feature")
    if feature not in NUMERIC_FEATURES:
        feature = NUMERIC_FEATURES[0]

    pdp_grid, pdp_curves = effects.partial_dependence(model, feature)
    ale_edges, ale_curves, ale_method, ale_gap = effects.accumulated_local_effects(model, feature)

    context = {
        "data": data,
        "kind": kind,
        "model": model,
        "model_label": MODEL_LABELS[kind],
        "other_kind": LOGREG if kind == TREE else TREE,
        "other_label": MODEL_LABELS[LOGREG if kind == TREE else TREE],
        "lam": lam,
        "lam_max": lam_max,
        "lam_step": lam_step,
        "table": table,
        "model_figure": model_figure,
        "coefficient_rows": coefficient_rows,
        "selection_figure": selection_figure,
        "classes": data.classes,

        "choices": data.choices(),
        "cf_index": cf_index,
        "cf_target": cf_target,
        "cf_k": cf_k,
        "cf_n": cf_n,
        "cf_rows": cf_rows,
        "cf_info": cf_info,
        "cf_predicted": predicted,
        "cf_actual": data.label_of(cf_index),
        "cf_original": counterfactuals.original_row(cf_index),
        "feature_names": [PRETTY[f] for f in counterfactuals.FEATURES],

        "feature": feature,
        "feature_label": PRETTY[feature],
        "numeric_features": [(f, PRETTY[f]) for f in NUMERIC_FEATURES],
        "pdp_figure": plots.effect_figure("pdp", pdp_grid, pdp_curves, feature),
        "ale_figure": plots.effect_figure("ale", ale_edges, ale_curves, feature, ale_method),
        "ale_method": ale_method,
        "ale_gap": None if ale_gap is None else round(ale_gap, 6),
    }
    return render(request, "project2/index.html", context)
