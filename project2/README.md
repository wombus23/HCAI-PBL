# Project 2: Explainability

One page at `/project2/` with four linked regions, built on the Palmer Penguins
dataset. Two controls at the top, model class and λ, decide which model every
region below is talking about.

## Files

- `data.py` — reads the CSV, fixes the encoding and the train/test split.
- `modelgrid.py` — fits the grid of models per class and applies the λ rule.
- `counterfactuals.py` — local sampling, MAD weighted ranking (task 4).
- `effects.py` — PDP and ALE, written out rather than imported (task 5).
- `plots.py` — the figures.
- `views.py` — one view; all state lives in the query string.

No database tables. The dataset is fixed, the model grids are fitted on first
use and cached in memory, and everything else is derived from the URL.

## The dataset

`sample_data/penguins.csv` ships with the app instead of being pulled in through
the `palmerpenguins` package, because the submission rules ask for no unusual
dependencies and an import that fails on the grader's machine is a zero rather
than a missing feature. It is the same file the package distributes.

333 of the 344 rows are complete. The other 11 are dropped rather than imputed:
filling in a penguin's sex or bill length would put invented values inside an
explanation, which is the one thing an explainability tool should not do. The
count is stated in the interface.

Year has three values and is treated as categorical. That is why the effect
plots offer four features rather than five: a partial dependence curve over
three integers is a bar chart in disguise.

## Tasks 1 to 3: interpretability and complexity

A grid of models is fitted per class, from heavily regularized to unconstrained,
and λ picks between the finished models by

    argmax_f  acc_test(f) − λ · Ω(f)

which is deliberately not the penalty used while fitting. The distinction is
called out in the interface, since it is the easiest thing to misread about the
task.

- **Decision tree.** `max_leaf_nodes` from 2 to unlimited, Ω is the leaf count
  the fitted tree actually has, which is not always the cap it was given.
- **Logistic regression.** An L1 penalty over a grid of C, so coefficients reach
  exactly zero. Ω is the number of original features still in use, counting a
  categorical feature once rather than once per one hot column. Counting
  features rather than raw coefficients keeps Ω on the same footing as the leaf
  count: both answer how many things a person has to read to understand the
  model.

The slider range is computed from the grid rather than hard coded, so both ends
of the trade off are always reachable: at λ = 0 the most accurate model wins, and
at the top of the range the simplest one does. Ties go to the simpler model.

The trade off figure plots every model as a point in (Ω, accuracy) and draws the
line of slope λ through the winner, which is what the argmax is doing.

## Task 4: counterfactuals

Sample N points around x, keep the ones the model assigns to the target class,
rank by distance, show the best k.

- **Noising continuous features**: gaussian noise scaled by the feature's
  standard deviation, clipped to the range observed in the dataset so the
  counterfactual is a penguin that could exist, then rounded to the precision
  the data is recorded in.
- **Noising categorical features**: island, sex and year are resampled, with
  probability p, uniformly among their other levels. Adding gaussian noise to a
  one hot column would produce rows that are not valid penguins at all.
- **Distance**: MAD weighted L1 over the four measurements, plus 1 for each
  categorical feature that differs. The MAD weighting is what makes 200 g of body
  mass comparable to 2 mm of bill, and the flat cost of 1 per changed category
  makes one changed island cost about one MAD of a measurement, which is a
  defensible exchange rate for someone reading the table.
- **When nothing is found**: the search widens over up to six rounds, doubling N
  and increasing both the gaussian spread and the flip probability, then reports
  what it tried instead of silently returning nothing. This happens for real at
  the far end of the λ slider, where the logistic model has been regularized down
  to zero features and predicts one class for everything: no amount of sampling
  produces a counterfactual, because the model ignores its input.

The region is driven by the same model as everything else, so moving the λ
slider changes the counterfactuals.

## Task 5: PDP and ALE

Both are computed here, without a library.

**PDP** forces the feature to each value on a grid across the whole dataset and
averages the predicted probability, one curve per species. Checked against
`sklearn.inspection.partial_dependence` during development: the curves match to
floating point. The library is not used at runtime.

**ALE** integrates the average partial derivative of the predicted probability
over the feature:

    ALE_c(v) = ∫ E[ ∂f_c/∂x_s | x_s = z ] dz

evaluated on quantile bins and centred so the curve averages to zero.

Which derivative, which is the question the sheet asks:

- **Logistic regression: exact.** With softmax probabilities p and coefficients
  W, and a feature standardized by `scale_s` before fitting,

      ∂p_c/∂x_s = ( 1 / scale_s ) · p_c · ( W[c,s] − Σ_k p_k W[k,s] )

  The app uses this, and also computes the discretized version to report the gap
  between the two as a check. It is around 1e-3 to 1e-2 depending on the feature,
  the larger gaps appearing where the probability turns fastest, which is what
  the rectangle rule would be expected to do.

- **Decision tree: discretization required.** The prediction is piecewise
  constant, so the derivative is zero almost everywhere and undefined on the
  split points. There is nothing to integrate, so the derivative inside a bin is
  replaced by the finite difference across that bin's edges.

Both plots are drawn for the same feature so they can be compared. PDP averages
over combinations that may not exist when features are correlated, while ALE only
compares predictions inside a narrow band of the feature. Flipper length and body
mass are strongly correlated in this dataset, so the difference is visible rather
than theoretical.

## Known limits

- One feature at a time in the effect plots; no two way PDP or ALE.
- The counterfactual search is random sampling, not optimisation, so it finds
  close counterfactuals rather than the closest one. Raising N narrows the gap.
- ALE uses 10 quantile bins, fixed. Fewer bins smooth the curve, more bins make
  it noisy at the tails where few penguins sit.
