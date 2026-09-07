# Project 3: Active learning for learning to defer

A classifier and a simulated human expert answering AG News articles together.
The interface at `/project3/` presents the results of all four tasks and serves
the required PDF report.

## Running it

The results are committed, so the page works straight after a clone. To
regenerate them:

    python manage.py run_experiments     # ~30 s, rewrites results/results.json
    python manage.py build_report        # rewrites the PDF, needs reportlab

Everything is seeded, so a rerun reproduces the committed numbers exactly.

## Why the experiments are cached

Vectorizing 24,000 documents and refitting the deferral model 400 times does not
belong inside an HTTP request. The management command writes
`results/results.json`, the views read it, and the figures are redrawn from those
numbers on each request. That keeps binary images out of the repository while
letting the page stay live, and it means the page and the PDF report can never
disagree, since both are generated from the same file.

## Files

- `data.py` — dataset, TF-IDF features, LSA, k-means regions, the task 1 classifier.
- `expert.py` — the simulated expert (task 2).
- `defer.py` — calibration, rejector, deferral policies and their evaluation (task 3).
- `active.py` — the four query strategies and the learning curves (task 4).
- `experiments.py` — runs all of it and writes the JSON.
- `report.py` — builds the PDF from that JSON.
- `plots.py`, `views.py`, `urls.py` — the interface.
- `prepare_data.py` — how the shipped CSVs were produced from the original AG News.

## The dataset

`sample_data/` holds the full 7,600 article test set and a stratified 24,000
article sample of the 120,000 article training set, as gzipped CSVs. Shipping the
data rather than pulling it through the `datasets` package follows the same rule
as project 2: no unusual dependencies. Sampling keeps the repository at a few
megabytes and every experiment runnable in well under a minute.

## Task 1: baseline

TF-IDF over words and bigrams, then multinomial logistic regression. Test
accuracy 0.9071.

A fine tuned transformer would score two to three points higher and would have
made the rest impractical, since task 4 refits the deferral model hundreds of
times. One property did matter for the choice: the model has to produce a usable
confidence, because deferral compares it against the expert. That rules out a
linear SVM's unnormalised decision values.

## Task 2: the expert

Competence is attached to k-means regions of the document space, not to the
label. Four strong regions (0.88 to 0.95), four middling, four weak (0.25 to
0.35), giving 0.655 overall.

Attaching competence to the label would have been simpler and would have ruined
task 4: the profile could then be inferred from the classifier's own predictions
without asking the expert anything. Regions lean towards topics without being
topics, and the measured per class accuracies are much flatter than the per
region ones — that gap is the design working.

Two further details. When the expert is wrong, the wrong answer comes from a
confusion profile rather than uniformly, so mistakes are plausible. And answers
are fixed per article, so asking twice gives the same answer; otherwise a query
strategy could buy accuracy by asking the same question repeatedly.

## Task 3: learning to defer

Defer when `P(expert correct | x) − P(classifier correct | x) > 0`.

The rejector estimating `P(expert correct | x)` is a logistic regression on the
same TF-IDF features, trained on whether the expert's answer was right. It never
sees the region assignments.

`P(classifier correct | x)` is **not** the model's maximum probability. A linear
model on TF-IDF is overconfident, and comparing a raw confidence against a
calibrated rejector puts the two sides of the inequality on different scales, so
the system defers far too rarely. The confidence is calibrated first, from out of
fold predictions on the training set.

Team accuracy 0.9233 against 0.9071 for the classifier alone, at an 8.5% deferral
rate, recovering 25.5% of the distance to the oracle.

The evaluation deliberately goes past team accuracy, as the sheet asks. Random
deferral at the same rate scores *below* the classifier alone: handing work to a
65% accurate expert is harmful unless it is aimed. And the learned rejector and a
confidence only rule land within 0.003 of each other on accuracy while behaving
quite differently — the expert is right on 0.802 of what the rejector defers
against 0.723 for the confidence rule. Confidence finds hard articles; the
rejector finds articles this expert is good at.

## Task 4: active learning

No expert answers at the start; each round buys 50, up to 1,000 (4% of the
training set), averaged over five seeds.

| Strategy | Team accuracy | Share of the full label gain | Questions to 75% |
| --- | --- | --- | --- |
| Deferral margin | 0.9208 | 0.85 | 300 |
| Random | 0.9199 | 0.79 | 600 |
| Classifier uncertainty | 0.9197 | 0.78 | 650 |
| Expert uncertainty | 0.9152 | 0.50 | not reached |

The result worth presenting is that the accuracy ranking and the AUC ranking
disagree. Random builds the best model of the expert's competence, because it
samples the input space representatively. Deferral margin builds a worse model
and a better team, because it spends its budget where the deferral decision is
actually in the balance: knowing the expert's competence precisely in a region
where the classifier is at 99% changes nothing.

Classifier uncertainty, the textbook baseline, is no better than random — it
selects articles that are hard to classify, which is a different question from
where the expert helps. Expert uncertainty is worse than random: chasing points
where the competence model is unsure skews its training sample and its deferral
rate collapses to 0.03.

## The report

`static/project3/project3_report.pdf`, nine pages, reachable from the download
button at the top of the interface, as the project sheet requires.

It is built by `report.py` from the same JSON the page reads. This is the only
part of the repository that needs a library outside numpy, sklearn, matplotlib
and pandas: reportlab, for the PDF layout. The built file is committed, so
nobody needs reportlab installed to read the report, run the app, or use the
download button — only to rebuild it.

## Known limits

- The expert is simulated in a way the rejector is well matched to: competence is
  piecewise constant over regions of the same feature space the rejector uses. A
  real expert would not be so conveniently shaped.
- The classifier is frozen during active learning. In a real deployment an
  expert's answer is also a label, and using it for both is the obvious extension.
- One expert, always available, with no cost per deferral at test time.
- Task 5, the interactive human-in-the-loop interface, is not implemented. It is
  marked optional in the project sheet.
