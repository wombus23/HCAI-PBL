# Project 1: Supervised learning interface

An interface for the basic supervised learning loop: upload a CSV, look at it,
train a model on it, read the scores.

## Running it

The app needs nothing beyond the libraries the course already assumes:

    pip install django pandas numpy scikit-learn matplotlib
    python manage.py migrate
    python manage.py runserver

Then open http://127.0.0.1:8000/project1/. Two CSVs are included in
`project1/sample_data/` to try it with: `iris.csv` (classification, three flower
species from four measurements) and `diabetes.csv` (regression). Add an `Id`
column to either one to see the app drop it on its own.

## The three steps

| URL | What happens |
| --- | --- |
| `/project1/` | Upload a CSV, or open one that was uploaded before |
| `/project1/<id>/explore/` | Preview, per column statistics, five kinds of figure |
| `/project1/<id>/train/` | Choose an algorithm, sweep one hyperparameter, score every fit |
| `/project1/run/<id>/` | The result of one sweep, kept so runs can be compared |

## Files

- `ml.py` — everything that is not django: reading the CSV, detecting the kind
  of problem, the algorithm and score registries, the figures, the sweep. No
  django imports beyond `settings`, so the pipeline can be run and tested on its
  own.
- `models.py` — `Dataset` and `TrainingRun`.
- `forms.py` — upload, figure and training forms, including validation of the
  hyperparameter values the user types in.
- `views.py` — thin: it validates input, calls `ml.py` and renders.

## What the user controls, and what the app decides

This was the question in Task 4. The split follows one rule: the user decides
anything that changes the answer, and the app handles anything where a wrong
choice would silently produce a misleading number.

**The user decides**

- Which CSV, and whether identifier columns are dropped.
- Whether the problem is classification or regression. The app guesses, states
  the guess and the reason for it in plain words, and offers a single button to
  overrule it.
- Which figure to draw and which columns go on which axis.
- The learning algorithm, out of five per problem type.
- Which hyperparameter values to try. The field is prefilled with a sensible
  range for the selected algorithm and refills when the algorithm changes, so
  the user starts from something reasonable without being locked into it.
- The size of the test set, the random seed for the split, and whether the split
  keeps the class balance.
- The score. Accuracy is not forced on anyone: balanced accuracy and macro F1
  are there for unbalanced classes, and each score carries a one line
  explanation of when it misleads.

**The app decides**

- Which column is the label: the last one, as the project description specifies.
- Feature scaling. It is applied only to the algorithms that need it (k nearest
  neighbours, logistic regression, support vector machines, ridge), and it is
  fitted inside the pipeline on the training split only, so no information leaks
  from the test set. Making this a checkbox would mostly produce quietly broken
  models.
- Missing values, filled with the median for numbers and the most frequent value
  for categories, again inside the pipeline.
- Text features, one hot encoded when they have at most 20 levels and left out
  otherwise. The run page lists exactly which columns went in each direction.
- Which value of the hyperparameter is the best one, since for RMSE and mean
  absolute error the best model is the one with the lowest number.

**What the app says out loud**

Every automatic decision is reported back rather than hidden: which columns were
dropped as identifiers, why the problem was read as classification, how many
rows ended up in each half, which features were scaled or encoded, and whether
the gap between the training and test score looks like overfitting. That is the
human centric part. The user does not have to control a step to need to know it
happened.

## Choices worth naming

- **Both problem types are supported.** The type is detected from the label
  column: text or few distinct whole numbers means classification, a spread of
  numeric values means regression. The rule is deliberately simple so it can be
  explained in one sentence in the interface, and it can always be overruled.
- **Training and test scores are both shown.** A single test number tells the
  user which model won but not why, and the gap between the two curves is the
  clearest way to show overfitting without a lecture on it.
- **Runs are stored.** `TrainingRun` keeps the settings, the per value results
  and the figures, so the user can compare what they tried an hour ago instead
  of retyping it. This is also why the sweep redirects to its own page rather
  than rendering the results into the form response.
- **Figures are matplotlib images written to the media directory**, following
  the approach shown in the `demos` app.

## Known limits

- The sweep covers one hyperparameter per algorithm, not a grid over several.
- The split is a single train and test split, not cross validation, which is
  what the lecture described and what keeps the runtime interactive.
- Uploads are capped at 10 MB and 15 hyperparameter values per sweep, so a
  single request cannot tie up the development server.
