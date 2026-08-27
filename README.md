# Human-Centric AI: projects

Course projects for Human-Centric Artificial Intelligence, built on the django
skeleton from https://github.com/ppaamm/HCAI-PBL.

## Group

| Name | Matriculation number |
| --- | --- |
| Muhammad Noor Ullah Ejaz | 000000 |

## Running it

Python 3.10 or newer. Nothing outside the standard scientific stack is used.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

The virtual environment is optional but keeps these packages out of your system
python. `.venv/` is ignored by git, so it never ends up in the repository.

Then open http://127.0.0.1:8000/home/. Every project is reachable from that
page.

`python manage.py migrate` is not optional: project 1 stores uploaded datasets
and training runs in the database, so the tables have to exist before the app
will load.

## Projects

### Project 1: Supervised learning interface

`/project1/` — upload a CSV, explore it, train models on it.

- **Upload.** Reads a CSV where the first row holds the column names and the
  last column holds the label. Columns that only number the rows are detected
  and dropped.
- **Explore.** A preview, a per column summary table, and five figures: a
  scatter plot of two features coloured by class, one feature against the label,
  a per class histogram, the distribution of the label, and a correlation
  heatmap.
- **Train.** Five algorithms per problem type, a sweep over one hyperparameter,
  a train and test split the user controls, and a choice of score. The results
  page plots the score against the hyperparameter for both halves of the split
  and shows a confusion matrix or a predicted against actual plot for the best
  model.
- Classification and regression are both supported. The type is detected from
  the label column and can be overruled from the interface.

`project1/README.md` covers the structure and the reasoning behind which parts
of the pipeline the user controls and which the app decides on its own.

Two sample datasets are in `project1/sample_data/`: `iris.csv` for
classification and `diabetes.csv` for regression.

### Project 2: Explainability

`/project2/` — one page, four linked regions, on the Palmer Penguins dataset.

- **Model and complexity.** A grid of decision trees and a grid of L1 penalised
  logistic regressions are fitted, and a λ slider picks between the finished
  models by maximising `accuracy − λ·Ω`. Ω is the number of leaves for a tree and
  the number of features still in use for logistic regression. The selected model
  is drawn, along with every model in the grid and the trade off line λ defines.
- **Counterfactuals.** Pick a penguin and a target species, and see the closest
  rows the model would put in that class, found by local sampling and ranked by
  MAD weighted L1 distance. Categorical features are resampled rather than
  noised, and contribute a flat cost when they change.
- **Feature effects.** PDP and ALE for each of the four measurements, three
  curves per plot, both written from scratch. ALE uses the exact analytic
  derivative for logistic regression and finite differences for the tree, since a
  tree has no useful derivative to integrate.

All three regions read the same model, so moving the λ slider changes every one
of them. `project2/README.md` covers the reasoning in more detail, including the
handling of missing rows and the choice of complexity measure.

The dataset ships as `project2/sample_data/penguins.csv` rather than being pulled
in through the `palmerpenguins` package, so the app needs nothing unusual
installed.
