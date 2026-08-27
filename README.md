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
