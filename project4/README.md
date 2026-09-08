# Project 4: Preference elicitation

A user study comparing two ways of asking someone what films they like, plus the
working instrument that would run it. The study was designed but not run, as the
project sheet specifies.

The landing page at `/project4/` does the two things the sheet asks for: a
download button for the PDF report covering tasks 1 to 3, and a link that starts
the study itself (task 4).

## Running it

    python manage.py migrate
    python manage.py run_simulation     # ~45 s, rewrites results/simulation.json
    python manage.py build_report4      # rewrites the PDF, needs reportlab

Both outputs are committed, so the page works straight after a clone. The
simulation is seeded and reproduces exactly.

## Files

- `catalogue.py` — the films and the feature representation (task 1).
- `preference.py` — Plackett-Luce, MAP fitting, held out scoring, recommendations (task 2).
- `protocol.py` — the study parameters, in one place so the report and the instrument agree.
- `simulation.py` — synthetic participants, used to size the study.
- `models.py`, `views.py`, templates — the instrument (task 4).
- `report.py` — builds the PDF from `protocol.py` and the simulation results.

## Task 1: the features

25 dimensions: 18 genre indicators, four standardized continuous features (year,
duration, IMDb score, log vote count) and three audience rating groups.

The binding constraint is the elicitation budget, not the dataset. Every feature
is a number that has to be estimated from a few dozen answers, so the
representation that would suit a recommender trained on millions of ratings is
useless here. Director and cast identity are the painful omission: they matter
enormously to real taste, but as one-hot features they would add thousands of
unidentifiable dimensions.

The catalogue is filtered to the 2,735 films with at least 25,000 IMDb votes.
That is a study decision rather than a modelling one: asking someone to rank ten
films they have never heard of measures their reading of the metadata, not their
taste.

## Task 2: Plackett-Luce

A ranking is read as a sequence of choices — pick a favourite from ten, then from
the remaining nine, and so on:

    P(i1 > ... > in) = prod_k exp(w'x_ik) / sum_{j>=k} exp(w'x_ij)

Three reasons for this formulation, in order of importance to the study:

1. **It contains Bradley-Terry exactly** at n = 2. The study compares two
   interfaces, so if each arm were fitted with a different model, any difference
   could be the model rather than the interface.
2. **The log likelihood is concave**, so with a Gaussian prior there is one
   optimum, no restarts, no seeds. Every fit is reproducible from the answers.
3. **It counts a ranking honestly.** Exploding a ranking of ten into its 45
   implied pairs and feeding plain Bradley-Terry would treat them as 45
   independent observations, overstating what a ranking is worth and biasing the
   study towards the design under test. Nine choice events is what it is.

Estimation is MAP under a Gaussian prior. The prior is load-bearing: with 25
features and 30 answers the unregularized likelihood often has no finite maximum,
since a participant who never saw a documentary has no finite best estimate for
that weight.

## Task 3: the study

Within participants, both interfaces, counterbalanced order, disjoint film sets.
Primary outcome is how well each block's fitted `w` predicts 10 held out
comparisons the model never saw. There is no ground truth `w` for a real person,
so recovery cannot be measured; prediction can, and it is what a recommender
needs anyway.

The sample size comes from simulating the study rather than guessing, which is
possible because the model is fully specified. The simulation produced the design
decision, not just a number:

| Matched on | Ranking advantage | Within-subjects n | Between-subjects n per group |
| --- | --- | --- | --- |
| Time (300 s) | +0.009 | 240 | 479 |
| Decisions (27) | +0.025 | 33 | — |

Matched on decisions made, ranking wins clearly. Matched on the participant's
time, the advantage nearly vanishes, because a ranking of ten costs far more than
nine times a single comparison. That gap is the substance of the study, and the
between-subjects column is why it is run within participants.

The report covers the hypotheses, recruitment, procedure, measures, the
preregistered analysis plan, threats to validity, and ethics.

## Task 4: the instrument

Consent → block A → held out block → ratings → block B → held out → ratings →
exit question → debrief.

- Ranking is drag and drop, with arrow buttons as an accessible fallback.
- Screen times are measured client side and stored per screen.
- Films are sampled without replacement across the entire session, held out
  blocks included, so the held out block measures prediction and not memory, and
  the second interface is not scored on films the participant already considered.
- Every answer is written as it is given, so a refresh, a back button or a closed
  tab neither loses nor duplicates a response.
- The debrief shows participants their own estimated taste profile and five
  recommendations, which is both an honest thank you and a useful sanity check on
  the model.
- Responses export as CSV in the shape the analysis plan expects, at
  `/project4/export/`.

A shortened demo run is linked from the landing page for inspection. It is
flagged `is_demo` in the database so it can be excluded from any analysis.

## Known limits

- Films are sampled uniformly at random, as the sheet specifies. Adaptive
  selection is the obvious extension and would probably help the ranking design
  more.
- The catalogue is mainstream by construction, so nothing here generalises to
  eliciting taste over an unfamiliar catalogue.
- Ten is one ranking size. The interesting curve is over the size; this measures
  one point on it.
- Rebuilding the PDF needs reportlab, which is outside the allowed library set.
  The built PDF is committed, so nothing needs it at runtime.
