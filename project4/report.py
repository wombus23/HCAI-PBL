"""
Builds the PDF report required by the project sheet: the method (tasks 1 and 2)
and the design of the user study (task 3).

    python manage.py build_report4

Numbers come from `protocol.py` and `results/simulation.json`, so the document
and the running instrument cannot drift apart. Like project 3, this is the only
part that needs reportlab, and the built PDF is committed.
"""

import os

from django.conf import settings
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from . import catalogue, plots, protocol, simulation

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "static", "project4", "project4_report.pdf")

BLUE = colors.HexColor("#275CB2")
GREY = colors.HexColor("#5a6473")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=25,
                                textColor=colors.HexColor("#1c2430"), spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=11,
                                   leading=15, textColor=GREY, alignment=1, spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18,
                             textColor=BLUE, spaceBefore=16, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=15,
                             textColor=colors.HexColor("#1c2430"), spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.8, leading=14.2,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "bullet": ParagraphStyle("bu", parent=base["BodyText"], fontSize=9.8, leading=14,
                                 leftIndent=14, bulletIndent=3, spaceAfter=4),
        "caption": ParagraphStyle("c", parent=base["Normal"], fontSize=8.4, leading=11.5,
                                  textColor=GREY, alignment=1, spaceAfter=12),
        "code": ParagraphStyle("code", parent=base["Normal"], fontName="Courier",
                               fontSize=8.6, leading=12.5,
                               backColor=colors.HexColor("#eef1f5"), borderPadding=6,
                               spaceAfter=9),
    }


CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.3, leading=10.6,
                      textColor=colors.HexColor("#1c2430"))
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", textColor=GREY)
BOLD = ParagraphStyle("boldcell", parent=CELL, fontName="Helvetica-Bold")


def _table(rows, widths, highlight=None):
    """Cells are Paragraphs so that long text wraps instead of running over."""
    wrapped = []
    for r, row in enumerate(rows):
        style_for = HEAD if r == 0 else (BOLD if r == highlight else CELL)
        wrapped.append([cell if hasattr(cell, "wrap") else Paragraph(str(cell), style_for)
                        for cell in row])
    rows = wrapped

    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (-1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#aab3c0")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, colors.HexColor("#dfe4ea")),
    ]
    if highlight is not None:
        style += [("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#e7edf7"))]
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle(style))
    return table


def _figure(url, width=14 * cm):
    relative = url.replace(settings.MEDIA_URL, "", 1)
    path = os.path.join(settings.MEDIA_ROOT, relative)
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        ratio = handle.height / handle.width
    return Image(path, width=width, height=width * ratio)


def build():
    sim = simulation.load()
    if sim is None:
        raise RuntimeError("Run `python manage.py run_simulation` first.")

    s = _styles()
    data = catalogue.catalogue()
    figures = plots.all_figures(sim)
    story = []
    P = lambda text, style="body": Paragraph(text, s[style])
    B = lambda text: Paragraph(text, s["bullet"], bulletText="\u2022")

    full = protocol.FULL
    power = sim["power"]

    # ------------------------------------------------------------- title
    story += [
        Spacer(1, 1.5 * cm),
        P("Comparing Two Ways of Asking", "title"),
        P("A study design for preference elicitation in a film recommender<br/>"
          "Human-Centric Artificial Intelligence · Project 4", "subtitle"),
        P("<b>The question.</b> A new user arrives at a film recommender that knows "
          "nothing about them. It can ask them to choose between two films, over and "
          "over, or it can ask them to rank ten films at a time. Both end with an "
          "estimate of the same thing: a preference vector w such that the utility of a "
          "film is w'x. Which way of asking produces a better estimate for the same "
          "amount of the user's patience?"),
        P("<b>What this document contains.</b> The feature representation (task 1), the "
          "extension of Bradley-Terry to rankings (task 2), and the design of a user "
          "study that would answer the question (task 3). The study was not run, as the "
          "project sheet specifies. The instrument that would run it is implemented and "
          "reachable from the project page (task 4)."),
        P("<b>The short version of the design.</b> %d participants, each doing both "
          "interfaces in a counterbalanced order on disjoint sets of films, with the "
          "primary outcome being how well the fitted preference vector predicts %d held "
          "out comparisons that the model never saw. The sample size comes from "
          "simulating the whole study first, which is possible here because the "
          "preference model is fully specified."
          % (power["within_total_80"], full["holdout_pairs"])),
    ]

    # ------------------------------------------------------------- task 1
    story += [
        P("1. Task 1: representing a film", "h1"),
        P("A film becomes a vector x of %d features, and a user's taste is a vector w of "
          "the same length. The catalogue is the IMDB 5000 dataset, reduced to the %s "
          "films with at least %s IMDb votes."
          % (len(catalogue.FEATURE_NAMES), format(data["n"], ","), format(catalogue.MIN_VOTES, ","))),
        P("<b>The constraint that decides everything.</b> Every feature is a number that "
          "has to be estimated from a few dozen answers. A representation that would be "
          "excellent for a recommender trained on millions of ratings is useless here: "
          "the elicitation budget, not the dataset, sets the dimension. That rules out "
          "the obvious rich features and leaves a deliberately small set."),
    ]

    feature_rows = [
        ["Block", "Count", "What it captures", "Why it is worth a dimension"],
        ["Genre indicators", "18", "Action, Comedy, Horror, ...",
         "The strongest and most self-explanatory axis of film taste. Multi-hot, since a film is often three at once. A weight on a genre is directly readable, which matters because participants are shown their own profile."],
        ["Release year", "1", "Era, standardized",
         "Separates the person who only watches recent films from the one who does not. Genre cannot express this."],
        ["Duration", "1", "Running time, standardized",
         "The commitment axis: some people will not start a two and a half hour film."],
        ["IMDb score", "1", "Critical consensus, standardized",
         "Distinguishes following the consensus from ignoring it."],
        ["Popularity", "1", "log10 vote count, standardized",
         "Blockbuster watcher against someone who has seen everything famous already. The log is essential: raw counts span four orders of magnitude."],
        ["Audience rating", "3", "family / teen / adult",
         "Collapsed from twenty-odd certificate strings, because nobody has an opinion about Approved against Passed."],
    ]
    story += [_table(feature_rows, [2.9 * cm, 1.3 * cm, 3.4 * cm, 8.0 * cm]), Spacer(1, 10)]

    story += [
        P("<b>What is deliberately left out.</b>", "body"),
        B("<b>Director and cast identity.</b> They matter enormously to real taste, and "
          "as one-hot features they would add thousands of dimensions that no elicitation "
          "budget could estimate. A study with a bigger budget could use learned "
          "embeddings of cast and crew; this one cannot afford them."),
        B("<b>Budget and box office.</b> Heavily missing in the dataset, and largely a "
          "proxy for the popularity feature already included."),
        B("<b>Plot keywords.</b> The same dimensionality problem as cast, with more noise."),
        Spacer(1, 4),
        P("<b>The catalogue filter is a study decision, not a modelling one.</b> Asking "
          "someone to rank ten films they have never heard of measures their reading of "
          "the metadata, not their taste. The vote threshold keeps the films plausibly "
          "recognisable. It also biases the catalogue towards the mainstream, which is a "
          "real limitation and is recorded as such: the study would generalise to "
          "eliciting taste over well known films, not over the long tail."),
    ]

    # ------------------------------------------------------------- task 2
    story += [
        PageBreak(),
        P("2. Task 2: from a pair to a ranking", "h1"),
        P("Bradley-Terry gives the probability that a user prefers film a to film b:"),
        Paragraph("P(a &gt; b) = exp(w'x_a) / ( exp(w'x_a) + exp(w'x_b) )", s["code"]),
        P("which is a softmax over two utilities. The natural extension to a full "
          "ranking i1 &gt; i2 &gt; ... &gt; in is the Plackett-Luce model, which reads a "
          "ranking as a sequence of choices: the participant picks a favourite out of all "
          "n, then a favourite out of the remaining n-1, and so on down the list."),
        Paragraph("P(i1 &gt; i2 &gt; ... &gt; in) = "
                  "prod over k=1..n-1 of  exp(w'x_ik) / sum over j&gt;=k of exp(w'x_ij)",
                  s["code"]),
        P("<b>Why this formulation.</b>", "body"),
        B("<b>It contains Bradley-Terry exactly.</b> With n = 2 the product has a single "
          "factor and reduces to the formula above. That matters more than usual here: "
          "the entire study compares two elicitation designs, and if each design were "
          "fitted with a different model, any difference in the results could be the "
          "model rather than the interface. One likelihood, two ways of feeding it, is "
          "what makes the comparison clean."),
        B("<b>The log likelihood is concave in w.</b> Each factor is a linear term minus "
          "a log-sum-exp. With a Gaussian prior the objective is strictly concave, so "
          "there is one optimum, no restarts and no seeds. Every participant's fit is "
          "reproducible from their answers."),
        B("<b>It counts a ranking honestly.</b> The tempting shortcut is to explode a "
          "ranking of ten into its 45 implied pairwise comparisons and pour them into "
          "plain Bradley-Terry. That treats 45 comparisons derived from one ranking as "
          "45 independent observations, which overstates the information in a ranking "
          "and would bias the study in favour of the design being tested. Plackett-Luce "
          "counts a ranking of ten as nine choice events, which is what it is."),
        P("<b>What the model assumes about people, and where that is wrong.</b> "
          "Plackett-Luce assumes independence of irrelevant alternatives: the relative "
          "odds of a and b do not change when c is added to the list. Humans violate "
          "this, and a ranking interface gives them more opportunity to, since context "
          "effects need context. If the ranking design underperforms in the study, this "
          "is one of the two explanations to distinguish between, the other being simple "
          "fatigue. The design accounts for it below."),
        P("<b>Estimation.</b> The estimate is the MAP under a Gaussian prior w ~ N(0, "
          "sigma^2 I):"),
        Paragraph("w_hat = argmax  sum_obs log P(obs | w)  -  ||w||^2 / (2 sigma^2)", s["code"]),
        P("The prior is not decoration. With %d features and roughly 30 answers, the "
          "unregularized likelihood often has no finite maximum: a participant who never "
          "saw a documentary has no finite best estimate for that weight, and the "
          "optimizer will happily send it to infinity. The prior holds unseen directions "
          "at zero, which is also the honest answer. Optimisation is L-BFGS on the exact "
          "gradient and takes milliseconds, so the fit can run inside the participant's "
          "session and be shown to them at the end."
          % len(catalogue.FEATURE_NAMES)),
    ]

    # ------------------------------------------------------------- task 3
    story += [
        PageBreak(),
        P("3. Task 3: the user study", "h1"),
        P("3.1 Hypotheses", "h2"),
        P("The comparison only means something once it is clear what is being held "
          "fixed, and the two obvious choices give opposite answers. That is the "
          "substance of the study rather than a technicality, so both are stated."),
        B("<b>H1 (primary, time matched).</b> For an equal amount of participant time, "
          "the ranking interface yields a preference vector that predicts held out "
          "comparisons more accurately than the pairwise interface does."),
        B("<b>H2 (secondary, decision matched).</b> Per choice event, ranking yields a "
          "more accurate preference vector. A ranking of ten is nine choice events."),
        B("<b>H3 (workload).</b> The ranking interface is rated as requiring more mental "
          "effort per screen."),
        B("<b>H4 (preference).</b> Participants asked which interface they would rather "
          "use again do not split evenly."),
        P("H1 is the one that matters for a product decision, because users spend "
          "minutes, not decisions. H2 is included because it is the version of the "
          "question the model answers cleanly, and the gap between them is the most "
          "interesting thing the study can report."),

        P("3.2 Design", "h2"),
        P("Within participants, two blocks, order counterbalanced. Every participant "
          "uses both interfaces; what is randomized is which comes first. Assignment is "
          "balanced rather than independent coin flips, since order enters the analysis "
          "as a factor and an unbalanced assignment loses power for nothing."),
        P("<b>Why within participants.</b> Taste is idiosyncratic and the achievable "
          "accuracy varies enormously from person to person: someone with sharp, "
          "consistent preferences is easier to model than someone who is genuinely "
          "indifferent. Between participants, that variation lands entirely in the error "
          "term. The simulation puts a number on the cost: %d participants per group "
          "between subjects against %d in total within subjects, for the same power on "
          "H1. Nobody should run this between participants."
          % (power["between_per_group_80"], power["within_total_80"])),
        P("<b>The two threats this creates, and the answers to them.</b> Carryover: "
          "having thought about their taste in block one, a participant is faster and "
          "more decisive in block two. Counterbalancing spreads this across conditions, "
          "and order is a factor in the model, so it is measured rather than assumed "
          "away. Film reuse: the second block must not reuse block one's films, or the "
          "second interface is scored on material the participant has already "
          "considered. The instrument samples without replacement across the whole "
          "session, held out films included."),

        P("3.3 Participants and recruitment", "h2"),
        B("<b>Sample.</b> %d participants recruited through Prolific, which gives "
          "screening, a paid participant pool used to short tasks, and no need to handle "
          "identities directly." % power["within_total_80"]),
        B("<b>Screening.</b> 18 or older, fluent in English, and self-reported film "
          "watching of at least one film a month. The last one matters: someone who does "
          "not watch films has no preference vector to elicit, and including them adds "
          "noise that no interface can fix."),
        B("<b>Payment.</b> At the platform's fair-pay rate for the measured median "
          "duration from the pilot, paid regardless of how they answer, and stated up "
          "front. Paying by completion rather than by answers is what keeps participants "
          "from optimising for looking consistent."),
        B("<b>Exclusions, decided before looking at the outcome.</b> Median screen time "
          "under 1.5 s in either block, or an identical answer to every Likert item in "
          "both blocks. Both are recorded automatically. Note what is not used: there is "
          "no attention check question, because a preference question has no correct "
          "answer and a fake one would only measure compliance."),

        P("3.4 Procedure", "h2"),
        P("The implemented instrument follows exactly this sequence."),
    ]

    procedure = [
        ["Step", "What happens", "Recorded"],
        ["1. Consent", "Information sheet, voluntary participation, what is stored, right to stop.", "Consent given"],
        ["2. Block A", "One interface: %d pairwise comparisons or %d rankings of ten."
         % (full["pairwise_tasks"], full["ranking_tasks"]), "Films shown, order chosen, time per screen"],
        ["3. Held out A", "%d pairwise comparisons on films not seen before." % full["holdout_pairs"],
         "Same, plus fitted w and its accuracy"],
        ["4. Ratings A", "Four 7-point items: effort, frustration, expressiveness, perceived accuracy.", "Four integers"],
        ["5-7. Block B", "The other interface, then its held out block and ratings.", "As above"],
        ["8. Exit", "Which interface they would rather use again; how many films they recognised.", "Two categorical answers"],
        ["9. Debrief", "Their estimated taste profile and five recommendations.", "Nothing"],
    ]
    story += [_table(procedure, [2.5 * cm, 7.3 * cm, 5.8 * cm]), Spacer(1, 10)]

    story += [
        P("<b>Why a held out block rather than comparing w to the truth.</b> There is no "
          "ground truth preference vector for a real person, so recovery of w cannot be "
          "measured. What can be measured is whether the estimate predicts choices the "
          "model has not seen, which is also exactly what a recommender needs it for. "
          "The held out films are drawn from the part of the catalogue the participant "
          "has not been shown, so the measure is prediction and not memory."),

        P("3.5 Measures", "h2"),
    ]

    measures = [
        ["Measure", "Type", "Definition"],
        ["Held out accuracy", "Primary",
         "Share of the %d held out comparisons the block's fitted w predicts correctly." % full["holdout_pairs"]],
        ["Held out log loss", "Primary (secondary form)",
         "The same, keeping the confidence rather than thresholding it. More sensitive at the same sample size."],
        ["Elicitation time", "Primary (denominator)",
         "Summed screen time for the block, used to test H1 as accuracy per minute."],
        ["Effort, frustration", "Secondary", "7-point items after each block."],
        ["Expressiveness, perceived accuracy", "Secondary",
         "7-point items. Perceived accuracy against measured accuracy is worth reporting on its own: people may not be able to tell."],
        ["Preferred interface", "Secondary", "Forced choice at the end, with a no preference option."],
        ["Films recognised", "Covariate", "Self-report, three levels. Expected to moderate everything."],
    ]
    story += [_table(measures, [3.6 * cm, 3.0 * cm, 9.0 * cm]), Spacer(1, 10)]

    story += [
        P("3.6 Analysis plan", "h2"),
        P("Fixed before collection, and this is what would be preregistered."),
        B("<b>H1.</b> Linear mixed model on held out accuracy: fixed effects for "
          "interface, order and their interaction, a random intercept per participant, "
          "elicitation time as an offset so the test is per unit of time. The interface "
          "coefficient is the answer. The interaction with order is the carryover check."),
        B("<b>H2.</b> The same model with choice events in place of time."),
        B("<b>H3, H4.</b> Wilcoxon signed rank on the paired Likert differences; an exact "
          "binomial test on the forced choice, excluding participants who answered no "
          "preference and reporting how many did."),
        B("<b>Multiplicity.</b> H1 is primary; H2 to H4 are reported with confidence "
          "intervals and Holm correction across the three."),
        B("<b>Stopping rule.</b> Fixed sample size decided in advance from the power "
          "analysis. No looking at the outcome and continuing."),

        P("3.7 How many participants, and where that number comes from", "h2"),
        P("The preference model is fully specified, so the study can be simulated rather "
          "than guessed at. %d synthetic participants with known taste vectors answered "
          "both designs at matched budgets, and each fitted vector was scored on %d held "
          "out comparisons. The simulated participants answer noisily, drawing from the "
          "same Plackett-Luce model, so the achievable ceiling is %.3f rather than 1."
          % (sim["meta"]["participants"], sim["meta"]["holdout_pairs"], sim["ceiling"])),
        Spacer(1, 4),
        _figure(figures["events"], 13 * cm),
        P("Figure 1. Matched on decisions made. A ranking of ten is nine choices, and "
          "each is made against a wider field than a pairwise comparison, so it carries "
          "more information.", "caption"),
        _figure(figures["time"], 13 * cm),
        P("Figure 2. Matched on the participant's time, using %s s per comparison and "
          "%s s per ranking. The advantage nearly disappears."
          % (sim["meta"]["seconds_pairwise"], sim["meta"]["seconds_ranking"]), "caption"),
    ]

    power_rows = [
        ["Matched on", "Ranking advantage", "SD of the difference", "Within-subjects n", "Between-subjects n per group"],
        ["Time (300 s)", "%+0.4f" % power["effect"], "%.4f" % power["sd"],
         str(power["within_total_80"]), str(power["between_per_group_80"])],
        ["Decisions (27)", "%+0.4f" % power["events_effect"], "%.4f" % power["events_sd"],
         str(power["events_within_total_80"]), "-"],
    ]
    story += [
        _table(power_rows, [2.7 * cm, 2.9 * cm, 3.0 * cm, 3.1 * cm, 3.9 * cm], highlight=1),
        Spacer(1, 8),
        P("Two things follow. The study is run within participants, because the between "
          "subjects sample for H1 is not fundable. And %d participants is the planning "
          "number for H1 at 80%% power. It is a floor, not an estimate: simulated "
          "participants vary only in their taste vector and their response noise, while "
          "real ones also vary in engagement, film knowledge and how carefully they "
          "read. The pilot exists to measure the real variance, and the timing constants "
          "the whole calculation rests on." % power["within_total_80"]),
        P("<b>Pilot.</b> Ten participants, run exactly as the main study. Its purpose is "
          "to replace the two assumed timing constants with measurements, to check that "
          "the ranking screen is usable on a phone, and to estimate the between "
          "participant variance in held out accuracy. If the measured times differ "
          "materially from the assumptions, the task counts are recalculated before the "
          "main run, and the pilot data is not pooled with it."),
    ]

    # ------------------------------------------------------------- validity
    story += [
        PageBreak(),
        P("4. What could go wrong, and what is done about it", "h1"),
    ]
    threats = [
        ["Threat", "Response"],
        ["Carryover between blocks",
         "Counterbalanced order, order as a factor in the model, disjoint film sets across blocks."],
        ["Fatigue inside the ranking block",
         "Screen time is recorded per task, so within block slowdown can be tested directly rather than assumed absent."],
        ["The two budgets are not really matched",
         "Time is measured, not assumed, and H1 is tested per unit of measured time. The assumed constants only set the task counts."],
        ["Unfamiliar films",
         "Catalogue filtered to films with at least %s votes; recognition asked at the end and used as a covariate." % format(catalogue.MIN_VOTES, ",")],
        ["Model misspecification favours one design",
         "One likelihood for both designs. If Plackett-Luce is wrong about people, it is wrong for both arms, though not necessarily by the same amount, which is why H2 is reported alongside H1."],
        ["Indifferent participants",
         "Screened on film watching frequency; held out accuracy near chance in both blocks is reported rather than dropped, since it is a real outcome."],
        ["Random film pairs are uninformative",
         "The sheet specifies uniform sampling and the study keeps it, so the comparison is between interfaces and not between selection strategies. Adaptive selection is the obvious follow-up and is listed below."],
        ["Experimenter degrees of freedom",
         "Hypotheses, exclusions, and the analysis model fixed in advance and preregistered; fixed sample size with no interim looks."],
    ]
    story += [_table(threats, [5.0 * cm, 10.6 * cm]), Spacer(1, 10)]

    story += [
        P("5. Ethics and data", "h1"),
        B("<b>Consent.</b> Information sheet before anything is recorded, stating what "
          "happens, how long it takes, what is stored and that stopping is allowed at any "
          "point. Nothing is written to the database until consent is given."),
        B("<b>What is stored.</b> A random identifier, the films shown, the orders "
          "chosen, screen times, the Likert answers and the two exit answers. No name, no "
          "email address, no IP address. The free text box records only its length, as an "
          "engagement signal, not its contents."),
        B("<b>Withdrawal.</b> Closing the tab stops participation and partial data is "
          "excluded from analysis. Because responses cannot be linked back to a person, "
          "they cannot be withdrawn after completion, and the information sheet says so "
          "rather than promising otherwise."),
        B("<b>Risk.</b> Minimal. The task is choosing between films. The one real "
          "consideration is the debrief screen, which shows a profile inferred from a few "
          "dozen answers; it is worded as what the model concluded, not as a fact about "
          "the person."),
        B("<b>Approval.</b> A real run would need institutional ethics approval, and the "
          "information sheet would carry the approval number and a contact address for "
          "complaints. The implemented consent page marks exactly where those go."),

        P("6. What the study cannot tell you", "h1"),
        B("Both interfaces sample films uniformly at random, as the sheet specifies. "
          "Adaptive selection would change the answer for both arms and probably by "
          "different amounts, since a ranking screen has more room for a clever "
          "selection to pay off. That is the first extension worth running."),
        B("The catalogue is mainstream by construction. Nothing here generalises to "
          "eliciting taste over an unfamiliar or long tail catalogue, which is precisely "
          "where a recommender needs help most."),
        B("The result is about a first session, not a relationship. A recommender that "
          "keeps learning from what a user actually watches faces a different problem, "
          "and a cold start interface that wins in one session may not be the right one "
          "to build."),
        B("Ten is one ranking size, chosen because it fits on a screen. The interesting "
          "curve is over the size, and this design measures one point on it."),

        P("7. The implemented instrument", "h1"),
        P("Task 4 is implemented and reachable from the project page. It is the study as "
          "described: consent, two counterbalanced blocks with held out comparisons and "
          "ratings after each, an exit question and a debrief. Ranking is drag and drop "
          "with arrow buttons as an accessible fallback, screen times are measured "
          "client side, and every answer is written to the database as it is given, so a "
          "refresh or a closed tab neither loses nor duplicates anything. Responses are "
          "exportable as CSV in the shape the analysis above expects."),
        P("A shortened demo run is available for inspection and is flagged in the data "
          "so it can be excluded. The numbers in this document come from the same "
          "configuration file the instrument reads, so the two cannot disagree."),
        Paragraph("python manage.py run_simulation   # rewrites results/simulation.json<br/>"
                  "python manage.py build_report4    # rewrites this PDF", s["code"]),
    ]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    document = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Project 4: Preference elicitation",
        author="Human-Centric AI",
    )
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return OUTPUT


def _page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(document.page))
    canvas.restoreState()
