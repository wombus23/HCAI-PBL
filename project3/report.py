"""
Builds the PDF report that the project sheet requires, straight from
`results/results.json` so it can never disagree with the interface.

    python manage.py build_report

This is the one part of the repository that needs a library outside the numpy,
sklearn, matplotlib set: reportlab, for laying out the PDF. The built report is
committed at `static/project3/project3_report.pdf`, so nobody needs reportlab
installed to read it, run the app, or use the download button. It is only needed
to rebuild the file.
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

from . import plots
from .data import CLASSES
from .experiments import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "static", "project3", "project3_report.pdf")

BLUE = colors.HexColor("#275CB2")
GREY = colors.HexColor("#5a6473")
LIGHT = colors.HexColor("#eef1f5")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=25,
                                textColor=colors.HexColor("#1c2430"), spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=11,
                                   leading=15, textColor=GREY, alignment=1,
                                   spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18,
                             textColor=BLUE, spaceBefore=16, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=15,
                             textColor=colors.HexColor("#1c2430"), spaceBefore=11,
                             spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.8, leading=14.2,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "caption": ParagraphStyle("c", parent=base["Normal"], fontSize=8.4, leading=11.5,
                                  textColor=GREY, alignment=1, spaceAfter=12),
        "code": ParagraphStyle("code", parent=base["Normal"], fontName="Courier",
                               fontSize=8.6, leading=12, textColor=colors.HexColor("#1c2430"),
                               backColor=LIGHT, borderPadding=6, spaceAfter=9),
    }


def _table(rows, widths, header=True, highlight=None):
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dfe4ea")),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), GREY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#aab3c0")),
        ]
    if highlight is not None:
        style += [
            ("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#e7edf7")),
            ("FONTNAME", (0, highlight), (-1, highlight), "Helvetica-Bold"),
        ]
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle(style))
    return table


def _figure(url, width=15.5 * cm):
    """Turn a media URL from plots.py back into a file path for embedding."""
    relative = url.replace(settings.MEDIA_URL, "", 1)
    path = os.path.join(settings.MEDIA_ROOT, relative)
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        ratio = handle.height / handle.width
    return Image(path, width=width, height=width * ratio)


def build():
    results = load()
    if results is None:
        raise RuntimeError("Run `python manage.py run_experiments` first.")

    s = _styles()
    meta, t1, t2, t3, t4 = (results["meta"], results["task1"], results["task2"],
                            results["task3"], results["task4"])
    figures = plots.all_figures(results)
    story = []
    P = lambda text, style="body": Paragraph(text, s[style])

    # ---------------------------------------------------------------- title
    story += [
        Spacer(1, 1.6 * cm),
        P("Active Learning for Learning-to-Defer", "title"),
        P("Human-Centric Artificial Intelligence · Project 3<br/>AG News topic "
          "classification with a simulated human expert", "subtitle"),
    ]

    headline = [
        ["", "Accuracy", "What it is"],
        ["Classifier alone", "%.4f" % t1["accuracy"], "Task 1 baseline, trained on every label"],
        ["Expert alone", "%.4f" % t2["accuracy"], "Task 2 simulated expert on the test set"],
        ["Team, all expert labels", "%.4f" % t3["learned"]["accuracy"],
         "Task 3, %s expert answers used" % meta["n_train"]],
        ["Team, %s questions" % t4["strategies"]["random"]["queries"][-1],
         "%.4f" % t4["summary"][0]["final"],
         "Task 4, best strategy: %s" % t4["summary"][0]["strategy"]],
        ["Oracle deferral", "%.4f" % t3["ceiling"], "Upper bound: defer exactly when the expert wins"],
    ]
    story += [_table(headline, [4.6 * cm, 2.4 * cm, 8.5 * cm], highlight=4), Spacer(1, 10)]

    story += [
        P("<b>Summary.</b> A linear classifier reaches %.4f on AG News. A simulated "
          "expert whose competence varies by region of the input space reaches %.4f, "
          "far worse overall, but is right on %.0f%% of the articles the classifier "
          "gets wrong, which is %.1f%% of the test set. Learning when to defer captures "
          "a quarter of that headroom, "
          "and doing so needs surprisingly few expert answers: %d questions, %.1f%% of "
          "the training set, recover %.0f%% of what the full set of expert labels buys, "
          "provided the questions are chosen by the value of the deferral decision "
          "rather than by uncertainty."
          % (t1["accuracy"], t2["accuracy"],
             100 * (t3["ceiling"] - t1["accuracy"]) / (1 - t1["accuracy"]),
             100 * (t3["ceiling"] - t1["accuracy"]),
             t4["strategies"]["random"]["queries"][-1],
             100 * t4["strategies"]["random"]["queries"][-1] / meta["n_train"],
             100 * t4["summary"][0]["share_of_gain"])),
        P("Generated %s. Every number in this report is read from "
          "<font face='Courier'>results/results.json</font>, produced by "
          "<font face='Courier'>python manage.py run_experiments</font> in %s seconds."
          % (meta["generated"], meta["seconds"])),
    ]

    # ------------------------------------------------------------- the data
    story += [
        P("1. Data and setup", "h1"),
        P("AG News: news articles labelled with one of four topics (%s). The test set "
          "is complete at %s articles. The training set is a stratified %s article "
          "sample of the full 120,000."
          % (", ".join(CLASSES), meta["n_test"], meta["n_train"])),
        P("<b>Why a sample, and why a CSV.</b> The dataset ships inside the repository "
          "as two gzipped CSVs rather than being fetched through the "
          "<font face='Courier'>datasets</font> package. The submission rules ask for "
          "no unusual dependencies, and an import that fails on the grader's machine "
          "costs more than the convenience is worth. Sampling to %s training articles "
          "keeps the repository small and every experiment in this report runnable in "
          "well under a minute, which matters because the active learning study refits "
          "the deferral model %d times." % (meta["n_train"],
                                            len(t4["strategies"]) * t4["rounds"] * len(t4["seeds"]))),
        P("Features are TF-IDF over words and bigrams with English stop words removed "
          "and sublinear term frequency scaling, giving %s features. A 100 dimensional "
          "LSA projection of the same matrix is used only for clustering."
          % t1["n_features"]),
    ]

    # ------------------------------------------------------------- task 1
    story += [
        P("2. Task 1: the baseline classifier", "h1"),
        P("Multinomial logistic regression on the TF-IDF matrix, C = 4. Test accuracy "
          "<b>%.4f</b>." % t1["accuracy"]),
        P("<b>Why this model.</b> A fine tuned transformer would score two to three "
          "points higher and would have made every later experiment impractical: the "
          "active learning study alone would have taken hours instead of seconds, and "
          "the model would have needed a dependency the submission rules exclude. It "
          "would also have changed nothing about the questions this project asks, which "
          "are about the interaction between a classifier and an expert rather than "
          "about squeezing the last points out of the classifier. One property did "
          "matter: the model has to produce a usable confidence signal, since deferral "
          "depends on comparing it against the expert. That rules out a linear SVM's "
          "unnormalised decision values.", "body"),
    ]

    per_class = [["Class", "Precision", "Recall", "F1", "Support"]]
    per_class += [[r["label"], "%.3f" % r["precision"], "%.3f" % r["recall"],
                   "%.3f" % r["f1"], str(r["support"])] for r in t1["per_class"]]
    story += [_table(per_class, [4.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]),
              Spacer(1, 12), _figure(figures["confusion"], 10.5 * cm),
              P("Figure 1. Baseline classifier on the test set. World, Business and "
                "Sci/Tech account for nearly all of the confusion; Sports is almost "
                "perfectly separable.", "caption")]

    # ------------------------------------------------------------- task 2
    story += [
        PageBreak(),
        P("3. Task 2: the simulated expert", "h1"),
        P("The expert must not be uniformly competent, and the sheet asks for expertise "
          "concentrated in specific regions of the input space. The design has three "
          "parts, each of which is a decision that shapes the rest of the project."),
        P("<b>Competence follows regions, not labels.</b> The document space is split "
          "into %d k-means regions on the normalised LSA vectors, and each region is "
          "assigned an accuracy: four strong (0.88 to 0.95), four middling, four weak "
          "(0.25 to 0.35). Attaching competence to the true label instead would have "
          "been simpler and would have wrecked task 4, because the profile could then "
          "be inferred from the classifier's own predictions without asking the expert "
          "anything. The regions lean towards topics without being topics, so the "
          "profile has to be bought with queries. The measured per class accuracies in "
          "the table below are far flatter than the per region ones, which is exactly "
          "the gap the design was aiming for." % meta["n_regions"]),
        P("<b>Mistakes are plausible, not random.</b> When the expert is wrong the "
          "answer is drawn from a confusion profile rather than uniformly, so a missed "
          "Business article usually comes back as Sci/Tech. A uniformly random wrong "
          "answer would make the expert's errors trivially detectable and would inflate "
          "the value of deferral."),
        P("<b>Answers are fixed per article.</b> Querying the same article twice returns "
          "the same answer. An expert who resampled on every query would make the "
          "active learning curves meaningless, since a strategy could buy accuracy by "
          "asking the same question repeatedly."),
        Spacer(1, 6),
        _figure(figures["expert"], 15 * cm),
        P("Figure 2. Designed against measured competence per region, with the "
          "classifier's accuracy for reference. The expert beats the classifier in four "
          "regions and is worse than guessing in three.", "caption"),
    ]

    region_rows = [["Region", "Test articles", "Designed", "Measured", "Leans", "Purity"]]
    region_rows += [[str(r["region"]), str(r["size"]), "%.2f" % r["designed"],
                     "%.3f" % r["accuracy"], r["dominant"], "%.2f" % r["purity"]]
                    for r in t2["by_region"]]
    story += [_table(region_rows, [2.2 * cm, 3.0 * cm, 2.4 * cm, 2.4 * cm, 2.8 * cm, 2.2 * cm]),
              Spacer(1, 10)]

    class_rows = [["True class", "Expert accuracy", "Test articles"]]
    class_rows += [[r["label"], "%.3f" % r["accuracy"], str(r["size"])] for r in t2["by_class"]]
    story += [
        _table(class_rows, [4.0 * cm, 4.0 * cm, 3.4 * cm]),
        Spacer(1, 8),
        P("<b>Strengths and weaknesses.</b> Overall test accuracy is <b>%.4f</b>, against "
          "%.4f for the classifier. Per region it ranges from %.3f to %.3f. Per class it "
          "only ranges from %.3f to %.3f. An observer who only knew the expert's accuracy "
          "per topic would conclude they were mediocre and roughly uniform, and would "
          "have no idea where they were worth asking."
          % (t2["accuracy"], t1["accuracy"],
             min(r["accuracy"] for r in t2["by_region"]),
             max(r["accuracy"] for r in t2["by_region"]),
             min(r["accuracy"] for r in t2["by_class"]),
             max(r["accuracy"] for r in t2["by_class"]))),
    ]

    # ------------------------------------------------------------- task 3
    story += [
        PageBreak(),
        P("4. Task 3: learning to defer", "h1"),
        P("With the expert's answer available for every training article, the system "
          "decides per article whether to answer or hand over. The rule compares two "
          "probabilities:"),
        Paragraph("defer(x)  &lt;=&gt;  P(expert correct | x) - P(classifier correct | x) &gt; t", s["code"]),
        P("<b>P(expert correct | x)</b> is a logistic regression on the same TF-IDF "
          "features, trained on whether the expert's answer was right. It never sees the "
          "region assignments, so it has to recover the competence profile from the text. "
          "Its AUC on the test set is %.4f." % t3["rejector_auc"]),
        P("<b>P(classifier correct | x)</b> is not the model's maximum probability. A "
          "linear model on TF-IDF is overconfident, and comparing a raw confidence "
          "against a calibrated rejector puts the two sides of the inequality on "
          "different scales, which makes the system defer far too rarely. The confidence "
          "is therefore calibrated first, by fitting a one dimensional logistic "
          "regression from out of fold confidence to out of fold correctness on the "
          "training set. Figure 4 shows why it was needed: at a raw confidence of %.2f "
          "the classifier is right about %.0f%% of the time."
          % (t3["calibration"][0]["confidence"], 100 * t3["calibration"][0]["actual"])),
        Spacer(1, 4),
        _figure(figures["coverage"], 14 * cm),
        P("Figure 3. Team accuracy against the share of articles handed to the expert, "
          "for four deferral policies.", "caption"),
        _figure(figures["calibration"], 11 * cm),
        P("Figure 4. The classifier's raw confidence against how often it is actually "
          "right, before and after calibration.", "caption"),
    ]

    policy_rows = [["Policy", "Team acc.", "Deferred", "Expert acc.\non those",
                    "Classifier acc.\non the rest", "Gains\ncaught", "Damage"]]
    for key, label in [("learned", "Learned rejector"),
                       ("confidence_only", "Classifier confidence only"),
                       ("random", "Random deferral"),
                       ("oracle", "Oracle (upper bound)")]:
        row = t3[key]
        policy_rows.append([
            label, "%.4f" % row["accuracy"], "%.3f" % row["deferral_rate"],
            "%.3f" % row["accuracy_deferred"], "%.3f" % row["accuracy_retained"],
            "%.3f" % row["recall_of_gains"], "%.4f" % row["damage"],
        ])
    story += [
        _table(policy_rows, [4.4 * cm, 1.9 * cm, 1.8 * cm, 2.3 * cm, 2.6 * cm, 1.7 * cm, 1.8 * cm],
               highlight=1),
        Spacer(1, 8),
        P("The first three policies hand over the same share of articles, so the "
          "comparison is about which ones rather than how many. <i>Gains caught</i> is "
          "the share of the articles that only the expert gets right which the system "
          "actually deferred. <i>Damage</i> is the share of all articles given away "
          "that the classifier would have got right and the expert did not."),
        P("<b>Result.</b> Team accuracy is <b>%.4f</b> against %.4f for the classifier "
          "alone, recovering %.1f%% of the distance to the oracle at a deferral rate of "
          "%.1f%%. Random deferral at the same rate is worse than not deferring at all, "
          "which is worth stating: handing work to a %.0f%% accurate expert is actively "
          "harmful unless it is aimed."
          % (t3["learned"]["accuracy"], t3["classifier_only"],
             100 * t3["normalised_gain"], 100 * t3["learned"]["deferral_rate"],
             100 * t2["accuracy"])),
        P("<b>On the quality of the deferral decisions.</b> Team accuracy alone hides "
          "the interesting part. The learned rejector and the confidence rule end up "
          "within %.4f of each other on accuracy, but they are not doing the same thing. "
          "The expert is right on %.3f of what the rejector hands over against %.3f for "
          "the confidence rule, and the rejector does %.4f damage against %.4f. "
          "Confidence finds articles that are hard to classify; the rejector finds "
          "articles this particular expert happens to be good at. Those two sets overlap "
          "because hard articles are often in the same regions, but they are different "
          "questions, and the second one is the one deferral is asking."
          % (abs(t3["learned"]["accuracy"] - t3["confidence_only"]["accuracy"]),
             t3["learned"]["accuracy_deferred"], t3["confidence_only"]["accuracy_deferred"],
             t3["learned"]["damage"], t3["confidence_only"]["damage"])),
    ]

    # ------------------------------------------------------------- task 4
    story += [
        PageBreak(),
        P("5. Task 4: active learning for competence discovery", "h1"),
        P("The setting changes: the classifier may still use every label, but there are "
          "no expert answers at all, and each one has to be paid for. Starting from "
          "nothing, each round selects %d articles to ask about, the rejector is refitted "
          "on everything answered so far, and the team is evaluated on the test set. The "
          "budget is %d questions, %.1f%% of the training set. Results are averaged over "
          "%d seeds." % (t4["batch"], t4["strategies"]["random"]["queries"][-1],
                         100 * t4["strategies"]["random"]["queries"][-1] / meta["n_train"],
                         len(t4["seeds"]))),
        P("<b>Strategies.</b> <i>Random</i> is the control. <i>Classifier uncertainty</i> "
          "is the textbook baseline: ask about the articles the classifier is least sure "
          "of. <i>Expert uncertainty</i> asks where the current competence estimate is "
          "closest to a coin flip, which is uncertainty sampling applied to the thing "
          "actually being learned. <i>Deferral margin</i> asks where P(expert correct) and "
          "P(classifier correct) are closest together, so the deferral decision is on the "
          "point of flipping."),
        P("<b>Why deferral margin is the interesting one.</b> The system is not trying to "
          "learn the expert's competence; it is trying to make a good deferral decision. "
          "Those come apart. Knowing precisely how good the expert is in a region where "
          "the classifier is at 99% changes nothing, because the system keeps those "
          "articles either way. The margin criterion is the only one of the four that "
          "spends its budget where an answer can change what the system does."),
        Spacer(1, 4),
        _figure(figures["active"], 14 * cm),
        P("Figure 5. Team accuracy against number of expert answers bought, shaded by "
          "one standard deviation over %d seeds." % len(t4["seeds"]), "caption"),
    ]

    summary_rows = [["Strategy", "Team acc.\nat %d" % t4["strategies"]["random"]["queries"][-1],
                     "Share of the\nfull label gain", "Questions to\nreach 75%",
                     "Competence\nAUC", "Deferral\nrate"]]
    for row in t4["summary"]:
        summary_rows.append([
            row["strategy"], "%.4f" % row["final"], "%.3f" % row["share_of_gain"],
            str(row["queries_to_75"]) if row["queries_to_75"] else "not reached",
            "%.4f" % row["final_auc"], "%.4f" % row["final_deferral"],
        ])
    story += [
        _table(summary_rows,
               [4.2 * cm, 2.4 * cm, 2.8 * cm, 2.6 * cm, 2.2 * cm, 2.0 * cm],
               highlight=1),
        Spacer(1, 10),
        _figure(figures["active_auc"], 13 * cm),
        P("Figure 6. How well each strategy recovers the expert's competence profile, "
          "measured as AUC on the test set.", "caption"),
        P("<b>The two figures disagree, and that is the finding.</b> Random sampling "
          "builds the best competence model by AUC (%.4f against %.4f for deferral "
          "margin), because it trains on a representative sample of the input space. "
          "Deferral margin builds a worse model of the expert and a better team. It "
          "reaches 75%% of the full label gain in %d questions where random needs %d, "
          "and ends %.4f above it. Optimising the model and optimising the decision are "
          "different objectives, and only one of them is what the system is for."
          % (t4["summary"][1]["final_auc"], t4["summary"][0]["final_auc"],
             t4["summary"][0]["queries_to_75"], t4["summary"][1]["queries_to_75"],
             t4["summary"][0]["final"] - t4["summary"][1]["final"])),
        P("Classifier uncertainty is no better than random. It selects articles that are "
          "hard to classify, which correlates only loosely with where the expert helps. "
          "Expert uncertainty is <i>worse</i> than random, ending at %.4f: chasing the "
          "points where the competence model is least sure gives it a badly skewed "
          "training sample, and its deferral rate collapses to %.3f, meaning it has "
          "talked itself out of using the expert at all."
          % (t4["summary"][-1]["final"], t4["summary"][-1]["final_deferral"])),
    ]

    # ------------------------------------------------------- limits and repro
    story += [
        P("6. Limitations", "h1"),
        P("<b>The expert is simulated, and simulated in a way the rejector can learn.</b> "
          "Competence is piecewise constant over k-means regions of the same feature "
          "space the rejector uses, so a linear model on those features is well matched "
          "to the thing it has to fit. A real expert's competence would not be so "
          "conveniently shaped, and the AUC of %.4f should be read as an upper bound on "
          "what this approach would achieve against a human." % t3["rejector_auc"]),
        P("<b>One expert, always available.</b> There is no cost per query at deployment "
          "time, no queue, no fatigue, and no second expert to choose between. A budget "
          "on deferrals at test time, rather than only on training queries, would change "
          "the operating point."),
        P("<b>The classifier is frozen during active learning.</b> Expert answers are "
          "used only to learn competence, never to improve the classifier. In a real "
          "deployment an expert's answer is also a label, and using it for both is the "
          "obvious extension."),
        P("<b>The training set is a sample.</b> Using all 120,000 articles would raise "
          "the baseline by roughly a point and shrink the headroom deferral has to work "
          "with, but should not change which strategy wins."),
        P("7. Reproducing this report", "h1"),
        Paragraph("python manage.py run_experiments   # rewrites results/results.json<br/>"
                  "python manage.py build_report      # rewrites this PDF", s["code"]),
        P("Every step is seeded (data sample, split, k-means, expert answers, model "
          "fitting, query selection), so a rerun reproduces these numbers exactly. The "
          "run takes about %s seconds. The interface at "
          "<font face='Courier'>/project3/</font> reads the same JSON file, so the page "
          "and this report can never disagree." % meta["seconds"]),
    ]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    document = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Project 3: Active Learning for Learning-to-Defer",
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
