"""Builds report/PathSnap_Technical_Report.pdf from the files in results/, maps/ and report/.
Every number, table and graph is read from the repository, so the report is reproducible.
The prose is a DRAFT: rewrite it in your own words and check every statement before submitting.
Run from the PathSnap folder:   pip install reportlab   then   python scripts/08_make_report.py"""
import os, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether)

# ------------------------------------------------------------------ fill these in
TEAM_NAME = "Team 7"
MEMBERS = ["Ramitha Nayak MV - PES1UG24AM392", "Roopesh - PES1UG24AM393", "Rushikesh - PES1UG24AM394", "Sakshi P Shetty - PES1UG24AM395"]
REPO_URL = "https://github.com/ramithanayak98-ar/ML-HACKATHON"
BETA = 50
OUT = "report/PathSnap_Technical_Report.pdf"
FIG = "report/figs"; os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ load results
inv = pd.read_csv("results/eda/dataset_inventory.csv", dtype={"trip_id": str})
mc = pd.read_csv("results/model_comparison.csv")
tuning = pd.read_csv("results/tuned_vs_untuned.csv")
ob = json.load(open("results/optuna_best.json"))
trials = pd.read_csv("results/optuna_trials.csv")
feat_test = pd.read_parquet("results/features_student_test.parquet", columns=["trip_id", "point_idx", "label"])
feat_train = pd.read_parquet("results/features_train.parquet", columns=["label"])
miss_rate = 1 - feat_test.groupby(["trip_id", "point_idx"]).label.max().mean()
hmm_model = {os.path.basename(p)[4:-4]: pd.read_csv(p, dtype={"trip_id": str})
             for p in sorted(glob.glob("results/hmm_*.csv")) if "vs_classifier" not in p}
beta_rows = {}
for p in glob.glob("results/hmm_vs_classifier*.csv"):
    beta_rows.update(pd.read_csv(p).drop(columns="trip_id").mean().to_dict())
betas = sorted({float(k.split("beta")[1]) for k in beta_rows if k.startswith("hmm_top1_beta")})
NICE = {"LogReg": "Logistic Regression", "GaussNB": "Gaussian Naive Bayes", "KNN": "K-Nearest Neighbours",
        "DecTree": "Decision Tree", "RandForest": "Random Forest", "HistGB": "HistGradientBoosting",
        "MLP": "MLP neural network", "final_model": "HistGB tuned (final)"}
pc = lambda x: f"{100*x:.1f}%"

# ------------------------------------------------------------------ extra figures (all from saved results)
def save(fig, name):
    path = f"{FIG}/{name}.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); return path

f, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(trials.number, trials.value, "o", ms=4, label="trial value")
ax.plot(trials.number, trials.value.cummax(), "-", label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("CV macro F1"); ax.set_title("Optuna TPE search history (training trips)"); ax.legend()
fig_optuna = save(f, "optuna_history")

if betas:
    f, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(betas, [beta_rows[f"hmm_top1_beta{b:g}"] for b in betas], "o-", label="Top-1 accuracy")
    ax.plot(betas, [beta_rows[f"hmm_cont_beta{b:g}"] for b in betas], "s-", label="Route continuity")
    ax.axhline(beta_rows["cls_top1"], ls="--", c="C0", alpha=.5, label="classifier-only Top-1")
    ax.axhline(beta_rows["cls_continuity"], ls="--", c="C1", alpha=.5, label="classifier-only continuity")
    ax.set_xscale("log"); ax.set_xlabel("beta (m, log scale)"); ax.set_title("HMM decoder: effect of beta (student-test trips)")
    ax.legend(fontsize=7); fig_beta = save(f, "hmm_beta")
else:
    fig_beta = None

if hmm_model:
    names = [m for m in ["LogReg", "GaussNB", "KNN", "DecTree", "RandForest", "MLP", "HistGB", "final_model"] if m in hmm_model]
    f, axs = plt.subplots(1, 2, figsize=(9, 3.4)); x = np.arange(len(names)); w = .38
    for ax, (a, b, t) in zip(axs, [("cls_top1", f"hmm_top1_beta{BETA}", "Top-1 accuracy"),
                                   ("cls_continuity", f"hmm_cont_beta{BETA}", "Route continuity")]):
        ax.bar(x - w/2, [hmm_model[n][a].mean() for n in names], w, label="classifier only")
        ax.bar(x + w/2, [hmm_model[n][b].mean() for n in names], w, label=f"+ HMM (beta={BETA})")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7); ax.set_title(t); ax.set_ylim(.6, 1)
    axs[0].legend(fontsize=7); fig_hmm_models = save(f, "hmm_per_model")
else:
    fig_hmm_models = None

# ------------------------------------------------------------------ pdf helpers
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.8, leading=13.6, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=20, spaceBefore=6, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)
C = ParagraphStyle("c", parent=B, fontSize=8.2, leading=10.4, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=B, fontSize=8.4, leading=11, textColor=colors.HexColor("#4b5563"), spaceAfter=8)
HD = ParagraphStyle("hd", parent=C, fontName="Helvetica-Bold", textColor=colors.white)
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
def bullets(items):
    for i in items: story.append(Paragraph(i, ParagraphStyle("l", parent=B, leftIndent=14, bulletIndent=3, spaceAfter=2), bulletText="•"))
def table(rows, widths):
    d = [[Paragraph(str(c), HD if r == 0 else C) for c in row] for r, row in enumerate(rows)]
    t = Table(d, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")])]))
    story.append(t); story.append(Spacer(1, 6))
def fig(path, caption, width=165):
    if path and os.path.exists(path):
        from PIL import Image as PI
        w, h = PI.open(path).size; wmm = width; hmm = wmm * h / w
        if hmm > 120: hmm = 120; wmm = hmm * w / h
        story.append(KeepTogether([Image(path, width=wmm * mm, height=hmm * mm), Paragraph(caption, CAP)]))
    else:
        P(f"<font color='#b91c1c'>[Missing figure: {path} - {caption}]</font>", CAP)

# ================================================================== TITLE
story += [Spacer(1, 30)]
P("<font size=22><b>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</b></font>", ParagraphStyle("t", parent=B, leading=28))
P("<font size=12>Technical Report - UE24CS352A Machine Learning Hackathon</font>")
story.append(Spacer(1, 10))
P(f"<b>{TEAM_NAME}</b><br/>" + "<br/>".join(MEMBERS))
P(f"Repository: {REPO_URL}")
P("All numbers, tables and graphs in this report were generated by <font face='Courier'>scripts/08_make_report.py</font> "
  "from the result files committed in the repository (<font face='Courier'>results/</font>, <font face='Courier'>maps/</font>).", CAP)

final_cls = hmm_model.get("final_model")
P("<b>Summary.</b> Each GPS observation is paired with every road arc within 50 m (candidate generation); a classifier scores each "
  "(observation, candidate) pair; an HMM with Viterbi decoding then selects a connected sequence of arcs. Seven classifiers from "
  f"seven algorithm families were compared; HistGradientBoosting was selected and tuned with Bayesian optimisation (Optuna TPE, {ob.get('n_trials')} trials). "
  + (f"On the 10 student-test trips the final pipeline matches {pc(final_cls[f'hmm_top1_beta{BETA}'].mean())} of GPS points to the reference arc "
     f"(classifier only: {pc(final_cls.cls_top1.mean())}) and raises route continuity from {pc(final_cls.cls_continuity.mean())} to "
     f"{pc(final_cls[f'hmm_cont_beta{BETA}'].mean())} (averages over trips)." if final_cls is not None else ""))
story.append(PageBreak())

# ================================================================== 2.2 EDA
P("2.2 Dataset Understanding and Exploratory Data Analysis", H1)
tr, st_ = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
P("<b>Partitions used.</b> All exploration statistics below use the <b>train</b> (80 trips) and <b>student_test</b> (10 trips) partitions; "
  "the partition is stated for every figure. Evaluator-testing trips were not opened during development.")
P("<b>Raw files (verified by inspection).</b> <font face='Courier'>.track</font>: longitude, latitude, time (s); "
  "<font face='Courier'>.nodes</font>: longitude, latitude (row index = node id); <font face='Courier'>.arcs</font>: from-node, to-node "
  "(row index = arc id, each arc is one-directional, so a two-way street appears as two arcs); <font face='Courier'>.route</font>: the reference "
  "sequence of arc ids. The route is <i>not</i> one label per GPS observation, and every trip has its own local road network.")
table([["Statistic", "train (80 trips)", "student_test (10 trips)"],
       ["GPS points per trip (min / median / max)", f"{tr.n_gps.min()} / {int(tr.n_gps.median())} / {tr.n_gps.max()}", f"{st_.n_gps.min()} / {int(st_.n_gps.median())} / {st_.n_gps.max()}"],
       ["Road arcs per trip (min / median / max)", f"{tr.n_arcs.min()} / {int(tr.n_arcs.median())} / {tr.n_arcs.max()}", f"{st_.n_arcs.min()} / {int(st_.n_arcs.median())} / {st_.n_arcs.max()}"],
       ["Missing values in GPS files (total)", str(int(tr.missing_values.sum())), str(int(st_.missing_values.sum()))],
       ["Trips with non-increasing timestamps", str(int((~tr.timestamps_sorted).sum())), str(int((~st_.timestamps_sorted).sum()))],
       ["Time gaps > 5 s (total)", str(int(tr.n_gaps_over_5s.sum())), str(int(st_.n_gaps_over_5s.sum()))],
       ["Largest time gap (s)", f"{tr.max_gap_s.max():.0f}", f"{st_.max_gap_s.max():.0f}"],
       ["Mean % of 1-second steps", f"{tr.pct_1s_steps.mean():.1f}%", f"{st_.pct_1s_steps.mean():.1f}%"],
       ["Weakly connected components (median)", f"{tr.weak_components.median():.0f}", f"{st_.weak_components.median():.0f}"],
       ["Isolated nodes (total)", str(int(tr.isolated_nodes.sum())), str(int(st_.isolated_nodes.sum()))]], [70, 50, 50])
gaps = inv.sort_values("max_gap_s", ascending=False).head(5)
P("<b>Missing-value analysis.</b> " + ("No missing values were found in any GPS file. " if inv.missing_values.sum() == 0 else
  f"{int(inv.missing_values.sum())} missing values were found. ") +
  "The main data-quality issue is irregular sampling: although most steps are exactly 1 s, some trips contain very long gaps. "
  "Largest gaps: " + "; ".join(f"trip {r.trip_id} ({r.partition}): {r.max_gap_s:,.0f} s" for r in gaps.itertuples()) + ".")
fig("results/eda/histograms.png", "Figure 1 - GPS points per trip, road arcs per trip and largest time gap per trip (train + student_test).")
P("<b>Preprocessing.</b>")
bullets(["<b>Projection to metres.</b> Longitude/latitude are angles, and trips lie in different countries, so a single fixed projection is "
         "unsuitable. Each trip is projected with a local equirectangular projection around its own mean position "
         "(x = R cos(lat<sub>0</sub>) (lon - lon<sub>0</sub>), y = R (lat - lat<sub>0</sub>), R = 6 371 km, angles in radians); over a few kilometres "
         "the distortion is far below GPS noise.",
         "<b>Observation labels.</b> Each GPS point is aligned to one arc of the reference route by a monotonic dynamic-programming alignment "
         "(route position never decreases) that minimises total point-to-segment distance. These labels are derived, so they are treated as "
         "reference rather than error-free truth.",
         "<b>Candidate generation.</b> Arcs are sampled every 10 m into a KD-tree; for each GPS point all arcs within <b>50 m</b> are kept "
         f"(at most 10, nearest first). Candidate miss rate on student_test: <b>{100*miss_rate:.2f}%</b> of GPS points have the reference arc "
         "outside their candidate set.",
         f"<b>Resulting tables.</b> {len(feat_train):,} training pairs ({100*feat_train.label.mean():.1f}% positive) and "
         f"{len(feat_test):,} student-test pairs ({100*feat_test.label.mean():.1f}% positive)."])
P("<b>Features and why.</b>")
table([["Feature", "Meaning", "Reason"],
       ["dist_m", "point-to-arc distance (m)", "GPS noise is a few metres; closer arcs are more likely"],
       ["dist_rank, dist_minus_min, dist_ratio", "distance relative to the point's other candidates", "the decision is relative between nearby arcs"],
       ["bearing_diff_deg, cos_bearing", "angle between vehicle heading and arc direction", "separates the two directions of a two-way street, which have equal distance"],
       ["speed_mps, low_speed", "speed from neighbouring points", "heading is unreliable when the vehicle is stopped"],
       ["arc_len_m, proj_t", "arc length, position of projection on the arc", "geometry context"],
       ["n_cand, gap_prev_s", "number of candidates, time since previous point", "ambiguity level and sampling gaps"]], [45, 55, 70])
fig("report/eda_map.png", "Figure 2 - Folium map of a training trip: raw GPS (blue), reference route (green), road network (grey). Partition: train.")
story.append(PageBreak())

# ================================================================== 3 MODELS
P("3. Model Experimentation and Comparison", H1)
P("Each (GPS point, candidate arc) pair is a binary classification example (1 = reference arc). Models were trained on the 80 training trips "
  "with <b>5-fold GroupKFold</b> cross-validation grouped by trip (so no trip appears in both training and validation folds), refitted on all "
  "training trips, and evaluated on the 10 student-test trips. Slow models were trained on a random row sample (column 'Train rows'). "
  "<b>Primary selection metric:</b> observation-level Top-1 accuracy on student_test, with log loss as a check of probability quality, "
  "because the HMM uses the probabilities as emission scores.")
P("3.1 Overall model comparison", H2)
rows = [["Model", "Family", "Key configuration", "CV macro F1", "Top-1", "Route cont. (cls / HMM)", "Test macro F1", "Log loss", "Inference (s)"]]
CONF = {"LogReg": "StandardScaler + L2, max_iter=1000", "GaussNB": "default (var_smoothing 1e-9)",
        "KNN": "k=25, distance-weighted, scaled; 100k rows", "DecTree": "max_depth=12",
        "RandForest": "200 trees, min_samples_leaf=5; 600k rows", "HistGB": "sklearn defaults",
        "MLP": "64-32 ReLU, Adam, early stopping; 400k rows"}
for r in mc.itertuples():
    h = hmm_model.get(r.model)
    cont = f"{pc(h.cls_continuity.mean())} / {pc(h[f'hmm_cont_beta{BETA}'].mean())}" if h is not None else "n/a"
    rows.append([r.model, r.family, CONF.get(r.model, ""), f"{r.cv_macro_f1_mean:.3f} ± {r.cv_macro_f1_std:.3f}", pc(r.test_top1), cont,
                 f"{r.test_macro_f1:.3f}", f"{r.test_log_loss:.3f}", f"{r.inference_time_s:.2f}"])
table(rows, [17, 17, 30, 19, 15, 22, 14, 13, 13])
P("Top-1 here pools all student-test GPS points; route continuity is the trip-averaged fraction of consecutive predicted arcs that are connected. "
  "Log loss is defined for every model because all seven output probabilities.", CAP)
best = mc.loc[mc.test_top1.idxmax()]; worst_ll = mc.loc[mc.test_log_loss.idxmax()]
P(f"<b>Observations.</b> The best student-test Top-1 is {best.model} ({pc(best.test_top1)}). The tree ensembles, gradient boosting and the MLP "
  f"are within about {100*(mc.test_top1.max() - mc[mc.model.isin(['HistGB','RandForest','MLP'])].test_top1.min()):.1f} percentage points of each other, "
  f"which is smaller than the fold-to-fold standard deviation of CV macro F1 (about {mc.cv_macro_f1_std.mean():.3f}). {worst_ll.model} has by far the "
  f"worst log loss ({worst_ll.test_log_loss:.3f}), i.e. poorly calibrated probabilities. <b>Selection:</b> HistGradientBoosting was selected because it is "
  "in the top group on Top-1, macro F1 and log loss, trains on the full training set in seconds, and its probabilities are well calibrated for the HMM. "
  "Random Forest was a close alternative but needed a row sample and about ten times longer training.")
P("3.2 Per-model questions", H2)
INFO = {
 "LogReg": ("A linear baseline: if distance and heading agreement separate correct from wrong candidates almost linearly, it should already work.",
            "Fast, stable, calibrated probabilities. Cannot model interactions such as 'heading matters only when moving'.",
            "C=1.0 (L2), max_iter=1000, features standardised.", "Binary cross-entropy (log loss) with L2 penalty."),
 "GaussNB": ("A probabilistic baseline that models each feature per class with a Gaussian.",
             "Extremely fast. Assumes independent, Gaussian features, which is false here (dist_m, dist_rank and dist_ratio are strongly correlated), "
             "so probabilities are over-confident.", "var_smoothing=1e-9.", "Maximum likelihood of class-conditional Gaussians (no iterative loss)."),
 "KNN": ("Instance-based: similar (distance, heading, speed) patterns should share labels.",
         "No training phase and non-linear. Prediction is slow (distance to every stored example) and sensitive to feature scaling.",
         "k=25, weights='distance', standardised features, trained on a 100 000-row sample.", "None (no training loss; majority vote weighted by distance)."),
 "DecTree": ("Non-linear threshold rules match decisions such as 'within x m and heading within y degrees'.",
             "Interpretable and fast, but high variance and prone to overfitting if grown deep.", "max_depth=12, other parameters default.",
             "Gini impurity reduction at each split."),
 "RandForest": ("Averages many decorrelated trees to reduce the variance of a single tree.",
                "Strong accuracy and good probabilities; slower and larger in memory.",
                "200 trees, min_samples_leaf=5, all cores, trained on a 600 000-row sample.", "Gini impurity per tree; prediction averages tree probabilities."),
 "HistGB": ("Gradient boosting fits trees sequentially to the errors of earlier trees; histogram binning makes it fast on 1.2 M rows.",
            "Top accuracy, well-calibrated probabilities, fast; more hyperparameters to tune.",
            "sklearn defaults: learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.",
            "Binary cross-entropy (log loss) minimised by gradient boosting."),
 "MLP": ("A neural network can learn smooth non-linear combinations of the features.",
         "Flexible; needs scaling, is slower to train and sensitive to initialisation.",
         "hidden layers (64, 32), ReLU, Adam, early_stopping=True, trained on a 400 000-row sample.", "Binary cross-entropy minimised with Adam."),
}
for r in mc.itertuples():
    why, sl, hp, loss = INFO.get(r.model, ("", "", "", ""))
    gap = r.test_macro_f1 - r.cv_macro_f1_mean
    P(f"3.2.{list(mc.model).index(r.model)+1} {NICE.get(r.model, r.model)} ({r.family})", H3)
    bullets([f"<b>Why suitable:</b> {why}", f"<b>Strengths / limitations:</b> {sl}", f"<b>Hyperparameters (initial values):</b> {hp}",
             f"<b>Loss / objective:</b> {loss}",
             f"<b>Student-test results:</b> Top-1 {pc(r.test_top1)}, macro F1 {r.test_macro_f1:.3f}, precision {r.test_precision:.3f}, "
             f"recall {r.test_recall:.3f}, log loss {r.test_log_loss:.3f}; training {r.train_time_s:.1f} s, inference {r.inference_time_s:.2f} s.",
             f"<b>Over/underfitting:</b> CV macro F1 {r.cv_macro_f1_mean:.3f} ± {r.cv_macro_f1_std:.3f} vs student-test {r.test_macro_f1:.3f} "
             f"(difference {gap:+.3f}). " + ("Validation and test agree, so there is no evidence of strong overfitting. " if abs(gap) < 0.03 else
             "The gap suggests a difference between validation and test behaviour. ") +
             "Training-set scores were not recorded, so the train/validation gap could not be measured directly."])
P("3.3 Model visualisations", H2)
fig("results/compare_cv_macro_f1_mean.png", "Figure 3 - Cross-validation macro F1 per model (train, GroupKFold).", 120)
fig("results/compare_test_top1.png", "Figure 4 - Student-test Top-1 accuracy per model.", 120)
fig("results/compare_test_macro_f1.png", "Figure 5 - Student-test macro F1 per model.", 120)
fig("results/compare_test_log_loss.png", "Figure 6 - Student-test log loss per model (lower is better).", 120)
fig("results/mlp_training_curve.png", "Figure 7 - MLP training loss and validation score per epoch (the only model with epoch-wise history).", 110)
cms = [p for p in sorted(glob.glob("results/cm_*.png"))]
if cms:
    imgs = [Image(p, width=55 * mm, height=45 * mm) for p in cms]
    grid = [imgs[i:i + 3] + [""] * (3 - len(imgs[i:i + 3])) for i in range(0, len(imgs), 3)]
    story.append(Table(grid)); P("Figure 8 - Confusion matrices on student_test (threshold 0.5).", CAP)
story.append(PageBreak())

# ================================================================== 4 BAYES OPT
P("4. Bayesian Hyperparameter Optimisation (HistGradientBoosting)", H1)
bp = ob.get("best_params", {})
bullets(["<b>Hyperparameters and ranges:</b> learning_rate 0.01-0.3 (log), max_iter 100-600, max_leaf_nodes 15-127, min_samples_leaf 20-500 (log), "
         "l2_regularization 1e-4 to 10 (log).",
         f"<b>Objective:</b> mean macro F1 of {ob.get('n_folds', 3)}-fold GroupKFold (grouped by trip) on a {ob.get('search_rows', 300000):,}-row sample of the "
         "training trips; the final model was refitted on all training rows. Student-test and evaluator trips were not used in the search.",
         f"<b>Method:</b> Optuna TPE sampler (sequential model-based / Bayesian optimisation), {ob.get('n_trials')} trials, "
         f"{ob.get('search_time_s', 0):.0f} s in total.",
         "<b>Best values:</b> " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp.items()) +
         f" (best CV macro F1 {ob.get('best_cv_macro_f1', 0):.4f})."])
table([["Model", "CV macro F1", "Student-test Top-1", "Student-test macro F1", "Log loss", "Inference (s)"]] +
      [[r.model, f"{r.cv_macro_f1:.4f}", pc(r.test_top1), f"{r.test_macro_f1:.4f}", f"{r.test_log_loss:.4f}", f"{r.inference_time_s:.2f}"] for r in tuning.itertuples()],
      [52, 22, 26, 28, 20, 22])
u, t_ = tuning.iloc[0], tuning.iloc[1]
P(f"<b>Did tuning help?</b> Only marginally: CV macro F1 {u.cv_macro_f1:.4f} -> {t_.cv_macro_f1:.4f}, Top-1 {pc(u.test_top1)} -> {pc(t_.test_top1)}, "
  f"log loss {u.test_log_loss:.4f} -> {t_.test_log_loss:.4f}, while inference became slower ({u.inference_time_s:.2f} s -> {t_.inference_time_s:.2f} s). "
  "The best configuration uses many shallow trees with a low learning rate and strong L2 regularisation. The default configuration was already close to optimal.")
if "HistGB" in hmm_model and "final_model" in hmm_model:
    P(f"After HMM decoding the difference disappears: untuned {pc(hmm_model['HistGB'][f'hmm_top1_beta{BETA}'].mean())} vs tuned "
      f"{pc(hmm_model['final_model'][f'hmm_top1_beta{BETA}'].mean())} Top-1 (trip average), within noise.")
fig(fig_optuna, "Figure 9 - Optuna trial values and best-so-far CV macro F1 (training trips).", 140)
story.append(PageBreak())

# ================================================================== 5 HMM
P("5. HMM and Viterbi Route Reconstruction", H1)
bullets(["<b>Hidden states:</b> at each GPS point t, the candidate arcs of that point (each state keeps its arc_id, so the candidate-to-arc link is preserved).",
         "<b>Emission score:</b> log p<sub>t</sub>(arc), the selected classifier's probability (clipped at 1e-6).",
         "<b>Transition score:</b> -|d<sub>route</sub> - d<sub>gps</sub>| / beta, where d<sub>gps</sub> is the straight-line distance between consecutive GPS "
         "points and d<sub>route</sub> the driving distance along the directed road graph from the projection on the previous arc to the projection on the "
         "next arc (Dijkstra, cached per node, searched up to 2 000 m).",
         "<b>Viterbi recurrence:</b> V<sub>t</sub>(b) = e<sub>t</sub>(b) + max<sub>a</sub> [V<sub>t-1</sub>(a) + T(a -> b)], with back-pointers to recover the best path.",
         "<b>Missing candidates / gaps:</b> GPS points with no candidate within 50 m are skipped. If no transition is reachable (e.g. after a long gap or a "
         "missing road), the current chain is closed and a new chain starts, so one break does not invalidate the whole trip.",
         "<b>Disconnected segments:</b> transitions between arcs that are not reachable within the search limit get probability 0 (score -inf), so the "
         "decoder cannot choose them unless it restarts a chain."])
if betas:
    table([["Setting", "Top-1 (trip avg)", "Route continuity"], ["Classifier only", pc(beta_rows["cls_top1"]), pc(beta_rows["cls_continuity"])]] +
          [[f"HMM beta = {b:g} m", pc(beta_rows[f"hmm_top1_beta{b:g}"]), pc(beta_rows[f"hmm_cont_beta{b:g}"])] for b in betas], [60, 50, 50])
    P(f"beta was tuned on the student-test trips (permitted for decoder parameters). Small beta enforces connectivity strictly but lowers Top-1; "
      f"Top-1 stops improving above beta = {BETA} while continuity keeps falling, so beta = {BETA} was chosen.")
fig(fig_beta, "Figure 10 - Effect of beta on Top-1 accuracy and route continuity (student_test, final model).", 140)
if hmm_model:
    table([["Model", "Top-1 classifier only", f"Top-1 + HMM", "Continuity classifier only", "Continuity + HMM"]] +
          [[NICE.get(n, n), pc(h.cls_top1.mean()), pc(h[f'hmm_top1_beta{BETA}'].mean()), pc(h.cls_continuity.mean()), pc(h[f'hmm_cont_beta{BETA}'].mean())]
           for n, h in hmm_model.items()], [48, 30, 28, 34, 30])
    fig(fig_hmm_models, f"Figure 11 - Classifier-only vs classifier + HMM (beta = {BETA}) for every model (student_test, trip averages).")
if final_cls is not None:
    d = final_cls.assign(delta=final_cls[f"hmm_top1_beta{BETA}"] - final_cls.cls_top1).sort_values("delta")
    n_up = int((final_cls[f"hmm_cont_beta{BETA}"] > final_cls.cls_continuity).sum())
    P(f"<b>Did HMM decoding help?</b> Route continuity improved on {n_up} of {len(final_cls)} trips; for Top-1 it helped most on trip {d.iloc[-1].trip_id} ({d.iloc[-1].delta*100:+.1f} points) "
      f"and {d.iloc[-2].trip_id} ({d.iloc[-2].delta*100:+.1f}), and did not help on trip {d.iloc[0].trip_id} ({d.iloc[0].delta*100:+.1f}) "
      f"and {d.iloc[1].trip_id} ({d.iloc[1].delta*100:+.1f}). The hardest trip is {final_cls.loc[final_cls.cls_top1.idxmin(), 'trip_id']} "
      f"(classifier-only {pc(final_cls.cls_top1.min())}); on such trips a strict decoder (small beta) can lock onto a wrong connected path when the true "
      "arc is unreachable within the search limit, which is why a larger beta works better overall.")
story.append(PageBreak())

# ================================================================== 6 FOLIUM
P("6. Geographical Visualisation Using Folium", H1)
P("Interactive maps are generated by <font face='Courier'>scripts/01_eda.py</font> and <font face='Courier'>scripts/07_make_maps.py</font> and committed "
  "in <font face='Courier'>maps/</font>. Layers: raw GPS (blue points), relevant road network (grey), reference route (green, thick), predicted route "
  "(red), with a layer control and a scale bar.")
fig("report/route_map_good.png", "Figure 12 - Predicted (red) vs reference (green) route on a well-matched student-test trip.")
fig("report/route_map_hard.png", "Figure 13 - The hardest student-test trip: where the predicted route departs from the reference.")
story.append(PageBreak())

# ================================================================== 7 API / UI
P("7. API, UI and Deployment", H1)
bullets(["<b>Laptop 1 (FastAPI, api/main.py):</b> loads the saved models at start-up; endpoints GET /health, GET /trips, GET /models, GET /dev_results, "
         "POST /predict {trip_id, model, use_hmm}. /predict runs candidate generation, features, classifier and Viterbi using only GPS and road network, "
         "then scores the prediction server-side against the reference route and logs every request.",
         "<b>Laptop 2 (Streamlit, ui/app.py - 'SnapRoute'):</b> lists the evaluation trips from the API, sends the selected trip id, and shows metrics and "
         "the predicted route on a map. It never receives reference geometry.",
         "<b>Run:</b> Laptop 1: <font face='Courier'>uvicorn api.main:app --host 0.0.0.0 --port 8000</font>; Laptop 2: "
         "<font face='Courier'>set API_URL=http://&lt;Laptop-1-IP&gt;:8000</font> and <font face='Courier'>streamlit run ui/app.py</font>."])
fig("report/ui_result.png", "Figure 14 - Evaluator UI showing the predicted route and metrics for an evaluation trip.")
fig("report/ui_compare.png", "Figure 15 - UI model-comparison view.")
fig("report/api_log.png", "Figure 16 - API request log on the command prompt of the receiving laptop (Laptop 1).")
story.append(PageBreak())

# ================================================================== 8 INTEGRITY
P("8. Academic Integrity and AI-Use Disclosure", H1)
bullets(["Evaluator-testing trips were not used for training, feature selection, model selection, hyperparameter search or choosing beta.",
         "No predictions are hard-coded; the API computes every route at request time.",
         "All reported numbers come from result files in the repository and can be regenerated with scripts 01-08.",
         "<b>AI assistance:</b> the source code (data processing, feature engineering, model training, Bayesian optimisation, HMM/Viterbi decoder, API and UI) and a first draft of this report were produced with an AI assistant (Claude, Anthropic). The team ran every experiment on its own machine, chose the decoder setting (beta = 50) from the measured results, and all numbers, tables and graphs come from those runs. The report text was reviewed and edited by the team."])

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="PathSnap Technical Report")
def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(18 * mm, 10 * mm, f"PathSnap Technical Report - {TEAM_NAME}"); c.drawRightString(192 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()
doc.build(story, onFirstPage=foot, onLaterPages=foot)
missing = [p for p in ["report/eda_map.png", "report/route_map_good.png", "report/route_map_hard.png", "report/ui_result.png",
                       "report/ui_compare.png", "report/api_log.png"] if not os.path.exists(p)]
print("saved", OUT)
if missing:
    print("Screenshots still missing (add them and run again):", *missing, sep="\n  ")
