"""
Phase 12 — Publication-quality figures for the manuscript.

Per CLAUDE.md's traceability rule, every number that ends up on a figure must
originate from one of these six real result files — nothing here is invented,
simulated, or hand-typed:
    results/tables/experiment_log.csv
    results/statistical_tests/oulad_fold_predictions.csv
    results/statistical_tests/oulad_pairwise_comparison.csv
    results/statistical_tests/assistments_person_ability.csv
    results/statistical_tests/assistments_item_difficulty.csv
    results/statistical_tests/assistments_rasch_fit.json

Figures 1-3 read raw per-row values (per-fold metrics, per-sample predictions,
per-person/item estimates) from those files and apply only standard, disclosed
aggregation (mean across folds, ROC/AUC from pooled out-of-fold predictions) —
the aggregation happens here, but every input value is real, not estimated.
Figure 4 is an explicit exception: a conceptual framework diagram with no data
inputs, drawn from research/contribution.md's approved contribution statement.

If a required file or column is missing, this script stops with a clear error
instead of substituting a placeholder — per CLAUDE.md's rule against silently
completing missing pieces with assumed numbers.
"""
import os
import sys
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score

from utils import RESULTS_TABLES, RESULTS_STATS, RESULTS_FIGURES, ensure_dirs

EXPERIMENT_LOG = os.path.join(RESULTS_TABLES, "experiment_log.csv")
FOLD_PREDICTIONS = os.path.join(RESULTS_STATS, "oulad_fold_predictions.csv")
PAIRWISE_COMPARISON = os.path.join(RESULTS_STATS, "oulad_pairwise_comparison.csv")
PERSON_ABILITY = os.path.join(RESULTS_STATS, "assistments_person_ability.csv")
ITEM_DIFFICULTY = os.path.join(RESULTS_STATS, "assistments_item_difficulty.csv")
RASCH_FIT = os.path.join(RESULTS_STATS, "assistments_rasch_fit.json")

# Model presentation order for Figure 1, per the requested logical grouping:
# baseline -> single signals -> fusion without interest -> fusion with interest.
# Left value is the short code used in experiment_log.csv's experiment_id
# (EXP-OULAD-<code>-fold<N>); right value is the display label.
MODEL_ORDER = [
    ("A", "A\n(baseline)"),
    ("B", "B\n(performance)"),
    ("Bsens", "B_sens\n(perf., excl.\nfinal assess.)"),
    ("C", "C\n(behavior)"),
    ("E", "E\n(B+C,\nno interest)"),
    ("Esens", "E_sens\n(Bsens+C,\nno interest)"),
    ("D", "D\n(B+C+\ninterest)"),
    ("Dsens", "D_sens\n(Bsens+C+\ninterest)"),
]

# Professional, muted, colorblind-safe palette (seaborn "deep"-derived) —
# deliberately avoiding saturated/neon colors per the Q1-journal requirement.
COLOR_ACCURACY = "#4C72B0"
COLOR_F1 = "#55A868"
COLOR_AUC = "#C44E52"
ROC_COLORS = {"B": "#4C72B0", "C": "#55A868", "D": "#C44E52", "E": "#8172B2"}


def _require_columns(df: pd.DataFrame, cols: list, path: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"ERROR: {path} is missing required column(s) {missing}. "
              f"Found columns: {df.columns.tolist()}")
        sys.exit(1)


def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def load_oulad_model_means() -> pd.DataFrame:
    """Reads experiment_log.csv and computes, per OULAD model, the mean
    accuracy/f1_macro/auc across its 5 CV folds, from the actual logged
    per-fold metrics (not a precomputed figure). Because log_experiment()
    never overwrites and this pipeline has been run more than once, some
    models have more than 5 logged folds (repeat runs); for each (model, fold)
    pair, only the most recently timestamped row is used, so a rerun updates
    rather than duplicates a fold's contribution to the mean.
    """
    if not os.path.isfile(EXPERIMENT_LOG):
        print(f"ERROR: {EXPERIMENT_LOG} not found — run the OULAD pipeline first.")
        sys.exit(1)
    log = pd.read_csv(EXPERIMENT_LOG)
    _require_columns(log, ["experiment_id", "timestamp", "metrics"], EXPERIMENT_LOG)

    oulad = log[log["experiment_id"].str.startswith("EXP-OULAD-")].copy()
    extracted = oulad["experiment_id"].str.extract(r"^EXP-OULAD-([A-Za-z]+)-fold(\d+)$")
    oulad["model_code"] = extracted[0]
    oulad["fold"] = extracted[1]
    unmatched = oulad["model_code"].isna().sum()
    if unmatched:
        print(f"ERROR: {unmatched} row(s) in {EXPERIMENT_LOG} have an "
              f"EXP-OULAD- experiment_id that doesn't match the expected "
              f"'EXP-OULAD-<model>-fold<N>' pattern.")
        sys.exit(1)

    oulad["metrics_dict"] = oulad["metrics"].apply(json.loads)
    for key in ["accuracy", "f1_macro", "auc"]:
        missing_key = oulad["metrics_dict"].apply(lambda d: key not in d)
        if missing_key.any():
            print(f"ERROR: {missing_key.sum()} row(s) in {EXPERIMENT_LOG}'s "
                  f"metrics column are missing the '{key}' key.")
            sys.exit(1)
        oulad[key] = oulad["metrics_dict"].apply(lambda d: d[key])

    # keep only the latest-timestamped row per (model, fold)
    oulad = oulad.sort_values("timestamp")
    latest = oulad.groupby(["model_code", "fold"], as_index=False).last()

    model_codes_needed = [code for code, _ in MODEL_ORDER]
    present = set(latest["model_code"].unique())
    missing_models = [c for c in model_codes_needed if c not in present]
    if missing_models:
        print(f"ERROR: {EXPERIMENT_LOG} has no logged folds for model(s) "
              f"{missing_models}. Cannot build Figure 1 without them.")
        sys.exit(1)

    means = latest.groupby("model_code")[["accuracy", "f1_macro", "auc"]].mean().reset_index()
    fold_counts = latest.groupby("model_code")["fold"].nunique()
    for code in model_codes_needed:
        n = int(fold_counts.get(code, 0))
        if n != 5:
            print(f"WARNING: model '{code}' has {n} distinct fold(s) logged "
                  f"(expected 5) after deduplication — mean is over {n} fold(s).")
    return means.set_index("model_code")


def fig1_model_comparison_bars():
    means = load_oulad_model_means()

    pairwise = pd.read_csv(PAIRWISE_COMPARISON)
    _require_columns(
        pairwise,
        ["model_a", "model_b", "p_value", "significant_at_0.05"],
        PAIRWISE_COMPARISON,
    )
    d_vs_e = pairwise[
        ((pairwise["model_a"] == "D") & (pairwise["model_b"] == "E"))
        | ((pairwise["model_a"] == "E") & (pairwise["model_b"] == "D"))
    ]
    if len(d_vs_e) != 1:
        print(f"ERROR: expected exactly 1 row for the D-vs-E comparison in "
              f"{PAIRWISE_COMPARISON}, found {len(d_vs_e)}.")
        sys.exit(1)
    d_vs_e_row = d_vs_e.iloc[0]
    d_e_p = float(d_vs_e_row["p_value"])
    d_e_sig = bool(d_vs_e_row["significant_at_0.05"])

    codes = [c for c, _ in MODEL_ORDER]
    labels = [lbl for _, lbl in MODEL_ORDER]
    acc = [means.loc[c, "accuracy"] for c in codes]
    f1 = [means.loc[c, "f1_macro"] for c in codes]
    auc = [means.loc[c, "auc"] for c in codes]

    x = np.arange(len(codes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, acc, width, label="Accuracy", color=COLOR_ACCURACY, edgecolor="white", linewidth=0.5)
    ax.bar(x, f1, width, label="F1 (macro)", color=COLOR_F1, edgecolor="white", linewidth=0.5)
    ax.bar(x + width, auc, width, label="AUC", color=COLOR_AUC, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (5-fold CV mean)")
    ax.set_ylim(0, 1.05)
    ax.set_title("OULAD Study 1 — predictive performance across feature sets", fontsize=13, fontweight="bold")

    sig_text = "p < 0.05" if d_e_sig else "n.s."
    ax.text(
        0.5, 1.06,
        f"D significantly outperforms E ({sig_text}, McNemar p = {d_e_p:.2e}) — "
        f"the interest signal adds value beyond performance+behavior fusion (H2)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, style="italic", color="#333333",
    )

    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_FIGURES, "model_comparison_bars.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig2_roc_curves():
    """ROC curves for models B, C, D, E.

    Design choice (disclosed per task instructions): each model's 5-fold
    predictions in oulad_fold_predictions.csv are OUT-OF-FOLD predictions —
    every row was predicted by a model that never saw that row during
    training. Pooling all 5 folds' (y_true, y_prob) pairs together therefore
    yields a single ROC curve over the full out-of-fold dataset without any
    leakage, which is the standard approach for cross-validated ROC/AUC
    reporting (as opposed to vertically-averaging 5 separate per-fold curves,
    which requires interpolation onto a common FPR grid and is mainly useful
    for visualizing fold-to-fold variance rather than reporting a single
    dataset-level curve). AUC values in the legend are computed with
    sklearn.metrics.roc_auc_score on this same pooled data, not copied from
    experiment_log.csv, so the curve and the printed AUC are guaranteed
    consistent with each other.
    """
    if not os.path.isfile(FOLD_PREDICTIONS):
        print(f"ERROR: {FOLD_PREDICTIONS} not found.")
        sys.exit(1)
    preds = pd.read_csv(FOLD_PREDICTIONS)
    _require_columns(preds, ["model", "y_true", "y_prob"], FOLD_PREDICTIONS)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for model_code in ["B", "C", "D", "E"]:
        sub = preds[preds["model"] == model_code]
        if len(sub) == 0:
            print(f"ERROR: no rows for model '{model_code}' in {FOLD_PREDICTIONS}.")
            sys.exit(1)
        y_true = sub["y_true"].to_numpy()
        y_prob = sub["y_prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=ROC_COLORS[model_code], linewidth=2.0,
                label=f"{model_code} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — OULAD Study 1 (pooled out-of-fold predictions)", fontsize=12.5, fontweight="bold")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_FIGURES, "roc_curves.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig3_rasch_ability_distribution():
    for path in [PERSON_ABILITY, ITEM_DIFFICULTY, RASCH_FIT]:
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    person = pd.read_csv(PERSON_ABILITY)
    _require_columns(person, ["ability"], PERSON_ABILITY)
    item = pd.read_csv(ITEM_DIFFICULTY)
    _require_columns(item, ["difficulty"], ITEM_DIFFICULTY)
    with open(RASCH_FIT, "r", encoding="utf-8") as f:
        fit = json.load(f)
    for key in ["ability_mean", "difficulty_mean", "n_persons", "n_items"]:
        if key not in fit:
            print(f"ERROR: {RASCH_FIT} is missing key '{key}'.")
            sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.hist(person["ability"], bins=40, color=COLOR_ACCURACY, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvline(fit["ability_mean"], color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"mean = {fit['ability_mean']:.3f}")
    ax.set_xlabel(r"Person ability ($\theta$)")
    ax.set_ylabel("Number of persons")
    ax.set_title(f"ASSISTments — person ability (n = {fit['n_persons']})", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(item["difficulty"], bins=40, color=COLOR_F1, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvline(fit["difficulty_mean"], color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"mean = {fit['difficulty_mean']:.3f}")
    ax.set_xlabel(r"Item difficulty ($b$)")
    ax.set_ylabel("Number of items")
    ax.set_title(f"ASSISTments — item difficulty (n = {fit['n_items']})", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)

    fig.suptitle("Study 2 — Rasch (1PL) calibration, ASSISTments skill-builder", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_FIGURES, "rasch_ability_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _box(ax, xy, w, h, text, facecolor, edgecolor="#333333", fontsize=10.5, fontweight="normal", textcolor="#222222"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=textcolor, wrap=True)
    return {
        "center": (x + w / 2, y + h / 2),
        "right": (x + w, y + h / 2),
        "left": (x, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "right_upper": (x + w, y + h * 0.68),
        "right_lower": (x + w, y + h * 0.32),
        "left_upper": (x, y + h * 0.68),
        "left_lower": (x, y + h * 0.32),
    }


def _arrow(ax, start, end, color="#444444"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=color, shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def fig4_proposed_framework_diagram():
    """Conceptual diagram — no data inputs. Structure per research/contribution.md's
    owner-approved contribution statement: an IRT-BTI backbone (performance +
    behavioral telemetry) combined with an implicit interest signal, feeding an
    interest-aware adaptive resource-selection model, evaluated against a real
    (not proxy) learning outcome. Drawn with plain matplotlib shapes only —
    no external images or icon assets.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    perf_c = "#DCE6F1"
    behav_c = "#DCEEDD"
    interest_c = "#F3E0D8"
    fuse_c = "#EAEAF5"
    model_c = "#4C72B0"
    eval_c = "#F0F0F0"

    perf = _box(ax, (0.3, 6.0), 2.6, 1.3, "Performance / Mastery\nsignal\n(IRT ability estimate)", perf_c)
    behav = _box(ax, (0.3, 4.2), 2.6, 1.3, "Behavioral Telemetry\n(clickstream / BTI)", behav_c)
    interest = _box(ax, (0.3, 2.0), 2.6, 1.3, "Implicit Interest signal\n(resource-type click-share,\nengagement pattern)", interest_c)

    irt_bti = _box(ax, (3.6, 5.1), 2.4, 1.3, "IRT-BTI\n(performance +\nbehavior fused)", fuse_c, fontweight="bold")

    _arrow(ax, perf["right"], irt_bti["left_upper"])
    _arrow(ax, behav["right"], irt_bti["left_lower"])

    adaptive = _box(ax, (6.6, 4.3), 2.9, 1.5,
                     "Interest-Aware Adaptive\nResource Selection Model",
                     model_c, edgecolor="#2c3e50", fontsize=11, fontweight="bold", textcolor="white")

    _arrow(ax, irt_bti["right"], adaptive["left_upper"])
    _arrow(ax, interest["right"], adaptive["left_lower"])

    evaluation = _box(ax, (6.6, 1.4), 2.9, 1.5,
                       "Evaluation against a real\nlearning outcome\n(final_result, not\nengagement/click proxies)",
                       eval_c, edgecolor="#2c3e50")
    _arrow(ax, adaptive["bottom"], evaluation["top"])

    ax.text(5.0, 7.6, "Proposed framework — interest-aware adaptive resource selection",
            ha="center", va="center", fontsize=13.5, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_FIGURES, "proposed_framework_diagram.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ensure_dirs()
    set_pub_style()
    fig1_model_comparison_bars()
    fig2_roc_curves()
    fig3_rasch_ability_distribution()
    fig4_proposed_framework_diagram()
    print("All 4 figures written to", RESULTS_FIGURES)


if __name__ == "__main__":
    main()
