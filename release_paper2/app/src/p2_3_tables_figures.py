"""
Paper 2, Step 3 (2026-10-14): tables T1-T6 (CSV + pipe text) and figures F1-F4 (PNG 300 dpi + SVG + data CSV),
plus p2_alt_texts.md. Reads existing output files only; no new statistics.
Palette: Okabe-Ito (colour-blind safe); every series also differs by marker / hatch / line style so the figures
remain readable in grayscale. PNG/SVG metadata (software, date) are stripped.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import p2_common as P
from utils import RESULTS_STATS

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "pink": "#CC79A7", "sky": "#56B4E9",
      "verm": "#D55E00", "black": "#000000"}
FAM_STYLE = {"LR": dict(color=OI["blue"], marker="o", linestyle="-"),
             "GBM": dict(color=OI["verm"], marker="s", linestyle="--")}
plt.rcParams.update({"font.size": 9, "svg.hashsalt": "p2", "axes.grid": True, "grid.color": "#dddddd",
                     "figure.dpi": 100})
OUT = P.OUT
TXT = []


def rd(name):
    return pd.read_csv(os.path.join(RESULTS_STATS, name), comment="#", low_memory=False)


def fmt(x, nd=4):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}"
    return str(x)


def table(df, name, title, notes=()):
    P.write_csv(df, f"{name}.csv", [title, *notes])
    lines = [title, *notes, "|".join(df.columns)]
    for _, r in df.iterrows():
        lines.append("|".join(fmt(v) for v in r.values))
    long_ = [l for l in lines if len(l) > 110]
    path = os.path.join(OUT, f"{name}.txt")
    if not P.SMOKE:
        assert not os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}; lines > 110 chars: {len(long_)}", flush=True)
    TXT.append("\n".join(lines))


def save(fig, name):
    for ext, meta in [("png", {"Software": None}), ("svg", {"Creator": None, "Date": None, "Format": None,
                                                              "Type": None})]:
        path = os.path.join(OUT, f"{name}.{ext}")
        if not P.SMOKE:
            assert not os.path.exists(path)
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=meta)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def tables():
    z = rd("p2_1_zbase_metrics_summary.csv")
    t1 = z[z.family == "LR"].drop_duplicates("setting")[["setting", "n_source_rows", "n_rows_used", "n_dropped_rows",
                                                          "dropped_reason", "base_columns"]].copy()
    t1["base"] = ["C" if "mean_score" not in b else "B+C" for b in t1.base_columns]
    t1["dropped_reason"] = ["risk set: unregistered by t" if r.startswith("unregistered") else
                            "no complete B/C/interest set" for r in t1.dropped_reason]
    assert set(t1.base_columns) <= {"total_clicks;active_days;mean_clicks_per_active_day",
                                    "mean_score;weighted_mean_score;n_assessments;score_std;total_clicks;active_days;"
                                    "mean_clicks_per_active_day"}, set(t1.base_columns)
    t1 = t1.drop(columns=["base_columns"]).rename(columns={"n_source_rows": "N_source", "n_rows_used": "N",
                                                          "n_dropped_rows": "dropped"})
    table(t1, "p2_T1_data_settings", "T1. Settings, N and dropped rows",
          ["Source: p2_1_zbase_metrics_summary.csv; FULL drop breakdown: p2_5_dropped_rows_breakdown.csv.",
           "Base columns (B+C): " + "B = mean_score, weighted_mean_score, n_assessments, score_std;",
           "C = total_clicks, active_days, mean_clicks_per_active_day (EARLY: computed on data up to day t)."])

    rob, ci1, dim = rd("robustness_stratcv_metrics_summary.csv"), rd("ci1_stratcv_metrics_summary.csv"), \
        rd("robustness_dimctrl_metrics_summary.csv")
    arms = [("E (base: B+C)", rob, "E"), ("D (E + 21 interest)", rob, "D"), ("course one-hot only", ci1, "course_only"),
            ("E_course", ci1, "E_course"), ("D_course", ci1, "D_course"), ("E_course + centered 21", ci1, "E_course_ctr"),
            ("E_course + shuffled centered 21", ci1, "E_course_ctrshuf"), ("E + shuffled 21 (within pres.)", dim, "E_shufint"),
            ("E + 21 noise columns", dim, "E_noise21")]
    t2 = []
    for lab, d, m in arms:
        row = {"arm": lab, "model": m}
        for fam in ["LR", "GBM"]:
            r = d[(d.family == fam) & (d.model == m)].iloc[0]
            row.update({f"{fam}_acc": r.accuracy_pooled, f"{fam}_f1": r.f1_macro_pooled, f"{fam}_auc": r.auc_pooled})
        row["N"] = int(r.n)
        t2.append(row)
    table(pd.DataFrame(t2), "p2_T2_full_main", "T2. FULL setting, pooled out-of-fold metrics, stratified 5-fold",
          ["Sources: robustness_stratcv_metrics_summary.csv, ci1_stratcv_metrics_summary.csv,",
           "robustness_dimctrl_metrics_summary.csv. Comparisons with CIs: p2_numbers.csv P03/P04."])

    c = rd("ml_1b_comparisons_bh.csv")
    t3 = c[["setting", "family", "model_a", "model_b", "n_common", "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high",
            "delta_auc", "odds_ratio", "p_value_bh"]].rename(columns={"n_common": "N", "family": "model"})
    table(t3, "p2_T3_within_between", "T3. Within/between decomposition (source: ml_1b_comparisons_bh.csv)",
          ["W = base + course-centered 21; M = base + course means of 21; WB = both.",
           "BH within setting x model (3). Centering / course means: students of that presentation",
           "in the analysed sample, features only, no labels."])

    s = rd("leak_breadth_SUMMARY.csv")
    s = s[(s.scheme == "stratified_5fold") & (s.comparison_id.isin(["C5", "C2"]))]
    t4 = s[["setting", "feature_family", "comparison", "model_family", "N", "delta_acc", "delta_acc_ci_low",
            "delta_acc_ci_high", "delta_auc", "bh_p"]].copy()
    nums = rd("p2_numbers.csv")
    rule = nums[nums.claim_id == "P09"].set_index("setting")["value"]
    t4["primary_rule_met"] = [int(rule[f"{a}_{b}"]) if c_ == "base_course->full21_ctr" else ""
                              for a, b, c_ in zip(t4.setting, t4.feature_family, t4.comparison)]
    zc = rd("p2_1_zbase_comparisons_bh.csv")
    zc = zc[zc.setting != "FULL"]
    t4z = pd.DataFrame({"setting": zc.setting.str.extract(r"(EARLY_t\d+)")[0],
                        "feature_family": zc.setting.str.extract(r"_(VLEonly|withAssess)")[0],
                        "comparison": zc.model_a + "->" + zc.model_b, "model_family": zc.family, "N": zc.n_common,
                        "delta_acc": zc.delta_acc, "delta_acc_ci_low": zc.delta_acc_ci_low,
                        "delta_acc_ci_high": zc.delta_acc_ci_high, "delta_auc": zc.delta_auc, "bh_p": zc.p_value_bh,
                        "primary_rule_met": ""})
    t4 = pd.concat([t4, t4z], ignore_index=True).sort_values(["setting", "feature_family", "comparison",
                                                              "model_family"], kind="stable")
    t4["comparison"] = t4["comparison"].str.replace("base_course_z_full21", "z_full21").str.replace(
        "base_course_z", "z_base").str.replace("base_course", "base_c")
    t4 = t4.rename(columns={"feature_family": "family", "model_family": "model", "delta_acc": "dAcc",
                            "delta_acc_ci_low": "ci_lo", "delta_acc_ci_high": "ci_hi", "delta_auc": "dAUC",
                            "primary_rule_met": "rule"})
    table(t4, "p2_T4_early_primary", "T4. EARLY comparisons. PRIMARY = base_c->full21_ctr (prereg rule)",
          ["base_c = base_course; z_base = base_course_z; z_full21 = base_course_z + full21_ctr.",
           "Sources: leak_breadth_SUMMARY.csv (BH over 12); p2_1_zbase_comparisons_bh.csv (BH over 4);",
           "p2_numbers.csv P09 (primary rule, re-derived). z-scoring: within presentation, features only."])

    m = rd("ml_3_module_comparisons_bh.csv")
    t5 = m[["setting", "module", "family", "n_common", "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high",
            "delta_acc_boot_se", "p_value_bh", "module_meets_criterion", "label_inconclusive"]].rename(
        columns={"n_common": "N", "family": "model"})
    t5["setting"] = t5["setting"].str.replace("EARLY_", "")
    t5 = t5.rename(columns={"delta_acc": "dAcc", "delta_acc_ci_low": "ci_lo", "delta_acc_ci_high": "ci_hi",
                            "delta_acc_boot_se": "se", "p_value_bh": "p_bh", "module_meets_criterion": "crit",
                            "label_inconclusive": "inconclusive"})
    re_ = rd("ml_3_random_effects.csv")
    pooled = pd.DataFrame({"setting": re_.setting.str.replace("EARLY_", ""), "module": "POOLED", "model": re_.family, "N": "",
                           "dAcc": re_.pooled_dl, "ci_lo": re_.ci_low_dl,
                           "ci_hi": re_.ci_high_dl, "se": re_.se_dl,
                           "p_bh": "", "crit": "",
                           "inconclusive": ["I2=" + fmt(a, 3) + ";tau2=" + fmt(b, 3) + ";HK " + fmt(c_, 3) + "/" + fmt(d, 3)
                                                  for a, b, c_, d in zip(re_.I2_percent, re_.tau2, re_.ci_low_hk,
                                                                         re_.ci_high_hk)]})
    table(pd.concat([t5, pooled], ignore_index=True), "p2_T5_module_replication",
          "T5. Per-module replication, base_course -> base_course_ctr",
          ["Sources: ml_3_module_comparisons_bh.csv, ml_3_random_effects.csv. BH over 28.",
           "POOLED rows: DL estimate, normal 95% CI, se = DL SE; last column: I2 (%), tau2,",
           "Hartung-Knapp CI (t, df 6). Labels per prereg_ml_2026-10-14.md."])

    f = rd("robustness_fairness_gender_fnr.csv")
    t6 = f[["family", "model", "n_rows", "n_actual_fail_F", "n_actual_fail_M", "fnr_F", "fnr_M", "gap_F_minus_M",
            "gap_ci_low", "gap_ci_high"]].rename(columns={"family": "model_family", "n_rows": "N"})
    table(t6, "p2_T6_fnr_gender", "T6. Gender FNR gap (F minus M), FULL, stratified 5-fold",
          ["Source: robustness_fairness_gender_fnr.csv. FNR = share predicted Pass among students",
           "who failed or withdrew. CI = student-level bootstrap (1000)."])


def figures():
    alt = []
    # F1
    ms = rd("ml_1b_metrics_summary.csv")
    arms = ["base", "base_course", "W", "M", "WB"]
    d1 = ms[ms.model.isin(arms)][["setting", "family", "model", "n", "accuracy_pooled"]]
    P.write_csv(d1, "p2_F1_data.csv", ["Source: ml_1b_metrics_summary.csv"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, st in zip(axs, ["FULL", "EARLY_t28", "EARLY_t56"]):
        for k, fam in enumerate(["LR", "GBM"]):
            v = d1[(d1.setting == st) & (d1.family == fam)].set_index("model").loc[arms, "accuracy_pooled"]
            sty = FAM_STYLE[fam]
            ax.plot(np.arange(5) + (k - 0.5) * 0.15, v.values, marker=sty["marker"], color=sty["color"],
                    linestyle="none", label=fam, markersize=5, markerfacecolor="white" if fam == "GBM" else sty["color"])
        ax.set_xticks(range(5))
        ax.set_xticklabels(arms, rotation=45, ha="right")
        ax.set_title(st)
    axs[0].set_ylabel("Pooled accuracy")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "p2_F1_accuracy_by_arm")
    alt.append(("F1", "p2_F1_accuracy_by_arm", "Dot plot in three panels (FULL, EARLY_t28, EARLY_t56). The x-axis "
                "lists five arms: base, base_course, W, M, WB. The y-axis shows pooled out-of-fold accuracy (each "
                "panel has its own y range). Each panel shows one point per arm for logistic regression (filled "
                "circles) and for gradient boosting (open squares)."))
    # F2
    icc = rd("ml_1a_icc.csv")
    d2 = icc[icc.grouping == "presentation"][["setting", "variable", "N", "n_groups", "icc1", "icc_defined"]]
    P.write_csv(d2, "p2_F2_data.csv", ["Source: ml_1a_icc.csv (grouping = presentation). icc1 empty = undefined."])
    vars_ = list(d2[d2.setting == "FULL"].variable)
    fig, ax = plt.subplots(figsize=(4.8, 5.6))
    for k, (st, mk, col) in enumerate([("FULL", "o", OI["black"]), ("EARLY_t28", "^", OI["blue"]),
                                       ("EARLY_t56", "s", OI["orange"])]):
        v = d2[d2.setting == st].set_index("variable").loc[vars_, "icc1"]
        y = np.arange(len(vars_)) + (k - 1) * 0.22
        ax.plot(v.values, y, marker=mk, color=col, linestyle="none", label=st, markersize=4,
                markerfacecolor="white" if k == 1 else col)
        for yy, val in zip(y, v.values):
            if np.isnan(val):
                ax.text(0.0, yy, "undefined", fontsize=6, va="center", color=col)
    ax.set_yticks(range(len(vars_)))
    ax.set_yticklabels([x.replace("click_share_", "") for x in vars_], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("ICC(1) across the 22 presentations")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "p2_F2_icc_by_feature")
    alt.append(("F2", "p2_F2_icc_by_feature", "Horizontal dot plot. The y-axis lists the binary outcome and the 21 "
                "interest columns (20 activity-type click shares and resource_diversity_entropy). The x-axis is "
                "ICC(1) across the 22 presentations. Three markers per row: FULL (black circle), EARLY_t28 (open "
                "triangle) and EARLY_t56 (orange square). Rows where the ICC is undefined because the column is "
                "constant are labelled 'undefined' instead of a marker."))
    # F3
    m = rd("ml_3_module_comparisons_bh.csv")
    re_ = rd("ml_3_random_effects.csv")
    d3 = pd.concat([m[["setting", "module", "family", "n_common", "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high"]],
                    pd.DataFrame({"setting": re_.setting, "module": "Pooled DL", "family": re_.family, "n_common": np.nan,
                                  "delta_acc": re_.pooled_dl, "delta_acc_ci_low": re_.ci_low_dl,
                                  "delta_acc_ci_high": re_.ci_high_dl, "hk_ci_low": re_.ci_low_hk,
                                  "hk_ci_high": re_.ci_high_hk})], ignore_index=True)
    P.write_csv(d3, "p2_F3_data.csv", ["Sources: ml_3_module_comparisons_bh.csv, ml_3_random_effects.csv."])
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    mods = sorted(m.module.unique())
    for i, st in enumerate(["EARLY_t28", "EARLY_t56"]):
        for j, fam in enumerate(["LR", "GBM"]):
            ax = axs[i, j]
            g = m[(m.setting == st) & (m.family == fam)].set_index("module").loc[mods]
            y = np.arange(len(mods))
            sty = FAM_STYLE[fam]
            ax.errorbar(g.delta_acc, y, xerr=[g.delta_acc - g.delta_acc_ci_low, g.delta_acc_ci_high - g.delta_acc],
                        fmt=sty["marker"], color=sty["color"], ecolor=sty["color"], capsize=2, markersize=4)
            p = re_[(re_.setting == st) & (re_.family == fam)].iloc[0]
            yp = len(mods) + 0.6
            ax.fill([p.ci_low_dl, p.pooled_dl, p.ci_high_dl, p.pooled_dl],
                    [yp, yp - 0.3, yp, yp + 0.3], color=OI["black"])
            ax.plot([p.ci_low_hk, p.ci_high_hk], [yp + 0.55] * 2, color=OI["black"], linestyle=":", linewidth=1.2)
            ax.axvline(0, color="#555555", linewidth=0.8)
            ax.set_yticks(list(y) + [yp])
            ax.set_yticklabels(mods + ["Pooled (DL)"], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{st}, {fam}", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("Delta accuracy (with - without)")
    fig.tight_layout()
    save(fig, "p2_F3_module_forest")
    alt.append(("F3", "p2_F3_module_forest", "Forest plot in four panels (EARLY_t28 and EARLY_t56 by logistic "
                "regression and gradient boosting). Each panel lists the seven modules AAA to GGG on the y-axis "
                "with the change in accuracy from adding the 21 course-centered interest columns "
                "(point and 95% bootstrap interval). A black diamond at the bottom shows the "
                "DerSimonian-Laird pooled estimate and its 95% interval, and a dotted line above it shows the "
                "Hartung-Knapp interval. A vertical reference line marks zero."))
    # F4
    s = rd("leak_breadth_SUMMARY.csv")
    s = s[(s.scheme == "stratified_5fold") & (s.comparison_id == "C5")]
    zc = rd("p2_1_zbase_comparisons_bh.csv")
    zc = zc[(zc.setting != "FULL") & (zc.model_b == "base_course_z_full21")]
    d4 = pd.concat([pd.DataFrame({"setting": s.setting + "_" + s.feature_family, "comparison": "base_course -> full21_ctr",
                                  "family": s.model_family, "delta_acc": s.delta_acc, "ci_low": s.delta_acc_ci_low,
                                  "ci_high": s.delta_acc_ci_high, "source": "leak_breadth_SUMMARY.csv"}),
                    pd.DataFrame({"setting": zc.setting, "comparison": "base_course_z -> + full21_ctr",
                                  "family": zc.family, "delta_acc": zc.delta_acc, "ci_low": zc.delta_acc_ci_low,
                                  "ci_high": zc.delta_acc_ci_high, "source": "p2_1_zbase_comparisons_bh.csv"})],
                   ignore_index=True)
    P.write_csv(d4, "p2_F4_data.csv", ["Sources in column 'source'."])
    sets = ["EARLY_t28_VLEonly", "EARLY_t28_withAssess", "EARLY_t56_VLEonly", "EARLY_t56_withAssess"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    offs = {("base_course -> full21_ctr", "LR"): -0.27, ("base_course -> full21_ctr", "GBM"): -0.09,
            ("base_course_z -> + full21_ctr", "LR"): 0.09, ("base_course_z -> + full21_ctr", "GBM"): 0.27}
    for (cmp_, fam), off in offs.items():
        g = d4[(d4.comparison == cmp_) & (d4.family == fam)].set_index("setting").loc[sets]
        sty = FAM_STYLE[fam]
        filled = cmp_.startswith("base_course ->")
        ax.errorbar(np.arange(4) + off, g.delta_acc, yerr=[g.delta_acc - g.ci_low, g.ci_high - g.delta_acc],
                    fmt=sty["marker"], color=sty["color"], capsize=2, markersize=5,
                    markerfacecolor=sty["color"] if filled else "white", label=f"{fam}: {cmp_}")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(sets, fontsize=8)
    ax.set_ylabel("Delta accuracy (95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "p2_F4_early_delta_acc")
    alt.append(("F4", "p2_F4_early_delta_acc", "Point-and-interval plot. The x-axis lists four EARLY settings "
                "(t28 and t56, each VLE-only and with assessments). The y-axis shows the change in pooled accuracy "
                "with its 95% student-level bootstrap interval, with a horizontal reference line at zero. Each "
                "setting has four points: logistic regression (circles) and gradient boosting (squares), for "
                "base_course to full21_ctr (filled) and for z-scored base_course to the same plus full21_ctr "
                "(open)."))
    path = os.path.join(OUT, "p2_alt_texts.md")
    if not P.SMOKE:
        assert not os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Alt texts for Paper 2 figures (factual descriptions of what is plotted)\n\n")
        for k, name, txt in alt:
            f.write(f"## {k} ({name}.png / .svg; data: p2_{k}_data.csv)\n\n{txt}\n\n")
    print(f"wrote {path}", flush=True)


def main():
    if os.environ.get("P2_FIGS_ONLY") != "1":
        tables()
    figures()
    print("\n\n".join(TXT))


if __name__ == "__main__":
    main()
