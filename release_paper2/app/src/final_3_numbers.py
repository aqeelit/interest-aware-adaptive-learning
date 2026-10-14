"""
Final round, Step 3 (2026-10-14): traceable numbers table for the exploratory note.

Values are read ONLY from the key-aligned outputs (robustness_*, ci*_, wd_, early_, leak_, breadth_,
final_). oulad_pairwise_comparison.csv and oulad_pairwise_comparison_corrected.csv are never read
(asserted). Every value is re-derived independently from the raw prediction files by
final_3_rederive.py (imports nothing from the project); rederived_ok = True only if every
re-derived component (value, ci_low, ci_high, p_bh, N where applicable) is within 1e-9.

source_row = 1-based physical line number in source_file (comment and header lines counted).
"""
import os
import numpy as np
import pandas as pd

import final_3_rederive as R
from utils import OULAD_RAW_DIR, RESULTS_STATS
from final_common import write_csv

TOL = 1e-9
FORBIDDEN = {"oulad_pairwise_comparison.csv", "oulad_pairwise_comparison_corrected.csv"}
ALLOWED_PREFIX = ("robustness_", "ci1_", "ci2_", "ci3_", "ci4_", "ci5_", "wd_", "early_", "leak_", "breadth_",
                  "final_")
SCH = "stratified_5fold"
LOPO = "leave_one_presentation_out"


def load(name):
    assert name not in FORBIDDEN and name.startswith(ALLOWED_PREFIX), name
    path = os.path.join(RESULTS_STATS, name)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n_comment = 0
    while lines[n_comment].startswith("#"):
        n_comment += 1
    assert not any(l.startswith("#") for l in lines[n_comment:]), f"comment line after header in {name}"
    d = pd.read_csv(path, comment="#")
    d["_line"] = np.arange(len(d)) + n_comment + 2
    return d


def one(d, **flt):
    m = np.ones(len(d), bool)
    for k, v in flt.items():
        m &= (d[k] == v).to_numpy()
    r = d[m]
    assert len(r) == 1, (flt, len(r))
    return r.iloc[0]


def main():
    rows, detail = [], []
    gender = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))[R.K + ["gender"]]

    def add(cid, text, src, fam, scheme, setting, red, src_p=None, N_col="n"):
        """src = (file, row); red = dict of re-derived components."""
        f, r = src
        val = {"value": float(r[red["_value_col"]])}
        if "_ci" in red:
            val["ci_low"], val["ci_high"] = float(r[red["_ci"][0]]), float(r[red["_ci"][1]])
        if src_p is not None:
            pf, pr, pcol = src_p
            val["p_bh"] = float(pr[pcol])
        N = int(r[N_col]) if N_col in r.index else np.nan
        ok = True
        for comp, v in val.items():
            rv = red[comp]
            diff = abs(v - rv)
            ok &= bool(diff <= TOL)
            detail.append({"claim_id": cid, "model_family": fam, "setting": setting, "component": comp,
                           "source_value": v, "rederived_value": rv, "abs_diff": diff, "ok": bool(diff <= TOL)})
        if "N" in red and not np.isnan(N):
            detail.append({"claim_id": cid, "model_family": fam, "setting": setting, "component": "N",
                           "source_value": N, "rederived_value": red["N"], "abs_diff": abs(N - red["N"]),
                           "ok": N == red["N"]})
            ok &= N == red["N"]
        sf = f if src_p is None else f"{f}; {src_p[0]}"
        sr = str(int(r["_line"])) if src_p is None else f"{int(r['_line'])}; {int(src_p[1]['_line'])}"
        rows.append({"claim_id": cid, "claim_text": text, "value": val["value"],
                     "ci_low": val.get("ci_low", np.nan), "ci_high": val.get("ci_high", np.nan),
                     "p_bh": val.get("p_bh", np.nan), "N": N, "model_family": fam, "scheme": scheme,
                     "setting": setting, "source_file": sf, "source_row": sr, "rederived_ok": bool(ok)})

    # C01: full-course D - E accuracy, stratified 5-fold
    boot = load("robustness_bootstrap_d_vs_e.csv")
    src = ["robustness_oof_predictions_stratcv.csv"]
    for fam, pf in [("LR", "robustness_lr_pairwise_mcnemar_bh_keyaligned.csv"),
                    ("GBM", "robustness_gbm_pairwise_mcnemar_bh.csv")]:
        r = one(boot, scheme=SCH, family=fam, model_a="E", model_b="D")
        pdf = load(pf)
        pr = pdf[((pdf.model_a == "D") & (pdf.model_b == "E")) | ((pdf.model_a == "E") & (pdf.model_b == "D"))]
        assert len(pr) == 1
        pr = pr.iloc[0]
        members = [(a, b, {"family": fam, "scheme": SCH}) for a, b in zip(pdf.model_a, pdf.model_b)]
        tgt = list(zip(pdf.model_a, pdf.model_b)).index((pr.model_a, pr.model_b))
        v, lo, hi, n, _ = R.acc_diff(src, "E", "D", family=fam, scheme=SCH)
        add("C01", "Full-course D minus E pooled accuracy (stratified 5-fold); McNemar BH over 21 pairs",
            ("robustness_bootstrap_d_vs_e.csv", r), fam, SCH, "FULL_course_D_E_rowset",
            {"_value_col": "acc_diff_b_minus_a", "_ci": ("acc_diff_ci_low", "acc_diff_ci_high"),
             "value": v, "ci_low": lo, "ci_high": hi, "p_bh": R.family_bh(src, members, tgt), "N": n},
            src_p=(pf, pr, "p_value_bh"), N_col="n_rows")

    # C02: course-only, E_course, D_course accuracy
    ms = load("ci1_stratcv_metrics_summary.csv")
    for model in ["course_only", "E_course", "D_course"]:
        for fam in ["LR", "GBM"]:
            r = one(ms, family=fam, model=model)
            a, n = R.accuracy(["ci1_stratcv_oof_predictions.csv"], model, family=fam)
            add("C02", f"Pooled accuracy of {model} (stratified 5-fold)", ("ci1_stratcv_metrics_summary.csv", r),
                fam, SCH, "FULL_course_D_E_rowset", {"_value_col": "accuracy_pooled", "value": a, "N": n})

    # C03a: dimensionality control E_shufint -> D (interest shuffled within presentation)
    db = load("robustness_dimctrl_bootstrap.csv")
    dm = load("robustness_dimctrl_mcnemar_bh.csv")
    src3 = ["robustness_dimctrl_oof_predictions.csv", "robustness_oof_predictions_stratcv.csv"]
    members = [(a, b, {"family": f, "scheme": SCH}) for a, b, f in zip(dm.model_a, dm.model_b, dm.family)]
    for fam in ["LR", "GBM"]:
        r = one(db, family=fam, model_a="E_shufint", model_b="D")
        pr = one(dm, family=fam, model_a="E_shufint", model_b="D")
        tgt = [i for i, (a, b, fl) in enumerate(members) if (a, b, fl["family"]) == ("E_shufint", "D", fam)][0]
        v, lo, hi, n, _ = R.acc_diff(src3, "E_shufint", "D", family=fam, scheme=SCH)
        add("C03", "Shuffled-interest control: E_shufint (interest block shuffled within presentation) -> D "
            "accuracy gain; BH over 8", ("robustness_dimctrl_bootstrap.csv", r), fam, SCH, "FULL_course_D_E_rowset",
            {"_value_col": "acc_diff_b_minus_a", "_ci": ("acc_diff_ci_low", "acc_diff_ci_high"),
             "value": v, "ci_low": lo, "ci_high": hi, "p_bh": R.family_bh(src3, members, tgt), "N": n},
            src_p=("robustness_dimctrl_mcnemar_bh.csv", pr, "p_value_bh"), N_col="n_rows")
    # C03b: course-centered version, E_course_ctrshuf -> E_course_ctr
    cc = load("ci1_stratcv_comparisons_bh.csv")
    src3b = ["ci1_stratcv_oof_predictions.csv"]
    members = [(a, b, {"family": f, "scheme": SCH}) for a, b, f in zip(cc.model_a, cc.model_b, cc.family)]
    for fam in ["LR", "GBM"]:
        r = one(cc, family=fam, model_a="E_course_ctrshuf", model_b="E_course_ctr")
        tgt = [i for i, (a, b, fl) in enumerate(members)
               if (a, b, fl["family"]) == ("E_course_ctrshuf", "E_course_ctr", fam)][0]
        v, lo, hi, n, _ = R.acc_diff(src3b, "E_course_ctrshuf", "E_course_ctr", family=fam, scheme=SCH)
        add("C03", "Within-course shuffled control, course-centered: E_course_ctrshuf -> E_course_ctr accuracy "
            "gain; BH over 6", ("ci1_stratcv_comparisons_bh.csv", r), fam, SCH, "FULL_course_D_E_rowset",
            {"_value_col": "boot_acc_diff_b_minus_a", "_ci": ("boot_acc_diff_ci_low", "boot_acc_diff_ci_high"),
             "value": v, "ci_low": lo, "ci_high": hi, "p_bh": R.family_bh(src3b, members, tgt), "N": n},
            src_p=("ci1_stratcv_comparisons_bh.csv", r, "p_value_bh"), N_col="n_common")

    # C04: stratified vs LOPO accuracy for E and D
    for f, sch, pf in [("robustness_stratcv_metrics_summary.csv", SCH, "robustness_oof_predictions_stratcv.csv"),
                       ("robustness_lopo_metrics_summary.csv", LOPO, "robustness_oof_predictions_lopo.csv")]:
        d = load(f)
        for model in ["E", "D"]:
            for fam in ["LR", "GBM"]:
                r = one(d, family=fam, model=model)
                a, n = R.accuracy([pf], model, family=fam)
                add("C04", f"Pooled accuracy of {model} ({sch})", (f, r), fam, sch, "FULL_course_D_E_rowset",
                    {"_value_col": "accuracy_pooled", "value": a, "N": n})

    # C05 / C06: EARLY base_course -> full21_ctr and breadth2_ctr -> full21_ctr
    summ = load("leak_breadth_SUMMARY.csv")
    bc = load("breadth_stratcv_comparisons_bh.csv")
    bsrc = ["breadth_stratcv_oof_predictions.csv"]
    for cid, comp, a, b in [("C05", "C5", "base_course", "full21_ctr"), ("C06", "C2", "breadth2_ctr", "full21_ctr")]:
        for t in [28, 56]:
            for ff in ["VLEonly", "withAssess"]:
                setting = f"EARLY_t{t}"
                fam_rows = bc[(bc.setting == setting) & (bc.feature_family == ff)]
                members = [(x, y, {"family": f, "scheme": SCH, "setting": setting, "feature_family": ff})
                           for x, y, f in zip(fam_rows.model_a, fam_rows.model_b, fam_rows.family)]
                for fam in ["LR", "GBM"]:
                    r = one(summ, setting=setting, feature_family=ff, scheme=SCH, comparison_id=comp, model_family=fam)
                    tgt = [i for i, (x, y, fl) in enumerate(members) if (x, y, fl["family"]) == (a, b, fam)][0]
                    v, lo, hi, n, _ = R.acc_diff(bsrc, a, b, family=fam, scheme=SCH, setting=setting,
                                                 feature_family=ff)
                    add(cid, f"EARLY {a} -> {b} accuracy gain ({ff}); BH over 12 within cutoff x feature family",
                        ("leak_breadth_SUMMARY.csv", r), fam, SCH, f"{setting}_{ff}",
                        {"_value_col": "delta_acc", "_ci": ("delta_acc_ci_low", "delta_acc_ci_high"),
                         "value": v, "ci_low": lo, "ci_high": hi, "p_bh": R.family_bh(bsrc, members, tgt), "N": n},
                        src_p=("leak_breadth_SUMMARY.csv", r, "bh_p"), N_col="N")

    # C07: gender FNR gap (F - M) for B, E, D
    fnr = load("robustness_fairness_gender_fnr.csv")
    for fam in ["LR", "GBM"]:
        for model in ["B", "E", "D"]:
            r = one(fnr, family=fam, model=model)
            v, lo, hi, n = R.fnr_gap("robustness_oof_predictions_stratcv.csv", model, fam, gender)
            add("C07", f"Gender FNR gap F minus M, model {model} (stratified 5-fold); student-level bootstrap CI",
                ("robustness_fairness_gender_fnr.csv", r), fam, SCH, "FULL_course_model_rowset",
                {"_value_col": "gap_F_minus_M", "_ci": ("gap_ci_low", "gap_ci_high"), "value": v, "ci_low": lo,
                 "ci_high": hi, "N": n}, N_col="n_rows")

    # C08: pass-rate range across the 22 presentations
    pc = load("ci4_lopo_per_course_d_vs_e.csv")
    pc = pc[pc.family == "LR"].reset_index(drop=True)
    assert len(pc) == 22
    full_red, full_n = R.course_passrates("robustness_oof_predictions_lopo.csv", family="LR", model="D")
    epr = load("early_course_passrates.csv")
    sets = [("FULL_course_D_E_rowset", "ci4_lopo_per_course_d_vs_e.csv", pc, full_red, full_n,
             int(pc["n_rows"].sum()))]
    for t in [28, 56]:
        e = epr[epr["sample"] == f"EARLY_t{t}"].reset_index(drop=True)
        assert len(e) == 22
        red, rn = R.course_passrates("breadth_stratcv_oof_predictions.csv", setting=f"EARLY_t{t}",
                                 feature_family="VLEonly", family="LR", model="base_course")
        sets.append((f"EARLY_t{t}", "early_course_passrates.csv", e, red, rn, int(e["n"].sum())))
    for setting, f, d, red, red_n, ntot in sets:
        assert len(red) == 22
        for kind in ["min", "max"]:
            i = d["pass_rate"].idxmin() if kind == "min" else d["pass_rate"].idxmax()
            r = d.loc[i].copy()
            r["_N"] = ntot
            add("C08", f"{kind} pass rate across the 22 presentations (N = all rows of the sample)", (f, r),
                "n/a", "n/a", setting,
                {"_value_col": "pass_rate", "value": float(getattr(red, kind)()), "N": red_n},
                N_col="_N")

    # C09: LOPO per-course D - E differences
    s9 = load("ci4_lopo_per_course_summary.csv")
    for fam in ["LR", "GBM"]:
        r = one(s9, family=fam)
        diffs, n = R.lopo_course_diffs("robustness_oof_predictions_lopo.csv", "E", "D", fam)
        red = {"n_positive_diff": int((diffs > 0).sum()), "n_zero_diff": int((diffs == 0).sum()),
               "n_negative_diff": int((diffs < 0).sum()), "mean_diff_unweighted": float(diffs.mean()),
               "sd_diff_unweighted_ddof1": float(diffs.std(ddof=1))}
        for col, rv in red.items():
            add("C09", f"LOPO per-course D minus E accuracy difference: {col} (22 presentations)",
                ("ci4_lopo_per_course_summary.csv", r), fam, LOPO, "FULL_course_D_E_rowset",
                {"_value_col": col, "value": rv, "N": n}, N_col="n_rows_total")

    out = pd.DataFrame(rows)
    det = pd.DataFrame(detail)
    write_csv(out, "final_paper_numbers.csv",
              ["Values read only from key-aligned outputs; never from oulad_pairwise_comparison*.csv.",
               "source_row = 1-based physical line in source_file (comment/header lines counted); two files = "
               "value file; p_bh file.",
               "rederived_ok = every re-derived component (value, CI, p_bh, N) within 1e-9 of the source "
               "(final_3_rederive.py, independent of project code). Details: final_paper_numbers_rederivation.csv."])
    write_csv(det, "final_paper_numbers_rederivation.csv",
              ["One row per re-derived component; abs_diff tolerance 1e-9 (N compared exactly)."])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(out.drop(columns=["claim_text"]).to_string(index=False))
    print("mismatches:")
    print(det[~det["ok"]].to_string(index=False))


if __name__ == "__main__":
    main()
