"""
Paper 2, Step 2 (2026-10-14): p2_numbers.csv (traceable numbers) and p2_analysis_inventory.csv.

Values are read from key-aligned outputs only (robustness_*, ci1_, ml_, leak_/breadth_, p2_); the
row-position-bug files oulad_pairwise_comparison*.csv are never read (asserted). Every value is re-derived
independently from raw prediction files by p2_rederive.py (no project imports); rederived_ok = True only if every
re-derived component (value, ci_low, ci_high, p_bh, N) is within 1e-9 (N exact).
source_row = 1-based physical line in source_file (comment and header lines counted).
"""
import glob
import json
import os
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import p2_rederive as R
from utils import OULAD_RAW_DIR, RESULTS_STATS
import p2_common as P

TOL = 1e-9
FORBIDDEN = {"oulad_pairwise_comparison.csv", "oulad_pairwise_comparison_corrected.csv"}
SCH = "stratified_5fold"
rows, detail = [], []


def load(name):
    assert name not in FORBIDDEN, name
    path = os.path.join(RESULTS_STATS, name)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n = 0
    while lines[n].startswith("#"):
        n += 1
    assert not any(l.startswith("#") for l in lines[n:])
    d = pd.read_csv(path, comment="#", low_memory=False)
    d["_line"] = np.arange(len(d)) + n + 2
    d["_file"] = name
    return d


def one(d, **flt):
    m = np.ones(len(d), bool)
    for k, v in flt.items():
        m &= (d[k] == v).to_numpy()
    assert m.sum() == 1, (d["_file"].iloc[0], flt, int(m.sum()))
    return d[m].iloc[0]


def add(cid, text, r, cols, red, fam, scheme, setting, p_row=None, p_col=None, n_col=None):
    """cols: dict component -> source column on row r. red: re-derived components."""
    val = {k: float(r[c]) for k, c in cols.items()}
    if p_row is not None:
        val["p_bh"] = float(p_row[p_col])
    ok = True
    for comp, v in val.items():
        d = abs(v - red[comp])
        good = bool(d <= TOL)
        ok &= good
        detail.append({"claim_id": cid, "setting": setting, "model_family": fam, "component": comp,
                       "source_value": v, "rederived_value": red[comp], "abs_diff": d, "ok": good})
    N = int(r[n_col]) if n_col else int(red["N"])
    if "N" in red:
        good = N == int(red["N"])
        ok &= good
        detail.append({"claim_id": cid, "setting": setting, "model_family": fam, "component": "N",
                       "source_value": N, "rederived_value": red["N"], "abs_diff": abs(N - red["N"]), "ok": good})
    sf, sr = r["_file"], str(int(r["_line"]))
    if p_row is not None and p_row["_file"] != r["_file"] or (p_row is not None and p_row["_line"] != r["_line"]):
        sf, sr = f"{sf}; {p_row['_file']}", f"{sr}; {int(p_row['_line'])}"
    rows.append({"claim_id": cid, "claim_text": text, "value": val["value"], "ci_low": val.get("ci_low", np.nan),
                 "ci_high": val.get("ci_high", np.nan), "p_bh": val.get("p_bh", np.nan), "N": N, "model_family": fam,
                 "scheme": scheme, "setting": setting, "source_file": sf, "source_row": sr, "rederived_ok": bool(ok)})


def comp_claim(cid, text, d, flt, srcs, pflt, members_rows, setting, fam, scheme=SCH, cols=None, pcol="p_value_bh",
               ncol="n_common"):
    """Comparison row with value / CI / BH p; members_rows = the BH family rows (with model_a, model_b, family)."""
    r = one(d, **flt)
    cols = cols or {"value": "delta_acc", "ci_low": "delta_acc_ci_low", "ci_high": "delta_acc_ci_high"}
    red = R.acc_diff(srcs, r["model_a"], r["model_b"], family=fam, **pflt)
    mem = [(a, b, {"family": f, **pflt}) for a, b, f in zip(members_rows["model_a"], members_rows["model_b"],
                                                            members_rows["family"])]
    tgt = [i for i, (a, b, f) in enumerate(mem) if (a, b, f["family"]) == (r["model_a"], r["model_b"], fam)][0]
    red["p_bh"] = R.family_bh(srcs, mem, tgt)
    add(cid, text, r, cols, red, fam, scheme, setting, p_row=r, p_col=pcol, n_col=ncol)
    return red


def numbers():
    gender = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))[R.K + ["gender"]]

    # P01 sample sizes (rows in the analysed sample) — from ml_1b and p2_1 metric summaries
    ms = load("ml_1b_metrics_summary.csv")
    for st in ["FULL", "EARLY_t28", "EARLY_t56"]:
        r = one(ms, setting=st, family="LR", model="base")
        a, n = R.accuracy(["ml_1b_oof_predictions.csv"], "base", setting=st, family="LR")
        add("P01", f"N analysed ({st}); dropped rows in column n_dropped_rows of the source", r,
            {"value": "n_rows_used"}, {"value": n, "N": n}, "n/a", SCH, st, n_col="n_rows_used")

    # P02 FULL accuracies of all arms (T2)
    for f, models, src in [("robustness_stratcv_metrics_summary.csv", ["E", "D"], ["robustness_oof_predictions_stratcv.csv"]),
                           ("ci1_stratcv_metrics_summary.csv", ["course_only", "E_course", "D_course", "E_course_ctr",
                                                                "E_course_ctrshuf"], ["ci1_stratcv_oof_predictions.csv"]),
                           ("robustness_dimctrl_metrics_summary.csv", ["E_noise21", "E_shufint"],
                            ["robustness_dimctrl_oof_predictions.csv"])]:
        d = load(f)
        for m in models:
            for fam in ["LR", "GBM"]:
                r = one(d, family=fam, model=m)
                a, n = R.accuracy(src, m, family=fam)
                add("P02", f"FULL pooled accuracy, arm {m} (stratified 5-fold)", r, {"value": "accuracy_pooled"},
                    {"value": a, "N": n}, fam, SCH, "FULL", n_col="n")

    # P03 FULL D - E (bootstrap file + 21-pair BH family)
    boot = load("robustness_bootstrap_d_vs_e.csv")
    src = ["robustness_oof_predictions_stratcv.csv"]
    for fam, pf in [("LR", "robustness_lr_pairwise_mcnemar_bh_keyaligned.csv"), ("GBM", "robustness_gbm_pairwise_mcnemar_bh.csv")]:
        r = one(boot, scheme=SCH, family=fam, model_a="E", model_b="D")
        pdf = load(pf)
        pr = pdf[((pdf.model_a == "D") & (pdf.model_b == "E")) | ((pdf.model_a == "E") & (pdf.model_b == "D"))].iloc[0]
        mem = [(a, b, {"family": fam, "scheme": SCH}) for a, b in zip(pdf.model_a, pdf.model_b)]
        tgt = list(zip(pdf.model_a, pdf.model_b)).index((pr.model_a, pr.model_b))
        red = R.acc_diff(src, "E", "D", family=fam, scheme=SCH)
        red["p_bh"] = R.family_bh(src, mem, tgt)
        add("P03", "FULL D minus E accuracy (stratified 5-fold); McNemar BH over 21 pairs", r,
            {"value": "acc_diff_b_minus_a", "ci_low": "acc_diff_ci_low", "ci_high": "acc_diff_ci_high"}, red, fam, SCH,
            "FULL", p_row=pr, p_col="p_value_bh", n_col="n_rows")

    # P04 FULL controls: E_shufint -> D, E_noise21 -> D (BH over 8)
    db, dm = load("robustness_dimctrl_bootstrap.csv"), load("robustness_dimctrl_mcnemar_bh.csv")
    src4 = ["robustness_dimctrl_oof_predictions.csv", "robustness_oof_predictions_stratcv.csv"]
    mem = [(a, b, {"family": f, "scheme": SCH}) for a, b, f in zip(dm.model_a, dm.model_b, dm.family)]
    for ctl in ["E_shufint", "E_noise21"]:
        for fam in ["LR", "GBM"]:
            r, pr = one(db, family=fam, model_a=ctl, model_b="D"), one(dm, family=fam, model_a=ctl, model_b="D")
            tgt = [i for i, (a, b, f) in enumerate(mem) if (a, b, f["family"]) == (ctl, "D", fam)][0]
            red = R.acc_diff(src4, ctl, "D", family=fam, scheme=SCH)
            red["p_bh"] = R.family_bh(src4, mem, tgt)
            add("P04", f"FULL control {ctl} -> D accuracy gain; BH over 8", r,
                {"value": "acc_diff_b_minus_a", "ci_low": "acc_diff_ci_low", "ci_high": "acc_diff_ci_high"}, red, fam,
                SCH, "FULL", p_row=pr, p_col="p_value_bh", n_col="n_rows")

    # P05 within/between arms accuracy and P06 comparisons (ml_1b)
    for st in ["FULL", "EARLY_t28", "EARLY_t56"]:
        for m in ["base", "base_course", "W", "M", "WB"]:
            for fam in ["LR", "GBM"]:
                r = one(ms, setting=st, family=fam, model=m)
                a, n = R.accuracy(["ml_1b_oof_predictions.csv"], m, setting=st, family=fam)
                add("P05", f"Within/between arm {m}: pooled accuracy", r, {"value": "accuracy_pooled"},
                    {"value": a, "N": n}, fam, SCH, st, n_col="n")
    c1 = load("ml_1b_comparisons_bh.csv")
    for st in ["FULL", "EARLY_t28", "EARLY_t56"]:
        for fam in ["LR", "GBM"]:
            fr = c1[(c1.setting == st) & (c1.family == fam)]
            for a, b in zip(fr.model_a, fr.model_b):
                comp_claim("P06", f"Within/between {a} -> {b} accuracy difference; BH over 3", c1,
                           {"setting": st, "family": fam, "model_a": a, "model_b": b}, ["ml_1b_oof_predictions.csv"],
                           {"setting": st}, fr, st, fam)

    # P07 primary EARLY base_course -> full21_ctr (C5) and P08 breadth2_ctr -> full21_ctr (C2)
    summ, bc = load("leak_breadth_SUMMARY.csv"), load("breadth_stratcv_comparisons_bh.csv")
    rule = {}
    for cid, comp in [("P07", "C5"), ("P08", "C2")]:
        for t in [28, 56]:
            for ff in ["VLEonly", "withAssess"]:
                st = f"EARLY_t{t}"
                fr = bc[(bc.setting == st) & (bc.feature_family == ff)]
                for fam in ["LR", "GBM"]:
                    r = one(summ, setting=st, feature_family=ff, scheme=SCH, comparison_id=comp, model_family=fam)
                    pf = {"setting": st, "feature_family": ff, "scheme": SCH}
                    red = R.acc_diff(["breadth_stratcv_oof_predictions.csv"], r["model_a"], r["model_b"], family=fam, **pf)
                    mem = [(a, b, {"family": f, **pf}) for a, b, f in zip(fr.model_a, fr.model_b, fr.family)]
                    tgt = [i for i, (a, b, f) in enumerate(mem) if (a, b, f["family"]) == (r["model_a"], r["model_b"], fam)][0]
                    red["p_bh"] = R.family_bh(["breadth_stratcv_oof_predictions.csv"], mem, tgt)
                    add(cid, f"EARLY {r['model_a']} -> {r['model_b']} accuracy difference; BH over 12", r,
                        {"value": "delta_acc", "ci_low": "delta_acc_ci_low", "ci_high": "delta_acc_ci_high"}, red, fam,
                        SCH, f"{st}_{ff}", p_row=r, p_col="bh_p", n_col="N")
                    if cid == "P07":
                        rule[(st, ff, fam)] = red
    # P09 primary rule (verbatim in p2_prereg): CI excludes 0 same direction for LR and GBM AND BH p < 0.05 (both)
    for (st, ff) in sorted({(a, b) for a, b, _ in rule}):
        lr, gb = rule[(st, ff, "LR")], rule[(st, ff, "GBM")]
        d = [int(x["ci_low"] > 0) - int(x["ci_high"] < 0) for x in (lr, gb)]
        met = int(d[0] != 0 and d[0] == d[1] and lr["p_bh"] < 0.05 and gb["p_bh"] < 0.05)
        r = one(summ, setting=st, feature_family=ff, scheme=SCH, comparison_id="C5", model_family="LR")
        s_d = [int(rr["delta_acc_ci_low"] > 0) - int(rr["delta_acc_ci_high"] < 0) for rr in
               [r, one(summ, setting=st, feature_family=ff, scheme=SCH, comparison_id="C5", model_family="GBM")]]
        g = one(summ, setting=st, feature_family=ff, scheme=SCH, comparison_id="C5", model_family="GBM")
        s_met = int(s_d[0] != 0 and s_d[0] == s_d[1] and r["bh_p"] < 0.05 and g["bh_p"] < 0.05)
        r2 = r.copy()
        r2["_rule"] = s_met
        add("P09", "Primary rule met (1/0): base_course -> full21_ctr, CI excludes 0 same direction LR and GBM AND "
            "BH p < 0.05 (both)", r2, {"value": "_rule"}, {"value": met, "N": lr["N"]}, "LR+GBM", SCH,
            f"{st}_{ff}", n_col="N")

    # P10 course-normalised base (p2_1)
    c2 = load("p2_1_zbase_comparisons_bh.csv")
    for st in c2.setting.unique():
        fr = c2[c2.setting == st]
        for fam in ["LR", "GBM"]:
            for a, b in zip(fr[fr.family == fam].model_a, fr[fr.family == fam].model_b):
                comp_claim("P10", f"Course-normalised base: {a} -> {b} accuracy difference; BH over 4", c2,
                           {"setting": st, "family": fam, "model_a": a, "model_b": b}, ["p2_1_zbase_oof_predictions.csv"],
                           {"setting": st}, fr, st, fam)

    # P11 leave-one-module-out (ml_2)
    c3 = load("ml_2_lomo_comparisons_bh.csv")
    for st in c3.setting.unique():
        for fam in ["LR", "GBM"]:
            fr = c3[(c3.setting == st) & (c3.family == fam)]
            comp_claim("P11", "Leave-one-module-out base -> base_ctr accuracy difference; BH over 1", c3,
                       {"setting": st, "family": fam}, ["ml_2_lomo_oof_predictions.csv"], {"setting": st}, fr, st, fam,
                       scheme="leave_one_module_out")

    # P12 per-module replication (ml_3; BH over all 28) and P13 random-effects pooling
    c4 = load("ml_3_module_comparisons_bh.csv")
    src = ["ml_3_module_oof_predictions.csv"]
    mem = [(a, b, {"family": f, "setting": s, "module": m}) for a, b, f, s, m in
           zip(c4.model_a, c4.model_b, c4.family, c4.setting, c4.module)]
    raw_p = [R.mcnemar_p(R.pair(src, a, b, **f)) for a, b, f in mem]
    bh_all = R.bh(raw_p)
    se = {}
    for i, (a, b, f) in enumerate(mem):
        r = one(c4, setting=f["setting"], module=f["module"], family=f["family"])
        red = R.acc_diff(src, a, b, **f)
        red["p_bh"] = float(bh_all[i])
        se[(f["setting"], f["family"], f["module"])] = (red["value"], red["se"])
        add("P12", f"Per-module ({f['module']}) base_course -> base_course_ctr accuracy difference; BH over 28", r,
            {"value": "delta_acc", "ci_low": "delta_acc_ci_low", "ci_high": "delta_acc_ci_high"}, red, f["family"],
            "stratified_5fold_within_module", f"{f['setting']}_{f['module']}", p_row=r, p_col="p_value_bh",
            n_col="n_common")
    re_ = load("ml_3_random_effects.csv")
    for _, r in re_.iterrows():
        mods = sorted(m for (s, f, m) in se if s == r["setting"] and f == r["family"])
        y = [se[(r["setting"], r["family"], m)][0] for m in mods]
        s_ = [se[(r["setting"], r["family"], m)][1] for m in mods]
        p = R.dl_hk(y, s_)
        base = {"N": None}
        for col, comp, extra in [("pooled_dl", "DL pooled delta-Accuracy (95% normal CI)", ("ci_low_dl", "ci_high_dl")),
                                 ("I2_percent", "I2 (%)", None), ("tau2", "tau2", None),
                                 ("pooled_dl", "DL pooled delta-Accuracy with Hartung-Knapp CI (t, df 6)",
                                  ("ci_low_hk", "ci_high_hk"))]:
            cols = {"value": col}
            red = {"value": p["pooled"] if col == "pooled_dl" else p["I2"] if col == "I2_percent" else p["tau2"]}
            if extra:
                cols.update({"ci_low": extra[0], "ci_high": extra[1]})
                red.update({"ci_low": p[extra[0]], "ci_high": p[extra[1]]})
            r2 = r.copy()
            r2["_n"] = len(mods)
            add("P13", f"Random effects over {len(mods)} modules: {comp}", r2, cols, red, r["family"],
                "stratified_5fold_within_module", r["setting"], n_col="_n")

    # P14 gender FNR gap (F - M) for B, E, D
    fnr = load("robustness_fairness_gender_fnr.csv")
    for fam in ["LR", "GBM"]:
        for m in ["B", "E", "D"]:
            r = one(fnr, family=fam, model=m)
            red = R.fnr_gap("robustness_oof_predictions_stratcv.csv", m, fam, gender)
            add("P14", f"Gender FNR gap F minus M, model {m}; student-level bootstrap CI", r,
                {"value": "gap_F_minus_M", "ci_low": "gap_ci_low", "ci_high": "gap_ci_high"}, red, fam, SCH,
                "FULL_model_rowset", n_col="n_rows")


ROUNDS = [(("oulad_", "assistments_", "construct_validity"), "original study", None),
          (("robustness_",), "robustness 2026-10-13", None),
          (("ci1_", "ci2_", "ci3_", "ci4_", "ci5_", "ci_"), "course-identity 2026-10-13", None),
          (("wd_", "early_"), "WD/EARLY 2026-10-13", "prereg_wd_early_2026-10-13.md"),
          (("leak_", "breadth_"), "leak/breadth 2026-10-13", "prereg_leak_breadth_2026-10-13.md"),
          (("final_",), "final 2026-10-14", "prereg_final_2026-10-14.md"),
          (("ml_",), "ml 2026-10-14", "prereg_ml_2026-10-14.md"),
          (("p2_",), "p2 2026-10-14", "p2_prereg_2026-10-14.md")]
RAW_P = ["p_value", "spearman_p", "spearman_p_unadjusted_descriptive"]
COPIES = {"leak_breadth_SUMMARY.csv", "oulad_pairwise_comparison_corrected.csv"}


def inventory():
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_STATS, "*"))):
        name = os.path.basename(path)
        if not name.endswith((".csv", ".json")) or name in ("p2_analysis_inventory.csv", "p2_numbers.csv",
                                                            "p2_numbers_rederivation.csv"):
            continue
        lab, pr = next(((l, p) for pre, l, p in ROUNDS if name.startswith(pre)), ("unassigned", None))
        if name.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                n = len(re.findall(r'"p_value"\s*:', fh.read()))
        else:
            d = pd.read_csv(path, comment="#", low_memory=False)
            pc = next((c for c in RAW_P if c in d.columns), None)
            n = int(d[pc].notna().sum()) if pc else 0
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if pr:
            with open(os.path.join(RESULTS_STATS, pr), encoding="utf-8") as fh:
                ts = datetime.fromisoformat(re.search(r"Written: (\S+?)[, ]", fh.read()).group(1))
            earlier = "yes" if ts < mtime else "no"
            ts_s = ts.isoformat()
        else:
            earlier, ts_s = "no prereg", ""
        out.append({"file": name, "round": lab, "rows_with_p": n, "unique_tests": 0 if name in COPIES else n,
                    "copy_of_tests_in_other_file": name in COPIES, "prereg_file": pr or "", "prereg_timestamp": ts_s,
                    "file_mtime_utc": mtime.isoformat(timespec="seconds"), "prereg_earlier_than_output": earlier})
    inv = pd.DataFrame(out)
    tot = {"rows_with_p": int(inv.rows_with_p.sum()), "unique_tests": int(inv.unique_tests.sum()),
           "unique_tests_prereg_earlier": int(inv.loc[inv.prereg_earlier_than_output == "yes", "unique_tests"].sum())}
    return inv, tot


def main():
    numbers()
    out, det = pd.DataFrame(rows), pd.DataFrame(detail)
    hdr = ["Values only from key-aligned outputs; oulad_pairwise_comparison*.csv never read.",
           "source_row = 1-based physical line in source_file (comment/header lines counted); 'a; b' = value file; "
           "p_bh file.", "rederived_ok = all re-derived components within 1e-9 (N exact); detail in "
           "p2_numbers_rederivation.csv. setting column = analysed setting label."]
    P.write_csv(out, "p2_numbers.csv", hdr)
    P.write_csv(det, "p2_numbers_rederivation.csv", ["One row per re-derived component."])
    inv, tot = inventory()
    P.write_csv(inv, "p2_analysis_inventory.csv",
                ["rows_with_p = rows with a non-missing raw p-value (p_value / spearman_p / "
                 "spearman_p_unadjusted_descriptive); JSON = count of p_value keys.",
                 "unique_tests = 0 for files that only copy tests counted in another file.",
                 f"TOTALS: {json.dumps(tot)}"])
    pd.set_option("display.width", 250)
    print(out.groupby("claim_id")["rederived_ok"].agg(["size", "sum"]).to_string())
    print("MISMATCHES:")
    print(det[~det["ok"]].to_string(index=False))
    print(inv.groupby("round", sort=False)[["rows_with_p", "unique_tests"]].sum().to_string())
    print("TOTALS", tot)


if __name__ == "__main__":
    main()
