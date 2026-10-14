"""
Leak audit, Step 1a-1d (2026-10-13). Pre-registration: prereg_leak_breadth_2026-10-13.md.

a. Sanity: rebuild EARLY t=28/56 with the existing code path and re-run all models of the
   earlier EARLY round (stratified and LOPO); compare to early_stratcv_oof_predictions.csv and
   early_lopo_oof_predictions.csv. Exit code 2 (stops run_leak_breadth.py) on any mismatch.
b. Provenance: per builder call, per student, max raw timestamp actually passed in; violations
   (> t) per group. Risk-set rule, missing date_unregistration, exclusions, label composition.
c. Independent re-implementation (leak_reimpl_early_features.py) vs pipeline, column by column.
d. Decode: pooled Accuracy/F1/AUC for base / +interest / base_course / (+interest)_course, and a
   search of earlier early_* result files for values that round to 0.9615.
Verdict per feature family -> leak_audit_verdict.csv.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import ensure_dirs, RESULTS_STATS, OULAD_RAW_DIR
from train_evaluate_oulad import KEY
from robustness_common import stratified_folds, course_folds, metrics
from leakbreadth_common import build_early, write_csv, CUTOFFS, FAMILIES
import leak_reimpl_early_features as R

FAMS = {"VLEonly": ("C", "C_t{t}", "CI_t{t}"), "withAssess": ("BC", "E_t{t}", "D_t{t}")}


def base_cols(s, code):
    return s["C"] if code == "C" else s["B"] + s["C"]


def compare_preds(new, old, labels):
    m = new.merge(old, on=KEY, suffixes=("_new", "_old"))
    same_order = len(new) == len(old) and (new[KEY].to_numpy() == old[KEY].to_numpy()).all()
    row = dict(labels)
    row.update({"n_new": len(new), "n_old": len(old), "n_matched": len(m), "row_order_identical": bool(same_order),
                "fold_identical": int((m["fold_new"].astype(str) == m["fold_old"].astype(str)).sum()),
                "y_pred_identical": int((m["y_pred_new"] == m["y_pred_old"]).sum()),
                "max_abs_prob_diff": float((m["y_prob_new"] - m["y_prob_old"]).abs().max())})
    row["match"] = bool(len(new) == len(old) == len(m) and same_order and row["fold_identical"] == len(m)
                        and row["y_pred_identical"] == len(m) and row["max_abs_prob_diff"] < 1e-9)
    return row


def main():
    ensure_dirs()
    data = W.load_clean()
    old_s = pd.read_csv(os.path.join(RESULTS_STATS, "early_stratcv_oof_predictions.csv"), comment="#")
    old_l = pd.read_csv(os.path.join(RESULTS_STATS, "early_lopo_oof_predictions.csv"), comment="#")

    sanity, decode, prov, riskrows, diffs, struct = [], [], [], [], [], []
    raw = R.load_raw()
    types = R.activity_type_labels(raw)
    for t in CUTOFFS:
        s_plain = W.build_sample("EARLY", cutoff=t, data=data)
        s, calls = build_early(t, data, instrument=True)
        assert s["frame"].drop(columns=s["breadth"]).equals(s_plain["frame"]), "instrumented build differs"
        f = s["frame"]
        y = f["target"].to_numpy()

        # ---- a + d: re-run the earlier EARLY models
        for fam_label, (code, bn, pn) in FAMS.items():
            base, plus = bn.format(t=t), pn.format(t=t)
            setting = f"EARLY_t{t}_{fam_label}"
            variants, _ = W.design(s_plain, base_cols(s_plain, code), base, plus)
            folds = stratified_folds(y)
            for name, X in variants.items():
                for fam in FAMILIES:
                    p, _ = W.run_cv(f, X, y, folds, fam, name, "stratified_5fold")
                    o = old_s[(old_s["setting"] == setting) & (old_s["family"] == fam) & (old_s["model"] == name)]
                    sanity.append(compare_preds(p, o, {"setting": setting, "scheme": "stratified_5fold",
                                                       "family": fam, "model": name}))
                    if name in (base, plus, f"{base}_course", f"{plus}_course"):
                        decode.append({"setting": setting, "cutoff_days": t, "feature_family": fam_label,
                                       "scheme": "stratified_5fold", "family": fam, "model": name, "n": len(p),
                                       **{f"{k}_pooled": v for k, v in
                                          metrics(p["y_true"], p["y_pred"], p["y_prob"]).items()}})
            lv, _ = W.lopo_design(s_plain, base_cols(s_plain, code), base)
            lf = course_folds(f)
            for name, X in lv.items():
                for fam in FAMILIES:
                    p, _ = W.run_cv(f, X, y, lf, fam, name, "leave_one_presentation_out")
                    o = old_l[(old_l["setting"] == setting) & (old_l["family"] == fam) & (old_l["model"] == name)]
                    sanity.append(compare_preds(p, o, {"setting": setting, "scheme": "leave_one_presentation_out",
                                                       "family": fam, "model": name}))
            print(f"sanity done t={t} {fam_label}", flush=True)

        # ---- b: provenance
        interest_calls = [c for c in calls if c["group"] == "I_interest"]
        assert len(interest_calls) == 2 and sum(c["group"] == "B_assessment" for c in calls) == 1 \
            and sum(c["group"] == "C_behaviour" for c in calls) == 1
        for i, c in enumerate(calls):
            role = "values"
            if c is interest_calls[0]:
                role = "column_labels_only (full_interest_columns: output used only for .columns)"
            mx = c["per_student_max"]
            prov.append({"cutoff_days": t, "call_seq": i, "builder": c["builder"], "feature_group": c["group"],
                         "role": role, "time_col": c["time_col"], "n_input_rows": c["n_input_rows"],
                         "n_students_with_records": int(len(mx)),
                         "max_timestamp_over_students": float(mx.max()) if len(mx) else np.nan,
                         "n_students_max_gt_t": int((mx > t).sum()),
                         "provenance_violations": int((mx > t).sum()) if role == "values" else np.nan})
        info = data[0]
        reg = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentRegistration.csv"))
        tg = info[KEY + ["final_result"]].merge(reg[KEY + ["date_unregistration"]], on=KEY, how="left")
        excl = tg["date_unregistration"] <= t
        kept = tg[~excl]
        comp = kept["final_result"].value_counts()
        riskrows.append({"cutoff_days": t,
                         "rule": "exclude if date_unregistration is non-missing and <= t (pandas: NaN <= t is False)",
                         "n_studentinfo_rows": len(tg),
                         "n_missing_date_unregistration": int(tg["date_unregistration"].isna().sum()),
                         "n_missing_date_unregistration_final_withdrawn":
                             int((tg["date_unregistration"].isna() & (tg["final_result"] == "Withdrawn")).sum()),
                         "n_excluded": int(excl.sum()), "n_risk_set": len(kept),
                         "n_risk_set_pipeline_frame": len(f),
                         **{f"n_{k}": int(comp.get(k, 0)) for k in ["Pass", "Distinction", "Fail", "Withdrawn"]}})

        # ---- c: independent re-implementation
        ri = R.reimplement(t, raw, types)
        same_rows = len(ri) == len(f)
        same_order = same_rows and (ri[KEY].to_numpy() == f[KEY].to_numpy()).all()
        cols = ["target"] + s["B"] + s["C"] + s["I"] + s["breadth"]
        struct.append({"cutoff_days": t, "n_rows_pipeline": len(f), "n_rows_reimpl": len(ri),
                       "same_row_count": bool(same_rows), "same_row_order": bool(same_order),
                       "n_keys_only_in_pipeline": len(f.merge(ri[KEY], on=KEY, how="left", indicator=True)
                                                      .query("_merge == 'left_only'")),
                       "n_keys_only_in_reimpl": len(ri.merge(f[KEY], on=KEY, how="left", indicator=True)
                                                    .query("_merge == 'left_only'")),
                       "pipeline_interest_columns_equal_reimpl": sorted(s["I"]) == sorted(
                           [c for c in ri.columns if c.startswith("click_share_")] + ["resource_diversity_entropy"])})
        m = f[KEY + cols].merge(ri[KEY + [c for c in cols if c in ri.columns]], on=KEY, suffixes=("_p", "_r"))
        for c in cols:
            group = ("B_assessment" if c in s["B"] else "C_behaviour" if c in s["C"] else
                     "I_interest" if c in s["I"] else "breadth" if c in s["breadth"] else "target")
            if f"{c}_r" not in m:
                diffs.append({"cutoff_days": t, "column": c, "feature_group": group, "present_in_reimpl": False})
                continue
            a, b = m[f"{c}_p"].astype(float).to_numpy(), m[f"{c}_r"].astype(float).to_numpy()
            nan_mis = int((np.isnan(a) != np.isnan(b)).sum())
            both = ~np.isnan(a) & ~np.isnan(b)
            d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            diffs.append({"cutoff_days": t, "column": c, "feature_group": group, "present_in_reimpl": True,
                          "n_compared": int(len(m)), "max_abs_diff": d, "n_nan_mismatch": nan_mis,
                          "n_rows_diff_gt_1e-9": int((np.abs(a[both] - b[both]) > 1e-9).sum()),
                          "match_le_1e-9": bool(d <= 1e-9 and nan_mis == 0)})

    sanity_df = pd.DataFrame(sanity)
    write_csv(sanity_df, "leak_1a_sanity_rebuild_vs_early_outputs.csv",
              ["match = same rows/order, folds, identical predicted classes, max |prob diff| < 1e-9."])
    if not sanity_df["match"].all():
        print("SANITY CHECK FAILED - stopping.", flush=True)
        print(sanity_df[~sanity_df["match"]].to_string(index=False), flush=True)
        sys.exit(2)
    print("SANITY CHECK PASSED", flush=True)

    prov_df, risk_df = pd.DataFrame(prov), pd.DataFrame(riskrows)
    diff_df, struct_df = pd.DataFrame(diffs), pd.DataFrame(struct)
    write_csv(prov_df, "leak_1b_provenance.csv",
              ["Per builder call: per-student max of the raw timestamp column actually passed in; violations = "
               "students with max > t (counted for value-producing calls only)."])
    write_csv(risk_df, "leak_1b_riskset.csv")
    write_csv(struct_df, "leak_1c_reimpl_rows.csv", ["Independent re-implementation: leak_reimpl_early_features.py."])
    write_csv(diff_df, "leak_1c_reimpl_column_diffs.csv", ["Independent re-implementation: leak_reimpl_early_features.py."])

    # Decode: earlier result cells that round to 0.9615
    hits = []
    for path in sorted(glob.glob(os.path.join(RESULTS_STATS, "early_*.csv"))):
        if "oof_predictions" in path:
            continue
        df = pd.read_csv(path, comment="#")
        for col in df.select_dtypes("number").columns:
            for i in np.flatnonzero(np.round(df[col].to_numpy(float), 4) == 0.9615):
                r = df.iloc[i]
                hits.append({"file": os.path.basename(path), "row": int(i), "column": col, "value": float(r[col]),
                             "setting": r.get("setting"), "family": r.get("family"),
                             "model": r.get("model", np.nan), "model_a": r.get("model_a", np.nan),
                             "model_b": r.get("model_b", np.nan), "fold": r.get("fold", np.nan)})
    write_csv(pd.DataFrame(decode), "leak_1d_decode_metrics.csv")
    write_csv(pd.DataFrame(hits), "leak_1d_decode_09615_locations.csv",
              ["All numeric cells in earlier early_*.csv result files (prediction files excluded) that round to 0.9615."])

    verdict = []
    for fam_label, groups in {"VLEonly": ["C_behaviour", "I_interest", "breadth"],
                              "withAssess": ["B_assessment", "C_behaviour", "I_interest", "breadth"]}.items():
        pv = prov_df[(prov_df["role"] == "values") & prov_df["feature_group"].isin(groups)]
        n_viol = int(pv["provenance_violations"].sum())
        dd = diff_df[diff_df["feature_group"].isin(groups + ["target"])]
        cols_ok = bool(dd["present_in_reimpl"].all() and dd["match_le_1e-9"].fillna(False).all())
        rows_ok = bool(struct_df["same_row_count"].all() and struct_df["same_row_order"].all())
        verdict.append({"feature_family": fam_label, "cutoffs": "28;56", "provenance_violations": n_viol,
                        "reimpl_columns_match": cols_ok, "reimpl_rows_and_order_match": rows_ok,
                        "max_abs_diff_any_column": float(dd["max_abs_diff"].max()),
                        "audit": "PASS" if (n_viol == 0 and cols_ok and rows_ok) else "FAIL"})
    vdf = pd.DataFrame(verdict)
    write_csv(vdf, "leak_audit_verdict.csv", ["Pre-registered PASS definition (prereg_leak_breadth_2026-10-13.md)."])
    print(vdf.to_string(index=False), flush=True)
    print(prov_df.to_string(index=False), flush=True)
    print(risk_df.to_string(index=False), flush=True)
    print(struct_df.to_string(index=False), flush=True)
    print(diff_df.to_string(index=False), flush=True)
    print(pd.DataFrame(hits).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
