"""
CI-4 (2026-10-13) — per-course breakdown of D vs E under leave-one-presentation-out.

Source: robustness_oof_predictions_lopo.csv (standard LOPO, robustness round).
Per held-out presentation x family: n rows, n unique students, pass rate
(mean target), accuracy of E and D, D - E. Then: count of positive differences,
mean and SD (ddof=1) over the 22 courses, pooled D - E over all rows, and the
Spearman correlation (scipy.stats.spearmanr) of per-course D - E with pass rate
(n = 22 courses).

Outputs: ci4_lopo_per_course_d_vs_e.csv, ci4_lopo_per_course_summary.csv
"""
import os
import pandas as pd
from scipy.stats import spearmanr

from utils import ensure_dirs, RESULTS_STATS
from train_evaluate_oulad import KEY
from ci_common import write_csv


def main():
    ensure_dirs()
    preds = pd.read_csv(os.path.join(RESULTS_STATS, "robustness_oof_predictions_lopo.csv"))
    preds = preds[preds["model"].isin(["D", "E"])]
    preds["correct"] = (preds["y_pred"] == preds["y_true"]).astype(int)
    rows, summ = [], []
    for fam in ["LR", "GBM"]:
        p = preds[preds["family"] == fam]
        w = p.pivot_table(index=KEY + ["y_true"], columns="model", values="correct").reset_index()
        assert w[["D", "E"]].notna().all().all()
        w["course"] = w["code_module"] + "_" + w["code_presentation"]
        for c, g in w.groupby("course"):
            rows.append({"family": fam, "course": c, "n_rows": len(g), "n_students": g["id_student"].nunique(),
                         "pass_rate": g["y_true"].mean(), "acc_E": g["E"].mean(), "acc_D": g["D"].mean(),
                         "diff_D_minus_E": g["D"].mean() - g["E"].mean()})
        r = pd.DataFrame([x for x in rows if x["family"] == fam])
        rho, pval = spearmanr(r["diff_D_minus_E"], r["pass_rate"])
        summ.append({"family": fam, "n_courses": len(r), "n_rows_total": int(r["n_rows"].sum()),
                     "n_positive_diff": int((r["diff_D_minus_E"] > 0).sum()),
                     "n_zero_diff": int((r["diff_D_minus_E"] == 0).sum()),
                     "n_negative_diff": int((r["diff_D_minus_E"] < 0).sum()),
                     "mean_diff_unweighted": r["diff_D_minus_E"].mean(),
                     "sd_diff_unweighted_ddof1": r["diff_D_minus_E"].std(ddof=1),
                     "pooled_diff_D_minus_E": w["D"].mean() - w["E"].mean(),
                     "spearman_rho_diff_vs_passrate": rho, "spearman_p": pval,
                     "spearman_n": len(r)})
    per = pd.DataFrame(rows)
    s = pd.DataFrame(summ)
    write_csv(per, "ci4_lopo_per_course_d_vs_e.csv", ["Source: robustness_oof_predictions_lopo.csv (standard LOPO)."])
    write_csv(s, "ci4_lopo_per_course_summary.csv", ["Spearman over n = 22 course presentations (spearman_n)."])
    print(per.to_string(index=False))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
