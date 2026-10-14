"""
Paper 2, Step 5 helper (2026-10-14): machine-readable facts for p2_methods_facts.md.
(a) p2_5_software_versions.csv: Python and package versions inside the Docker image.
(b) p2_5_dropped_rows_breakdown.csv: why the FULL (D/E) row set has fewer rows than the target table:
    target rows (cleaned studentInfo) missing from B (no scored assessment), from C/D (no VLE record), both.
"""
import os
import sys
import platform
import importlib
import pandas as pd

from utils import DATA_PROCESSED
from train_evaluate_oulad import KEY, load_target
import p2_common as P


def main():
    vers = [{"package": "python", "version": sys.version.split()[0]}, {"package": "platform", "version": platform.platform()}]
    for m in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "pyarrow"]:
        vers.append({"package": m, "version": importlib.import_module(m).__version__})
    P.write_csv(pd.DataFrame(vers), "p2_5_software_versions.csv", ["Versions inside the Docker image used for the runs."])

    t = load_target()
    info = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_info_clean.parquet"))
    files = {"B": "oulad_features_B_performance.parquet", "C": "oulad_features_C_behavior.parquet",
             "D": "oulad_features_D_interest.parquet"}  # D here = interest table alone (not the fused D)
    sets = {k: pd.read_parquet(os.path.join(DATA_PROCESSED, f))[KEY] for k, f in files.items()}
    m = t[KEY].copy()
    for k, d in sets.items():
        m[f"in_{k}"] = m.merge(d.assign(_x=1), on=KEY, how="left")["_x"].notna().to_numpy()
    m = m.merge(info[KEY + ["final_result"]], on=KEY, how="left")
    m["in_DE"] = m["in_B"] & m["in_C"] & m["in_D"]
    rows = [{"category": "target rows (cleaned studentInfo)", "n": len(m)},
            {"category": "in D/E row set (B, C and interest present)", "n": int(m["in_DE"].sum())},
            {"category": "dropped", "n": int((~m["in_DE"]).sum())},
            {"category": "dropped: no B (no scored assessment) but has VLE", "n": int((~m["in_B"] & m["in_C"]).sum())},
            {"category": "dropped: no VLE record (no C/D) but has B", "n": int((m["in_B"] & ~m["in_C"]).sum())},
            {"category": "dropped: neither B nor VLE", "n": int((~m["in_B"] & ~m["in_C"]).sum())},
            {"category": "check: same rows in C and interest table (1 = yes)", "n": int((m["in_C"] == m["in_D"]).all())}]
    for fr, g in m[~m["in_DE"]].groupby("final_result"):
        rows.append({"category": f"dropped with final_result = {fr}", "n": len(g)})
    P.write_csv(pd.DataFrame(rows), "p2_5_dropped_rows_breakdown.csv",
                ["FULL row set = target inner-joined with B, C and D feature tables (ci_common.load_base)."])
    print(pd.DataFrame(vers).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
