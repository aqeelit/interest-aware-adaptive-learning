"""
Final round, Step 4 (2026-10-14; read-only): inventory of every analysis output file in
results/statistical_tests (all rounds).

n_tests = rows with a non-missing raw p-value. Raw p-value column = the first of
p_value, spearman_p, spearman_p_unadjusted_descriptive present in the file (BH-adjusted and OLD
columns are not counted separately). JSON: number of keys named p_value.
bh_family = bh_family_size values in the file, plus every '#' header line that mentions BH, verbatim;
if neither exists: whether a BH-adjusted column exists, with "family not recorded in file".
Pre-registration: round assigned from the file prefix; prereg timestamp = its 'Written:' line;
prereg_earlier = prereg timestamp < file modification time.
copies_tests_of: files whose p-values are copies of another file's tests (verified by value).
"""
import glob
import json
import os
import re
from datetime import datetime, timezone
import pandas as pd

from utils import RESULTS_STATS
from final_common import write_csv

RAW_P = ["p_value", "spearman_p", "spearman_p_unadjusted_descriptive"]
ROUNDS = [(("oulad_", "assistments_", "construct_validity"), "original study (pre-2026-10-13)", None),
          (("robustness_",), "robustness 2026-10-13", None),
          (("ci1_", "ci2_", "ci3_", "ci4_", "ci5_", "ci_"), "course-identity 2026-10-13", None),
          (("wd_", "early_"), "WD/EARLY 2026-10-13", "prereg_wd_early_2026-10-13.md"),
          (("leak_", "breadth_"), "leak/breadth 2026-10-13", "prereg_leak_breadth_2026-10-13.md"),
          (("final_",), "final 2026-10-14", "prereg_final_2026-10-14.md")]


def prereg_time(name):
    with open(os.path.join(RESULTS_STATS, name), encoding="utf-8") as fh:
        m = re.search(r"Written: (\S+?),? ", fh.read())
    return datetime.fromisoformat(m.group(1).rstrip(","))


def header_lines(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            out.append(line[1:].strip())
    return out


def main():
    rows = []
    files = sorted(f for f in glob.glob(os.path.join(RESULTS_STATS, "*"))
                   if f.endswith((".csv", ".json")) and os.path.basename(f) != "final_analysis_inventory.csv")
    pvals = {}
    for path in files:
        name = os.path.basename(path)
        rnd = [(lab, pr) for pre, lab, pr in ROUNDS if name.startswith(pre)]
        lab, pr = rnd[0] if rnd else ("unassigned", None)
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        row = {"file": name, "round": lab, "file_mtime_utc": mtime.isoformat(timespec="seconds")}
        if name.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                s = fh.read()
            row.update({"p_column": "p_value (JSON keys)", "n_tests": len(re.findall(r'"p_value"\s*:', s)),
                        "bh_family": "no BH adjustment in file", "n_rows": None})
        else:
            d = pd.read_csv(path, comment="#", low_memory=False)
            pc = next((c for c in RAW_P if c in d.columns), None)
            n = int(d[pc].notna().sum()) if pc else 0
            if pc:
                pvals[name] = d[pc].dropna().to_numpy()
            bh = []
            if "bh_family_size" in d.columns:
                bh.append("bh_family_size=" + "/".join(str(int(x)) for x in sorted(d["bh_family_size"].dropna().unique())))
            bh += [h for h in header_lines(path) if "BH" in h]
            bhcols = [c for c in d.columns if "bh" in c.lower() and c not in ("bh_family_size",)]
            if not bh:
                bh = [f"BH column(s) {bhcols} present; family not recorded in file"] if bhcols and n \
                    else (["no BH adjustment in file"] if n else [""])
            row.update({"p_column": pc or "", "n_tests": n, "bh_family": " | ".join(bh), "n_rows": len(d)})
        if pr:
            pt = prereg_time(pr)
            row.update({"prereg_file": pr, "prereg_written": pt.isoformat(),
                        "prereg_earlier": bool(pt < mtime)})
        else:
            row.update({"prereg_file": "none", "prereg_written": "", "prereg_earlier": False})
        rows.append(row)
    inv = pd.DataFrame(rows)

    # copies of tests already counted elsewhere (verified by p-value equality)
    def same(a, b_list):
        b = pd.concat([pd.Series(pvals[x]) for x in b_list]).to_numpy()
        return len(pvals[a]) == len(b) and sorted(pvals[a]) == sorted(b)
    copies = {"leak_breadth_SUMMARY.csv": ["breadth_stratcv_comparisons_bh.csv", "breadth_lopo_comparisons_bh.csv"],
              "oulad_pairwise_comparison_corrected.csv": ["oulad_pairwise_comparison.csv"]}
    inv["copies_tests_of"] = ""
    inv["copy_verified_by_p_values"] = ""
    for a, bl in copies.items():
        i = inv.index[inv["file"] == a]
        if len(i) and a in pvals and all(x in pvals for x in bl):
            inv.loc[i, "copies_tests_of"] = "; ".join(bl)
            inv.loc[i, "copy_verified_by_p_values"] = str(same(a, bl))
    inv["n_tests_unique"] = inv["n_tests"].where(inv["copies_tests_of"] == "", 0)

    tot = {"all_rows_with_p": int(inv["n_tests"].sum()), "unique": int(inv["n_tests_unique"].sum()),
           "unique_in_prereg_rounds": int(inv.loc[inv["prereg_file"] != "none", "n_tests_unique"].sum()),
           "unique_prereg_earlier_than_file": int(inv.loc[inv["prereg_earlier"], "n_tests_unique"].sum())}
    write_csv(inv, "final_analysis_inventory.csv",
              ["n_tests = rows with a non-missing raw p-value (first of p_value / spearman_p / "
               "spearman_p_unadjusted_descriptive); JSON = count of p_value keys.",
               "prereg_earlier = prereg 'Written:' timestamp earlier than the file's modification time.",
               "n_tests_unique = 0 for files that only copy tests counted in another file.",
               f"TOTALS: {json.dumps(tot)}"])
    by_round = inv.groupby("round", sort=False).agg(n_files=("file", "size"), n_tests=("n_tests", "sum"),
                                                     n_tests_unique=("n_tests_unique", "sum")).reset_index()
    write_csv(by_round, "final_analysis_inventory_by_round.csv", [f"TOTALS: {json.dumps(tot)}"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(inv[["file", "round", "n_tests", "n_tests_unique", "prereg_earlier", "copies_tests_of"]].to_string(index=False))
    print(by_round.to_string(index=False))
    print("TOTALS", tot)


if __name__ == "__main__":
    main()
