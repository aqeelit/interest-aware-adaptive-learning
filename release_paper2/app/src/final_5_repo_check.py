"""
Final round, Step 5 (2026-10-14; read-only, nothing is pushed, the patch is NOT applied).

Input: stdin = list of files tracked in the repository at HEAD (git ls-tree -r --name-only HEAD,
produced on the host; HEAD equals the remote main). The project root is mounted read-only at /repo.

(a) final_5_patch_affected_files.csv: tracked files in results/ and app/src that would change or become
    misleading if app/src/statistics_tests_oulad_keyaligned.patch were applied and the pipeline re-run:
    the patched script, the output files it writes (file names found in its source next to to_csv),
    tracked scripts that read those outputs, and the figure files those scripts write from them.
(b) final_5_publication_scan.csv: every file under results/ and app/src (tracked and untracked),
    scanned for: a user_id column / text; an id_student column; column sets of raw OULAD tables or raw
    ASSISTments logs. Data files are read with pandas; other files are scanned as text.
"""
import os
import re
import sys
import pandas as pd

from final_common import write_csv

REPO = "/repo"
RAW_OULAD = {"studentInfo": {"gender", "region", "highest_education", "imd_band", "age_band"},
             "studentVle": {"id_site", "date", "sum_click"},
             "studentAssessment": {"id_assessment", "date_submitted", "is_banked", "score"},
             "studentRegistration": {"date_registration", "date_unregistration"},
             "vle": {"id_site", "activity_type", "week_from", "week_to"},
             "assessments": {"id_assessment", "assessment_type", "weight"},
             "courses": {"module_presentation_length"}}
RAW_ASSIST = {"user_id", "problem_id", "correct"}


def read_text(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main():
    tracked = [l.strip() for l in sys.stdin if l.strip()]
    tr_set = set(tracked)

    # (a)
    patched = "app/src/statistics_tests_oulad.py"
    src = read_text(os.path.join(REPO, patched))
    outs = sorted(set(re.findall(r'"(oulad_[a-z_]+\.csv)"', src)) - {"oulad_fold_predictions.csv"})
    aff = [{"file": patched, "tracked": patched in tr_set,
            "reason": "the patch modifies this script (row_index -> KEY alignment)"}]
    for o in outs:
        f = f"results/statistical_tests/{o}"
        aff.append({"file": f, "tracked": f in tr_set,
                    "reason": f"written by {patched}; values would be recomputed on KEY-aligned pairs"})
    for f in tracked:
        if f.startswith("app/src/") and f.endswith(".py") and f != patched:
            s = read_text(os.path.join(REPO, f))
            for o in outs:
                if o in s:
                    aff.append({"file": f, "tracked": True, "reason": f"reads {o}"})
                    for fig in sorted(set(re.findall(r'"([a-z_]+\.png)"', s))):
                        # report only figures written by a function that reads the output
                        for m in re.finditer(r"def (\w+)\(.*?(?=\ndef |\Z)", s, re.S):
                            body = m.group(0)
                            if fig in body and ("PAIRWISE" in body or o in body):
                                ff = f"results/figures/{fig}"
                                aff.append({"file": ff, "tracked": ff in tr_set,
                                            "reason": f"written by {f}:{m.group(1)}, which reads {o}"})
    affdf = pd.DataFrame(aff).drop_duplicates()
    affdf = affdf[affdf["file"].str.startswith(("results/", "app/src/"))]

    # (b)
    rows = []
    for top in ["results", "app/src"]:
        for dp, _, fns in os.walk(os.path.join(REPO, top)):
            if "__pycache__" in dp:
                continue
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, REPO).replace(os.sep, "/")
                row = {"file": rel, "tracked_at_HEAD": rel in tr_set, "has_user_id": False,
                       "has_id_student_column": False, "n_rows": None, "raw_oulad_table_signature": "",
                       "raw_assistments_signature": False, "scan": ""}
                if fn.endswith(".csv"):
                    try:
                        cols = set(pd.read_csv(p, comment="#", nrows=0).columns)
                        row["n_rows"] = int(sum(1 for _ in open(p, encoding="utf-8", errors="replace")
                                                if not _.startswith("#")) - 1)
                        row["scan"] = "csv header"
                    except Exception as e:  # noqa: BLE001
                        cols, row["scan"] = set(), f"csv unreadable: {e}"
                    row["has_user_id"] = "user_id" in cols
                    row["has_id_student_column"] = "id_student" in cols
                    row["raw_oulad_table_signature"] = ";".join(k for k, v in RAW_OULAD.items() if v <= cols)
                    row["raw_assistments_signature"] = RAW_ASSIST <= cols
                elif fn.endswith((".py", ".json", ".md", ".txt", ".patch", ".yml", ".yaml")):
                    s = read_text(p)
                    row["has_user_id"] = bool(re.search(r"\buser_id\b", s))
                    row["scan"] = "text search for 'user_id' (code/text mentions included)"
                else:
                    row["scan"] = "binary, not scanned"
                rows.append(row)
    scan = pd.DataFrame(rows)
    write_csv(affdf, "final_5_patch_affected_files.csv",
              ["Read-only; the patch was NOT applied. Tracked = present at HEAD (= remote main)."])
    write_csv(scan, "final_5_publication_scan.csv",
              ["has_user_id for .py/.md/.txt etc. = the token 'user_id' appears in the text (e.g. code), not data.",
               "id_student = original OULAD student identifier; raw signatures = column sets of raw tables."])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(affdf.to_string(index=False))
    flagged = scan[scan["has_user_id"] | scan["has_id_student_column"] | (scan["raw_oulad_table_signature"] != "")
                   | scan["raw_assistments_signature"]]
    print(flagged.drop(columns=["scan"]).to_string(index=False))


if __name__ == "__main__":
    main()
