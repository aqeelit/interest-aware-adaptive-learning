"""
Paper 2, Step 6 (2026-10-14): anonymised reproducibility folder release_paper2/ (project root mounted at /repo).

  python src/p2_6_release.py build     copy scripts + aggregate results, write README / LICENSE / DATA_NOTE
  (host)  git apply --directory=release_paper2 app/src/statistics_tests_oulad_keyaligned.patch
  python src/p2_6_release.py finalize  identity scan (release folder + manuscript-facing p2_ files, incl. file
                                       metadata), fix hits ONLY inside release_paper2/, re-scan, MANIFEST.csv, zip

Nothing outside release_paper2/ is modified except the new scan reports p2_6_identity_scan*.csv.
"""
import csv
import hashlib
import io
import os
import re
import shutil
import struct
import sys
import zipfile
import zlib
import pandas as pd

REPO = "/repo"
REL = os.path.join(REPO, "release_paper2")
STATS = os.path.join(REPO, "results", "statistical_tests")
SRC = os.path.join(REPO, "app", "src")
PATTERNS = ["aqeel", "hussein", "khafaji", "adptivio", "aqeelit", "hilyoon", "al-hussein", "karbala", "@",
            "c:\\", "/users/", "/home/"]
NAME_RX = re.compile(r"(?i)(dr\.?\s+)?aqeel(\s+mohammed)?(\s+ali)?(\s+al-hussein|\s+hussein)?|al-hussein|hussein|"
                     r"khafaji|adptivio|aqeelit|hilyoon|karbala")
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PATH_RX = re.compile(r"(?i)(c:\\[^\s\"',;)]*|/users/[^\s\"',;)]*|/home/[^\s\"',;)]*)")

SCRIPT_PREFIX = ("robustness_", "ci_", "wd_", "early_", "leak_", "breadth_", "final_", "ml_", "p2_", "run_")
SCRIPT_EXTRA = ["utils.py", "data_cleaning_oulad.py", "feature_engineering_oulad.py", "train_evaluate_oulad.py",
                "statistics_tests_oulad.py", "wdearly_common.py", "leakbreadth_common.py"]
SCRIPT_EXCLUDE = {"run_pipeline.py", "p2_6_release.py"}  # pipeline also runs ASSISTments; packaging tool holds the scan patterns
RESULT_PREFIX = ("robustness_", "ci1_", "ci2_", "ci3_", "ci4_", "ci5_", "wd_", "early_", "leak_", "breadth_",
                 "final_", "ml_", "p2_", "prereg_")
RESULT_EXT = (".csv", ".json", ".md", ".png", ".svg", ".bib", ".txt")

README = """# Reproducibility package: click-based activity features beyond course context (OULAD)

This folder contains the analysis code and aggregate results for the paper
"How Much Do Click-Based Activity Features Add Beyond Course Context? A Within/Between
Decomposition and Early-Prediction Audit on OULAD". Author identities are removed for double-blind review.

## Contents
- `app/` Docker set-up (`Dockerfile`, `docker-compose.yml`, `requirements.txt`) and `app/src/` analysis scripts.
  `app/src/statistics_tests_oulad.py` has the key-aligned pairing fix applied (pairs matched on
  id_student, code_module, code_presentation instead of positional row index).
- `results/statistical_tests/` aggregate outputs only (metrics, comparisons, tables, figures, pre-registration
  files, verified references). No per-student prediction files and no student identifiers are included.
- `DATA_NOTE.md` how to obtain the OULAD data (not redistributed here).
- `MANIFEST.csv` SHA-256 and size of every file in this folder (except MANIFEST.csv itself).

## How to reproduce
1. Download OULAD (see `DATA_NOTE.md`) and place the seven CSV files in `data/raw/oulad/`.
2. From the folder root: `docker compose -f app/docker-compose.yml build`
3. Base pipeline (cleaning, features, base models):
   `docker compose -f app/docker-compose.yml run --rm research-app src/data_cleaning_oulad.py`, then
   `src/feature_engineering_oulad.py`, `src/train_evaluate_oulad.py`, `src/statistics_tests_oulad.py`.
4. Analysis rounds, in order: `src/run_robustness.py`, `src/run_course_identity.py`, `src/run_wd_early.py`,
   `src/run_leak_breadth.py`, the `src/final_*.py` scripts (1, 2, 3, 4, 5), `src/run_ml.py`, and the
   `src/p2_*.py` scripts (1, 5, 4, 2, 3).
   Every script uses seed 42. Output files refuse to overwrite existing files, so start from an empty
   `results/statistical_tests/` to regenerate everything.

All analyses are exploratory and post hoc; see the `prereg_*.md` files for what was fixed before each round.
In this package, names in the pre-registration files are replaced by [ANONYMIZED], so their SHA-256 values
differ from the values quoted inside them.
"""
LICENSE = """MIT License

Copyright (c) 2026 [ANONYMIZED]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
DATA_NOTE = """# Data note

This package uses the Open University Learning Analytics Dataset (OULAD):
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

OULAD is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).
The raw data are NOT included in this package.

How to obtain: download the dataset from the Open University Learning Analytics Dataset page
(https://analyse.kmi.open.ac.uk/open_dataset) or from the UCI Machine Learning Repository
("Open University Learning Analytics dataset"), and place the seven CSV files (assessments.csv, courses.csv,
studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv, vle.csv) in
`data/raw/oulad/`. The scripts read them from there (mounted at /workspace/data inside the container).
"""
DOCKERFILE = """FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Data and results are mounted as volumes at runtime.
ENTRYPOINT ["python"]
"""
COMPOSE = """services:
  research-app:
    build: .
    volumes:
      - ../data:/workspace/data
      - ../results:/workspace/results
"""


def has_id_column(path):
    try:
        cols = pd.read_csv(path, comment="#", nrows=0).columns
    except Exception:  # noqa: BLE001
        return False
    return bool({"id_student", "user_id", "userid_DI", "row_index"} & set(cols))


def build():
    assert not os.path.exists(REL), "release_paper2/ already exists"
    os.makedirs(os.path.join(REL, "app", "src"))
    os.makedirs(os.path.join(REL, "results", "statistical_tests"))
    log = []
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py") and fn not in SCRIPT_EXCLUDE and (fn.startswith(SCRIPT_PREFIX) or fn in SCRIPT_EXTRA):
            shutil.copy2(os.path.join(SRC, fn), os.path.join(REL, "app", "src", fn))
            log.append(("script", fn, "copied"))
        elif fn.endswith(".py"):
            log.append(("script", fn, "excluded (not needed for these rounds)"))
    shutil.copy2(os.path.join(REPO, "app", "requirements.txt"), os.path.join(REL, "app", "requirements.txt"))
    for name, text in [("Dockerfile", DOCKERFILE), ("docker-compose.yml", COMPOSE)]:
        with open(os.path.join(REL, "app", name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    for fn in sorted(os.listdir(STATS)):
        p = os.path.join(STATS, fn)
        reason = None
        if not fn.startswith(RESULT_PREFIX) or not fn.endswith(RESULT_EXT):
            reason = "excluded (not part of these rounds / type)"
        elif "oof_predictions" in fn or "console" in fn:
            reason = "excluded (per-student predictions or console log)"
        elif fn.endswith(".txt") and not fn.startswith("p2_T"):
            reason = "excluded (text log)"
        elif fn.endswith(".csv") and has_id_column(p):
            reason = "excluded (student identifier / row-level column)"
        if reason:
            log.append(("result", fn, reason))
            continue
        shutil.copy2(p, os.path.join(REL, "results", "statistical_tests", fn))
        log.append(("result", fn, "copied"))
    for name, text in [("README.md", README), ("LICENSE", LICENSE), ("DATA_NOTE.md", DATA_NOTE)]:
        with open(os.path.join(REL, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    df = pd.DataFrame(log, columns=["kind", "file", "action"])
    with open(os.path.join(STATS, "p2_6_release_file_selection.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("# Paper 2 Step 6: which scripts / result files were copied into release_paper2/ and why others not.\n")
        df.to_csv(f, index=False)
    print(df["action"].value_counts().to_string())


def png_text(b):
    out, pos = [], 8
    while pos + 8 <= len(b):
        n, typ = struct.unpack(">I4s", b[pos:pos + 8])
        data = b[pos + 8:pos + 8 + n]
        if typ in (b"tEXt", b"iTXt"):
            out.append(data.replace(b"\x00", b"=").decode("latin-1", "replace"))
        elif typ == b"zTXt":
            k, rest = data.split(b"\x00", 1)
            out.append(k.decode("latin-1") + "=" + zlib.decompress(rest[1:]).decode("latin-1", "replace"))
        pos += 12 + n
    return out


def metadata_text(path):
    low = path.lower()
    if low.endswith(".png"):
        with open(path, "rb") as f:
            return png_text(f.read())
    if low.endswith(".svg"):
        with open(path, encoding="utf-8", errors="replace") as f:
            s = f.read()
        return re.findall(r"<metadata>.*?</metadata>", s, re.S) + re.findall(r"<!--.*?-->", s, re.S)
    if low.endswith(".docx"):
        with zipfile.ZipFile(path) as z:
            return [z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.startswith("docProps/")]
    if low.endswith(".pdf"):
        with open(path, "rb") as f:
            b = f.read()
        return [m.decode("latin-1") for m in re.findall(rb"/(?:Author|Creator|Producer|Title|Subject)\s*\([^)]*\)", b)]
    return []


def scan(paths, scope):
    hits = []
    for p in paths:
        rel = os.path.relpath(p, REPO).replace(os.sep, "/")
        texts = []
        if p.lower().endswith((".png", ".pdf", ".docx", ".zip")):
            texts = [("metadata", t) for t in metadata_text(p)]
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                texts = [("content", f.read())] + [("metadata", t) for t in metadata_text(p)]
        for where, t in texts:
            for ln, line in enumerate(t.splitlines(), 1):
                low = line.lower()
                for pat in PATTERNS:
                    if pat in low:
                        kind = ("email" if pat == "@" and EMAIL_RX.search(line) else
                                "at-sign (non-email)" if pat == "@" else
                                "path" if pat in ("c:\\", "/users/", "/home/") else "name")
                        hits.append({"scope": scope, "file": rel, "where": where, "line": ln, "pattern": pat,
                                     "kind": kind, "context": line.strip()[:160]})
    return hits


def all_files(root):
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.join(dp, f) for f in sorted(fns)]
    return sorted(out)


def fix_file(p):
    if p.lower().endswith((".png", ".zip")):
        return False
    with open(p, encoding="utf-8", errors="strict") as f:
        s = f.read()
    t = NAME_RX.sub("[ANONYMIZED]", s)
    t = EMAIL_RX.sub("[ANONYMIZED-EMAIL]", t)
    t = PATH_RX.sub("[LOCAL-PATH]", t)
    if t != s:
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(t)
        return True
    return False


def write_scan(df, name, note):
    path = os.path.join(STATS, name)
    assert not os.path.exists(path), path
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {note}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)")


def finalize():
    rel_files = all_files(REL)
    p2_files = [os.path.join(STATS, f) for f in sorted(os.listdir(STATS)) if f.startswith("p2_")
                and not f.startswith("p2_6_identity_scan")]
    before = pd.DataFrame(scan(rel_files, "release_paper2") + scan(p2_files, "manuscript-facing p2_ (not modified)"),
                          columns=["scope", "file", "where", "line", "pattern", "kind", "context"])
    write_scan(before, "p2_6_identity_scan_before_fix.csv",
               "Identity scan BEFORE fixes (case-insensitive patterns incl. '@' and local paths; file metadata).")
    fixed = []
    for p in rel_files:
        hit_kinds = set(before[(before.scope == "release_paper2") &
                               (before.file == os.path.relpath(p, REPO).replace(os.sep, "/"))]["kind"])
        if hit_kinds & {"name", "email", "path"} and fix_file(p):
            fixed.append(os.path.relpath(p, REPO).replace(os.sep, "/"))
    after = pd.DataFrame(scan(all_files(REL), "release_paper2"),
                         columns=["scope", "file", "where", "line", "pattern", "kind", "context"])
    after["fixed_files"] = ""
    write_scan(after, "p2_6_identity_scan_after_fix.csv",
               "Identity scan of release_paper2/ AFTER fixes. Files fixed: " + "; ".join(fixed))
    rows = []
    for p in all_files(REL):
        if os.path.basename(p) == "MANIFEST.csv":
            continue
        with open(p, "rb") as f:
            b = f.read()
        rows.append({"file": os.path.relpath(p, REL).replace(os.sep, "/"), "sha256": hashlib.sha256(b).hexdigest(),
                     "bytes": len(b)})
    with open(os.path.join(REL, "MANIFEST.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256", "bytes"])
        w.writeheader()
        w.writerows(rows)
    zp = os.path.join(REPO, "release_paper2.zip")
    assert not os.path.exists(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in all_files(REL):
            z.write(p, os.path.join("release_paper2", os.path.relpath(p, REL)))
    print(f"fixed files: {fixed}")
    print(f"manifest rows: {len(rows)}; zip bytes: {os.path.getsize(zp)}")
    print(before.groupby(["scope", "kind"]).size().to_string())
    print("after:", after.groupby("kind").size().to_string() if len(after) else "no hits")


if __name__ == "__main__":
    {"build": build, "finalize": finalize}[sys.argv[1]]()
