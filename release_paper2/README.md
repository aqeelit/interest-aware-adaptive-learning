# Reproducibility package: click-based activity features beyond course context (OULAD)

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
