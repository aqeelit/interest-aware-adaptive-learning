# Running the Pipeline (Docker)

## 1. Place the downloaded data first
See ../data/README.md for exact paths. Nothing here will run without real data —
the scripts check for required files and exit with a clear message if missing.

## 2. Build and run
From inside this `app/` folder:

```
docker compose build
docker compose run --rm research-app
```

This runs the full pipeline in order:
1. OULAD data cleaning
2. OULAD feature engineering (feature sets A/B/C/D)
3. OULAD train & evaluate (Study 1 — 5-fold CV, logistic regression)
4. OULAD pairwise statistical comparison (McNemar's test)
5. ASSISTments Rasch/1PL validation (Study 2)

Results land in `../results/` and `../data/processed/` on your host machine
(mounted as volumes — nothing is trapped inside the container).

## 3. Run a single stage instead of the full pipeline
```
docker compose run --rm research-app src/data_cleaning_oulad.py
docker compose run --rm research-app src/feature_engineering_oulad.py
docker compose run --rm research-app src/train_evaluate_oulad.py
docker compose run --rm research-app src/statistics_tests_oulad.py
docker compose run --rm research-app src/irt_rasch_assistments.py
```

## 4. Without Docker Compose (plain docker)
```
docker build -t research-agent-app .
docker run --rm -v "%cd%/../data:/workspace/data" -v "%cd%/../results:/workspace/results" research-agent-app
```
(On Windows PowerShell use `${PWD}` instead of `%cd%` if `%cd%` doesn't expand.)

## Reproducibility
Every run appends to `results/tables/experiment_log.csv` with experiment_id,
dataset_version, random_seed, model, parameters, metrics, and timestamp — per
CLAUDE.md. Nothing is overwritten; every run is preserved for audit.
