# Data

## Where to put downloaded data (exact paths expected by app/src/*.py)

### OULAD (Open University Learning Analytics Dataset)
Download from: https://analyse.kmi.open.ac.uk/open_dataset
Extract the CSVs directly into:

    data/raw/oulad/studentInfo.csv
    data/raw/oulad/studentVle.csv
    data/raw/oulad/vle.csv
    data/raw/oulad/studentAssessment.csv
    data/raw/oulad/assessments.csv
    data/raw/oulad/courses.csv
    data/raw/oulad/studentRegistration.csv

### ASSISTments (skill-builder release, 2009 or 2017)
Download from: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data
(or the 2017 competition release, same schema requirements)
Place the file at:

    data/raw/assistments/skill_builder_data.csv

Required columns at minimum: `user_id`, `problem_id`, `correct`.
If your downloaded file has different column names, rename them to match before
running the pipeline, or edit app/src/irt_rasch_assistments.py's `build_response_matrix`.

## raw/
Original, untouched datasets exactly as obtained. Never edit files in this folder directly.

## processed/
Cleaned/feature-engineered data produced by scripts in app/src/. Every file here is
reproducible by re-running the pipeline against raw/ — see app/README.md.

## Provenance log
| File | Source | Date obtained | Processing script | Notes |
|------|--------|---------------|--------------------|-------|
| | | | | |
