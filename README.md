# Does Implicit Learner Interest Improve Outcome Prediction Beyond Performance and Behavior?

Code and results for a research study asking whether an implicit learner-interest
signal (resource-type click-share and engagement-diversity entropy) adds
predictive value over a performance-and-behavior baseline when predicting a real,
recorded academic outcome — using the Open University Learning Analytics Dataset
(OULAD) as the primary predictive study and ASSISTments as an independent
item-level IRT validation sub-study.

Repository: https://github.com/aqeelit/interest-aware-adaptive-learning

This repository contains the full data-processing and statistical-analysis
pipeline and the resulting summary/aggregate outputs (tables, figures,
statistics) — code and results only. **It does not contain any raw or
per-student data**, and, ahead of journal decision, does not include the
manuscript draft, internal review notes, or other pre-publication research
planning documents. See [Data availability](#data-availability) below.

## Repository structure

```
app/                 Dockerized Python pipeline (source of truth for all numbers)
  src/               Pipeline scripts — see app/README.md for the full run order
  Dockerfile, docker-compose.yml, requirements.txt

data/
  README.md          Where to download OULAD/ASSISTments and place them locally
  raw/, processed/   NOT included in this repository (see below) — created
                     locally when you run the pipeline against your own copy
                     of the downloaded datasets

results/
  tables/            experiment_log.csv — append-only, reproducible run log
  statistical_tests/ Aggregate statistical outputs (McNemar comparisons, FDR
                     correction, odds ratios, Rasch calibration, construct-
                     validity checks, subgroup fairness breakdown)
  figures/           Publication-quality figures (300 DPI PNG)
```

## Reproducing the results

### 1. Obtain the data (not included in this repository)

Both datasets are third-party research datasets released under CC-BY 4.0 and
must be downloaded directly from their original sources — see
[`data/README.md`](data/README.md) for exact download links and the file paths
the pipeline expects.

### 2. Run the pipeline via Docker

Once the raw data is in place, see [`app/README.md`](app/README.md) for exact
commands (`docker compose build` then `docker compose run --rm research-app`,
or individual pipeline stages). Every reported number traces to a specific
script and output file — see `results/tables/experiment_log.csv` for the full
reproducibility log (experiment ID, dataset version, random seed, parameters,
metrics, timestamp for every run).

## Data availability

- **OULAD** (Open University Learning Analytics Dataset): Kuzilek, J., Hlosta,
  M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
  *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171 —
  released under CC-BY 4.0, available at
  https://analyse.kmi.open.ac.uk/open_dataset
- **ASSISTments** (2009 skill-builder release): released under CC-BY 4.0,
  available at
  https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data

This repository redistributes neither dataset. `results/` contains only
aggregate/summary statistics (model-level metrics, pairwise test statistics,
Rasch item/person parameter estimates, subgroup-level fairness metrics). The
one file that is naturally one-row-per-student
(`assistments_person_ability.csv`, from the Rasch calibration) is published
only in a de-identified form
(`assistments_person_ability_deidentified.csv`), with ASSISTments' real
`user_id` replaced by a sequential anonymous ID — the original, keyed by real
`user_id`, is excluded from this repository and is regenerated locally only
when you run the pipeline yourself against your own downloaded copy of
ASSISTments.

## License

Code in this repository is released under the MIT License — see
[`LICENSE`](LICENSE). This license covers the code only; it does not grant any
rights to the OULAD or ASSISTments datasets themselves, which remain governed
by their own CC-BY 4.0 terms from their original publishers.

## AI-assistance disclosure

Generative AI (Claude, Anthropic) was used substantially in this project's data
pipeline development and statistical analysis. A complete, dated log of
AI-assisted actions taken during this project is maintained privately and is
available on request; a summary is included in the associated manuscript's
AI-use disclosure statement.
