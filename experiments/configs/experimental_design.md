# Experimental Design

Status: APPROVED framework (2026-08-30) — parameter values below are defaults pending
Phase 9 data-loading confirmation (e.g., exact OULAD module/presentation subset used).

## Task framing
Reframed from live A/B intervention to **offline predictive validity**: which feature set
best predicts actual recorded learning outcomes? This matches the observational nature of
both datasets and the precedent set by comparable published work (see literature/evidence_table.csv
P02, P04, P15).

## Study 1 — OULAD predictive comparison (primary study)

### Independent variable: feature set (4 levels)
| Model | Features |
|-------|----------|
| A — Random/Majority baseline | No informative features; majority-class or random prediction |
| B — Performance-only | Assessment-score-derived ability proxy (aggregate, not item-level IRT) |
| C — Behavior-only | VLE clickstream aggregates: total clicks, click count per resource type, active days |
| D — Proposed (fused) | B + C + implicit interest features: resource-type click-share distribution, relative time-of-engagement patterns per resource type |

### Dependent variable
- Primary: `final_result`, binarized as Pass/Distinction (1) vs Fail/Withdraw (0)
- Secondary (if data supports): continuous final score

### Evaluation protocol
- Stratified k-fold cross-validation (k=5 default; confirm at implementation time based on
  class balance per module-presentation)
- Metrics: Accuracy, F1 (macro), AUC-ROC
- Statistical comparison across feature sets: paired comparison across folds (e.g.,
  corrected paired t-test or McNemar's test on held-out predictions), not just point-estimate
  comparison — required before claiming any model "wins"
- Model family: kept simple and identical in structure across A-D (e.g., logistic
  regression or gradient boosting) so that differences are attributable to FEATURES, not
  model complexity — this isolates the variable of interest per CLAUDE.md's rule against
  untraceable claims

### Scope decision needed at implementation time
Which OULAD module(s)/presentation(s) to use (single module vs pooled across modules) —
defer to Phase 9 when the raw data is loaded and class balance is inspected.

## Study 2 — ASSISTments IRT validation (secondary/methodological sub-study)

### Purpose
Validate that a 1PL (Rasch) item response model produces sound ability estimates at true
item-level granularity, since OULAD's ability proxy (Study 1) is coarser (assessment-level,
not item-level).

### Method
- Fit 1PL/Rasch model to ASSISTments response matrix (student × item, correct/incorrect)
- Report standard Rasch diagnostics: item fit statistics (infit/outfit), person separation
  reliability, item separation reliability
- This sub-study's ability estimates are NOT merged into Study 1 — the two datasets are
  kept separate. Study 2 exists to methodologically justify the IRT-style reasoning used
  in Study 1's ability proxy, not to supply data into it directly.

## Addendum — approved 2026-08-30, after inspecting first real results

First run (models A/B/C/D only) produced: A acc=0.528, B acc=0.831/AUC=0.901,
C acc=0.786/AUC=0.873, D acc=0.882/AUC=0.941, all pairwise McNemar comparisons
significant (p<0.05). Two concerns were raised before proceeding to Phase 12-13:

1. **H2 not yet isolated.** D = B+C+interest was compared only against B alone and
   C alone, not against a B+C-without-interest control — so D's superiority over B
   and C individually does not by itself show that the INTEREST component adds
   value beyond ordinary fusion. Added:
   - **Model E** = B+C fused, no interest features. Compare D vs E to isolate H2.

2. **Circularity concern with feature set B.** OULAD's `final_result` is partly
   computed from assessment scores in many module-presentations, so predicting it
   from those same scores risks being tautological rather than genuinely
   predictive, inflating B's (and therefore D's) apparent performance. Added a
   sensitivity analysis:
   - **Model B_sens** = performance signal excluding each student's final/exam
     assessment (assessment_type=='Exam' where present, else the assessment with
     the latest date_submitted per course).
   - **Model D_sens** = B_sens + C + interest (fused).
   - **Model E_sens** = B_sens + C (fused, no interest) — the sensitivity-analysis
     counterpart to E, needed to re-check H2 under the sensitivity condition too.

All five original models (A/B/C/D) remain unchanged and their already-obtained
results stand; E/B_sens/D_sens/E_sens are additive, not replacements.

