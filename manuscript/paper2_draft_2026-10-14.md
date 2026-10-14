# DRAFT: course-context evaluation paper (paper 2)

Status (2026-10-14): Sections 1-3 drafted. Abstract and Sections 4-7 pending.
Rules for this file:
- Numbers appear only if they were read from a verified output file (final_paper_numbers.csv or the round files named in Appendix A). Anything waiting on the ml_ round is in [square brackets].
- Sources are cited only at the level of verification recorded in Appendix B. "Cited only" sources must be opened before submission.
- No author identifiers anywhere in this file (double-blind readiness). The code link is a placeholder.
- Venue is NOT chosen. Reference style, AI-disclosure wording, blinding and format are venue-specific and must be applied after the choice.

Working title:
How Much Do Click-Based Activity Features Add Beyond Course Context? A Within/Between Decomposition and Early-Prediction Audit on OULAD

---

## Abstract
[pending results]

---

## 1. Introduction

Predictive models that flag students at risk of failing or withdrawing are a standard application of learning analytics and educational data mining. Many are built from clickstream logs of the virtual learning environment (VLE), and a common design breaks clicks down by the type of resource or activity a student used, such as forum, quiz or course content, in addition to total activity (Lim, 2026; [earlier Open University work using click counts by week and content type, arXiv:1811.06369, authors to verify]). The Open University Learning Analytics Dataset (OULAD; Kuzilek et al., 2017) is the most widely reused public benchmark for this purpose. It links daily VLE interaction counts by activity type to assessment records and final results for 32,593 enrolments in 22 module presentations.

A recurring finding in learning analytics is that such models depend on the course context in which they were built. Comparisons of course-specific and generalised models (Gašević et al., 2016), an analysis of 17 blended courses in one institution (Conijn et al., 2017), and a feasibility study of general models of course failure (Cui et al., 2020) all report that predictive power or informative predictors differ between courses, or that one general model is not desirable. Kleter et al. (2024) compare generalisation within a cohort with generalisation within a course. On OULAD itself, Lim (2026) reports that evaluating on a held-out module, rather than on randomly partitioned student-disjoint folds, lowers day-28 AUROC from 0.716 to 0.661, and attributes the difference to contextual optimism in validation that mixes courses.

This matters for how the benefit of a new feature family should be judged. Under a random split that mixes courses, a feature block can raise accuracy for two different reasons. It can carry information about how a student differs from classmates in the same course (a within-course signal), or it can tell the model which course the student is in, and therefore which base rates and feature meanings apply (a between-course, contextual signal). The two have different implications for transfer: only the first is plausibly portable to a new course. Multilevel and mixed-effects regression have been used to separate course-level from student-level variation in LMS-based prediction (Conijn et al., 2017; Jovanović et al., [venue to verify]). In the literature we searched, we did not find a study that applies this distinction to the gain from adding activity-type click features in a classification setting on OULAD. That search was brief and not systematic, so we make no novelty claim on this basis.

This paper asks four questions.

- RQ1. How much of the apparent gain from activity-type click features is attributable to course context rather than to within-course differences between students?
- RQ2. When prediction uses only information available at an early cutoff (day 28 and day 56), does a within-course signal remain, and does it exceed what total engagement and breadth of use already provide?
- RQ3. Which activity types carry that signal, and is the pattern consistent across modules and presentations?
- RQ4. Does the choice of feature family change gender differences in false-negative rates?

We compare logistic regression and histogram gradient boosting under stratified cross-validation, leave-one-presentation-out validation and leave-one-module-out validation. Explicit controls are course identity, within-course centering, within-course shuffling of the click-share block, random noise columns of equal dimension, and a breadth-of-engagement baseline. Early-prediction settings use cutoff-first truncation and a risk set of students still registered at the cutoff. All comparisons are paired at the student level, uncertainty is quantified by a student-level bootstrap, and multiplicity is handled within declared families.

The contributions are the following.

1. A control-based protocol that separates course-level from student-level contributions of a feature block, released with open code.
2. Evidence from OULAD. Under stratified validation, most of the apparent full-course gain of activity-type features does not survive controls for course context. In early prediction, a small within-course signal remains for click-only features and is fragile once assessment information is included. [Add ml_ results: within/between decomposition, leave-one-module-out, per-module replication.]
3. A fairness check showing that the gender gap in false-negative rate is present for every feature family examined, so it cannot be attributed to any one feature block.
4. A transparent account of analytic flexibility: the study is exploratory, and we report the total number of tests run across all analysis rounds and which rounds had a written plan before running.

We do not claim a new method, a causal effect, or any link between the features and motivation or interest; they are behavioural traces. Results come from one institution's dataset and are hypothesis-generating.

Section 2 reviews related work, Section 3 describes data and methods, Sections 4 and 5 report and discuss results, Section 6 states limitations, and Section 7 concludes.

---

## 2. Related Work

### 2.1 Clickstream features and outcome prediction on OULAD

OULAD (Kuzilek et al., 2017) links demographics, registration, assessments, and daily VLE interactions for seven anonymised modules. Within-dataset prediction with deep and ensemble models has been reported to be strong (Waheed et al., 2020; Al-Zawqari et al., 2022, as characterised by Lim, 2026). Lim (2026) builds VLE features from total, recent and first-week clicks, active days, unique resources and clicks in 14 activity categories. In that study a VLE-only specification preserved discrimination relative to adding submission features while improving precision-recall and calibration. Le et al. (2026, preprint) benchmark cutoff-based early prediction on OULAD under a leakage-excluded protocol (cutoff-first truncation, with provenance checks). They report best ROC-AUC rising from 0.7151 at day 7 to 0.8602 at day 56, and a shift in feature importance from engagement volume early on to assessment scores later. Their leaky variants show that post-cutoff assessment information can inflate apparent early performance. Both studies treat the availability of each feature at prediction time as a validity condition, which we adopt in the early-prediction settings.

### 2.2 Course context and cross-course generalisation

The dependence of LMS-based models on course context has been documented repeatedly. Liu et al. (2015) found real differences in the predictive power of an engagement-analytics plugin between three courses and linked them to differences in the extent and structure of the LMS activities captured. Gašević et al. (2016) studied nine blended courses (n = 4,134) and found differences in predictive power and significant predictors between course-specific and generalised models. Conijn et al. (2017) analysed 17 blended courses with 4,989 students in one institution and reported strongly varying results, low portability, and little additional value of LMS data when in-between assessment grades were available. Cui et al. (2020) concluded that building a general model of course failure across variable courses was not desirable. Marras et al. (2021) found that the best feature sets differ with course setting and structure and depend on course activities. Kleter et al. (2024) found better model stability within a course than across courses or cohorts, with no difference in performance between within-cohort and within-course generalisation. A study of 15 courses and 50 offerings with mixed-effects models found that behaviour indicators explained a low proportion of variance, with much variability attributed to learners' internal conditions (Jovanović et al., [venue to verify]). Schwerter et al. (2026) report that models strong within one course did not automatically generalise across courses or institutions, and that calibration declined between institutions with different at-risk rates.

Most of these studies compare whole models or whole predictor sets across courses. The multilevel analyses (Conijn et al., 2017; Jovanović et al.) decompose variance for regression outcomes. We apply the same within/between logic to a specific feature block in classification, on the benchmark dataset most often used for click-based prediction.

### 2.3 Validation design and leakage

Lim (2026) separates both modules and learners between training and test (removing from training every learner who also appears in the held-out module) and compares this with student-disjoint random folds. The held-out-module design lowered day-28 AUROC by 0.055 and increased calibration error more than fourfold, and calibration varied sharply across modules. The author notes that repeated learners can carry student-level regularities across a nominal train-test boundary. Leakage and distribution mismatch have been argued to inflate claims in machine-learning-based science more generally (Kapoor & Narayanan, 2023, as cited by Lim, 2026), and time-aware validation for educational prediction has been proposed (Esbenshade et al., 2024, as cited by Le et al., 2026). We use cutoff-first truncation, student-disjoint variants and held-out presentation and module validation, and report each against stratified random folds.

### 2.4 Fairness of at-risk prediction

Riazy and Simbeck (2019) showed on OULAD that at-risk models with satisfactory accuracy perform differently across student subgroups, especially by gender and for students with disabilities. Calibration and fairness in early dropout prediction have also been examined (Karimi-Haghighi et al., 2021, as listed by Lim, 2026). Recent work on bias metrics stresses statistical power and false-positive risks when conclusions rest on finite samples (an EDM 2025 short paper on ABROCA; [authors to verify]), which motivates reporting uncertainty for group gaps. We report the female-minus-male false-negative-rate gap with student-level bootstrap intervals, and ask whether it changes with the feature family.

### 2.5 Positioning

The study is an evaluation rather than a new predictor. It re-examines a common feature family, activity-type click shares, under controls drawn from the context-generalisation literature, early-prediction settings drawn from the leakage literature, and an uncertainty-aware fairness check. Its contribution is empirical and limited to the dataset studied.

---

## 3. Data and Methods

### 3.1 Data

We use OULAD (Kuzilek et al., 2017), released under CC BY 4.0. It contains 32,593 enrolments in 22 module presentations of seven modules. A module presentation (code_module plus code_presentation) is the unit we call a course. We use seven tables: studentInfo, studentRegistration, studentVle, vle, studentAssessment, assessments, courses. The data are anonymised and public, and we contacted no participants. A student can enrol in more than one presentation; in our main row set, 2,345 of 23,326 students do.

The outcome is binary: success (final result Pass or Distinction) versus non-success (Fail or Withdrawn). In one sensitivity setting, Withdrawn students are excluded.

Cleaning removed records with missing final result, duplicate student-module-presentation entries, VLE records with no resolvable activity type, negative click counts, and assessment scores outside 0-100.

### 3.2 Analysis settings

| Setting | Features computed from | Sample | N | Success rate |
|---|---|---|---|---|
| FULL | whole-course records | enrolments retained by the inner join of the three feature tables | 25,770 | [0.597, derived from 10,390 non-successes; confirm from file] |
| WD | whole-course records | FULL with Withdrawn excluded | 21,117 | 0.7283 |
| EARLY t28 | records up to day 28 | enrolments still registered at day 28 | 27,538 | 15,385 of 27,538 |
| EARLY t56 | records up to day 56 | enrolments still registered at day 56 | 26,522 | 0.5801 |

Success rates by presentation range from 0.469 to 0.764 in FULL, from 0.443 to 0.751 at t28 and from 0.469 to 0.762 at t56. The feature join drops 6,823 of the 32,593 enrolments in FULL [state the reason after checking the code; an inner join drops enrolments missing from any source table].

### 3.3 Features

Three blocks are built per student.

- Performance block (B), from assessment records: mean score, weight-adjusted mean score, number of scored assessments, and standard deviation of scores.
- Behaviour block (C), from the VLE: total clicks, number of distinct active days, and mean clicks per student-resource-day record (named mean_clicks_per_active_day in the code) [verify definition in both FULL and EARLY code; the FULL code averages over records, not over active days].
- Activity-type block (I), 21 columns: for each of the 20 activity types, the share of the student's clicks on that type, plus the entropy of the share distribution, H = -sum_k s_k ln s_k with 0 ln 0 = 0. Shares are 0 for a student with no clicks. Some columns are constant in some settings (folder and repeatactivity are never used at day 28), so the effective dimension is smaller than 21.

Feature families: E = B + C; D = E + I; in VLE-only settings the base is C and the extended set is C + I. We call block I "activity-type click shares". We do not call it interest, because nothing in the data tests a motivational interpretation.

Breadth baseline: two columns, the entropy and the number of activity types with at least one click. A secondary variant adds the number of distinct resources used.

### 3.4 Early-prediction protocol

For cutoff t in {28, 56} days, studentVle (date <= t, including negative pre-course days) and studentAssessment (date_submitted <= t) are truncated before any join, aggregation, share or entropy computation. The risk set excludes students whose date_unregistration is present and <= t; a missing date_unregistration means the student did not unregister, and students who withdraw after t remain in. Students with no clicks by t get zero click-shares and zero behaviour counts; students with no scored submission by t get a score standard deviation of 0 and missing mean scores (imputed with the median in logistic regression, handled natively by gradient boosting). OULAD does not record grade release time, so scores are treated as observable at submission; this is an explicit assumption.

Two families are analysed per cutoff: VLE-only (no assessment information) and with-assessments (B_early added). Provenance was audited: the maximum timestamp of every raw record feeding each feature group was <= t for every student (zero violations), and an independent from-scratch implementation of all 31 compared feature columns matched the pipeline within 1e-9 (maximum absolute difference 7.1e-15).

### 3.5 Controls for course context

- Course identity: one-hot encoding of the 22 presentations.
- Course centering: each column of block I minus the mean of that column over the students of the same presentation in the analysed sample. This is unsupervised and uses test-fold features but no labels. In leave-one-out validation, the held-out course is centered using only its own students.
- Course means: each student receives the presentation mean of the same columns (planned decomposition arm).
- Within-course shuffling: the whole block I (centered or raw) is permuted across students within the same presentation, which preserves its joint distribution within course but removes its link to the individual student.
- Noise columns: 21 independent standard normal columns, to match dimensionality.

### 3.6 Models and validation

Logistic regression (median imputation, standardisation, L2 penalty, max_iter 1000) and histogram gradient boosting (scikit-learn HistGradientBoostingClassifier, library defaults, untuned). Random seed 42 throughout. Validation schemes:

1. Stratified 5-fold cross-validation.
2. Leave-one-presentation-out (22 held-out presentations; no course one-hot).
3. Student-disjoint variants: GroupKFold by student (5 folds), and leave-one-presentation-out with removal from training of every student enrolled in the held-out presentation (5,100 training rows removed in total across the 22 folds).
4. Leave-one-module-out (7 folds) with the same removal [ml_ round].

Metrics are pooled out-of-fold accuracy, macro-F1 and AUC.

### 3.7 Statistical comparison and uncertainty

Models are compared pairwise on the same students (matched by student, module and presentation). We use McNemar's test on correctness [state exact-versus-asymptotic rule and continuity correction after checking robustness_common.py], the odds ratio of discordant counts with a 95% Wald interval, and a student-level cluster bootstrap (1,000 resamples of students) for differences in pooled accuracy and AUC. Benjamini-Hochberg correction is applied within each declared family (setting x model family x comparison set), and each file reports its family. All inference is exploratory.

### 3.8 Within/between decomposition and replication across modules [ml_ round]

- ICC(1) for the outcome and for each column of block I, across the 22 presentations and across the 7 modules.
- Arms per setting: base, base + presentation one-hot, W (base + course-centered block I, no course identity), M (base + course means), WB (both). Comparisons: W vs base, M vs base, WB vs M.
- Leave-one-module-out comparison of base vs base + course-centered block I.
- Per-module replication in each of the 7 modules, with random-effects pooling of module-level accuracy differences (DerSimonian-Laird and Hartung-Knapp with 6 degrees of freedom, using bootstrap standard errors), tau-squared and I-squared.
- Reading rules fixed before running (prereg_ml_2026-10-14.md): [copy verbatim from that file once written].

### 3.9 Fairness analysis

The false-negative rate is the proportion of non-successful students predicted to succeed. We report the female-minus-male gap for models B, E and D under both model families, with student-level bootstrap intervals, and bootstrap intervals for differences in gaps between models. OULAD records gender as a binary attribute, which we use as given.

### 3.10 Analysis transparency and reproducibility

The analyses were run in successive rounds on the same dataset. Across all rounds to date there are 538 unique hypothesis tests (32 original study, 92 robustness, 48 course-identity, 66 early/without-Withdrawn, 88 leak/breadth, 212 attribution and no-click sensitivity) [add the ml_ round]. Rounds after the first had a written plan with a timestamp and hash before running, but because the dataset had been analysed before, every round is exploratory and post hoc. An earlier version of our pairing script matched predictions by row position; every comparison reported here is key-aligned, and every reported number was re-derived independently from raw prediction files (58 claims, 194 components, all within 1e-9). Code and result files: [anonymised repository link placeholder; the personal repository name is identifying and must not appear in a blinded submission].

### 3.11 Venue-specific disclosures

[AI-assistance disclosure, data and code availability statement, funding and competing interests: apply the chosen venue's template.]

---

## 4. Results [pending]
Sources per subsection (see Appendix A):
- 4.1 Full-course gain and its decomposition: C01-C04, ci1_*, robustness_*, [ml_ Step 1].
- 4.2 Validation design: C04, ci2_*, ci3_*, [ml_ Step 2].
- 4.3 Early prediction: C05, C06, early_*, leak_*, breadth_*.
- 4.4 Activity-type attribution: final_1_* files (only one of the units was flagged under the declared rule in leave-one-out at t56; confirm all flag columns).
- 4.5 Heterogeneity and replication: C09, ci4_*, [ml_ Step 3].
- 4.6 Gender false-negative gaps: C07.

## 5. Discussion [pending]
## 6. Limitations [pending]
Must include at least: one dataset and institution; post hoc and exploratory analyses with 538+ tests; features are behavioural proxies (no construct validation); full-course settings can encode when a student left; the score-observability assumption in EARLY; binary gender; no causal claim; modest absolute effect sizes; heterogeneity across presentations.
## 7. Conclusion [pending]

---

## Appendix A: claim-to-evidence map (for the author, remove before submission)

| ID | Claim | Source files |
|---|---|---|
| C01 | Full-course D minus E accuracy gain | robustness_bootstrap_d_vs_e.csv; robustness_*_pairwise_mcnemar_bh*.csv |
| C02 | Course-only, E_course, D_course accuracy | ci1_stratcv_metrics_summary.csv |
| C03 | Shuffled-interest controls | robustness_dimctrl_*.csv; ci1_stratcv_comparisons_bh.csv |
| C04 | Stratified vs leave-one-presentation-out accuracy | robustness_stratcv_metrics_summary.csv; robustness_lopo_metrics_summary.csv |
| C05 | Early: base_course vs base_course + centered block I | leak_breadth_SUMMARY.csv |
| C06 | Early: breadth2 vs full block I | leak_breadth_SUMMARY.csv |
| C07 | Gender FNR gaps (B, E, D) | robustness_fairness_gender_fnr.csv |
| C08 | Success-rate range across presentations | ci4_lopo_per_course_d_vs_e.csv; early_course_passrates.csv |
| C09 | Per-course D minus E under leave-one-presentation-out | ci4_lopo_per_course_summary.csv |
All of the above are collected, with independent re-derivation, in results/statistical_tests/final_paper_numbers.csv.

## Appendix B: reference verification status (remove before submission)

FULL TEXT read: Lim (2026) IJACSA 17(9):617-623; Le, Abel and Laforge (2026) arXiv:2605.25794 (preprint).
ABSTRACT/RECORD seen: Kuzilek et al. (2017) doi:10.1038/sdata.2017.171; Gašević et al. (2016) doi:10.1016/j.iheduc.2015.10.002; Conijn et al. (2017) doi:10.1109/TLT.2016.2616312 (authors and DOI via Lim's list); Kleter, Matzat and Conijn (2024) doi:10.1109/TLT.2024.3443079 (via Lim's list); Cui, Chen and Shiri (2020) doi:10.1108/ILS-05-2019-0041; Liu, Froissard, Richards and Atif (2015) AMCIS; Marras, Vignoud and Käser (2021) EDM; Schwerter et al. (2026) doi:10.1016/j.compedu.2026.105670 (via Lim's list), arXiv:2604.22812; Riazy and Simbeck (2019) doi:10.18420/delfi2019_305; Jovanović et al. (venue and year not verified).
CITED ONLY (not opened; open before use): Waheed et al. (2020) doi:10.1016/j.chb.2019.106189; Al-Zawqari et al. (2022) doi:10.1016/j.caeai.2022.100103; Kapoor and Narayanan (2023) doi:10.1016/j.patter.2023.100804; Esbenshade, Vitale and Baker (2024) EDM pp. 602-609; Karimi-Haghighi et al. (2021) LAK21 companion proceedings pp. 294-304.
INCOMPLETE (authors or title not captured): arXiv:1811.06369; EDM 2025 ABROCA power paper (arXiv 2501.04683); a meta-analytic study in Studies in Higher Education 47(12):2370-2391.

## Appendix C: open verification items before submission
1. Open every CITED ONLY and INCOMPLETE source and complete its record.
2. Confirm McNemar implementation (exact threshold, continuity correction) in robustness_common.py.
3. Confirm exact column lists for B, C, B_early, C_early; confirm the definition behind mean_clicks_per_active_day.
4. State the reason for the 6,823 rows dropped by the feature join.
5. Replace the personal repository link with an anonymised one for review.
6. Apply venue template: style, AI disclosure, blinding, word limit.
7. Insert ml_ results, then fill Abstract and Sections 4-7.
