# Course identity / cross-course generalization: literature check

Date: 2026-10-13
Trigger: robustness runs showed most of the D-vs-E gain survives shuffling the interest columns within course presentation, and mostly disappears once course identity is controlled (see results/statistical_tests/ci1_* to ci5_*).

## Scope and limits (read first)
- 11 web searches plus 2 full-text fetches. Most sources were seen at abstract/snippet level only.
- This is NOT a systematic review. Not finding a direct precedent here is NOT evidence that none exists.
- Verification levels: FULL TEXT = opened and read by Claude; ABSTRACT/RECORD = abstract or bibliographic record seen; CITED ONLY = appears in another paper's reference list, not opened.

## Sources

| Source | What it shows (as seen) | Level |
|---|---|---|
| Gasevic, Dawson, Rogers, Gasevic (2016). Learning analytics should not promote one size fits all. Internet and Higher Education 28:68-84. doi:10.1016/j.iheduc.2015.10.002 | Nine blended undergraduate courses (n=4134): predictive power and significant predictors differ between course-specific and generalized models. | ABSTRACT/RECORD |
| Conijn, Snijders, Kleingeld, Matzat (2017). Predicting student performance from LMS data: a comparison of 17 blended courses using Moodle LMS. IEEE TLT 10(1):17-29. doi:10.1109/TLT.2016.2616312 | 17 courses, 4,989 students, one institution: results vary strongly across courses; portability of models is low; LMS data add little when in-between assessment grades are used. (Authors/DOI taken from Lim 2026's reference list; abstract from TU/e record.) | ABSTRACT/RECORD |
| Kleter, Matzat, Conijn (2024). Boundary conditions of generalizing predictive models for academic performance: within cohort versus within course. IEEE TLT 17:2183-2194. doi:10.1109/TLT.2024.3443079 | Model stability better within course; performance degradation after generalization depends on initial performance; no performance difference between within-cohort and within-course. (Citation details from Lim 2026.) | ABSTRACT/RECORD |
| Cui, Chen, Shiri (2020). Scale up predictive models for early detection of at-risk students: a feasibility study. Information and Learning Sciences 121(3/4):97-116. doi:10.1108/ILS-05-2019-0041 | Three large courses at one Canadian university: concluded it is not desirable to build a general model of course failure across variable courses. | ABSTRACT/RECORD |
| Liu, Froissard, Richards, Atif (2015). Validating the effectiveness of the Moodle Engagement Analytics Plugin. AMCIS 2015 | Real differences in predictive power between three courses, attributed to differences in extent/structure of LMS learning activities. | ABSTRACT/RECORD |
| Marras, Vignoud, Kaser (2021). Can feature predictive power generalize? Benchmarking early predictors of student success across flipped and online courses. EDM 2021 | Optimal feature ensembles vary with course setting/structure; predictive performance depends on course activities. | ABSTRACT/RECORD |
| Jovanovic et al. Students matter the most in learning analytics: the effects of internal and instructional conditions in predicting academic success | 15 courses / 50 offerings, mixed-effects models: behavior indicators explain a low proportion of variance; much variability attributed to learners' internal conditions. First author per repository record; venue/year NOT verified. | ABSTRACT/RECORD |
| (Title not captured) Studies in Higher Education 47(12):2370-2391 | Snippet: meta-analytic study of learning-behavior indicators across 50 course offerings (n=2,385); moderate variability within iterations of a course and across courses. Title/authors not captured. | PARTIAL SNIPPET |
| Schwerter et al. (2026). Cross-course generalizability of SRL-aligned predictive models using digital learning traces. Computers & Education 253:105670. doi:10.1016/j.compedu.2026.105670; arXiv 2604.22812 | Models strong within one course did not automatically generalize across courses or institutions; calibration declined between institutions with different at-risk rates (news summary + arXiv snippet). | ABSTRACT/RECORD |
| Lim (2026). Student-disjoint cross-module benchmarking for calibrated early warning in online learning. IJACSA 17(9):617-623. https://thesai.org/Downloads/Volume17No9/Paper_58-Student_Disjoint_Cross_Module_Benchmarking.pdf | OULAD, 32,593 enrollments, 22 presentations. Leave-one-module-out with removal of learners who also appear in the held-out module (2 to 1,772 training rows removed per fold) vs student-disjoint random 5-fold: day-28 AUROC 0.661 vs 0.716 (0.055 optimism gap from context mixing). Logistic regression transferred better than RF/HGB under fixed settings. VLE features include clicks in 14 activity categories. Calibration varied sharply by module. Limitations: seven modules, one institution; calls for replication on an independent institutional dataset. Does NOT decompose a feature-family gain into course-identity vs individual signal. Single-author IJACSA paper; venue standing not assessed. | FULL TEXT |
| Le, Abel, Laforge (2026). When can we trust early warnings? Leakage-excluded early outcome prediction from LMS interaction logs. arXiv:2605.25794 (preprint) | OULAD, temporal leakage under cutoff-based early prediction; assessment information drives most inflation; uses random 80/20 stratified splits; no course-identity analysis. | FULL TEXT |
| Riazy, Simbeck (2019). Predictive algorithms in learning analytics and their fairness. DELFI 2019, pp. 223-228. doi:10.18420/delfi2019_305 | OULAD: models with satisfactory accuracy perform differently across subgroups, especially gender and students with disabilities. Direction/size of gaps not seen. | ABSTRACT/RECORD |
| Toward sufficient statistical power in algorithmic bias assessment: a test for ABROCA. EDM 2025 short paper (arXiv 2501.04683) | Statistical-power and false-positive issues when testing bias metrics; motivates reporting uncertainty for fairness gaps. Authors not captured. | ABSTRACT/RECORD |
| "Modelling student online behaviour in a virtual learning environment" (arXiv 1811.06369) | Open University VLE features built from click counts by week and content type, plus activity flags: activity-type click aggregation is an established feature family. Authors not captured. | SNIPPET (method section) |
| Xu, Zhang, Blake, Stigler (2026). Modelability as a strategy for improving generalizability and scalability of predictive models. JLA 13(1):89-109. doi:10.18608/jla.2026.9099 | Cited by Lim 2026. | CITED ONLY |
| Santos, Henriques (2023). Accurate, timely, and portable: course-agnostic early prediction of student performance from LMS logs. Computers and Education: AI 5:100175 | Cited by Le et al. 2026. | CITED ONLY |
| Esbenshade, Vitale, Baker (2024). Non-overlapping leave future out validation (NOLFO). EDM 2024, pp. 602-609 | Cited by Le et al. 2026. | CITED ONLY |

## Reading (Claude's interpretation, not a source claim)
1. Course/context heterogeneity in LMS-based prediction is well established (2015 onward). A finding that course identity absorbs most of an apparent behavioral gain is consistent with this literature, not surprising within it.
2. On OULAD specifically, Lim (2026) already shows that mixing course contexts inflates results even when students are disjoint, and uses activity-category click features.
3. Gender/disability disparities in OULAD at-risk prediction are already documented in general terms (Riazy and Simbeck 2019).
4. Not found in these searches: a paper that attributes the gain of activity-type click-share features to course identity via within-course shuffle and course-centered controls. Search depth was shallow; do not claim novelty on this basis.
5. Within-course centering / multilevel decomposition is standard technique (Conijn and Jovanovic used multilevel/mixed models), so the method is not novel; any contribution would be empirical and small.

## Open risks in our own pipeline (NOT yet tested)
- Features are full-course aggregates and the target counts Withdrawn as failure, so features can encode when a student left (n_assessments, total clicks, active days). Lim (2026) and Le et al. (2026) both treat temporal availability as a central validity issue. Untested: effect on D-vs-E when excluding Withdrawn or using early windows.
- The "interest" label has no theoretical or empirical support beyond click-share of activity types; it duplicates a long-standing feature family under a new name.
- Many analyses were run after seeing results; no pre-registration.
- statistics_tests_oulad.py row-position bug (public GitHub repo still has the old script and old results).
