# Reviewer #1 (Adversarial Simulation)

Status: COMPLETE (2026-08-31). Hostile Q1-journal review focused on methodology,
statistics, novelty, and reproducibility. Manuscript was NOT modified in producing
this review, per protocol -- these are findings for the owner to triage.

## Summary recommendation: Major Revision

The core ablation design (Model D vs. E) is the paper's strongest asset and is
executed more carefully than most papers in this literature (real sensitivity
analysis, real McNemar tests, disclosed AI use). However, several issues below must
be addressed before this would be acceptable at a Q1 venue.

## Major issues

**1. No correction for multiple comparisons.** The pairwise statistical comparison
protocol (Section 3.6) runs McNemar's test across many model pairs (the underlying
results file has 28 pairwise comparisons across 8 models) at an uncorrected
alpha=0.05. The manuscript's central claim (D vs. E, Section 4.1) is one comparison
among many that were run; with 28 tests at alpha=0.05, roughly 1-2 false positives
are expected by chance alone even under a fully null scenario. The paper should
report a Bonferroni- or Benjamini-Hochberg-corrected significance threshold, or at
minimum justify why the central D-vs-E comparison was pre-specified rather than
selected post hoc from the full comparison matrix (the manuscript's own Section 3.6
admits this comparison was added "after inspecting initial results," which is the
opposite of pre-specification and needs to be addressed directly, not just
disclosed).

**2. Effect sizes are not reported alongside p-values.** With OULAD's sample size,
extremely small p-values (e.g., 5.92e-57) are expected even for modest,
practically limited differences. The manuscript reports accuracy differences of
roughly 2-3 percentage points (D=0.882 vs. E=0.856) as its "central result" without
an effect-size statistic (e.g., odds ratio, Cohen's g, or a confidence interval on
the accuracy difference). A Q1 reviewer will ask: is a 2.6-point accuracy gain
practically meaningful for an adaptive system, independent of its p-value? As
written, statistical significance is doing rhetorical work that effect size should
be doing.

**3. Single model family, no robustness check.** All eight feature-set conditions
use only logistic regression (Section 3.5). This is a reasonable design choice for
isolating feature effects, but the paper never tests whether the D-vs-E finding
survives a different model class (e.g., gradient boosting, which is standard in this
literature per the paper's own Related Work). If the interest signal's advantage is
an artifact of logistic regression's inductive bias (e.g., its sensitivity to
additional linearly-informative features) rather than a genuine property of the
data, this would not be caught by the current design.

**4. No held-out or cross-institution validation.** All results come from 5-fold
cross-validation within the same pooled OULAD sample. There is no test on a
temporally or institutionally held-out cohort. Given that OULAD spans multiple
modules and presentations, a stronger design would validate on module-presentations
held out entirely from training, not just row-level folds -- the current design may
leak module-specific patterns across folds.

**5. No hyperparameter tuning reported.** Section 3.5 states default-adjacent
logistic regression settings (L2, max_iter=1000) with no grid search or
regularization-strength tuning per feature set. Differential susceptibility to
under/overfitting across feature sets of different dimensionality (B and C are
low-dimensional; D adds a variable number of resource-type click-share features)
is not ruled out as a confound.

**6. Interest-signal dimensionality is never stated.** Section 3.4 describes the
interest signal as "click-share distribution across VLE resource types" but never
states how many resource types (i.e., how many additional features) this adds. This
matters directly for interpreting whether D's improvement over E is a real signal
or a capacity effect from added dimensions -- especially relevant given issue #3
above.

**7. "IRT-BTI-style backbone" terminology inflates apparent novelty.** The
Introduction and Discussion repeatedly invoke "IRT-BTI-style" framing for the
performance-and-behavior fusion, but Study 1's actual performance signal (Model B)
is an assessment-level aggregate, not an IRT-derived ability estimate -- true IRT
calibration is confined to the entirely separate Study 2 sub-study on ASSISTments,
which never feeds into Study 1's numbers. Branding Study 1's fusion as
"IRT-BTI-style" when no IRT parameter is actually used in it risks readers
overestimating the sophistication of the performance signal that produced the
paper's headline numbers.

**8. No code/data availability statement.** The manuscript discloses a
Docker-based reproducible pipeline in Section 3.8 but never states, anywhere,
whether the code itself (not just the logged parameters) will be made available
(e.g., a GitHub repository, Zenodo archive). Most Q1 journals in this space now
require an explicit Data/Code Availability statement; its absence will likely
generate an editorial query before review even begins.

## Minor issues

- No confidence intervals or standard deviations across the 5 folds are reported
  for any metric in Table 1 -- only point estimates. Fold-to-fold variability is
  invisible to the reader.
- The claim in Section 4.2 that E vs. E_sens "suggests combining behavioral
  features... reduces sensitivity" is stated as a finding but is really a single
  non-significant p-value (0.137) being interpreted as evidence of an effect
  (absence of significance is not evidence of absence); this should be softened.
- Novelty relative to prior work using similar feature classes (e.g., Bulathwela
  et al. 2022, which already fuses interest with knowledge state) could be stated
  more explicitly and defended, since a reviewer familiar with that literature may
  ask what is genuinely new here beyond the specific ablation framing.

