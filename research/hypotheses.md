# Hypotheses

Status: DRAFTED — reframed as predictive-validity hypotheses (owner-approved 2026-08-30),
consistent with the observational nature of OULAD/ASSISTments (no live intervention data).

## Primary hypotheses (OULAD predictive task) — VERIFIED 2026-08-30
All numbers below are read directly from results/statistical_tests/oulad_pairwise_comparison.csv
and results/tables/experiment_log.csv — none estimated.

- **H1 — SUPPORTED.** E (B+C fused, no interest) significantly outperforms B alone
  (statistic=47.157, p=6.55e-12) and C alone (statistic=148.364, p=3.95e-34).
  Accuracy: B=0.8312, C=0.7858, E=0.8562.
- **H2 — SUPPORTED.** D (B+C+interest) significantly outperforms E (B+C only)
  (D_only_correct=1226 vs E_only_correct=554, McNemar p=5.92e-57 — confirmed via
  results/figures/model_comparison_bars.png subtitle, sourced from the same
  oulad_pairwise_comparison.csv row). Accuracy: E=0.8562, D=0.8823.
  This is the paper's central result: the interest signal specifically — not just
  fusion in general — adds predictive value.
- **H3 — SUPPORTED.** Every model (B, Bsens, C, D, Dsens, E, Esens) significantly
  outperforms the majority-class baseline A (all p=0.0 as literally stored in the
  source CSV, i.e. below floating-point precision — report as p<0.001 in the
  manuscript, not literally "0").

### Sensitivity analysis (circularity check, approved 2026-08-30)
- B vs B_sens: significant (large n), small effect size — accuracy drops
  0.8312→0.8222 (~0.9 percentage points) when the final/exam assessment is excluded
  from the performance signal.
- D vs D_sens: also significant, small effect size — accuracy 0.8823→0.8743.
- E vs E_sens: NOT significant (p=0.137) — the behavior+performance-without-interest
  model is insensitive to excluding the final assessment, unlike B and D which use
  the raw performance signal more directly. Worth a Discussion note, not overclaiming.
- **Conclusion for Limitations:** the circularity concern is real (final_result does
  partly derive from assessment scores) but does not explain away the paper's central
  finding — H1 and H2 both hold in direction and remain large in magnitude after the
  sensitivity correction (D_sens still clearly above E and E_sens).

## Secondary hypothesis (ASSISTments IRT validation sub-study)
- **H4 — REWORDED to match actual computation (owner decision 2026-08-30).** Original
  wording promised "acceptable model fit assessed via standard Rasch fit statistics"
  (infit/outfit, separation reliability) — these were NOT computed and owner decided
  not to add them. Final wording for the manuscript:
  "A 1PL (Rasch) item response model was calibrated on ASSISTments data (n_items=2137,
  n_persons=1707), producing ability and difficulty parameter estimates (ability_mean=
  -0.01232, ability_sd=0.81144; difficulty_mean=0.29578, difficulty_sd=1.21751;
  difficulty standard errors via 100-iteration bootstrap, mean SE=0.25949), establishing
  that item-level IRT calibration is feasible on this class of data. Formal fit
  diagnostics (infit/outfit, separation reliability) were not assessed and are noted
  as a limitation, not claimed as verified."
  Do NOT write "acceptable model fit" or "good fit" anywhere in the manuscript for
  Study 2 — only report the parameter estimates above, and name the missing
  diagnostics explicitly in Limitations.

## Notes
- These are predictive-validity hypotheses, not causal/interventional claims. The
  manuscript must not claim that fusing interest signals causally improves learning
  outcomes — only that it improves outcome PREDICTION from historical data. This
  distinction must be preserved through Results, Discussion, and Limitations.
