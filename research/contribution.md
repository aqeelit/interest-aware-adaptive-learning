# Proposed Contribution

Status: DRAFT — awaiting owner approval before Phase 8 (Experiment Design) begins.

## Contribution statement (draft)
An adaptive resource-selection model that extends the existing IRT-BTI (Item Response
Theory — Behavioral Telemetry Index) signal with an explicit learner-interest component,
producing a three-input personalization model:

    Behavioral Telemetry (IRT-BTI)
              +
    Performance / Mastery signal (IRT ability estimate)
              +
    Explicit/implicit Learner Interest signal
              ↓
    Interest-Aware Adaptive Resource Selection Model
              ↓
    Evaluated against verified learning-gain outcomes
    (not engagement/click-through proxies alone)

## Why this fills the identified gap
- Addresses the gap in research/gap.md: interest is rarely used as a primary adaptation
  source, and where used, is validated against engagement rather than learning gain.
- Reuses IRT-BTI (already built, validated to high numerical precision) as the
  behavioral+performance backbone, rather than building new instrumentation from scratch —
  this is a genuine reuse of existing infrastructure, not a new system built for the paper.
- Positions the paper as scientific grounding for the Adptivio platform itself, not a
  purely academic exercise.

## Comparison to be tested (draft — needs Phase 8 refinement)
- Baseline A: Random resource assignment
- Baseline B: Traditional performance-only adaptive model (IRT ability estimate alone)
- Baseline C: Behavior-only adaptive model (BTI alone)
- Proposed: Interest-aware model (IRT-BTI + interest signal)

## Decisions (owner-approved 2026-08-30)
1. **Interest signal: implicit only.** Dwell time, click patterns, and resource-type
   preference — no explicit survey/self-report component in this study.
2. **Data source: public benchmark dataset first.** Used to validate the modeling approach
   methodologically before any real Moodle-plugin or pilot-course data is used. Real-data
   validation is deferred to a follow-up study.
3. Outcome measure (learning gain) — still open, to be resolved during Phase 8 dataset
   selection, since it depends on which benchmark dataset has the needed fields.

## Dataset decision (owner-approved 2026-08-30)
No single public benchmark dataset provides high-fidelity item-level IRT data AND rich
implicit interest signals (resource-type click behavior) AND a clear final learning
outcome, all at once. Owner approved a dual-dataset design to resolve this:

- **OULAD** (Open University Learning Analytics Dataset) — used for the interest signal
  (VLE clickstream by resource type: forum/quiz/homepage/oucontent/subpage) and the final
  learning outcome (final_result: Pass/Fail/Distinction/Withdraw). Assessment scores here
  are used as a coarser (assessment-level, not item-level) ability proxy.
- **ASSISTments** (2009 or 2017 version) — used separately to validate the IRT ability
  estimation component itself at proper item-level granularity, since it provides
  individual question-response records with correctness, response time, and skill tags.

This tradeoff (no single dataset serving all three roles at high fidelity) must be stated
explicitly in the manuscript's Limitations section — it is a genuine methodological
constraint of using public benchmark data before pilot/real data, not an oversight.
