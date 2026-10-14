# Research Gap

Status: IDENTIFIED and DOI-VERIFIED (2026-08-30) — all three core supporting studies now have
confirmed real DOIs, safe to cite directly in the manuscript.

## Proposed gap
Most adaptive e-learning systems select resources primarily from cognitive/performance-based
signals (learning style, pre-knowledge quizzes, mastery level), while explicit learner INTEREST
is comparatively rarely used as an adaptation source — and where interest IS combined with
behavioral telemetry, the outcome measured is typically engagement/click-through rather than
verified learning gain.

## Supporting evidence (≥3 independent studies, per CLAUDE.md rule) — DOIs verified
- P29 — Martin et al. 2020, systematic review of 61 articles (2009-2018): learning style was
  the most-used adaptive source; adaptive feedback/navigation the most-used adaptive targets.
  Interest-based adaptation not among the dominant strategies identified.
  DOI: 10.1007/s11423-020-09793-2 (confirmed via CrossRef).
- P30 — du Plooy et al. 2024, scoping review of 69 studies (2012-2024): pre-knowledge quizzes
  are the most common adaptation trigger; only 36% of studies report engagement increases,
  suggesting interest/engagement is a secondary, not primary, adaptation driver.
  DOI: 10.1016/j.heliyon.2024.e39630 (manually confirmed by Claude via web search 2026-08-30,
  after CrossRef's automated search failed to surface it — verified independently via
  ScienceDirect, Semantic Scholar, and DOAJ).
- P34 — Raj et al. 2021, systematic review of 52 publications (2015-2020): cognitive learner
  attributes (style, preference, knowledge) are used far more often than non-cognitive
  attributes (social tags, trust) — interest-as-signal falls outside the cognitive attributes
  this review found dominant.
  DOI: 10.1007/s40692-021-00199-4 (confirmed via CrossRef).

## Reinforcing (not sufficient alone, but consistent)
- P13 — Bulathwela et al. 2022: one of the few models integrating learner interest, knowledge,
  novelty, and popularity together — but validates against engagement prediction on a video
  platform, not verified learning outcomes.
- P22 — Amin et al. 2023: sequential/future-interest modeling via RL, validated on MOOC
  interaction sequences — again engagement/path-completion framed, not learning-gain framed.

## Confidence
Medium-high. The three core systematic reviews are independent, span different time windows
(2009-2018, 2012-2024, 2015-2020) and methodologies, and converge on the same pattern.
Owner should verify this gap still holds by checking for any 2024-2026 systematic review
specifically on interest-based adaptation before finalizing — the field moves fast (see
literature/papers.csv 2025-2026 entries already showing more interest-aware methods emerging).

## Candidate contribution (draft, needs owner approval)
An adaptive resource-selection model that fuses (a) explicit/implicit learner interest signals
with (b) real-time behavioral telemetry (e.g., IRT-BTI-derived signals), evaluated against
verified learning-gain outcomes rather than engagement proxies alone.
