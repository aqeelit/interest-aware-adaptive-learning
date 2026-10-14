# CLAUDE.md — Research Project Operating Rules

## Project
Adaptive E-Learning research paper — first submission-ready manuscript.
Owner and sole scientific authority: Dr. Aqeel Mohammed Ali Hussein.

## Non-negotiable rules for any AI agent working in this repository

1. **Never write the manuscript directly from a research question.** The manuscript is
   the last artifact produced, only after research/, literature/, data/, experiments/,
   and results/ are populated with real, traceable content.

2. **Never invent citations, DOIs, datasets, or numerical results.** If a citation is
   needed but not yet in literature/zotero_export.bib or literature/papers.csv, do not
   fabricate one — flag it as `MISSING_CITATION` and stop.

3. **Mark every unverified claim or assumption explicitly** with `ASSUMPTION:` inline.
   Do not present an assumption as an established fact.

4. **All numbers in the manuscript must trace to a file.** Every statistic, accuracy
   figure, p-value, or effect size cited in manuscript/manuscript.md must be traceable
   to a specific file in results/ (table, figure, or statistical_tests output),
   produced by a script in experiments/. If a number cannot be traced, flag it as
   `UNTRACEABLE_CLAIM` and do not include it.

5. **Do not run or simulate statistical analysis yourself.** Statistical tests are
   executed in Python (experiments/scripts or experiments/notebooks) and their real
   output is what gets used. Do not report a statistical result you have not seen
   printed from an actual script run.

6. **Do not claim a research gap without evidence.** A gap may only be proposed if
   supported by at least three independent studies in literature/evidence_table.csv,
   with the supporting papers cited explicitly.

7. **Do not mark a task "Done" without producing the artifact.** If asked to complete
   a phase, the phase is only complete when the corresponding file(s) exist and are
   populated — not when a text description of completion is given.

8. **Record AI involvement.** Any AI assistance used in research design, methodology,
   coding, or analysis must be logged in manuscript/ai_disclosure_log.md as it happens,
   not reconstructed at the end. Verify the target journal's current AI-use disclosure
   policy directly on the journal's site before finalizing manuscript/methodology text —
   do not rely on a remembered or general description of any publisher's policy.

9. **Stay in your lane.** Research question, hypotheses, methodology decisions, gap
   interpretation, and final manuscript approval are the owner's decisions. The agent's
   role is to draft, analyze, calculate, organize, and audit — not to decide.

## Repository map
- research/       — question, objectives, hypotheses, gap (owner-approved)
- literature/      — Zotero export, paper screening, evidence table
- data/            — raw and processed datasets
- experiments/     — notebooks, scripts, configs (Python = source of truth for numbers)
- results/         — tables, figures, statistical test outputs (all reproducible from experiments/)
- manuscript/      — outline, manuscript draft, references, AI disclosure log
- reviews/         — adversarial reviewer simulations (reviewer1, reviewer2)
- submission/       — target journal materials, final submission package

## Current phase
Phase 22 -- POST-REJECTION ROBUSTNESS WORK. No submission is currently pending.

Submission history:
1. Journal of Learning Analytics (JLA): submitted 2026-09-04 via OJS; desk-rejected
   (scope).
2. Journal of Educational Data Mining (JEDM): submitted; desk-rejected 2026-10-12
   (formatting, novelty, single dataset/robustness).

Work status:
- Robustness phase: COMPLETED 2026-10-13 (second model family, leave-one-presentation-out,
  student-level bootstrap CIs, dimensionality controls, gender FNR gaps; second-dataset
  shortlist prepared, owner decision pending). Scripts: app/src/robustness_*.py;
  outputs: results/statistical_tests/robustness_*.
- Course-identity follow-up: IN PROGRESS (tests whether the interest columns mainly
  encode course membership; also key-aligned re-derivation of the manuscript's
  pairwise statistics after the row-position alignment bug found in
  app/src/statistics_tests_oulad.py). Scripts: app/src/ci_*.py; outputs:
  results/statistical_tests/ci*_*.
- Interpretation of all new results, any change to research/ or manuscript/, and the
  choice of next target journal are the owner's decisions.

See manuscript/ai_disclosure_log.md for the full chronological record, and
submission/target_journal.md for the earlier journal-selection history.
