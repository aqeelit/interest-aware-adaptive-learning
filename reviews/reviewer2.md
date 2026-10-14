# Reviewer #2 (Adversarial Simulation — Independent Angle)

Status: COMPLETE (2026-08-31). Distinct critical stance from Reviewer #1: this
review focuses on theoretical framing, construct validity, and generalizability
rather than statistical technique. Manuscript was NOT modified in producing this
review, per protocol.

## Summary recommendation: Major Revision

Reviewer #1's statistical concerns are valid but orthogonal to a deeper issue: this
paper critiques the field for lacking theoretical grounding (citing Bernacki et al.
2021 approvingly on this point) while its own central construct -- "learner
interest" -- is defined purely operationally, with no engagement with the
psychological literature on interest itself. This undercuts the paper's own
framing device.

## Major issues

**1. Construct validity of "interest" is not established, and a competing
explanation is not ruled out.** The interest signal is operationalized entirely as
resource-type click-share and an entropy measure over that distribution (Section
3.4). Nowhere does the manuscript engage with established interest constructs from
educational psychology (e.g., Hidi and Renninger's phase model of situational vs.
individual interest) to justify why click-share is a valid proxy for interest
specifically, as opposed to, say, course design (some resource types are mandatory
and will be clicked regardless of interest) or access constraints (device/bandwidth
limitations shaping which resource types a student can engage with). The paper
never rules out that its "interest" signal is actually capturing structural or
socioeconomic variation in how students use the VLE, not interest. Given OULAD
includes demographic and disadvantage indicators, a construct-validity check (e.g.,
does the interest signal correlate strongly with IMD/deprivation band?) is a
natural test this paper does not run, and its absence is a significant gap for a
higher-education-equity-focused journal like this one.

**2. Ironic self-inconsistency in theoretical grounding.** Section 2.4 and the
Discussion cite Bernacki et al. (2021) to criticize the field for proceeding
"without a priori theoretical grounding." This paper's own interest construct is
introduced with no a priori theoretical justification either -- it is defined by
what was computable from OULAD's clickstream, not derived from a theory of what
learner interest is or how it should manifest behaviorally. A reviewer sympathetic
to Bernacki et al.'s critique will notice this paper does not meet the standard it
holds the rest of the field to.

**3. Generalizability is much narrower than the framing suggests.** The Abstract
and Conclusion speak broadly of "adaptive e-learning systems" and make a
recommendation applicable to the field generally. But the entire empirical base is
one distance-learning institution (Open University, UK, adult learners, historical
cohorts) plus one unrelated dataset (ASSISTments, US K-12/early-secondary math
tutoring) used only for an independent sub-study that never touches the main
finding. There is no campus-based, synchronous, or non-Anglophone higher-education
context represented at all, despite this journal's stated scope. The Limitations
section (Section 6) acknowledges "single-institution, historical data" but the
Abstract's and Conclusion's language ("adaptive e-learning systems... have a
measurable incentive") is broader than this scope supports and should be narrowed
to match.

**4. Mismatch between "adaptive resource selection" framing and what was actually
studied.** The Introduction and Title(via the paper's positioning, not literally in
the title itself) repeatedly invoke "adaptive resource selection" as the target
application, and Figure 4 depicts a full "Interest-Aware Adaptive Resource
Selection Model." But no resource-selection mechanism of any kind was built,
deployed, or evaluated -- the actual empirical contribution is an outcome-prediction
ablation study. The gap between the framing (an adaptive system) and the
contribution (a predictive model comparison) is larger than the Limitations section
currently acknowledges; Section 6 should state explicitly that no resource-selection
logic was implemented or tested, not just that causal claims are unsupported.

**5. Coherence question: why is Study 2 in this paper at all?** Study 2's Rasch
calibration on ASSISTments is methodologically sound but, by the paper's own
admission (Sections 3.7, 4.3, 6), never feeds into or validates anything used in
Study 1's actual results -- it establishes only that "item-level IRT estimation is
feasible on data of this type," a fairly generic methodological point that does not
depend on this paper's specific research question. A reviewer will ask whether
Study 2 belongs in this paper at all, versus being a separate methods note; as
currently framed, it reads as scope-padding that doesn't sharpen the paper's actual
claim about interest signals.

**6. Equity and fairness dimension entirely unexamined.** Given this journal's
focus on higher education broadly (not just technical ML benchmarking), the absence
of any subgroup analysis (e.g., by disability status, socioeconomic indicator, or
prior educational attainment, all present in OULAD's `studentInfo` data) is a
significant omission. If implicit interest signals are to be recommended as an
additional input to real adaptive systems (as the Conclusion suggests), a reviewer
concerned with equitable deployment will want to know whether the signal's
predictive value -- or its errors -- are distributed evenly across student
subgroups before endorsing its incorporation into real systems.

## Minor issues

- The Discussion's claim that this is "a comparatively rare demonstration of
  interest's outcome-relevance using this particular evaluative standard" is a
  novelty claim that would benefit from a more systematic comparison against the
  interest-modeling literature already reviewed in Section 2.2, several of which
  (e.g., Bulathwela et al. 2022) get closer to outcome-relevance than the current
  phrasing acknowledges.
- "Educational data mining" and "predictive validity" as keywords may undersell
  the paper's higher-education-specific framing to this journal's typical reader;
  consider whether a keyword signaling the equity/generalizability angle (once
  addressed) would better position the paper for this venue.

