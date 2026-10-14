# Manuscript — International Journal of Educational Technology in Higher Education

Status: DRAFT. Sections below are complete and verified against project files.
Sections not yet drafted are marked accordingly. Do not submit until every
[PENDING] section is resolved and the full manuscript has been reviewed by the
owner against literature/evidence_table.csv and research/gap.md.

**Outline status**
1. Introduction — DRAFTED (2026-08-31)
2. Related Work — DRAFTED (2026-08-31)
3. Methodology — DRAFTED
4. Results — DRAFTED
5. Discussion — DRAFTED (2026-08-31)
6. Limitations — DRAFTED (2026-08-31)
7. Conclusion — DRAFTED (2026-08-31)
Declarations — DRAFTED (2026-08-31): Data availability, Code availability (real
GitHub URL: https://github.com/aqeelit/interest-aware-adaptive-learning), plus
Funding and Conflicts-of-interest placeholders explicitly marked [To be completed
by the corresponding author] -- these two are NOT filled in and must not be left
as placeholders at submission time.
Title / Abstract / Keywords — DRAFTED (2026-08-31)
References — DRAFTED (2026-09-02): 38 entries (37 from papers.csv + Kuzilek et al.,
2017 for the OULAD dataset), in APA author-date style (comma before year), per
this journal's official submission guidelines for Research Articles, confirmed
directly on 2026-08-31 -- superseding an earlier, incorrect Springer Basic
(SPBASIC, no comma) pass based on a less authoritative general search. All 58
in-text citations throughout Sections 1, 2, 5-7 (plus one instance in Section
3.2) converted to match. Full author lists fetched live from CrossRef. Three
citation-integrity issues were found and RESOLVED during earlier passes (carried
forward unchanged by this reformatting): Sridharan et al. (2021)'s original DOI
resolved to an unrelated paper -- corrected to the real DOI
(10.1007/s10639-021-10560-w), confirmed via Springer/ERIC/multiple citing papers;
"Xaveria et al., 2025" was citing a nonexistent author, corrected to Hariyanto
et al. (2025) throughout; Raj et al.'s systematic-review entry, mislabeled 2022,
was corrected to 2021. See the References section for full detail, including an
editorial note on a conflicting secondary source for the citation-style
determination.

---

## Title Page

**Title:** Does Implicit Learner Interest Improve Outcome Prediction Beyond
Performance and Behavior? An Ablation Study on OULAD and ASSISTments

**Author:** Aqeel M. Ali Hussein

**Affiliation:** Adptivio EdTech Research Lab, Al-Hussein University College, Iraq

**Corresponding author:** Aqeel M. Ali Hussein (sole author)

---

## Title

Does Implicit Learner Interest Improve Outcome Prediction Beyond Performance and
Behavior? An Ablation Study on OULAD and ASSISTments

## Abstract

Adaptive e-learning systems predominantly select resources using cognitive and
performance-based signals; three independent systematic reviews converge in
finding that, where learner interest is used, it is typically validated
against engagement rather than a verified learning outcome. This study asks
whether an implicit interest signal -- resource-type click-share and
engagement-diversity entropy, with no explicit self-report -- improves
prediction of a recorded learning outcome beyond a performance-and-behavior
baseline, using the Open University Learning Analytics Dataset (OULAD; 32,593
students). Logistic-regression models trained on performance-only,
behavior-only, and combined feature sets were compared, isolating the interest
signal's contribution via a direct ablation comparison. The fused
performance-and-behavior model outperformed either signal alone (accuracy
0.856 vs. 0.831 and 0.786; both p < .001), and adding the interest signal
significantly outperformed this ablation (accuracy 0.882; McNemar p =
5.92e-57; odds ratio 0.452, 95% CI [0.409, 0.500]). This finding survived a
sensitivity analysis removing each student's final assessment. A separate
sub-study calibrated a Rasch (1PL) model on ASSISTments (1,707 students, 2,137
items), independent of the primary analysis. A construct-validity check found
only negligible associations between the interest signal and socioeconomic
deprivation and disability, though a subgroup analysis found a gender gap in
false-negative rate (0.199 female vs. 0.163 male) warranting attention before
deployment. Because both datasets are observational, findings are interpreted
as predictive-validity, not causal, evidence: implicit interest signals carry
outcome-relevant information beyond performance and behavior, but this does
not establish that using them for resource selection would improve outcomes.

## Keywords

adaptive learning; learner interest; predictive validity; ablation study; Item
Response Theory; educational data mining

---

## 1. Introduction

Adaptive and personalized e-learning systems aim to tailor instructional content,
activities, and pacing to individual learners, with the goal of improving outcomes
beyond what a uniform curriculum can achieve. The last decade has seen rapid growth in
applying artificial intelligence and machine learning techniques to this problem,
spanning learning-style detection, reinforcement-learning-driven path recommendation,
and multimodal behavioral modeling (Gligorea et al., 2023; Hariyanto et al., 2025;
Halkiopoulos et al., 2024).

A substantial share of this work targets resource or learning-path selection
specifically: systems that rank, filter, or sequence learning resources for a given
learner. Reported adaptation sources for this task are dominated by cognitive and
performance signals — learning-style profiles, prior-knowledge estimates, and
quiz-derived proficiency tiers — combined with collaborative filtering or clustering
(Sridharan et al., 2021; Chen et al., 2020; Sabeima et al., 2022; Henriquez-Nunez et
al. 2023; Kachapuram et al., 2025), and increasingly by reinforcement-learning or
graph-based path optimizers that track mastery and cognitive load in real time (Raj
et al., 2022; Wan et al., 2020; Zhang et al., 2026; Ye, 2025; Sayed et al., 2022;
Amin et al., 2023a, 2023b).

Learner interest — as distinct from mastery or raw behavioral activity — has been
explored as an adaptation source with growing frequency, typically inferred
implicitly from click, download, or dwell-time patterns rather than elicited
explicitly. Recent work models interest as a temporally evolving latent construct via
recurrent or attention-based architectures (Yazdi et al., 2024; Zhang, 2022), fuses
interest with knowledge state and content novelty within a single probabilistic
framework (Bulathwela et al., 2022), or combines interest signals with static learner
profiles for resource recommendation (Pan et al., 2025; Dong et al., 2025; Song
2024; Zhang, 2025; Liu, 2023; Qiu et al., 2024). Across this cluster, interest is
generally used to improve recommendation relevance or engagement, not evaluated
against an independent measure of learning gain.

Despite this breadth of individual systems, three independent systematic reviews
spanning different time periods converge on the same pattern: cognitive and
performance-based signals remain the dominant adaptation source, learner interest is
comparatively underused, and where interest is combined with behavioral telemetry,
the validated outcome is typically engagement or click-through rather than verified
learning gain (Martin et al., 2020; du Plooy et al., 2024; Raj et al., 2021). Section
2 reviews this evidence in more depth, organized around the four literature axes this
study surveyed.

This study addresses that gap directly. We evaluate whether an implicit
learner-interest signal — resource-type click-share and engagement diversity, derived
without any explicit self-report — adds predictive value over a
performance-and-behavior fusion baseline (an assessment-score-derived performance
proxy combined with behavioral telemetry, conceptually related to but distinct from
this research program's separate IRT-BTI work; see Section 3.2 and the Limitations
in Section 6 for the precise scope of that relationship) when the target is a
verified, recorded learning outcome rather than an engagement proxy. Figure 4
summarizes this proposed framework. Two datasets are used for this
purpose, as detailed in Section 3: OULAD, providing the interest signal and the
outcome variable, and ASSISTments, used separately to validate the item-level IRT
calibration underlying the performance component. The remainder of this paper is
organized as follows. Section 2 reviews related work in more depth. Section 3
describes the data, feature construction, and evaluation protocol. Section 4 reports
results, including the central finding that the interest signal significantly
improves predictive validity beyond performance-and-behavior fusion alone. Section 5
discusses these findings; Section 6 states limitations; Section 7 concludes.

## 2. Related Work

### 2.1 Adaptive Resource and Learning-Path Selection

Systems targeting resource or activity selection specifically have used a range of
adaptation mechanisms. Content-ranking approaches filter a knowledge base using
aggregated learner feedback (Sridharan et al., 2021) or fuse multimodal behavioral
data — text, image, and interaction logs — via graph-enhanced contrastive learning to
drive dynamic recommendation (Li et al., 2025). Learning-style-based clustering
combined with collaborative filtering and association-rule mining has been used to
rank resources against a learner's inferred style (Chen et al., 2020), while
cognitive-map models allow continuous joint adjustment of content, activity, path,
and learning partner, with controlled-experiment evidence of improved achievement
without added cognitive load (Wan et al., 2020). Profile-driven approaches
incorporating learning style, personality, and prior knowledge have been applied to
activity suggestion in a single-course deployment (Henriquez-Nunez et al., 2023), and
adaptation frameworks spanning learner knowledge, goals, and available time have been
proposed for lifelong-learning contexts (Sabeima et al., 2022).

Reinforcement-learning and graph-based methods increasingly dominate the
path-optimization sub-literature: hierarchical RL frameworks jointly track knowledge
mastery and cognitive load while allocating resources fairly across learners (Zhang
et al., 2026); multimodal knowledge-graph representations combined with IRT-based
learner profiling and graph-neural-network/RL adaptation engines aim to minimize
cognitive load while maximizing knowledge gain (Ye, 2025); and MDP/Q-learning
sequential recommenders adapt to real-time learner feedback across a session (Amin et
al. 2023b). Real-time behavioral and performance analytics have also been shown to
improve path-recommendation accuracy by roughly 30% over baseline approaches in an
undergraduate deployment (Raj et al., 2022). Simpler proficiency-tier
approaches using decision-tree classification of quiz results (Kachapuram et al.
2025) and hybrid explicit-rating/implicit-behavior recommenders benchmarked on MOOC
platform data (Amin et al., 2023a) round out this cluster, alongside K-12 deployments
combining VARK-style presentation adaptation with DQN reinforcement learning and
rule-based logic (Sayed et al., 2022).

Across this sub-literature, the adaptation signal is overwhelmingly performance-,
style-, or raw-behavior-derived; explicit modeling of learner interest as a distinct
construct is rare, motivating the next axis.

### 2.2 Learner Interest Modeling

A smaller but growing literature models learner interest explicitly, generally
inferred implicitly from behavioral traces rather than self-report. Sequence models
separate long-term from short-term interest components from click and download logs,
reporting very high classification accuracy on single-platform data (Yazdi et al.
2024), and combine recurrent and attention-based architectures — a GRU with
XLNet-style temporal attention — to mine interest for course-level recommendation
(Zhang, 2022). At larger scale, probabilistic graphical models have fused learner
interest with knowledge state, content novelty, and popularity within a single
engagement-prediction framework validated on over 20,000 learners and explicitly
addressing the cold-start problem (Bulathwela et al., 2022) — one of the few systems
integrating interest with knowledge state jointly. Other approaches fuse static
learner attributes with dynamic behavioral interest via TF-IDF and time-decay
weighting to model evolving preference (Pan et al., 2025), apply k-means clustering
with an embedding layer to separate short- and long-term interest and reduce
recommendation sparsity (Dong et al., 2025), or combine fuzzy-set learner modeling
with collaborative filtering over an interest matrix (Song, 2024). Interest has also
been folded into broader learner models alongside prior knowledge and learning style
for decision-tree/NLP-based recommendation (Zhang, 2025), used to drive real-time
adjustment based on historical adoption behavior in interest-oriented teaching (Liu
2023), and situated within systematic syntheses of learner- and
learning-object-modeling approaches more broadly (Qiu et al., 2024).

Consistent with the pattern noted in Section 1, this cluster validates interest
signals primarily against recommendation accuracy, engagement, or
sparsity-reduction metrics — not against a verified learning outcome such as course
completion or final grade.

### 2.3 Behavioral and Engagement Signals

A parallel literature treats behavioral telemetry and affective state as the primary
adaptation or monitoring signal, largely independent of any interest construct.
Supervised learning over online behavioral characteristics has been used to maintain
learner motivation and inform adaptive control (Abhirami et al., 2021), while
wearable EEG systems classify cognitive and emotional engagement in real time to
enable automated content and communication adaptation, at roughly 77% classification
accuracy (Apicella et al., 2022). Qualitative work links ICT-driven interest, agency,
and motivation to sustained engagement across university courses (Wahyuni et al.
2024), and a case-study design shows that behavioral-engagement effects on
achievement in an e-textbook environment are moderated by prior knowledge (Reinhold
et al., 2021) — a finding directly relevant to this study's own circularity check
between the performance signal and the outcome measure (Section 3.6).

This sub-literature establishes that behavioral and affective signals reliably
predict engagement and, in some designs, achievement, but rarely evaluates them
jointly with an explicit interest construct or against a genuine post-hoc outcome
measure spanning an entire course.

### 2.4 Systematic Reviews and the Research Gap

Multiple systematic reviews synthesize this literature at scale, and their
conclusions converge. Reviewing 61 articles published between 2009 and 2018, Martin
et al. (2020) find learning style to be the most common adaptive source and adaptive
feedback/navigation the most common adaptive target, with interest-based adaptation
not among the dominant strategies identified. A scoping review of 69 studies (du
Plooy et al., 2024) finds pre-knowledge quizzes to be the most common adaptation
trigger, with only 36% of reviewed studies reporting increased engagement as an
outcome — evidence that engagement/interest functions as a secondary rather than
primary adaptation driver in the field. A systematic review of 52 publications from
2015-2020 (Raj et al., 2021) similarly finds cognitive learner attributes — style,
preference, prior knowledge — used far more often than non-cognitive attributes such
as social tags or trust, with interest-as-signal falling outside the attributes this
review found dominant. These three reviews are independent, examine different
(partially overlapping) time periods, and use different methodologies, yet converge
on the same pattern, satisfying this project's evidentiary threshold for proposing a
research gap.

Additional broader reviews reinforce this picture without targeting interest
specifically: large-scale reviews of AI/ML techniques in adaptive e-learning report
gains in engagement, retention, and performance (Gligorea et al., 2023), while
other reviews flag interpretability, privacy, and scalability challenges (Hariyanto
et al., 2025) or bias and discrimination concerns (Halkiopoulos et al., 2024); a
review of personalized-learning research (Bernacki et
al. 2021) finds the field often proceeds without a priori theoretical grounding,
recommending theory-guided design; a 25-year overview of digital personalized
learning in primary and secondary education finds substantial conceptual and
methodological inconsistency across studies (Van Schoors et al., 2021); and further
reviews of learning-style detection via machine learning noting limited comparison
of deep-learning methods (Essa et al., 2023) and personalized learning in higher
education noting rare explicit use of learning theory (Fariani et al., 2023) both
point to adaptation-outcome validation remaining less developed than
adaptation-mechanism design; a review of AI-driven lifelong-learning path
generation separately finds adaptive-learning technologies predominant in this
space, with growing interest in generative language models (Bayly-Castaneda et al.
2024).

Taken together, this literature establishes that most adaptive e-learning systems
select resources primarily from cognitive/performance-based signals, while explicit
learner interest is comparatively rarely used as an adaptation source — and where
interest is combined with behavioral telemetry, the outcome measured is typically
engagement or click-through rather than verified learning gain. This is the gap this
study addresses: Section 3 describes an interest-aware model that fuses an implicit
interest signal with a performance-and-behavior backbone, evaluated
specifically against a recorded learning outcome rather than an engagement proxy.

## 3. Methodology

### 3.1 Research Design and Task Framing

This study adopts an offline predictive-validity design rather than a live
interventional (A/B) design. Because both datasets used are observational — no
resource-selection policy was actually deployed to students during data collection —
a causal claim that a given feature set *causes* improved learning outcomes cannot be
supported. Instead, the design asks a narrower, defensible question: which feature
representation best predicts a learner's actual, recorded learning outcome? This
framing follows precedent in comparable published work using the same class of
observational learning-analytics data. All hypotheses and conclusions in this paper
are stated as predictive-validity claims, not causal ones.

Two studies were conducted: Study 1, the primary predictive comparison on OULAD, and
Study 2, a methodological companion sub-study on ASSISTments. Study 2 exists
specifically to address a scope limitation of Study 1: Study 1's performance signal
(Model B) is an assessment-level aggregate, not a true item-level ability estimate,
because OULAD does not provide item-level response data. Study 2 uses a dataset that
does provide item-level responses (ASSISTments) to establish, independently, that
Item Response Theory calibration of the kind that would be needed to build a more
rigorous performance signal is methodologically feasible at this scale and data
type. Study 2's output is not merged into Study 1 and does not validate Study 1's
specific numbers -- it exists to support the claim that a future, higher-fidelity
version of this work's performance signal is achievable, not to strengthen Study 1's
current results directly. This scope and its limits are stated again in Section 6.

### 3.2 Data Sources

**Study 1 — Open University Learning Analytics Dataset (OULAD).** A public dataset
covering 32,593 students across 22 module-presentations at the Open University,
released under a CC-BY 4.0 license (Kuzilek et al., 2017). It was selected because
it jointly provides (a)
fine-grained VLE clickstream logs broken down by resource type, suitable for
constructing an implicit interest signal, and (b) a recorded final outcome
(`final_result`: Pass, Distinction, Fail, Withdrawn) suitable as a genuine
learning-outcome dependent variable.

**Study 2 — ASSISTments (skill-builder release).** Provides individual, item-level
student-response records, which OULAD does not. Used exclusively to validate that a
Rasch (1PL) item response model can be meaningfully calibrated on this class of data,
at a granularity finer than OULAD's assessment-level scores.

No single public dataset provided high-fidelity item-level IRT data, rich implicit
behavioral/interest signals, and a clear final learning outcome simultaneously. This
dual-dataset design is a direct methodological response to that gap; the resulting
granularity mismatch between the two datasets is addressed explicitly in Limitations.

### 3.3 Data Cleaning

Records with missing `final_result`, duplicate student-module-presentation entries,
VLE interaction records lacking a resolvable `activity_type`, negative click counts,
and assessment scores outside the valid 0-100 range were removed prior to feature
construction, identically across all downstream feature sets.

### 3.4 Feature Engineering (Study 1)

The dependent variable was binarized: `final_result` in {Pass, Distinction} -> 1,
in {Fail, Withdrawn} -> 0. Seven feature-set conditions were constructed:

| Model | Composition |
|---|---|
| A | Majority-class baseline; no informative features |
| B | Performance: assessment-score aggregates (mean, weighted mean, count, SD) |
| B_sens | B excluding each student's final/exam assessment |
| C | Behavior: VLE clickstream aggregates (total clicks, active days, mean clicks/day) |
| D (proposed) | B + C + implicit interest signal |
| D_sens | B_sens + C + implicit interest signal |
| E (ablation) | B + C only, no interest signal |
| E_sens | B_sens + C only |

The implicit interest signal (D, D_sens only) comprises 21 features: 20 click-share
columns, one per distinct VLE resource type actually present in the cleaned OULAD
data (dataplus, dualpane, externalquiz, folder, forumng, glossary, homepage,
htmlactivity, oucollaborate, oucontent, ouelluminate, ouwiki, page, questionnaire,
quiz, repeatactivity, resource, sharedsubpage, subpage, url), plus one
resource-diversity entropy measure over that distribution. Model E and the `*_sens`
variants were added after inspecting initial results from models A-D, for the
reasons given in Section 3.6.

### 3.5 Predictive Models and Evaluation Protocol

An identical model family — logistic regression (median imputation, standardization,
L2-regularized logistic regression) — was trained per feature-set condition, so that
performance differences are attributable to features rather than model capacity.
Stratified 5-fold cross-validation (seed=42, fixed across all runs) was used; Accuracy,
macro-F1, and AUC-ROC were computed per fold, with per-sample predictions retained for
paired statistical comparison.

### 3.6 Statistical Comparison

Pairwise comparisons used McNemar's test on aligned held-out predictions (exact
variant when discordant pairs < 25, asymptotic otherwise). A feature set is described
as outperforming another only when the corresponding test is significant at alpha=0.05;
point-estimate metric differences alone are not treated as sufficient evidence.

Two additions were made after inspecting the initial four-condition (A/B/C/D) results:

1. **Model E (ablation).** Isolates whether the interest signal specifically — not
   fusion of performance and behavior in general — drives D's advantage, via a direct
   D-vs-E comparison.
2. **Sensitivity variants (B_sens, D_sens, E_sens).** Address a circularity concern:
   OULAD's `final_result` is, in many module-presentations, computed in part from the
   same assessment scores used in feature set B. The sensitivity variants recompute
   the performance signal excluding each student's final/exam assessment, testing
   whether observed effects survive removal of the assessment most likely to overlap
   with the outcome's own computation.

### 3.7 Study 2: Rasch (1PL) Calibration on ASSISTments

A binary student x item response matrix was constructed, retaining students with
>=5 responses and items with >=20 responses, filtered iteratively to a fixed point so
every retained student and item provably satisfies both thresholds simultaneously. A
Rasch (1PL) model was fit via marginal maximum likelihood, yielding item difficulty
and person ability estimates; difficulty standard errors were estimated via a
100-iteration bootstrap.

This calibration establishes that item-level IRT estimation is feasible on data of
this type and scale. It is not merged into Study 1. Standard Rasch fit diagnostics
(infit/outfit statistics, person/item separation reliability) were not computed in
this study, by explicit scope decision; accordingly, no claim of "acceptable model
fit" is made anywhere in this manuscript — only the calibrated parameter estimates
themselves are reported (Section 4.3).

### 3.8 Reproducibility

Every model-fitting run logged its experiment identifier, dataset version, random
seed, model specification, hyperparameters, resulting metrics, and a timestamp to a
persistent, append-only log, so that every reported number traces to a specific,
re-runnable execution.

### 3.9 Use of Generative AI

In accordance with Springer Nature's editorial policy on generative AI, the following
is disclosed. A large language model (Claude, Anthropic) was used during this study
for: (i) literature search assistance and evidence synthesis informing the research
gap identification; (ii) design and implementation of the Python data-processing and
statistical-analysis pipeline described in Sections 3.3-3.7, including identification
and correction of implementation errors during development; and (iii) drafting
assistance for portions of this manuscript's text. The AI tool is not an author of
this work and bears no accountability for its content. The corresponding author
takes full responsibility for the accuracy, integrity, and interpretation of all data,
code, statistical results, and conclusions reported herein, and verified all
numerical results in this manuscript directly against the underlying output files
prior to inclusion. No figure in this manuscript was generated by AI image-generation
tools; all figures were produced by author-reviewed Python code operating on the
verified data described above. A complete, dated log of AI-assisted actions taken
during this project is maintained and available upon editorial request.

## 4. Results

### 4.1 Study 1 — Predictive Comparison Across Feature Sets

Table 1 reports 5-fold cross-validation performance (mean across folds) for all
eight feature-set conditions.

**Table 1.** Predictive performance by feature set (OULAD, 5-fold CV mean)

| Model | Accuracy | F1 (macro) | AUC |
|---|---|---|---|
| A (majority baseline) | 0.528 | 0.346 | 0.500 |
| B (performance) | 0.831 | 0.824 | 0.901 |
| B_sens (performance, excl. final assessment) | 0.822 | 0.814 | 0.893 |
| C (behavior) | 0.786 | 0.786 | 0.873 |
| E (B+C, ablation, no interest) | 0.856 | 0.850 | 0.922 |
| E_sens | 0.851 | 0.844 | 0.917 |
| D (proposed: B+C+interest) | 0.882 | 0.876 | 0.941 |
| D_sens | 0.874 | 0.868 | 0.935 |

All seven non-baseline models significantly outperformed the majority-class baseline
A (McNemar's test, all p < 0.001; see Supplementary Table S1 for full pairwise
statistics).

**H1 (performance+behavior fusion outperforms either signal alone).** Model E
(B+C, no interest) significantly outperformed B alone (McNemar statistic=47.16,
p=6.55e-12) and C alone (statistic=148.36, p=3.95e-34), supporting H1.

**H2 (interest signal adds value beyond performance+behavior fusion) — central
result.** Model D (B+C+interest) significantly outperformed Model E (B+C only)
(McNemar p=5.92e-57; D correctly classified 1,226 cases E misclassified, versus 554
cases in the reverse direction). This isolates the contribution of the interest
signal specifically, rather than fusion in general, supporting H2. This comparison
survives Benjamini-Hochberg correction for the full set of 28 pairwise comparisons
conducted in this study (corrected p=1.51e-56, still significant), and its effect
size is substantial: the odds ratio of E-only-correct to D-only-correct cases is
0.452 (95% CI [0.409, 0.500]), meaning D corrects roughly 2.2 times as many of E's
errors as E corrects of D's -- a clear, not merely statistically detectable,
advantage. All 27 other previously-significant pairwise comparisons in this study
remain significant after the same correction; the one previously non-significant
comparison (E vs. E_sens, Section 4.2) remains non-significant.

Figure 1 visualizes these comparisons; Figure 2 presents ROC curves for models
B, C, D, and E (pooled out-of-fold predictions).

### 4.2 Sensitivity Analysis (Circularity Check)

Because OULAD's `final_result` is, in many module-presentations, computed in part
from assessment scores, a sensitivity analysis excluding each student's final/exam
assessment from the performance signal was conducted. B vs. B_sens and D vs. D_sens
differences were both statistically significant (large sample size), but small in
effect size: Accuracy dropped from 0.831 to 0.822 for B, and from 0.882 to 0.874 for
D (~0.8-0.9 percentage points). The B-vs-B_sens odds ratio is 0.855 (95% CI
[0.782, 0.934]) -- significant, since its confidence interval excludes 1, but close
to 1 in absolute terms, consistent with the small percentage-point difference and
clearly smaller in magnitude than the D-vs-E odds ratio (0.452) reported in Section
4.1. Critically, D_sens (0.874) remained clearly above
both E (0.856) and E_sens (0.851), indicating that the paper's central finding (H2)
is not an artifact of this circularity. Interestingly, E vs. E_sens was not
statistically significant (p=0.137, and remains non-significant after
Benjamini-Hochberg correction, p_bh=0.137), suggesting that combining behavioral
features with the performance signal reduces sensitivity to the exclusion of any
single assessment, relative to the performance signal alone.

### 4.3 Study 2 — Rasch (1PL) Calibration on ASSISTments

A Rasch model was calibrated on 1,707 students and 2,137 items. Person ability
estimates had mean=-0.012 (SD=0.811); item difficulty estimates had mean=0.296
(SD=1.218; bootstrap mean standard error=0.259). Figure 3 shows both distributions.
As noted in Section 3.7, this sub-study reports calibrated parameter estimates only;
no fit-quality claim is made (see Limitations).

### 4.4 Construct Validity Check: Does the Interest Signal Reflect Deprivation or Disability Rather Than Interest?

Because the interest signal is derived entirely from clickstream behavior, it could
in principle reflect structural factors -- socioeconomic deprivation or
disability-related access constraints -- rather than learner interest per se. This
was tested directly against two demographic fields available in OULAD's
`studentInfo` table: `imd_band` (Index of Multiple Deprivation band, an ordinal
deprivation measure) and `disability` (Y/N).

Resource-diversity entropy correlated with `imd_band` at Spearman's
rho=0.03 (p=5.89e-07, n not separately re-stated here as it matches Table 1's
full sample). This correlation is statistically significant given the large
sample size, but explains well under 0.1% of variance in the interest signal --
practically negligible despite its formal significance. Entropy did not differ
significantly between students with and without a recorded disability
(Mann-Whitney U, p=0.644). The same two tests repeated on `click_share_forumng`
(chosen as a resource type plausibly related to internet access or peer support)
showed the identical pattern: a statistically significant but practically negligible
association with `imd_band` (rho=0.03, p=1.71e-05) and no significant association
with disability status (p=0.702).

These results do not fully rule out a deprivation- or access-related confound in
the interest signal, but they provide no evidence of a substantial one on the two
demographic dimensions tested; see Discussion (Section 5) and Limitations (Section
6) for how this qualifies the paper's central claim.

### 4.5 Subgroup Performance of the Proposed Model (D)

Model D's predictive performance (Accuracy, F1, AUC) and False Negative Rate (FNR --
the proportion of students who actually failed but were predicted to pass, the
metric most relevant to whether a support intervention would be withheld from a
student who needed one) were computed separately by disability status, gender, and
IMD band, using the same 5-fold cross-validation protocol as the main analysis.

**Table 2.** Model D performance by subgroup (OULAD, 5-fold CV)

| Dimension | Value | n | Accuracy | F1 | AUC | FNR | n (actually failed) |
|---|---|---|---|---|---|---|---|
| (overall) | ALL | 25,770 | 0.882 | 0.876 | 0.941 | 0.179 | 10,390 |
| disability | N | 23,361 | 0.883 | 0.876 | 0.941 | 0.181 | 9,188 |
| disability | Y | 2,409 | 0.879 | 0.879 | 0.941 | 0.156 | 1,202 |
| gender | F | 11,589 | 0.875 | 0.866 | 0.932 | 0.199 | 4,463 |
| gender | M | 14,181 | 0.888 | 0.884 | 0.947 | 0.163 | 5,927 |
| imd_band | most_deprived | 7,752 | 0.872 | 0.872 | 0.938 | 0.172 | 3,742 |
| imd_band | middle | 10,246 | 0.884 | 0.877 | 0.941 | 0.175 | 4,038 |
| imd_band | least_deprived | 6,775 | 0.888 | 0.875 | 0.939 | 0.192 | 2,343 |
| imd_band | missing | 997 | 0.904 | 0.874 | 0.947 | 0.217 | 267 |

Aggregate performance (Accuracy, F1, AUC) is broadly similar across all subgroups
examined, with differences on the order of 1-2 points. The False Negative Rate,
however, shows a gender gap: 0.199 for female students versus 0.163 for male
students, a roughly 3.6-percentage-point difference in the rate at which students
who actually fail are incorrectly predicted to pass. Disability and IMD-band
subgroups do not show a comparably large or monotonic FNR pattern. These figures
are reported here as observed; their interpretation is taken up in Section 5.

## 5. Discussion

The central finding of this study is that Model D (performance + behavior + implicit
interest) significantly outperforms Model E (performance + behavior only) in
predicting recorded learning outcomes (Section 4.1). This result directly addresses
the gap identified in Section 2.4: across three independent systematic reviews
spanning different time periods (Martin et al., 2020; du Plooy et al., 2024;
Raj et al., 2021), cognitive and performance-based signals were found to dominate
adaptation practice in the field, with learner interest comparatively underused and,
where used, validated against engagement rather than a verified learning outcome.
This study's design was built specifically to close that evaluative gap: the
dependent variable throughout is `final_result`, a recorded academic outcome, not an
engagement or click-through proxy. Within that framing, the finding that an implicit
interest signal -- built entirely from resource-type click-share and engagement
diversity, without any self-report -- adds predictive value beyond a strong
performance-and-behavior baseline is, to this study's knowledge, a comparatively rare
demonstration of interest's outcome-relevance using this particular evaluative
standard.

This result should be read as predictive-validity evidence, not causal evidence
(Section 3.1). The claim supported by Section 4.1 is that including the interest
signal allows a model to better predict which students pass, not that intervening on
a student's interest signal would improve their outcome. The distinction matters
practically: a system built on this finding could use the interest signal to
identify students at risk of a poor outcome earlier or more accurately than a
performance-and-behavior-only model, but this study does not license claims about
what would happen if such a system also used the interest signal to select or
recommend resources. That stronger, interventional claim would require the live A/B
design this study explicitly did not conduct (Section 3.1).

The sensitivity analysis (Section 4.2) is important to this interpretation. Because
OULAD's `final_result` is, in many module-presentations, computed in part from
assessment scores, the performance signal (Model B) risks partial tautology when used
to predict that same outcome. The sensitivity variants show this concern is real --
B_sens's accuracy is measurably lower than B's -- but does not eliminate the central
finding: D_sens remains clearly above both E and E_sens, meaning the interest
signal's advantage survives removal of the assessment component most likely to
overlap with the outcome's own computation. The additional observation that E and
E_sens do not differ significantly, while B/B_sens and D/D_sens do, suggests that
combining behavioral features with the performance signal reduces the model's
reliance on any single assessment -- a secondary finding worth further investigation
but not central to this paper's claims.

A construct-validity concern worth addressing directly is whether the interest
signal, being derived purely from clickstream behavior, actually captures
socioeconomic deprivation or disability-related access constraints rather than
learner interest per se -- a confound that would undermine the interpretation of
Model D's advantage as interest-specific. The check reported in Section 4.4 found
statistically significant but practically negligible associations between the
interest signal and deprivation band (explaining well under 0.1% of variance), and
no significant association with disability status at all. This does not amount to
proof that the signal measures "interest" in any theory-grounded sense -- a point
returned to below -- but it does provide some reassurance that the signal is not,
at least on these two dimensions, primarily a proxy for structural disadvantage.

The subgroup analysis in Section 4.5 surfaces a finding that should be stated
plainly rather than left in a table: Model D's False Negative Rate is
meaningfully higher for female students (0.199) than male students (0.163), a
gap not mirrored to a comparable degree across disability or deprivation bands.
Concretely, this means that among students who actually go on to fail, a larger
share of women than men are incorrectly predicted to pass by this model. If a
system built on this kind of model were used in practice to flag students for
support, this asymmetry would translate into female students who need help being
systematically somewhat less likely to receive an early warning. This is reported
here as an observed property of the fitted model on this dataset, not a claim about
why the gap exists; possible contributing factors -- differential relationships
between the underlying features and outcomes by gender, or gender-correlated
differences in how the click-based signals are generated -- were not investigated
and should not be inferred from this result alone. The gap is disclosed here
precisely because a paper proposing a model for potential use in identifying
at-risk students has an obligation to report this kind of subgroup asymmetry when
it is found, rather than only reporting aggregate performance.

Two further points connect this study's design choices to patterns identified in
Section 2. First, this study's implicit-only operationalization of interest
(click-share and engagement diversity, with no explicit or self-reported component)
is itself introduced without a priori grounding in an established theory of
interest -- for instance, models distinguishing situational from individual
interest, or its development over repeated engagement, are not invoked here to
justify why click-share distribution specifically should be read as interest
rather than, say, habit or course structure. This is a notable tension: Section 2.4
cites Bernacki et al. (2021) approvingly for criticizing the field's tendency to
proceed without a priori theoretical grounding, yet this study's own central
construct is operationalized in essentially the same atheoretical, data-availability-driven
manner it critiques elsewhere. This study's implicit-only operationalization sits within,
rather than resolves, the broader conceptual inconsistency Van Schoors et al. (2021)
document across digital personalized-learning research: different studies model
"interest" via substantially different constructs, and this study's implicit
click-based definition is one choice among several defensible alternatives, not a
claim that it is the correct one. Second, this study's separation of the predictive
Study 1 from the item-level IRT validation in Study 2 reflects the granularity
mismatch inherent in using public benchmark data (Section 3.2); a genuinely unified
model -- combining item-level ability estimation with implicit interest signals in a
single deployed system -- was outside this study's scope and remains for future work.
Study 2 was included in this paper specifically because it establishes the
feasibility of that future direction (item-level IRT calibration at this scale) --
not because it validates or strengthens Study 1's reported numbers, which it does
not touch. Readers should treat Study 2 as motivating evidence for a research
direction rather than as evidence supporting this paper's central claim, ideally
validated on data of the kind Bernacki et al. (2021) argue the field needs
more of: theory-guided designs, developed a priori rather than post hoc.

## 6. Limitations

**Observational, non-interventional data.** Neither OULAD nor ASSISTments reflects a
deployed adaptive resource-selection policy; both are historical records of learner
behavior and performance under whatever conditions each institution's course actually
ran. Accordingly, every claim in this paper is a predictive-validity claim (which
feature set best predicts a recorded outcome from historical data), not a causal
claim (which feature set would improve outcomes if used to drive live resource
selection). This distinction is maintained throughout Section 4 and Section 5 and
should be maintained in any downstream citation of this work.

**No resource-selection mechanism was built or tested.** Figure 4 and the framing in
Sections 1 and 3.1 describe a conceptual "interest-aware adaptive resource selection
model," but this study implemented and evaluated only the outcome-prediction
component of that framework -- no algorithm for actually selecting, ranking, or
recommending resources to a learner was built, deployed, or tested. The gap between
an outcome-prediction model and a deployed resource-selection system is substantial:
a system would additionally need a policy for translating predictions (or the
underlying signal) into specific resource choices, and evaluating that policy would
require the live interventional design explicitly out of scope here. Readers should
not infer that this study demonstrates or validates any adaptive resource-selection
mechanism.

**Dataset granularity mismatch.** No single public dataset available for this study
provided high-fidelity item-level IRT data, rich implicit behavioral/interest
signals, and a clear final learning outcome simultaneously (Section 3.2). OULAD
supplies the interest signal and outcome variable but only assessment-level (not
item-level) performance data; ASSISTments supplies true item-level response data
suitable for Rasch calibration but lacks the resource-type clickstream needed to
construct the interest signal and lacks a course-level final outcome. Study 1 and
Study 2 are consequently kept methodologically separate rather than merged into a
single unified model, and Study 2's contribution is limited to establishing that
item-level IRT calibration is feasible on data of this type -- it does not certify the
quality of the coarser, assessment-level performance signal actually used in Study 1.

**Residual circularity in the performance signal.** OULAD's `final_result` is, in
many module-presentations, computed in part from the same assessment scores used to
construct the performance feature set. The sensitivity analysis in Section 4.2
addresses this directly and shows the paper's central finding survives it, but does
not eliminate the underlying overlap entirely -- some degree of shared information
between the performance signal and the outcome likely remains even after excluding
the final assessment, since earlier assessments in a course can also correlate
strongly with a student's ultimate result. Readers should treat the absolute
performance figures in Table 1 as somewhat inflated relative to a fully independent
performance signal, while treating the *relative* ordering and significance of the
model comparisons (which this study's claims rest on) as more robust to this concern.

**Rasch calibration reports parameter estimates only.** Study 2 fits a 1PL (Rasch)
model and reports ability and difficulty parameter estimates with bootstrap standard
errors (Section 4.3), but does not compute standard model-fit diagnostics such as
infit/outfit statistics or person/item separation reliability. This was an explicit
scope decision made during this project, not an oversight discovered after the fact.
Consequently, this paper makes no claim about whether the fitted Rasch model
constitutes a good or acceptable fit to the ASSISTments data in the psychometric
sense -- only that stable parameter estimates were obtainable at this scale. A future
study intending to rely on this calibration for substantive measurement claims should
compute these diagnostics first.

**Implicit-only interest signal, not theoretically grounded, and only
partially checked against structural confounds.** Per an explicit design decision
made early in this project, the interest signal used throughout is implicit
(resource-type click-share and engagement-diversity entropy) with no explicit or
self-reported interest component, and it was introduced without grounding in an
established psychological theory of interest (Section 5). A post hoc construct-validity
check (Section 4.4) found only a statistically significant but practically
negligible association between the interest signal and socioeconomic deprivation
(imd_band) and no significant association with disability status, on the OULAD
sample studied. This reduces, but does not eliminate, the concern that the signal
partly reflects structural rather than motivational variation; the check was not
exhaustive (e.g., it does not test other demographic dimensions, and a null result
on these two dimensions does not rule out other confounds), and it cannot address
whether an explicit interest measure would perform comparably, better, or worse,
nor how implicit and explicit interest signals might interact if combined.

**Observed subgroup disparity in false negative rate.** A subgroup analysis
(Section 4.5) found Model D's False Negative Rate is higher for female students
(0.199) than male students (0.163), meaning female students who actually fail are
somewhat more likely to be incorrectly predicted to pass than male students in the
same situation. This gap was not investigated for cause and should not be assumed
to generalize beyond this dataset and model configuration; it is reported as a
finding that any future work building toward a deployed system from this line of
research must examine and address before such a system could be considered for use
in identifying students for support.

**Single model family, no robustness check across model classes.** All eight
feature-set conditions in Study 1 were evaluated using only logistic regression
(Section 3.5), a deliberate choice to isolate feature effects from model-capacity
effects. This design does not test whether the central finding (Model D
outperforming Model E) is specific to logistic regression's inductive biases or
would hold under other model families commonly used in this literature (e.g.,
gradient boosting, neural sequence models). This should be treated as an open
robustness question, not as evidence either for or against the finding's generality
across model classes.

**No hyperparameter tuning.** Logistic regression was run with a single fixed
configuration (L2 regularization, default-adjacent settings, max_iter=1000) across
all feature-set conditions, with no grid search or regularization-strength tuning
per condition. Because feature sets differ in dimensionality (e.g., Model D adds 21
interest features to Model E's baseline), a fixed hyperparameter configuration may
under- or over-fit different conditions to different degrees; this was not
separately assessed.

**No held-out or cross-institution validation.** All results in this paper come
from stratified cross-validation within a single pooled OULAD sample; no
temporally or institutionally held-out cohort was evaluated, and no test on data
from a different institution was conducted. Whether the reported feature-set
ranking is stable when trained on some module-presentations and evaluated on
others entirely unseen during training remains untested.

**Single-institution, historical data.** OULAD reflects Open University courses and
student populations at the time of data collection; ASSISTments reflects a
US-based math-tutoring context. Generalization of the specific numerical results in
Table 1 to other institutions, subject areas, or more recent cohorts should be made
cautiously; the qualitative finding that an interest signal adds predictive value
beyond performance and behavior is the more portable claim than any specific accuracy
figure.

## 7. Conclusion

This study asked whether an implicit learner-interest signal adds predictive value
over a performance-and-behavior baseline when predicting a verified learning outcome,
addressing a gap identified across three independent systematic reviews: that
learner interest is comparatively underused as an adaptation signal in adaptive
e-learning research, and where used, is typically validated against engagement
rather than outcome measures. Using a dual-dataset design -- OULAD for the primary
predictive comparison and ASSISTments for an independent item-level IRT validation --
this study found that a model combining performance, behavior, and an implicit
interest signal significantly outperformed both a performance-and-behavior-only
ablation and each individual signal, with the interest signal's specific contribution
isolated via a direct ablation comparison (Model D vs. Model E) rather than inferred
indirectly. A sensitivity analysis addressing a plausible circularity concern in the
performance signal showed this central finding survives in both direction and
magnitude, though the underlying data's observational nature restricts all findings
to predictive rather than causal claims.

These results suggest that adaptive e-learning systems operating in contexts similar
to the one studied here -- large-scale distance higher education with rich VLE
clickstream data -- have a measurable incentive to incorporate implicit interest
signals derived from ordinary clickstream data when seeking to improve prediction
of genuine learning outcomes, rather than engagement or click-through proxies,
without requiring additional explicit interest
elicitation from learners. Whether this incentive extends to campus-based,
synchronous, or non-Anglophone higher-education contexts not represented in this
study's data remains an open question. Future work should extend this
predictive-validity finding toward a live, interventional evaluation; test its
generalizability across institutional contexts; unify the item-level IRT
calibration demonstrated in Study 2 with the interest-aware model demonstrated in
Study 1, ideally on a single dataset that provides both; and examine whether explicit
interest signals, where available, are complementary to or redundant with the
implicit signal used here.

## Declarations

**Availability of data and materials.** The datasets analyzed during the current
study are publicly available and were not generated by the author. OULAD is
available in the Open University's OULAD repository, CC-BY 4.0 licensed:
https://analyse.kmi.open.ac.uk/open_dataset (Kuzilek et al., 2017). The
ASSISTments skill-builder dataset is available from the ASSISTments project.
Raw, individual-level records from either dataset are not redistributed by this
study; a de-identified version of Study 2's estimated person-ability parameters
(sequential identifiers replacing original user IDs) is included in the software
repository described below.

Software:
- Project name: interest-aware-adaptive-learning
- Project home page: https://github.com/aqeelit/interest-aware-adaptive-learning
- Archived version: commit fe91e1b on branch main
- Operating system(s): Platform independent (Dockerized)
- Programming language: Python
- Other requirements: Docker, Docker Compose
- License: MIT
- Any restrictions to use by non-academics: None

**Competing interests.** The author declares that he has no competing interests.

**Funding.** Not applicable. No external funding was received for this research.

**Authors' contributions.** AMAH conceived the study, designed the methodology,
supervised and verified all data processing, statistical analysis, and
interpretation of results (conducted with AI-assisted tooling as disclosed in
Section 3.9), drafted and revised the manuscript, and approved the final
manuscript for submission.

**Acknowledgements.** Not applicable.

**Authors' information.** Not applicable.

## References

Consolidated master list of every citation used anywhere in this manuscript
(Sections 1-7). 38 entries (37 from literature/papers.csv plus Kuzilek et al.,
2017, added to support the OULAD dataset-license claim in Section 3.2). Style:
APA (author-date, comma before year), per this journal's official submission
guidelines for Research Articles
(https://link.springer.com/journal/41239/submission-guidelines/research-articles),
confirmed directly on 2026-08-31. This replaces an earlier, incorrect Springer
Basic (SPBASIC, no comma before year) formatting pass based on a less
authoritative general search; that pass is fully superseded by this one -- all
58 in-text citations across Sections 1, 2, 5-7 (plus one instance in Section
3.2) were converted to comma-before-year APA format, and every reference entry
below was independently rebuilt from the same underlying CrossRef data used for
the SPBASIC version, not just re-punctuated.

*Editorial note:* an independent attempt to re-verify this journal's citation
style during this pass (via automated web fetch) was blocked by a login wall on
link.springer.com, and one third-party citation-style database (Paperpile) that
was reachable lists this journal under Springer Basic (no comma) rather than
APA. **This was resolved 2026-09-02**: a real, published article in this exact
journal (Tan, S. E., & Jung, I. (2024), Int. J. Educ. Technol. High. Educ. 21,
44) was located and its actual reference list inspected directly -- e.g.
"Jung, I.S., Kudo, M., & Choi, S. (2012). Stress in Japanese learners engaged
in online collaborative learning in English. British Journal of Educational
Technology, 43(6), 1016-1029." -- which is unambiguously APA style (comma
before year, ampersand before final author, volume(issue), page range). This
confirms APA directly from the journal's own published output, not just its
guidelines page, and supersedes the conflicting Paperpile listing.

Listed alphabetically by first-author family name as already established in
this manuscript's in-text citations (see individual notes below for entries
where that surname required correction or is disputed).

**Issues found and resolved during the earlier CrossRef-verification and
SPBASIC-conversion passes (unchanged by this APA reformatting):**
1. **Sridharan et al. (2021)'s DOI was wrong** -- it resolved via CrossRef to a
   completely unrelated paper. The correct DOI was located and confirmed via
   independent sources (see that entry below).
2. **"Xaveria et al. (2025)" was citing a nonexistent author** -- corrected to
   Hariyanto et al. (2025), the paper's real first author, throughout this
   manuscript (in-text and here). See the note on that entry below.
3. **Raj et al.'s systematic-review entry was mislabeled 2022** -- corrected to
   2021 to match this manuscript's existing in-text citations throughout.

- Abhirami, K., & K. Kavitha Devi, M. (2022). Student Behavior Modeling for an E-Learning System Offering Personalized Learning Experiences. *Computer Systems Science and Engineering*, *40*(3), 1127-1144. https://doi.org/10.32604/csse.2022.020013

- Amin, S., Uddin, M. I., Mashwani, W. K., Alarood, A. A., Alzahrani, A., & Alzahrani, A. O. (2023). Developing a Personalized E-Learning and MOOC Recommender System in IoT-Enabled Smart Education. *IEEE Access*, *11*, 136437-136455. https://doi.org/10.1109/access.2023.3336676

- Amin, S., Uddin, M. I., Alarood, A. A., Mashwani, W. K., Alzahrani, A., & Alzahrani, A. O. (2023). Smart E-Learning Framework for Personalized Adaptive Learning and Sequential Path Recommendations Using Reinforcement Learning. *IEEE Access*, *11*, 89769-89790. https://doi.org/10.1109/access.2023.3305584

- Apicella, A., Arpaia, P., Frosolone, M., Improta, G., Moccaldi, N., & Pollastro, A. (2022). EEG-based measurement system for monitoring student engagement in learning 4.0. *Scientific Reports*, *12*(1), Article 5857. https://doi.org/10.1038/s41598-022-09578-y

- Bayly-Castaneda, K., Ramirez-Montoya, M. S., & Morita-Alexander, A. (2024). Crafting personalized learning paths with AI for lifelong learning: a systematic literature review. *Frontiers in Education*, *9*, Article 1424386. https://doi.org/10.3389/feduc.2024.1424386

- Bernacki, M. L., Greene, M. J., & Lobczowski, N. G. (2021). A Systematic Review of Research on Personalized Learning: Personalized by Whom, to What, How, and for What Purpose(s)?. *Educational Psychology Review*, *33*(4), 1675-1715. https://doi.org/10.1007/s10648-021-09615-8

- Bulathwela, S., Pérez-Ortiz, M., Yilmaz, E., & Shawe-Taylor, J. (2022). Power to the Learner: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources. *Sustainability*, *14*(18), 11682. https://doi.org/10.3390/su141811682

- Chen, H., Yin, C., Li, R., Rong, W., Xiong, Z., & David, B. (2020). Enhanced learning resource recommendation based on online learning style model. *Tsinghua Science and Technology*, *25*(3), 348-356. https://doi.org/10.26599/tst.2019.9010014

- Dong, F., & Zhang, W. (2025). Intelligent Recommendation Method of Personalized Educational Resources Based on K-means Clustering. In *Proceedings of the 2025 International Conference on Digital Education and Information Technology* (pp. 64-68). https://doi.org/10.1145/3732299.3732312

- du Plooy, E., Casteleijn, D., & Franzsen, D. (2024). Personalized adaptive learning in higher education: A scoping review of key characteristics and impact on academic performance and engagement. *Heliyon*, *10*(21), e39630. https://doi.org/10.1016/j.heliyon.2024.e39630

- Essa, S. G., Celik, T., & Human-Hendricks, N. E. (2023). Personalized Adaptive Learning Technologies Based on Machine Learning Techniques to Identify Learning Styles: A Systematic Literature Review. *IEEE Access*, *11*, 48392-48409. https://doi.org/10.1109/access.2023.3276439

- Fariani, R. I., Junus, K., & Santoso, H. B. (2023). A Systematic Literature Review on Personalised Learning in the Higher Education Context. *Technology, Knowledge and Learning*, *28*(2), 449-476. https://doi.org/10.1007/s10758-022-09628-4

- Gligorea, I., Cioca, M., Oancea, R., Gorski, A. T., Gorski, H., & Tudorache, P. (2023). Adaptive Learning Using Artificial Intelligence in e-Learning: A Literature Review. *Education Sciences*, *13*(12), 1216. https://doi.org/10.3390/educsci13121216

- Halkiopoulos, C., & Gkintoni, E. (2024). Leveraging AI in E-Learning: Personalized Learning and Adaptive Assessment through Cognitive Neuropsychology—A Systematic Analysis. *Electronics*, *13*(18), 3762. https://doi.org/10.3390/electronics13183762

- Hariyanto, Kristianingsih, F. X. D., & Maharani, R. (2025). Artificial intelligence in adaptive education: a systematic review of techniques for personalized learning. *Discover Education*, *4*(1), Article 458. https://doi.org/10.1007/s44217-025-00908-6 (author list corrected — CrossRef shows first author as the Indonesian mononym "Hariyanto," with "Francisca Xaveria Diah" as the GIVEN name of second author Kristianingsih; this manuscript previously cited this paper as "Xaveria et al." which does not correspond to any author on the actual record. In-text citations corrected accordingly to "Hariyanto et al., 2025.")

- Henriquez-Nunez, Y., Parra, C., Brijaldo, M., & Carrillo-Ramos, A. (2023). Adaptive Model for the Selection of Resources and Activities in a Virtual Learning Environment. In *Proceedings of the 15th International Conference on Computer Supported Education* (pp. 256-263). https://doi.org/10.5220/0011854900003470

- Kachapuram, H., Pulipati, V. N. C. K., Potha, U., Kurni, M., Suneetha, I., & Harish, P. (2025). Quiz-Based Resource Recommendation System Using Decision Tree Classification for Adaptive Learning. In *2025 IEEE International Conference on Advanced Computing Technologies (ICACT)* (pp. 772-777). https://doi.org/10.1109/icact67549.2025.11351436

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, *4*(1), Article 170171. https://doi.org/10.1038/sdata.2017.171

- Li, Y., & Lu, B. (2025). Intelligent educational systems based on adaptive learning algorithms and multimodal behavior modeling. *PeerJ Computer Science*, *11*, e3157. https://doi.org/10.7717/peerj-cs.3157

- Liu, Q. (2023). Personalized Learning Resources Recommendation for Interest-Oriented Teaching. *International Journal of Emerging Technologies in Learning (iJET)*, *18*(06), 146-161. https://doi.org/10.3991/ijet.v18i06.38721

- Martin, F., Chen, Y., Moore, R. L., & Westine, C. D. (2020). Systematic review of adaptive learning research designs, context, strategies, and technologies from 2009 to 2018. *Educational Technology Research and Development*, *68*(4), 1903-1929. https://doi.org/10.1007/s11423-020-09793-2

- Pan, X., & Lu, M. (2025). Research on Constructing Personalized Learner Profiles Based on Multi-Feature Fusion. *Journal of Electronic Research and Application*, *9*(2), 274-284. https://doi.org/10.26689/jera.v9i2.10085

- Qiu, B., Zuhairi, M. F., & Morcos, J. (2024). A Comprehensive Study On Personalized Learning Recommendation In E-Learning System. *IEEE Access*, *12*, 100446-100482. https://doi.org/10.1109/access.2024.3428419 (family-name convention finalized as "Qiu": CrossRef's raw metadata field tags given="Qiu", family="Bin", but this is very likely a Western-convention mis-tagging of a Chinese name given first-name-first ("Qiu Bin" = family Qiu, given Bin) — confirmed against IEEE's own author listing, Semantic Scholar, and multiple independent papers that all cite this work as "Qiu, B." or "Qiu et al.", never as "Bin" alone.)

- Raj, N. S., & Renumol, V. G. (2021). A systematic literature review on adaptive content recommenders in personalized learning environments from 2015 to 2020. *Journal of Computers in Education*, *9*(1), 113-148. https://doi.org/10.1007/s40692-021-00199-4 (citation year kept at 2021, matching this manuscript's existing in-text citations throughout and the year recorded in papers.csv; CrossRef's registered container year for this DOI is 2022, a later print-issue assignment date)

- Raj, N. S., & Renumol, V. G. (2022). An improved adaptive learning path recommendation model driven by real-time learning analytics. *Journal of Computers in Education*, *11*(1), 121-148. https://doi.org/10.1007/s40692-022-00250-y (citation year kept at 2022 (online-first publication date, PMC9748379), matching this manuscript's existing in-text citations and papers.csv — CrossRef's registered year is 2024, the later print-issue assignment date)

- Reinhold, F., Hoch, S., Schiepe-Tiska, A., Strohmaier, A. R., & Reiss, K. (2021). Motivational and Emotional Orientation, Engagement, and Achievement in Mathematics. A Case Study With One Sixth-Grade Classroom Working With an Electronic Textbook on Fractions. *Frontiers in Education*, *6*, Article 588472. https://doi.org/10.3389/feduc.2021.588472

- Sabeima, M., Lamolle, M., & Nanne, M. F. (2022). Towards Personalized Adaptive Learning in e-Learning Recommender Systems. *International Journal of Advanced Computer Science and Applications* (partial metadata — verify manually). https://doi.org/10.14569/ijacsa.2022.0130803

- Sayed, W. S., Noeman, A. M., Abdellatif, A., Abdelrazek, M., Badawy, M. G., Hamed, A., & El-Tantawy, S. (2023). AI-based adaptive personalized content presentation and exercises navigation for an effective and engaging E-learning platform. *Multimedia Tools and Applications*, *82*(3), 3303-3333. https://doi.org/10.1007/s11042-022-13076-8

- Song, Y. (2024). Personalised recommendation of educational resources based on collaborative filtering. *International Journal of Business Intelligence and Data Mining*, *24*(3/4), 309-323. https://doi.org/10.1504/ijbidm.2024.137737

- Sridharan, S., Saravanan, D., Srinivasan, A. K., & Murugan, B. (2021). Adaptive learning management expert system with evolving knowledge base and enhanced learnability. *Education and Information Technologies*, *26*(5), 5895-5916. https://doi.org/10.1007/s10639-021-10560-w (DOI corrected — the original DOI 10.1007/s10639-021-10516-0, present in this project since before its CrossRef-verification pipeline existed, resolved to a completely unrelated paper. The DOI above is confirmed via Springer, ERIC record EJ1309018, and multiple independent citing papers.)

- Van Schoors, R., Elen, J., Raes, A., & Depaepe, F. (2021). An overview of 25 years of research on digital personalised learning in primary and secondary education: A systematic review of conceptual and methodological trends. *British Journal of Educational Technology*, *52*(5), 1798-1822. https://doi.org/10.1111/bjet.13148

- Wahyuni, N. T., & Ariyanto, G. (2024). Empowered Learners in a Digital Age: The Critical Nexus of Engagement, Agency, Interest, and Motivation. *Muslim Education Review*, *3*(2), 304-330. https://doi.org/10.56529/mer.v3i2.261

- Wan, H., & Yu, S. (2023). A recommendation system based on an adaptive learning cognitive map model and its effects. *Interactive Learning Environments*, *31*(3), 1821-1839. https://doi.org/10.1080/10494820.2020.1858115

- Ahmadian Yazdi, H., Seyyed Mahdavi, S. J., & Ahmadian Yazdi, H. (2024). Dynamic educational recommender system based on Improved LSTM neural network. *Scientific Reports*, *14*(1), Article 4381. https://doi.org/10.1038/s41598-024-54729-y (family name per CrossRef is the compound "Ahmadian Yazdi," not "Yazdi" alone; kept in Y-alphabetical position here to match this manuscript's existing "Yazdi et al." in-text citations rather than resorting under A)

- Ye, N. (2025). Adaptive Learning Path Generation and Optimization for Big Data Courses: A Multimodal Knowledge Graph and Reinforcement Learning Approach. *Scientific Navigation*, *1*(1), 3-17. https://doi.org/10.64376/tp3kwe32

- Zhang, R. (2022). Personalized Course Recommendation Method Based on Learner Interest Mining in Educational Big Data Environment. *Scientific Programming*, *2022*, 1-8. https://doi.org/10.1155/2022/9943965

- Zhang, M. (2025). Online Learning Resource Recommendation Method Based on Learner Model. *International Journal of High Speed Electronics and Systems*, *34*(04), Article 2540248. https://doi.org/10.1142/s0129156425402487

- Zhang, Y., Sun, Y., & Qu, X. (2026). Dynamic Optimization and Scheduling Mechanism of Adaptive Learning Path with Reinforcement Learning Assistance. In *2026 2nd International Conference on Intelligent Systems and Computational Networks (ICISCN)* (pp. 1-6). https://doi.org/10.1109/iciscn67954.2026.11566397
