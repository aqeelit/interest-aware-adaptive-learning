# Target Journal Selection

Status: SUPERSEDED 2026-09-02 -- see "CURRENT selection" below. Original ETHE
selection kept in full underneath for history/reference.

## CURRENT selection (2026-09-02): Journal of Learning Analytics (JLA)
- Publisher: UTS ePress, for the Society for Learning Analytics Research (SoLAR)
- No submission or processing charges whatsoever (confirmed via official FAQ)
- Scopus/SJR: Q1 in Computer Science Applications, Developmental and Educational
  Psychology, Artificial Intelligence, Education. SJR ~0.997, Impact Factor ~3.6-3.8
  (notably lower/less competitive than ETHE's 16-22, by design)
- ISSN: 1929-7750 (online)
- Reason for switch: ETHE's acceptance rate was found to be ~3% (2023 data) with a
  very high, rapidly-rising impact factor -- extremely competitive for this paper's
  technical scope (single institution, logistic regression, modest effect size).
  JLA is topically a much closer match: it is the flagship journal of the learning
  analytics community specifically, and this paper's actual toolkit (OULAD,
  ASSISTments, McNemar's test, ablation design, subgroup fairness analysis) is
  exactly the kind of methodologically-rigorous-but-not-ML-flashy work this
  community values, per direct review of the field's own literature during Phase 3.

## JLA submission mechanics (verified 2026-09-02 via official author guidelines)
- Reference style: APA 7th edition, using JLA's own APA7 template. Comma before
  year -- same convention already applied to the manuscript for ETHE, transferred
  directly.
- Double-blind peer review, but structured DIFFERENTLY from ETHE: there is NO
  separate title-page file. The single manuscript document's own first page must
  contain ONLY title, abstract, and keywords -- no author name/affiliation
  anywhere in the file at all. Author identity (name, affiliation, contact) is
  entered exclusively into the OJS online-submission-system fields, not into any
  uploaded document.
- Word limit: <=8,000 words for research papers (excludes abstract, keywords,
  reference list; text embedded in tables/figures DOES count). This manuscript's
  body (~6,450-6,711 words) is comfortably within range.
- Abstract limit: <=200 words (per the official LaTeX template) -- stricter than
  ETHE's 250. Manuscript's abstract was re-tightened to 199 words for JLA.
- "Notes for Practice" section is MANDATORY for all research paper submissions,
  immediately after the abstract/keywords: bullet points aimed at practitioners,
  summarizing established knowledge, what the study adds, and practical
  implications. Added to the JLA-formatted manuscript (5 bullet points).
- Must use JLA's official Word or LaTeX layout template -- the produced docx
  approximates the required structure/content but has NOT been verified against
  the actual downloadable template file (not fetched during this pass). Owner
  should download and cross-check against the real template before submission.
- AI-use disclosure: no JLA-specific written policy was located during this pass.
  Section 3.9 was generalized to a common (COPE/ICMJE-style) disclosure with an
  explicit note flagging that this needs direct verification against any
  JLA-specific policy before submission.
- If submitting as an extension of a prior conference paper: requires 25-30% new
  contribution and an explanatory cover letter. Not applicable here.

## Final deliverable for JLA
manuscript/../[outputs]/Manuscript_for_JLA_submission.docx (delivered to owner
2026-09-02): single blinded document, no author name anywhere, Title + Abstract
(199 words) + Keywords + Notes for Practice + Sections 1-7 + Declarations (data
availability only, non-identifying) + References (APA7). Author's name,
affiliation, and contact are NOT in this file -- owner must enter them directly
into JLA's OJS submission-system fields at upload time:
  Author: Aqeel M. Ali Hussein
  Affiliation: Adptivio EdTech Research Lab, Al-Hussein University College, Iraq

## Previous selection (2026-08-30, superseded): ETHE
**International Journal of Educational Technology in Higher Education (ETHE)**
- Publisher: Springer (fully sponsored by Universitat Oberta de Catalunya — no APC,
  genuine Diamond Open Access)
- ISSN: 2365-9440
- Scopus/SJR quartile: Q1 in Education, Computer Science Applications, and
  E-learning specifically
- Scope fit: publishes adaptive/personalized learning systems and AI-in-higher-education
  research regularly — directly matches this paper's topic

## Why this journal (verified 2026-08-30 via web search, not assumed)
- Zero cost to the author — appropriate given no institutional APC budget confirmed
  for this project.
- Q1 specifically in the "E-learning" SJR category, not just broad "Education" —
  closest topical match found among free-to-publish options.
- Springer Nature's general AI-use policy (which governs this journal) is a
  documented, moderate-disclosure policy that this project's workflow was already
  designed to satisfy from Phase 0 onward (see CLAUDE.md's AI disclosure rules).

## Springer Nature AI policy — key points governing this manuscript (verified 2026-08-30)
1. AI tools cannot be listed as authors; the human author bears full accountability
   for content, sources, and results.
2. Disclosure of generative-AI use IS required in the Methods section (or a suitable
   alternative section) whenever AI was used for content generation, drafting,
   rewriting, summarizing, or had any creative/editorial role — this covers this
   project's use of AI for code generation, research-design assistance, and drafting.
3. Disclosure is NOT required for minor copy-editing (grammar/spelling/tone/formatting)
   alone — not directly relevant here since AI use in this project goes well beyond that.
4. AI-generated images/figures are prohibited UNLESS they are a documented part of the
   research methodology itself. This project's figures are all matplotlib outputs
   generated from verified real data by code the author reviewed and ran — not
   AI-generated imagery — so they are permissible, but the pipeline's AI-assisted
   construction should still be named in Methods for full transparency.
5. Authors must be able to provide detailed evidence of AI use upon request by
   editors — this is exactly what manuscript/ai_disclosure_log.md exists to provide.

## Submission mechanics (verified 2026-09-02, critical for file preparation)
- **This journal uses DOUBLE-BLIND peer review** (confirmed via ETHE's own
  peer-review-policy page and an independent published analysis of the journal's
  20-year history). This means:
  - The reviewable manuscript file must NOT contain the author's name, affiliation,
    or any other identifying detail (including self-identifying links, e.g. a
    GitHub username derived from the author's name).
  - A SEPARATE title page file must be submitted containing: title, author name(s),
    affiliation(s), corresponding-author contact, and the full Declarations
    (Availability of data and materials, Competing interests, Funding, Authors'
    contributions, Acknowledgements, Authors' information).
  - manuscript/manuscript.md's single combined draft is NOT submission-ready as-is
    for this reason — see the two split Word files described in
    manuscript/ai_disclosure_log.md's 2026-09-02 entries for the corrected version.
- Reference style: APA (comma before year), confirmed both via the official
  guidelines page and by inspecting a real published article's reference list in
  this exact journal — not Springer Basic (SPBASIC), despite one third-party
  database (Paperpile) listing it that way.
- Word count: 4,500-8,000 words for the main text (Introduction-Conclusion).
  Abstract: 150-250 words.

## Backup candidates (not yet selected, for reference only if ETHE desk-rejects on scope)
Not researched in depth — to be investigated only if needed.
