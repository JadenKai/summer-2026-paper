# Literature Overview — AI-Assisted Essay/Exam Grading

Seven papers, each in its own folder with a one-page `summary.txt`. This file maps them
to the project's four research questions and synthesizes the evidence.

## Research questions
1. **Better/faster?** — Is AI-assisted grading actually better and faster than human-only grading?
2. **Schema/role-based grading** — schema-based grading that encodes domain + rubric details,
   using role-based prompting, not reliant on free natural language.
3. **Citing components** — How good is an LLM at citing/identifying the components of an essay?
4. **MLA text parser** — a preprocessor that formats an MLA essay into structured input for the LLM.

## Coverage map
`●` = primary evidence · `○` = secondary · blank = not addressed

| Folder | Paper (short) | Q1 | Q2 | Q3 | Q4 |
|---|---|:--:|:--:|:--:|:--:|
| `agreement-synthesis` | Li et al. — 65-study research synthesis | ● | ○ | | |
| `floden-human-vs-ai` | Flodén — ChatGPT vs teachers (BERJ 2025) | ● | ○ | | |
| `detailed-rubric` | Yoshida — do we need a detailed rubric? | ○ | ● | | |
| `autoscore` | AutoSCORE — structured JSON component agent | | ● | ○ | |
| `roundtable-res` | RES — multi-agent role/dialectical scoring | ○ | ● | | |
| `gradehitl-human-in-loop` | GradeHITL — human-in-the-loop rubric refine | ● | ○ | | |
| `argument-mining-small-llms` | Favero et al. — argument mining in essays | | ○ | ● | ● |

## Synthesis by question

**Q1 — Better/faster than human-only?**
The honest answer is "not consistently, and 'faster' is barely measured." The 65-study
synthesis (`agreement-synthesis`) shows LLM–human agreement is highly context-dependent
(QWK 0.00–0.97; the operational bar of QWK≥0.70 is frequently unmet). Flodén's direct
head-to-head (`floden-human-vs-ai`) confirms it on real exams: plausible-looking grades
but only 30% exact-grade agreement, a pull toward middle scores, and ~40–45% of grades
changing on re-runs. The "assisted" framing fares better than full automation: GradeHITL
(`gradehitl-human-in-loop`) shows human-in-the-loop rubric refinement beating every
fully-automated baseline. **Across all of these, speed/latency is asserted, not quantified**
— RES (`roundtable-res`) is the only one with timing, and it is ~170× *slower* per essay
than a single prompt. So "better" often costs "slower."

**Q2 — Schema-based / role-based grading.**
Strong support. AutoSCORE (`autoscore`) is the closest match to the project's idea: a
component-extraction agent emits a typed JSON schema (Boolean/count/text-span per rubric
dimension) and a separate scoring agent consumes it — yielding +16–74% QWK, most on weak
models. RES (`roundtable-res`) supplies ready-to-adapt role-based JSON prompt templates
(persona → rubric → rationale-scoring → moderator). Yoshida (`detailed-rubric`) tells you
how much schema detail to encode: a lean rubric is usually as accurate as an elaborate one
and far cheaper in tokens — but behavior is model-specific, so validate per model.

**Q3 — Citing components of an essay.**
Favero et al. (`argument-mining-small-llms`) is the direct evidence: identifying/classifying
essay components reaches ~0.80 F1 *given clean segmentation* but drops to ~0.51 end-to-end,
and quality assessment is weak (~0.45). Models cite conclusions/evidence well, rebuttals/
counterclaims poorly. AutoSCORE's extraction step is a second, schema-driven take on the
same "pull out the components" task.

**Q4 — MLA text parser.**
No paper centers on a parser — this is largely the project's novel contribution. The
justification comes from `argument-mining-small-llms`: **segmentation is the dominant error
source** (B-token/start precision only 66%) and boundary errors cascade into classification.
Supplying clean, pre-segmented structure (what an MLA parser would do) is exactly what
separates the ~0.51 end-to-end regime from the ~0.80 gold-segmentation regime.

## Where the project can contribute
- **Q1:** measure *speed* rigorously — every paper assumes it; almost none reports it.
- **Q4:** the MLA-formatting / segmentation preprocessor is essentially unexplored, yet the
  argument-mining results show it targets the single biggest bottleneck in component-level
  grading. This is the most defensible novel piece.
