<!--
DRAFT — research paper scaffold. All model names, numbers, datasets, and citations are
GENERIC PLACEHOLDERS to be filled in later.

Placeholder conventions:
  - Models:        M1 … M10, each tagged with a generic spec, e.g. <small / open-weights>.
  - Datasets:      <DATASET-A>, <DATASET-B>.
  - Numeric cells: XX.X  (use 0.XX for kappa-style metrics).
  - Citations:     [REF-n]  (see References section).
  - Open items:    "> TODO: …" blockquotes.
-->

# <TITLE: e.g., "Schema-Driven, Role-Based Rubric Grading of Essays: A 10-Model Study">

**<Author Name>** · **<Affiliation>** · **<Contact email>**

> TODO: finalize title, author list, and venue/format (Markdown draft → ACL/IEEE template later).

---

## Abstract

Automated essay grading with large language models (LLMs) is increasingly proposed as a
faster, cheaper complement to human raters, yet evidence that it is both *more accurate* and
*faster* remains mixed. We present a **schema-driven, role-based rubric grading system** and
evaluate it across **ten models (M1–M10)**. The system (i) preprocesses MLA-formatted essays
with a lightweight **text parser** that produces structured input, (ii) encodes the rubric as a
**typed schema** capturing domain-specific aspects rather than free-form natural-language
instructions, (iii) scores essays via **role-based prompting** with constrained structured
output, and (iv) **identifies and cites the components** of each essay. Across `<DATASET-A>`
and `<DATASET-B>`, the schema-based configuration reaches QWK `XX.X` versus `XX.X` for a
free-text baseline, and component-citing F1 improves from `XX.X` to `XX.X` when the MLA parser
supplies clean segmentation. We further report per-model latency (`XX.X` s/essay) and cost
(`$XX.X`/essay), quantifying an accuracy–speed trade-off that prior work rarely measures.

> TODO: tighten abstract once result placeholders are filled.

**Keywords:** automated essay scoring, rubric-based grading, LLMs, role-based prompting,
structured output, argument component identification.

---

## 1. Introduction

Grading written work is time-consuming and subjective, and the release of capable LLMs has
renewed interest in automating it [REF-1], [REF-2]. Two claims are commonly made about
AI-assisted grading: that it is **more consistent/accurate** and that it is **faster** than
human-only grading. The first claim is contested—reported LLM–human agreement varies widely
and is highly context-dependent [REF-3]. The second claim is almost universally *assumed* but
**rarely measured**: most studies report agreement metrics with no latency or cost data.

A second open problem is *how* the rubric is conveyed to the model. Much prior work passes the
rubric as free natural-language text, which is brittle: small wording changes shift scores, and
domain-specific criteria are easily misread [REF-4]. An alternative is to encode the rubric as a
**structured schema** of typed, domain-aware fields and to elicit scores through **role-based
prompting** with machine-readable output, reducing reliance on free-form generation [REF-5].

Finally, fine-grained feedback requires the model to **locate and cite the components** of an
essay (e.g., thesis, claims, evidence, conclusion). Component identification degrades sharply
when the model must also segment the text itself; clean upstream segmentation is a known
bottleneck [REF-6]. Because student essays in our setting follow **MLA formatting**, a simple
deterministic parser can supply that structure before the model ever scores the text.

**Contributions.**
1. A schema-driven, role-based rubric grading pipeline with an MLA text-parser front end
   (Section 3).
2. A **10-model** comparison (M1–M10) spanning size/access/context-length classes under a
   single protocol (Section 4).
3. Ablations isolating (a) schema vs. free-text rubric prompting and (b) with vs. without the
   MLA parser, plus **latency/cost** measurements that make the accuracy–speed trade-off
   explicit (Section 5).

---

## 2. Related Work

**Automated essay scoring (AES).** Early AES used surface features and supervised models trained
on human-scored essays [REF-1]; transformer encoders later improved in-domain agreement but
generalized poorly across prompts and genres [REF-2].

**LLM rubric scoring and rubric detail.** Zero-shot LLM scoring removes the need for task-specific
training [REF-3]. Recent work questions *how much* rubric detail is needed, finding that concise
rubrics often match elaborate ones at lower token cost, with model-specific exceptions [REF-4].
Our schema design is informed by this: encode enough structure to be unambiguous, no more.

**Role-based and multi-agent scoring.** Assigning the model an explicit evaluator role and
constraining it to structured (e.g., JSON) output improves alignment and interpretability; multi-
agent and dialectical variants further raise agreement at the cost of latency [REF-5], [REF-7].

**Argument mining / component identification.** Locating and typing essay components (claims,
evidence, etc.) is well studied; accuracy is strong given gold segmentation but drops when the
model must segment the text itself, making upstream structure the key bottleneck [REF-6].

> TODO: expand each paragraph to ~4–6 sentences and attach real citations to [REF-n].

---

## 3. System Design

### 3.1 Overview

The pipeline has four stages:

```
MLA essay (raw)
   │
   ▼
[1] MLA Text Parser ──► structured essay object (sections, paragraphs, citations)
   │
   ▼
[2] Schema Builder ───► typed rubric schema (domain aspects + score fields)
   │
   ▼
[3] Role-Based Scorer ► constrained structured output (per-aspect scores + rationale)
   │
   ▼
[4] Component Citer ──► identified/cited components mapped to rubric aspects
```

> TODO: replace with a proper figure (Figure 1) once drafting is done.

### 3.2 MLA Text Parser

A deterministic preprocessor converts a raw MLA-formatted essay into a structured object before
any model call. It extracts the **header block** (`<author>`, `<instructor>`, `<course>`,
`<date>`), the **title**, ordered **paragraphs**, and **in-text citations / Works Cited**
entries, emitting:

```json
{
  "header": { "author": "<...>", "course": "<...>", "date": "<...>" },
  "title": "<...>",
  "paragraphs": [ { "id": 0, "role": "<intro|body|conclusion>", "text": "<...>" } ],
  "citations": [ { "id": 0, "span": "<...>", "source_ref": "<...>" } ]
}
```

Keeping segmentation deterministic (rather than asking the model to infer it) is intended to
remove the dominant error source in downstream component identification [REF-6].

> TODO: specify parser rules for non-conforming MLA input and a fallback path.

### 3.3 Schema-Based Rubric Representation

Each rubric criterion is expressed as a **typed field** rather than a sentence of instructions.
A field carries a domain aspect, a value type, and a score scale:

```json
{
  "aspect": "<thesis_clarity>",
  "type": "<ordinal>",            // ordinal | boolean | count | span
  "scale": [0, 1, 2, 3, 4],
  "domain_note": "<discipline-specific expectation placeholder>",
  "evidence_fields": ["<span_ids>"]
}
```

Types mirror the structure of the criterion: ordinal for graded quality, boolean for
presence/absence, count for enumerable elements, and span for text the model must point to.
This makes the rubric machine-checkable and reduces reliance on free-text interpretation.

### 3.4 Role-Based Prompting

The scorer is given an explicit evaluator **role** and must return **structured output** keyed to
the schema; prose is confined to a `rationale` field and is never the score carrier.

```
SYSTEM: You are <ROLE: e.g., a strict rubric grader for <discipline> essays at <level>>.
        Score only against the provided schema. Output valid JSON. Do not add fields.

USER:   Rubric schema: <schema_json>
        Structured essay: <parsed_essay_json>
        For each aspect: assign a score on its scale and cite evidence span_ids.

OUTPUT: {
  "scores": [ { "aspect": "<...>", "score": <int>, "evidence": ["<span_ids>"],
               "rationale": "<short text>" } ],
  "holistic": <int>
}
```

> TODO: add the moderator/aggregation variant if multi-agent scoring is included.

### 3.5 Component Identification and Citing

Using the parsed spans, the model labels each component (`<thesis>`, `<claim>`, `<evidence>`,
`<counterclaim>`, `<conclusion>`, …) and links it to the rubric aspect(s) it satisfies, so every
score is **traceable to cited text**. Because segmentation is supplied by the parser (§3.2), the
model performs typing/citing rather than typing *and* segmenting.

---

## 4. Experimental Setup

### 4.1 Datasets

- **`<DATASET-A>`** — `<N_A>` essays, `<level>`, `<genre>`, human rubric scores on `<scale>`.
- **`<DATASET-B>`** — `<N_B>` essays, used for `<cross-prompt / generalization>` evaluation.

> TODO: confirm licensing and human-rater reliability for each dataset.

### 4.2 Models

We evaluate ten models under one protocol. Names are generic; specs are placeholders.

| ID  | Size class            | Access            | Context        | Notes                 |
|-----|-----------------------|-------------------|----------------|-----------------------|
| M1  | `<small>`             | `<open-weights>`  | `<ctx>`        | `<...>`               |
| M2  | `<small>`             | `<open-weights>`  | `<ctx>`        | `<...>`               |
| M3  | `<mid>`               | `<open-weights>`  | `<ctx>`        | `<...>`               |
| M4  | `<mid>`               | `<open-weights>`  | `<ctx>`        | `<...>`               |
| M5  | `<mid>`               | `<proprietary>`   | `<ctx>`        | `<...>`               |
| M6  | `<large>`             | `<proprietary>`   | `<ctx>`        | `<...>`               |
| M7  | `<large>`             | `<proprietary>`   | `<ctx>`        | `<...>`               |
| M8  | `<large>`             | `<open-weights>`  | `<ctx>`        | `<...>`               |
| M9  | `<reasoning>`         | `<proprietary>`   | `<ctx>`        | `<...>`               |
| M10 | `<reasoning>`         | `<proprietary>`   | `<ctx>`        | `<...>`               |

### 4.3 Metrics

- **QWK** — quadratic weighted kappa vs. human scores (primary agreement metric).
- **Accuracy** — exact-match score agreement.
- **Cohen's κ** — chance-corrected agreement.
- **Component-F1** — macro-F1 of component identification/citing.
- **Latency** — seconds/essay.
- **Cost** — `$`/essay.

### 4.4 Protocol and Conditions

All models run with temperature `<T>` and fixed prompts; each essay scored `<k>` times to assess
self-consistency. Conditions:

- **C1 — Free-text rubric** (baseline): rubric passed as natural-language prose.
- **C2 — Schema rubric** (ours, §3.3): typed-field schema, role-based structured output.
- **C3 — Schema + MLA parser** (full system): C2 with deterministic segmentation (§3.2).

> TODO: state hardware/API conditions used for latency/cost measurement.

---

## 5. Results

> All cells are placeholders (`XX.X` / `0.XX`). Fill from experiment logs.

### 5.1 Per-Model Grading Agreement (full system, C3)

| Model | QWK   | Accuracy | Cohen's κ |
|-------|-------|----------|-----------|
| M1    | 0.XX  | XX.X     | 0.XX      |
| M2    | 0.XX  | XX.X     | 0.XX      |
| M3    | 0.XX  | XX.X     | 0.XX      |
| M4    | 0.XX  | XX.X     | 0.XX      |
| M5    | 0.XX  | XX.X     | 0.XX      |
| M6    | 0.XX  | XX.X     | 0.XX      |
| M7    | 0.XX  | XX.X     | 0.XX      |
| M8    | 0.XX  | XX.X     | 0.XX      |
| M9    | 0.XX  | XX.X     | 0.XX      |
| M10   | 0.XX  | XX.X     | 0.XX      |
| **Mean** | **0.XX** | **XX.X** | **0.XX** |

*Readout placeholder:* Agreement ranged from `0.XX` (M`<x>`) to `0.XX` (M`<y>`); `<larger /
reasoning>` models tended to score higher, but with `<diminishing / inconsistent>` returns.

### 5.2 Ablation — Schema vs. Free-Text Rubric (mean over M1–M10)

| Condition                | QWK   | Accuracy | Cohen's κ |
|--------------------------|-------|----------|-----------|
| C1 — Free-text rubric    | 0.XX  | XX.X     | 0.XX      |
| C2 — Schema rubric       | 0.XX  | XX.X     | 0.XX      |
| Δ (C2 − C1)              | +0.XX | +XX.X    | +0.XX     |

*Readout placeholder:* The schema condition improved mean QWK by `+0.XX`, with the largest gains
on `<smaller>` models and `<more complex>` rubrics.

### 5.3 Component-Citing — Effect of the MLA Parser

| Condition                     | Component-F1 |
|-------------------------------|--------------|
| C2 — Schema, model-segmented  | XX.X         |
| C3 — Schema + MLA parser      | XX.X         |
| Δ                             | +XX.X        |

*Readout placeholder:* Supplying deterministic segmentation raised component-F1 from `XX.X` to
`XX.X`, consistent with segmentation being the dominant error source [REF-6].

### 5.4 Latency and Cost (full system, C3)

| Model | Latency (s/essay) | Cost ($/essay) |
|-------|-------------------|----------------|
| M1    | XX.X              | 0.XXXX         |
| …     | …                 | …              |
| M10   | XX.X              | 0.XXXX         |

*Readout placeholder:* The most accurate configuration was `<NN>×` slower and `<NN>×` more
expensive per essay than the fastest model, making the accuracy–speed trade-off explicit.

---

## 6. Discussion

**Q1 — Is AI-assisted grading better *and* faster?** Our `XX.X` mean QWK `<is/ is not>` within
the human-rater band, and §5.4 shows the highest-agreement setting is materially slower/costlier.
"Better" and "faster" do not necessarily co-occur, and reporting only agreement hides this.

**Q2 — Schema and role-based prompting.** §5.2 indicates that encoding the rubric as a typed
schema and constraining output `<improves / matches>` free-text prompting while reducing
free-form reliance; gains concentrate on `<weaker>` models, suggesting structure substitutes for
capability.

**Q3 — Citing components.** §5.3 shows competent component identification once segmentation is
clean, but `<weaker>` performance on `<counterclaim/rebuttal>`-type components, mirroring prior
findings [REF-6].

**Q4 — MLA parser.** Deterministic MLA segmentation `<closed / narrowed>` the gap between model-
segmented and structurally-supplied input, supporting a parser-as-bottleneck-fix view.

**Per-model variability.** Behavior was not uniform: `<one model>` `<regressed>` under added
structure, underscoring the need for per-model validation rather than one-size-fits-all prompts.

---

## 7. Limitations

- Single rubric family / `<discipline>`; generalization to other genres untested.
- Human scores treated as ground truth despite inter-rater disagreement.
- MLA parser assumes conforming formatting; robustness to malformed input is `<TBD>`.
- Latency/cost depend on `<hardware / API>` conditions and may not transfer.

> TODO: add threats to validity and ethical/deployment considerations.

---

## 8. Conclusion

We described a schema-driven, role-based rubric grading system with an MLA text-parser front end
and evaluated it across ten models. Placeholder results suggest that structured rubrics and clean
upstream segmentation `<improve>` agreement and component citing, while explicit latency/cost
reporting reveals an accuracy–speed trade-off that agreement-only studies obscure. Future work
will `<fill: fine-tuning, human-in-the-loop refinement, broader genres>`.

---

## References

> Replace placeholders with real citations; keep numbering aligned to [REF-n] in the text.

- [REF-1] `<Author>`, "`<Foundations of automated essay scoring>`," `<Venue>`, `<Year>`.
- [REF-2] `<Author>`, "`<Transformer-based essay scoring>`," `<Venue>`, `<Year>`.
- [REF-3] `<Author>`, "`<LLM–human agreement synthesis>`," `<Venue>`, `<Year>`.
- [REF-4] `<Author>`, "`<Rubric detail and token efficiency>`," `<Venue>`, `<Year>`.
- [REF-5] `<Author>`, "`<Structured-component / schema scoring>`," `<Venue>`, `<Year>`.
- [REF-6] `<Author>`, "`<Argument mining / component identification>`," `<Venue>`, `<Year>`.
- [REF-7] `<Author>`, "`<Role-based / multi-agent essay scoring>`," `<Venue>`, `<Year>`.
- [REF-8] `<Author>`, "`<Human-in-the-loop grading>`," `<Venue>`, `<Year>`.
- [REF-9] `<Author>`, "`<Human vs. AI exam grading>`," `<Venue>`, `<Year>`.
- [REF-10] `<Author>`, "`<...>`," `<Venue>`, `<Year>`.
