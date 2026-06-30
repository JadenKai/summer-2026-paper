# Summer Research 2026 — Rubric/Schema-Based LLM Essay Grading

A research project investigating whether large language models can grade essays
**accurately, faster, and transparently** using a **schema-driven, role-based rubric**
system — with an **MLA text-parser** front end that supplies clean structure before scoring.

> **New to this project? Start here**, then read [`papers/00-overview.md`](papers/00-overview.md)
> (literature map) and [`draft/rubric-grading-draft.md`](draft/rubric-grading-draft.md)
> (the working paper). This README is the single source of truth for project state and decisions.

---

## Research questions

1. **Better/faster?** — Is AI-assisted grading actually *more accurate* **and** *faster* than
   human-only grading? (Note: "faster" is widely assumed but rarely measured — a gap we target.)
2. **Schema + role-based grading** — Encode the rubric as a **typed schema** (domain + rubric
   details) and score via **role-based prompting** with structured output, *not* reliant on
   free-form natural language.
3. **Citing components** — How well can an LLM **identify and cite the components** of an essay
   (thesis, claims, evidence, rebuttal, conclusion)?
4. **MLA text parser** — Does a deterministic **MLA preprocessor** (structured input) improve
   downstream scoring and component identification?

---

## Repository layout

```
README.md                       # this file — project overview & state
datasets.txt                    # dataset shortlist: free vs paid (with prices) + task mapping
draft/
  rubric-grading-draft.md       # working paper draft — generic placeholders, 10 models (M1–M10)
papers/
  00-overview.md                # cross-paper map to the 4 research questions (+ coverage matrix)
  links.txt                     # the 7 paper URLs
  <7 folders>/                  # each: original PDF + concise summary.txt
models/                         # Python venv w/ Hugging Face tooling — TOOLING, not project source
```

---

## Literature (7 papers)

Full detail + the Q1–Q4 coverage matrix is in [`papers/00-overview.md`](papers/00-overview.md).
Each folder has the PDF + a one-page `summary.txt`.

| Folder | Paper (short) | Strongest link |
|---|---|---|
| `agreement-synthesis` | Li et al. — 65-study PRISMA synthesis of LLM–human agreement | Q1 |
| `floden-human-vs-ai` | Flodén (BERJ 2025) — ChatGPT vs teachers on real exams | Q1 |
| `detailed-rubric` | Yoshida — how much rubric detail is needed? | Q2 |
| `autoscore` | AutoSCORE — structured-JSON component agent + scorer | Q2 / Q3 |
| `roundtable-res` | RES — multi-agent role-based + dialectical scoring | Q2 |
| `gradehitl-human-in-loop` | GradeHITL — human-in-the-loop rubric refinement | Q1 (assisted) |
| `argument-mining-small-llms` | Favero et al. — argument component mining in essays | Q3 / Q4 |

**Key findings & the gaps we target:**
- LLM–human agreement is **highly context-dependent** (QWK ranged 0.00–0.97 across studies);
  the operational bar QWK ≥ 0.70 is frequently unmet → no clean "yes" to Q1.
- **Speed/cost is almost never measured** — only RES reports timing, and it's *slower* per
  essay. Rigorous latency/cost reporting is an open contribution (Q1).
- **Segmentation is the bottleneck** for component identification (type-classification F1 drops
  ~0.80 → ~0.51 when the model must segment the text itself). Clean upstream structure is what
  the **MLA parser** provides → the most novel piece (Q4); no paper centers on it.

---

## Proposed method

Four-stage pipeline (see [`draft/rubric-grading-draft.md`](draft/rubric-grading-draft.md) §3):

```
MLA essay (raw)
  → [1] MLA Text Parser     → structured essay (sections, paragraphs, citations)
  → [2] Schema Builder      → typed rubric schema (aspect + ordinal/boolean/count/span)
  → [3] Role-Based Scorer   → constrained structured (JSON) output: per-aspect scores + rationale
  → [4] Component Citer     → components mapped to rubric aspects, traceable to cited spans
```

Evaluated across **10 models (M1–M10)** spanning size / access / context classes, with ablations:
**schema vs free-text rubric**, and **with vs without the MLA parser**. Metrics: QWK, Accuracy,
Cohen's κ, Component-F1, **latency (s/essay)**, **cost ($/essay)**.

---

## Datasets

Full shortlist with prices, licenses, and links: [`datasets.txt`](datasets.txt).

**Recommended free core combo** (one dataset per question, no redundancy):

| Dataset | Why | Question |
|---|---|---|
| **DREsS** | Purpose-built rubric scoring; 3 rubric dims (content/org/language) → maps to schema fields | Q2 |
| **PERSUADE / Feedback-Prize** | Only set with span-level argument-component + effectiveness labels | Q3 / Q4 |
| **ASAP 2.0** | Large (~24k), modern, single 1–6 holistic scale → clean 10-model comparison | Q1 |
| *ELLIPSE (optional)* | 6 analytic traits + demographics → generalization / fairness check | Q2 |

**Notes:**
- All four above are **free** (CC BY / MIT / open; DREsS needs a consent form).
- **TOEFL11 is the only paid option** (LDC; per-corpus fee gated behind login, or LDC membership
  from **$2,400** non-profit) — free if your institution is already an LDC member. Prefer ELLIPSE.
- **No public corpus ships in raw MLA format.** The Q4 evaluation needs a small consented MLA set
  *or* synthesized MLA scaffolding over PERSUADE/ASAP essays (controlled with/without-parser ablation).

---

## Environment

- `models/` is a **Python virtualenv** with Hugging Face tooling (`hf`, `tiny-agents`, etc.) —
  it is tooling for pulling models/datasets, **not** project source (has its own `.gitignore`).
- Activate: `source models/bin/activate` (then `hf`, `python`, etc.).

---

## Status & next steps

- [x] Literature review — 7 papers read, summarized, mapped to Q1–Q4 (`papers/`).
- [x] Paper draft scaffold — generic placeholders, 10 models (`draft/rubric-grading-draft.md`).
- [x] Dataset shortlist — free vs paid with prices (`datasets.txt`).
- [ ] **Select + download datasets** (recommended: DREsS + PERSUADE + ASAP 2.0).
- [ ] **Build the MLA text parser** (raw MLA → structured essay object).
- [ ] **Define the typed rubric schema** + role-based scoring prompts.
- [ ] **Run the 10-model evaluation**; capture QWK / Acc / κ / Component-F1.
- [ ] **Measure latency & cost** per model (the under-reported Q1 contribution).
- [ ] **MLA-parser ablation** (synthesized or curated MLA inputs).
- [ ] **Fill draft placeholders** + replace `[REF-n]` with real citations from `papers/links.txt`.

---

*Last updated: 2026-06-11.*
