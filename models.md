## Design axes (what the roster is built to test)

- **Size ladders within a family** — quality changes are attributable to *scale*, not a confounded
  lab/architecture swap:
  - **Qwen3:** 4B → 8B → 32B → 235B (4-point sweep)
  - **Gemma 3:** 4B → 12B → 27B (3-point sweep)
  - **gpt-oss:** 20b → 120b
- **Reasoning vs. standard** — does explicit reasoning help rubric scoring, and at what cost?
  Dedicated reasoners (Phi-4-reasoning, DeepSeek-V3.2, Qwen3 thinking mode, gpt-oss reasoning-effort
  knob) sit alongside standard models, and **Phi-4-mini (standard) vs Phi-4-reasoning** is a
  within-family reasoning contrast.
- **All-open / self-hostable** — student essays raise FERPA/privacy issues, so an on-prem-deployable
  roster (Apache-2.0 / MIT, plus Gemma terms) keeps the whole study reproducible without API lock-in.
- **Cost/speed spectrum (the under-measured Q1 contribution)** — from Phi-4-mini (3.8B) to
  Qwen3-235B; note the gpt-oss MoEs have tiny *active* params, so they are fast/cheap despite size.

---

## Roster

### Tier 1 — Small (≤ ~9B) — 5
| Model | Params | Lab | License | Why this model |
|---|---|---|---|---|
| Qwen3-4B | 4B dense | Alibaba | Apache-2.0 | Lower rung of the Qwen size ladder; strong-for-size, has a thinking mode |
| Qwen3-8B | 8B dense | Alibaba | Apache-2.0 | Next rung up — isolates the 4B→8B scale step within one family |
| Gemma 3 4B | 4B dense | Google | Gemma terms | Best small **multilingual/multimodal** option; runs on ~8GB (L2/ESL essays) |
| Phi-4-mini | 3.8B dense | Microsoft | MIT | Strong small **standard** model; capability floor; the non-reasoning half of the Phi contrast |
| Ministral 3 8B | 8B dense | Mistral | Apache-2.0 | Non-Qwen efficient 8B point; strong cost/perf (verify current revision) |

### Tier 2 — Medium (~12–34B) — 6
| Model | Params (total / active) | Lab | License | Why this model |
|---|---|---|---|---|
| **gpt-oss-20b** ★ | 21B / 3.6B | OpenAI | Apache-2.0 | Required. Open MoE; tiny active count → fast/cheap; reasoning-effort control (low/med/high) |
| Phi-4-reasoning | 14B dense | Microsoft | MIT | Recent (Apr 2025) **reasoning** model with structured `<think>` traces; pairs with Phi-4-mini for a within-family reasoning contrast |
| Gemma 3 12B | 12B dense | Google | Gemma terms | Middle rung of the Gemma ladder (4B→12B→27B) |
| Mistral Small 3 (24B) | 24B dense | Mistral | Apache-2.0 | Non-Qwen/Gemma medium for family diversity; competitive 24B |
| Qwen3-32B | 32B dense | Alibaba | Apache-2.0 | Upper-medium rung of the Qwen ladder |
| Gemma 3 27B | 27B dense | Google | Gemma terms | Top of the Gemma ladder; strong multilingual mid model |

### Tier 3 — Large (≥ ~100B total) — 4
| Model | Params (total / active) | Lab | License | Why this model |
|---|---|---|---|---|
| **gpt-oss-120b** ★ | 117B / 5.1B | OpenAI | Apache-2.0 | Required. Large open MoE; fits one 80GB H100; small active count keeps it fast |
| DeepSeek-V3.2 | 671B / 37B (reasoning) | DeepSeek | MIT | Adds the major DeepSeek open lab; flagship open **reasoner**; high-capability anchor |
| GLM-4.5-Air | 106B / 12B | Zhipu / Z.ai | MIT | Large MoE that still runs on modest hardware; lab diversity (GLM-4.6 355B is the heavier swap) |
| Qwen3-235B-A22B | 235B / 22B | Alibaba | Apache-2.0 | Top of the Qwen ladder; the study's high-capability ceiling |

★ = explicitly requested (`gpt-oss`).

---

## Why these choices (overall rationale)

1. **All-open is the right call for this project.** Student essays raise FERPA/privacy issues —
   schools often can't send them to a closed API. An all-self-hostable roster (Apache-2.0 / MIT,
   plus Gemma terms) makes the whole study deployable on-prem and reproducible without API lock-in.
2. **Controlled size ladders** (Qwen 4-point, Gemma 3-point, gpt-oss 2-point) let you attribute
   grading-quality changes to **scale** rather than a confounded lab/architecture swap. Answers
   "how small can a self-hosted grader be?"
3. **A clean reasoning axis.** Phi-4-reasoning, DeepSeek-V3.2, Qwen3 thinking mode, and gpt-oss
   reasoning-effort (low/med/high) test whether explicit reasoning improves rubric scoring and at
   what speed cost. **Phi-4-mini vs Phi-4-reasoning** gives a within-family standard-vs-reasoning
   contrast.
4. **Cost/speed spectrum (Q1).** From Phi-4-mini (3.8B) to Qwen3-235B / gpt-oss-120b — and the
   gpt-oss MoEs run with very small active params, so they are fast/cheap despite large total size.
   Lets you make the accuracy-vs-speed tradeoff visible, which the literature rarely measures.
5. **Lab + multilingual diversity.** Seven labs (Alibaba, OpenAI, Google, Microsoft, Mistral,
   DeepSeek, Zhipu); Gemma 3 and Qwen are strong multilingually, matching the L2/ESL skew of AES data.

---

## Caveats
- **Pin exact revisions.** Open-weight repos update in place; record the HF commit/revision per
  model or the study isn't reproducible.
- **License nuance.** Qwen3 / gpt-oss / Mistral = Apache-2.0; Phi-4-mini / Phi-4-reasoning / GLM /
  DeepSeek = MIT; **Gemma 3 = Google's Gemma terms** (open weights, but custom license, not
  OSI-approved) — check before redistribution.
- **Hardware.** Small/medium pull easily via the `models/` venv (`hf`, `tiny-agents`). The large
  models need an 80GB GPU (gpt-oss-120b, GLM-4.5-Air) or multi-GPU / hosted endpoint (Qwen3-235B,
  DeepSeek-V3.2).
- **Reasoning models cost more tokens/latency.** Phi-4-reasoning, DeepSeek-V3.2, and high-effort
  gpt-oss/Qwen runs trade speed for depth — capture that in the Q1 latency/cost numbers, don't hide it.
- **Provider concentration is intentional:** 4 Qwen, 3 Gemma, 2 gpt-oss, 2 Mistral, 2 Phi, 1 GLM,
  1 DeepSeek — the repeats are deliberate size ladders / the reasoning contrast, not redundancy.
- **Prices intentionally omitted** — these are open weights, so cost depends on your host
  (self-host GPU-hours vs. Together / Fireworks / DeepInfra). Measure cost per essay on your actual
  serving setup for the Q1 numbers.

## Swap options
- Tighter Qwen ladder: add **Qwen3-14B** (→ 4B/8B/14B/32B/235B).
- Heavier large tier: **GLM-4.6** (355B/32B) in place of GLM-4.5-Air.
- More reasoning contrast: **Phi-4-reasoning-plus** (the higher-effort sibling).

See [`datasets.txt`](datasets.txt) for the data side and
[`draft/rubric-grading-draft.md`](draft/rubric-grading-draft.md) §4.2 for how the roster slots into
the experimental setup.

## Sources
- gpt-oss model card (120b/20b, Aug 2025, Apache-2.0): https://arxiv.org/pdf/2508.10925 · https://help.openai.com/en/articles/11870455-openai-open-weight-models-gpt-oss
- Phi-4-reasoning (14B, MIT, Apr 2025, `<think>` traces): https://huggingface.co/microsoft/Phi-4-reasoning
- DeepSeek-V3.2 (MIT, reasoning-first): https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- Qwen3 sizes/licensing: https://github.com/QwenLM/qwen3
- Gemma 3 sizes: https://huggingface.co/blog/daya-shankar/open-source-llms
- GLM-4.5-Air: https://huggingface.co/zai-org/GLM-4.5-Air
