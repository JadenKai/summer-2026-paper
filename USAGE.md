# Usage

All Python commands run through `uv` so the project's `.venv` (and NixOS `LD_LIBRARY_PATH`
fixup) is used. `./run.sh <args>` is `exec uv run <args>` — the two forms below are
interchangeable; pick whichever you have muscle memory for.

```
./run.sh python src/grade_essays.py ...
uv run python src/grade_essays.py ...
```

---

## `run.sh`

Wrapper that exports the nix-ld `LD_LIBRARY_PATH` fix (needed on NixOS for numpy/pandas
native extensions to load) and then runs `uv run "$@"`.

```
./run.sh <any command>
```

No flags of its own — everything after `run.sh` is passed straight to `uv run`.

---

## `src/grade_essays.py`

Scores a sample of ASAP 2.0 essays with one LLM and reports agreement with human scores
(accuracy, adjacent accuracy, QWK, Cohen's κ, MAE, bias, confusion matrix, timing).

```
uv run python src/grade_essays.py [--n N] [--model MODEL] [--backend {anthropic,ollama,vllm}]
                                   [--seed SEED] [--verbose]
```

| Flag | Default | What it does |
|---|---|---|
| `--n N` | `20` | Number of essays to sample from `dataset/ASAP2_0/train.csv`. |
| `--model MODEL` | backend-dependent | Model to grade with. E.g. `claude-opus-4-8`, `gpt-oss:20b`, or (with `--backend vllm`) the HF repo id / local path the vLLM server is serving. Required when the resolved backend has no default (`ollama_local`, `vllm`). |
| `--backend {anthropic,ollama,vllm}` | auto-detected | Force a backend instead of the default priority (`ANTHROPIC_KEY` set → Anthropic; else `OLLAMA_KEY` set → Ollama Cloud; else local Ollama at `OLLAMA_HOST`). `vllm` is never auto-selected — it must be requested explicitly, and targets `VLLM_HOST` (default `http://localhost:8000`). |
| `--seed SEED` | `42` | Random seed for essay sampling (reproducibility). |
| `--verbose` | off | Print raw model output for unparseable responses. |

Outputs:
- Appends a row of run metrics to `results/runs.csv`.
- Writes per-essay predictions to `results/csv/<model>.csv` (`/` and `:` in the model name are replaced with `-`).

Examples:
```
uv run python src/grade_essays.py
uv run python src/grade_essays.py --n 20 --model gpt-oss:20b --seed 42
uv run python src/grade_essays.py --n 20 --model claude-opus-4-8 --backend anthropic
uv run python src/grade_essays.py --n 20 --backend vllm --model Qwen/Qwen3-8B
```

---

## `src/pull_model.py`

Downloads a model snapshot from Hugging Face (for later serving with vLLM). Reads
`HF_TOKEN` from `.env`/the environment for gated or rate-limited repos; public repos work
without it. Pre-staging is optional — `vllm serve` can also pull a repo id itself.

```
uv run python src/pull_model.py REPO_ID [--revision REVISION] [--out OUT] [--allow-bin]
```

| Argument / flag | Default | What it does |
|---|---|---|
| `REPO_ID` (positional, required) | — | Hugging Face repo id, e.g. `Qwen/Qwen3-8B`. |
| `--revision REVISION` | `main` (HF default) | Pin a specific git revision/commit of the repo. |
| `--out OUT` | `models_cache/<repo-name>` | Destination directory for the downloaded snapshot. |
| `--allow-bin` | off (safetensors only) | Also fetch `.bin`/`.pt`/etc. checkpoint files instead of skipping them when safetensors are present. |

Examples:
```
uv run python src/pull_model.py openai/gpt-oss-20b
uv run python src/pull_model.py Qwen/Qwen3-8B --revision main
uv run python src/pull_model.py google/gemma-3-4b-it --out models_cache/gemma-3-4b-it
```

---

## `serve_vllm.sh`

Launches a vLLM OpenAI-compatible server for one model, so `grade_essays.py --backend vllm`
can grade against it. Requires `vllm` installed on the (GPU) host running this — it is not
part of this project's `uv` environment. `MODEL` can be a local snapshot directory (e.g. from
`src/pull_model.py`) or a bare HF repo id (vLLM downloads it itself on first launch, using
`HF_TOKEN` from `.env` if present).

```
./serve_vllm.sh MODEL [--port PORT] [-- EXTRA_VLLM_ARGS...]
```

| Argument / flag | Default | What it does |
|---|---|---|
| `MODEL` (positional, required) | — | Local model path or HF repo id to serve. |
| `--port PORT` | `8000` (or `$VLLM_PORT`) | Port to serve the OpenAI-compatible API on. |
| `-- EXTRA_VLLM_ARGS...` | — | Anything after `--` is passed through to `vllm serve` verbatim (e.g. `--tensor-parallel-size 2`). |

Examples:
```
./serve_vllm.sh Qwen/Qwen3-8B
./serve_vllm.sh models_cache/gpt-oss-20b --port 8001
VLLM_PORT=8001 ./serve_vllm.sh openai/gpt-oss-20b -- --tensor-parallel-size 2
```

---

## `src/plot_results.py`

Generates plots/tables from accumulated grading runs: per-model predicted-score
distribution, human-vs-model score counts, a metrics table (MAE / adjacent accuracy /
params / eval time), and an MAE-vs-parameter-count bubble chart. Reads `results/runs.csv`,
`results/csv/<model>.csv`, and `results/model_params.csv`; if a model's parameter count
isn't recorded yet, it's prompted for interactively and saved.

```
uv run python src/plot_results.py [--model MODEL ...] [--out OUT_DIR]
```

| Flag | Default | What it does |
|---|---|---|
| `--model MODEL` | all models in `results/runs.csv` | Restrict plotting to specific model(s); repeat the flag to select more than one. |
| `--out OUT_DIR` | `results/plots` | Output directory for generated PNG/CSV files. |

Examples:
```
uv run python src/plot_results.py
uv run python src/plot_results.py --model gpt-oss:20b --model glm-5.2:cloud
uv run python src/plot_results.py --out results/plots/
```

---

## `src/ollama_test.py`

Smoke test for the Ollama connection — sends one prompt to a model and prints the reply.
Uses Ollama Cloud if `OLLAMA_KEY` is set, otherwise a local Ollama server.

```
uv run python src/ollama_test.py [--model MODEL] [--prompt PROMPT]
```

| Flag | Default | What it does |
|---|---|---|
| `--model MODEL` | `gpt-oss:20b` on cloud; first installed model if local | Model to send the test prompt to. |
| `--prompt PROMPT` | `"In one short sentence, confirm you are reachable and name your model."` | Prompt text to send. |

Examples:
```
uv run python src/ollama_test.py
uv run python src/ollama_test.py --model gpt-oss:20b
uv run python src/ollama_test.py --prompt "Say hello in one sentence."
```

---

## `src/probe_model.py`

Diagnostic one-shot script: sends a fixed test essay to a hardcoded model
(`glm-5.2:cloud`) on Ollama Cloud with the score/rationale JSON schema, and prints the raw
response. No CLI arguments — edit the `model=` / `messages=` values in the file directly to
probe a different model or prompt.

```
uv run python src/probe_model.py
```
