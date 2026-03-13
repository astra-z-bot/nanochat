# Nanochat Repository Structure (High-Level)

This chapter is a structural map of the repository.
Goal: know what each file is responsible for before diving into implementation details.

---

## 1) Mental model: how the repo is organized

Nanochat is split into six practical layers:

1. **Core library (`nanochat/`)**
   - model, data loading, optimization, inference, common utilities.
2. **Entrypoint scripts (`scripts/`)**
   - runnable stage programs: train, eval, SFT, RL, chat serving.
3. **Task definitions (`tasks/`)**
   - benchmark/dataset adapters and scoring logic.
4. **Experiment presets (`runs/`)**
   - shell recipes wiring stages + hyperparams for reproducible runs.
5. **Validation (`tests/`)**
   - behavioral correctness tests on critical subsystems.
6. **Research/dev support (`dev/`)**
   - experiment logs, utilities, analysis notebooks, assets.

---

## 2) Top-level files

| File | Purpose |
|---|---|
| `README.md` | Main project doc: goals, usage patterns, key workflows. |
| `pyproject.toml` | Python project metadata and dependency declarations. |
| `uv.lock` | Fully resolved dependency lockfile for reproducibility. |
| `.python-version` | Interpreter version pin used by local tooling. |
| `.gitignore` | Ignore policy (env files, generated artifacts, caches, etc.). |
| `LICENSE` | Project license terms. |

---

## 3) Core package: `nanochat/`

This is the implementation center.

| File | High-level responsibility |
|---|---|
| `nanochat/__init__.py` | Package boundary (currently minimal). |
| `nanochat/gpt.py` | Transformer model definition and core forward logic. |
| `nanochat/flash_attention.py` | Unified attention backend path (FA3/SDPA switching and compatibility handling). |
| `nanochat/fp8.py` | Minimal FP8 training utilities and low-precision support path. |
| `nanochat/engine.py` | Inference engine (decode loop, token flow, generation runtime behavior). |
| `nanochat/tokenizer.py` | Tokenizer stack (BPE-related train/runtime encode-decode interfaces). |
| `nanochat/dataset.py` | Pretraining dataset utilities (dataset access and processing helpers). |
| `nanochat/dataloader.py` | Distributed data loader and packing/chunking logic for pretraining. |
| `nanochat/optim.py` | Optimizer implementations/composition (AdamW/Muon mix path). |
| `nanochat/checkpoint_manager.py` | Save/load checkpoint lifecycle and state persistence logic. |
| `nanochat/loss_eval.py` | Base-model loss-side evaluation helpers/metrics computations. |
| `nanochat/core_eval.py` | CORE metric evaluation logic used in benchmarking flow. |
| `nanochat/execution.py` | Sandboxed code-execution utility path for LLM-generated code use cases. |
| `nanochat/report.py` | Report-card generation helpers for training/evaluation outputs. |
| `nanochat/common.py` | Shared utilities: dtype/runtime detection, logging, common helpers, locking/download helpers. |
| `nanochat/ui.html` | Web chat frontend UI served by chat web runtime. |
| `nanochat/logo.svg` | Vector logo asset used by project/UI/docs. |

---

## 4) Stage entrypoints: `scripts/`

Each script is a runnable stage.

| File | Stage |
|---|---|
| `scripts/base_train.py` | Base pretraining stage (core model training loop). |
| `scripts/base_eval.py` | Base checkpoint evaluation stage (CORE + related eval flow). |
| `scripts/chat_sft.py` | Supervised finetuning stage for chat behavior. |
| `scripts/chat_rl.py` | RL stage for chat model refinement. |
| `scripts/chat_eval.py` | Chat-model evaluation stage. |
| `scripts/chat_cli.py` | CLI chat inference runtime. |
| `scripts/chat_web.py` | FastAPI web server for UI/API chat inference. |
| `scripts/tok_train.py` | Tokenizer training stage. |
| `scripts/tok_eval.py` | Tokenizer evaluation/compression analysis stage. |

---

## 5) Task layer: `tasks/`

Task modules define data formatting + scoring for evaluation/training datasets.

| File | Purpose |
|---|---|
| `tasks/common.py` | Shared task abstraction contract and common task utilities. |
| `tasks/mmlu.py` | MMLU task adapter/evaluation path. |
| `tasks/arc.py` | ARC task adapter/evaluation path. |
| `tasks/gsm8k.py` | GSM8K task adapter/evaluation path. |
| `tasks/humaneval.py` | HumanEval-style coding task evaluation path. |
| `tasks/smoltalk.py` | SmolTalk conversational dataset task adapter. |
| `tasks/spellingbee.py` | Counting/spelling-focused task (targeted capability shaping/eval). |
| `tasks/customjson.py` | Custom JSONL conversation/task ingestion adapter. |

---

## 6) Preset experiment recipes: `runs/`

Shell scripts that orchestrate end-to-end experiment flows.

| File | Role |
|---|---|
| `runs/speedrun.sh` | Main fast-path full pipeline recipe (pretrain + finetune/eval profile). |
| `runs/miniseries.sh` | Model-depth series recipe (family/sweep style runs). |
| `runs/scaling_laws.sh` | Scaling-law oriented experiment recipe (flop/budget sweeps). |
| `runs/runcpu.sh` | CPU/MPS-compatible sanity path for exercising codepaths without multi-GPU setup. |

---

## 7) Tests: `tests/`

Small but high-value checks on critical behavior.

| File | Verifies |
|---|---|
| `tests/test_attention_fallback.py` | Attention backend parity/correctness across flash/fallback implementations. |
| `tests/test_engine.py` | Core inference engine behavior and runtime correctness checks. |

---

## 8) Dev/research support: `dev/`

Not primary runtime entrypoints, but important for research workflow and maintenance context.

| File | Purpose |
|---|---|
| `dev/LOG.md` | Running experiment log and findings timeline. |
| `dev/LEADERBOARD.md` | Leaderboard documentation and target metric context. |
| `dev/repackage_data_reference.py` | Utility to repackage datasets into training-friendly shard format. |
| `dev/gen_synthetic_data.py` | Synthetic data generation helper script. |
| `dev/scaling_analysis.ipynb` | Notebook for scaling analysis workflows/results. |
| `dev/estimate_gpt3_core.ipynb` | Notebook for CORE-related estimation/analysis. |
| `dev/generate_logo.html` | Utility page/script for logo generation. |
| `dev/nanochat.png` | Raster logo/image asset. |
| `dev/scaling_laws_jan26.png` | Static scaling-law visualization asset. |

---

## 9) Additional project metadata folder

| File | Purpose |
|---|---|
| `.claude/skills/read-arxiv-paper/SKILL.md` | Local skill metadata/instructions related to reading arXiv papers (tooling/meta, not core runtime). |

---

## 10) Practical “what to read first” order

For fast onboarding, read in this sequence:

1. `README.md`
2. `scripts/base_train.py` + `nanochat/gpt.py`
3. `nanochat/dataloader.py` + `nanochat/tokenizer.py`
4. `scripts/base_eval.py` + `nanochat/core_eval.py` + `tasks/common.py`
5. `scripts/chat_sft.py`, `scripts/chat_rl.py`, `scripts/chat_eval.py`
6. `nanochat/engine.py`, `scripts/chat_cli.py`, `scripts/chat_web.py`, `nanochat/ui.html`
7. `runs/speedrun.sh`, `runs/miniseries.sh`, `runs/scaling_laws.sh`

This gives a top-down understanding of both code ownership and execution flow.
