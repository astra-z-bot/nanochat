# Nanochat Repository Structure

Nanochat can be read as an eight-stage system:

1. project bootstrap
2. data and tokenizer preparation
3. base model definition
4. base model training
5. base model evaluation
6. chat post-training
7. inference and serving
8. experiment orchestration and validation

The repository is easiest to understand in that order.

---

## 1. Project bootstrap

These files define the runtime and the project boundary.

| File | Function in the system |
|---|---|
| `README.md` | Main project document. Describes the intended workflow, major scripts, and experimental targets. |
| `pyproject.toml` | Python package definition and dependency list. |
| `uv.lock` | Fully resolved dependency graph for reproducible environments. |
| `.python-version` | Interpreter version pin for local tooling. |
| `.gitignore` | Generated, local, and environment-specific files that should not be versioned. |
| `LICENSE` | Legal terms. Not part of runtime behavior. |

This stage exists to stabilize everything that follows: dependency versions, interpreter version, and project entrypoints.

---

## 2. Data and tokenizer preparation

This stage converts raw text into token batches consumable by the model.

Pipeline:

**text -> tokenizer artifacts -> token ids -> packed batches**

### Core files

| File | Function in the system |
|---|---|
| `nanochat/tokenizer.py` | Tokenizer interface used by the project. Bridges tokenizer artifacts and runtime encode/decode behavior. |
| `scripts/tok_train.py` | Trains a tokenizer. Produces tokenizer artifacts from text data. |
| `scripts/tok_eval.py` | Evaluates tokenizer quality, mainly through compression-oriented metrics. |
| `nanochat/dataset.py` | Utilities for reading and organizing the pretraining dataset. |
| `nanochat/dataloader.py` | Builds distributed training batches from tokenized data. Handles packing/chunking logic. |

### Supporting files

| File | Function in the system |
|---|---|
| `dev/repackage_data_reference.py` | Repackages datasets into the parquet shard format expected by the training pipeline. |
| `dev/gen_synthetic_data.py` | Generates synthetic examples used in some data workflows. |

At the end of this stage, the model no longer sees text. It sees token ids arranged into training-ready batches.

---

## 3. Base model definition

This stage defines the network that will be trained.

### Core files

| File | Function in the system |
|---|---|
| `nanochat/gpt.py` | Core transformer model definition. The primary model file in the repository. |
| `nanochat/flash_attention.py` | Attention backend adapter. Selects the fast attention path when hardware/runtime conditions allow it and falls back otherwise. |
| `nanochat/fp8.py` | Minimal FP8 support path for low-precision training. |
| `nanochat/common.py` | Shared runtime helpers used across the training and inference stack, including dtype/runtime detection and common utility functions. |

This stage answers two concrete questions:
- what network is being trained
- what numerical/runtime path is used to execute it

---

## 4. Base model training

This stage optimizes the transformer on token batches.

Pipeline:

**batch -> forward -> loss -> backward -> optimizer step -> checkpoint/report**

### Core files

| File | Function in the system |
|---|---|
| `scripts/base_train.py` | Main pretraining entrypoint. Wires together model, data path, optimizer, checkpointing, and reporting. |
| `nanochat/optim.py` | Optimizer implementation and parameter-update logic. |
| `nanochat/checkpoint_manager.py` | Save/load/resume path for checkpoints. |
| `nanochat/loss_eval.py` | Loss-side evaluation helpers for base-model training. |
| `nanochat/report.py` | Produces structured training and evaluation summaries. |

This stage is the operational center of pretraining. It is where model definition, data flow, and optimization are combined into a runnable training job.

---

## 5. Base model evaluation

This stage scores the pretrained model on benchmark tasks.

### Evaluation entrypoints

| File | Function in the system |
|---|---|
| `scripts/base_eval.py` | Main entrypoint for evaluating base checkpoints. |
| `nanochat/core_eval.py` | CORE metric implementation used for model comparison in the repo. |

### Task layer

| File | Function in the system |
|---|---|
| `tasks/common.py` | Shared task abstraction and common task utilities. |
| `tasks/mmlu.py` | MMLU evaluation adapter. |
| `tasks/arc.py` | ARC evaluation adapter. |
| `tasks/gsm8k.py` | GSM8K evaluation adapter. |
| `tasks/humaneval.py` | HumanEval-style coding evaluation adapter. |
| `tasks/spellingbee.py` | Small targeted task for counting/spelling-type capability. |
| `tasks/smoltalk.py` | Conversational dataset/task adapter used in chat-oriented workflows. |
| `tasks/customjson.py` | Custom JSONL ingestion path for user-defined task data. |

This stage measures the base model before any chat-specific post-training.

---

## 6. Chat post-training

The base model is adapted into a chat model in this stage.

### Core files

| File | Function in the system |
|---|---|
| `scripts/chat_sft.py` | Supervised fine-tuning stage for conversational behavior. |
| `scripts/chat_rl.py` | Reinforcement-learning stage for further chat behavior refinement. |
| `scripts/chat_eval.py` | Evaluation path for the chat model. |
| `nanochat/execution.py` | Sandboxed execution utilities used in code-producing or tool-using workflows. |

### Related task/data files

| File | Function in the system |
|---|---|
| `tasks/smoltalk.py` | Conversational data source used in chat-oriented training. |
| `tasks/customjson.py` | Custom conversation/task data ingestion path for chat tuning and chat eval. |

At this stage the pipeline becomes:

**base model -> SFT -> RL (optional/iterative) -> chat evaluation**

---

## 7. Inference and serving

This stage turns trained checkpoints into an interactive system.

### Core files

| File | Function in the system |
|---|---|
| `nanochat/engine.py` | Inference engine. Handles token-by-token generation runtime. |
| `scripts/chat_cli.py` | Command-line chat interface. |
| `scripts/chat_web.py` | Web server for chat UI and API serving. |
| `nanochat/ui.html` | Browser UI for the web serving path. |
| `nanochat/logo.svg` | UI/docs asset used by the web-facing side of the project. |

This stage answers the runtime question: once the model has been trained, how is it actually used?

---

## 8. Experiment orchestration

These files are not the core logic themselves. They are the wiring for repeatable experiments.

### Files

| File | Function in the system |
|---|---|
| `runs/speedrun.sh` | Main end-to-end fast-path recipe. The most direct experiment script to understand first. |
| `runs/miniseries.sh` | Structured model-series workflow, useful for family/sweep runs. |
| `runs/scaling_laws.sh` | Scaling-law style experiment script for controlled budget/model comparisons. |
| `runs/runcpu.sh` | CPU/MPS-compatible sanity path for local testing of codepaths without full GPU setup. |

These scripts encode how the repo is intended to be used in practice.

---

## 9. Validation and correctness checks

### Files

| File | Function in the system |
|---|---|
| `tests/test_attention_fallback.py` | Verifies attention backend parity and correctness across fast/fallback implementations. |
| `tests/test_engine.py` | Verifies inference engine behavior. |

The tests identify which subsystems the repo treats as correctness-critical.

---

## 10. Dev and research support material

These files are not the central runtime path, but they explain the surrounding research workflow.

| File | Function in the system |
|---|---|
| `dev/LOG.md` | Running experiment log and findings timeline. |
| `dev/LEADERBOARD.md` | Leaderboard rules and target-metric framing. |
| `dev/repackage_data_reference.py` | Data-prep support utility. |
| `dev/gen_synthetic_data.py` | Synthetic-data support utility. |
| `dev/scaling_analysis.ipynb` | Scaling-analysis notebook. |
| `dev/estimate_gpt3_core.ipynb` | CORE/GPT-scale analysis notebook. |
| `dev/generate_logo.html` | Logo-generation utility page. |
| `dev/nanochat.png` | Raster logo/image asset. |
| `dev/scaling_laws_jan26.png` | Static scaling-law visualization asset. |

Additional metadata/support file:

| File | Function in the system |
|---|---|
| `.claude/skills/read-arxiv-paper/SKILL.md` | Local skill metadata/instructions. Not part of core training or inference runtime. |

---

## 11. Reading order

A clean reading order is:

### Pass 1 — base stack
1. `README.md`
2. `nanochat/tokenizer.py`
3. `nanochat/dataset.py`
4. `nanochat/dataloader.py`
5. `nanochat/gpt.py`
6. `scripts/base_train.py`

### Pass 2 — evaluation
7. `scripts/base_eval.py`
8. `nanochat/core_eval.py`
9. `tasks/common.py`
10. the individual files under `tasks/`

### Pass 3 — chat stack
11. `scripts/chat_sft.py`
12. `scripts/chat_rl.py`
13. `scripts/chat_eval.py`
14. `nanochat/execution.py`

### Pass 4 — serving stack
15. `nanochat/engine.py`
16. `scripts/chat_cli.py`
17. `scripts/chat_web.py`
18. `nanochat/ui.html`

### Pass 5 — experiment workflow
19. `runs/speedrun.sh`
20. `runs/miniseries.sh`
21. `runs/scaling_laws.sh`
22. `tests/test_attention_fallback.py`
23. `tests/test_engine.py`
24. `dev/LOG.md`
25. `dev/LEADERBOARD.md`

---

## 12. Condensed picture of the repo

Nanochat is a compact LLM training and experimentation stack.

- `nanochat/` contains the implementation.
- `scripts/` runs each stage.
- `tasks/` defines benchmark/task behavior.
- `runs/` encodes reproducible experiment recipes.
- `tests/` protects correctness-critical paths.
- `dev/` records supporting research workflow and utilities.

That is the structural map for the rest of the textbook.
