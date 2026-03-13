# Nanochat Repository Structure — Following the Build Flow of a Chat AI

Nanochat is easiest to understand as a staged system:

1. set up the project
2. prepare text and tokenizer
3. build and train the base model
4. evaluate the base model
5. turn it into a chat model
6. serve it through CLI/web
7. run experiments and compare results

---

## 1. Project bootstrapping

Before any model code matters, the project needs a reproducible runtime.

### Files

- `README.md`
  - the human entrypoint to the repo
  - explains the intended workflow, main scripts, and experimental goals
- `pyproject.toml`
  - declares the Python project and dependencies
  - tells you what external packages nanochat depends on
- `uv.lock`
  - exact locked dependency graph for reproducibility
- `.python-version`
  - expected Python version for local tooling
- `.gitignore`
  - what the repo treats as generated or environment-specific junk
- `LICENSE`
  - legal terms, not part of runtime logic

### Why this stage matters

Nanochat is an experiment-heavy repo. If the environment drifts, all downstream observations become noisy:
- performance numbers change
- precision/kernel paths change
- tokenizer/training/eval behavior may differ

So conceptually, this repo begins here, even though no “AI logic” is in these files.

---

## 2. Data and tokenizer: how raw text becomes model input

A chat model starts as token prediction over text. So the first real technical stage is:

**raw text -> tokenizer -> token ids -> training batches**

### Core files

#### `nanochat/tokenizer.py`
High-level role:
- owns tokenization interfaces used by the project
- connects tokenizer training artifacts to runtime encode/decode behavior
- acts as the boundary between raw text and model vocabulary space

#### `scripts/tok_train.py`
High-level role:
- trains a tokenizer for the project
- used when you want to create tokenizer artifacts from data

#### `scripts/tok_eval.py`
High-level role:
- evaluates tokenizer quality, mainly through compression-oriented metrics
- helps compare tokenizer choices before model training

#### `nanochat/dataset.py`
High-level role:
- defines how pretraining data is stored and accessed
- includes utilities around parquet-based dataset handling

#### `nanochat/dataloader.py`
High-level role:
- turns tokenized data into distributed training batches
- handles packing/chunking/batching details for pretraining

### Related support files

#### `dev/repackage_data_reference.py`
- utility for reorganizing raw datasets into parquet shards that match the expected training format

#### `dev/gen_synthetic_data.py`
- helper for generating synthetic examples
- not the main pretraining path, but relevant to data generation and augmentation workflows

### What this means in pipeline terms

The corresponding execution flow is:

1. define or gather text data
2. possibly repackage it into the repo’s preferred storage format
3. train a tokenizer or reuse one
4. encode text into token ids
5. batch those ids for model training

This is the first place where the repo becomes an ML system instead of just a Python project.

---

## 3. The base model: the transformer itself

Once text is tokenized, the next stage is the model.

This repo’s model core lives in `nanochat/`.

### Core files

#### `nanochat/gpt.py`
High-level role:
- defines the transformer model itself
- this is the heart of the repo
- if you want to understand “what is being trained?”, this is the primary file

It is the file that answers:
- what the model architecture is
- how tokens become hidden states and logits
- how attention/MLP/residual paths are structured

#### `nanochat/flash_attention.py`
High-level role:
- attention backend switching layer
- decides how attention is executed depending on available kernel/runtime support
- exists to make attention fast when possible, without changing higher-level model code

#### `nanochat/fp8.py`
High-level role:
- low-precision training support
- encodes special logic for FP8-style pathways and related constraints

#### `nanochat/common.py`
High-level role:
- runtime utilities shared across the system
- includes things like dtype/runtime detection, logging helpers, shared convenience functions
- not “the model,” but heavily used by the model/training runtime

### Why this stage matters

At this point, the repo answers:

> Given token ids, what network do we run, and what numerical path do we run it through?

This is the model-definition stage, before optimization enters the picture.

---

## 4. Base training: how the base model learns

After model definition, the next stage is optimization.

Pipeline view:

**batches -> forward pass -> loss -> backward pass -> optimizer step -> checkpoint/logging**

### Core files

#### `scripts/base_train.py`
High-level role:
- main pretraining entrypoint
- orchestrates the base model training run
- wires together tokenizer/data/model/optimizer/checkpointing/reporting

This is the runtime backbone of pretraining.

#### `nanochat/optim.py`
High-level role:
- optimizer implementation/composition
- contains the logic for how parameters are updated

#### `nanochat/checkpoint_manager.py`
High-level role:
- save/load/resume behavior for checkpoints
- essential for long runs and experiment continuity

#### `nanochat/loss_eval.py`
High-level role:
- computes training-adjacent evaluation/loss metrics for base models
- used to interpret whether learning is actually happening

#### `nanochat/report.py`
High-level role:
- report generation and summary outputs for runs
- translates raw run signals into consumable experiment artifacts

### Why this stage matters

This is where nanochat becomes an actual trainer rather than just a model definition.

The repo’s base-training stage is the combination of:
- model structure (`gpt.py`)
- data path (`dataset.py` / `dataloader.py` / tokenizer)
- optimizer/checkpoint/reporting
- orchestration (`base_train.py`)

---

## 5. Base evaluation: can the pretrained model do useful things?

A model is not useful just because it trains.
You need to score it.

### Core files

#### `scripts/base_eval.py`
High-level role:
- entrypoint for evaluating base checkpoints
- coordinates evaluation runs and metric collection

#### `nanochat/core_eval.py`
High-level role:
- implements the CORE metric/evaluation logic used by the repo
- this is central to how model quality is compared in nanochat

### Task layer

The actual tasks used for evaluation live under `tasks/`.

#### `tasks/common.py`
- shared task abstraction
- defines the common contract used by specific tasks

#### `tasks/mmlu.py`
- MMLU evaluation task

#### `tasks/arc.py`
- ARC evaluation task

#### `tasks/gsm8k.py`
- GSM8K evaluation task

#### `tasks/humaneval.py`
- HumanEval-style coding evaluation task

#### `tasks/spellingbee.py`
- smaller targeted task for counting/spelling-like capabilities

#### `tasks/smoltalk.py`
- conversational dataset/task adapter
- used more in the chat-oriented side of the repo, but still belongs to the task ecosystem

#### `tasks/customjson.py`
- user-defined JSONL task ingestion path
- extension point for custom data/evaluation

### Why this stage matters

At a system level, this stage answers:

> After training, how do we know whether the model improved?

So the repo structure now looks like:

- model creation
- model training
- model evaluation

That is already a full base-model stack.

---

## 6. Chat post-training: turning the base model into a chat model

A base model predicts next tokens.
A chat model needs further shaping.

Nanochat splits that into multiple post-training stages.

### Core files

#### `scripts/chat_sft.py`
High-level role:
- supervised fine-tuning (SFT) stage
- uses conversation-style/task-style data to teach the model better interaction behavior

#### `scripts/chat_rl.py`
High-level role:
- reinforcement-learning stage for chat behavior refinement
- sits after or alongside SFT depending on workflow

#### `scripts/chat_eval.py`
High-level role:
- evaluates the chat model specifically
- this is different from base eval because interactive behavior is the target now

#### `nanochat/execution.py`
High-level role:
- sandboxed execution utilities for code emitted by the model
- relevant to chat/eval/tool-using scenarios where the model output must be executed or validated

### Related task/data files

#### `tasks/smoltalk.py`
- useful conversational data source for SFT-style workflows

#### `tasks/customjson.py`
- custom conversation/task ingestion for chat tuning and eval

### Why this stage matters

This is the point where the repo moves from:
- “train a language model”

into:
- “shape a usable assistant/chat model”

So the flow is now:

1. pretrain base model
2. evaluate base model
3. SFT chat model
4. RL refine chat model
5. evaluate chat model

---

## 7. Inference and serving: using the trained model

Once you have a chat model, the next question is how it is used at runtime.

### Core files

#### `nanochat/engine.py`
High-level role:
- inference engine
- the runtime layer that handles generation behavior efficiently
- this is the bridge from trained weights to actual token-by-token output

#### `scripts/chat_cli.py`
High-level role:
- CLI interface for chatting with the model
- simplest interactive serving surface

#### `scripts/chat_web.py`
High-level role:
- web server that exposes UI/API chat functionality
- serving/runtime orchestration for browser-based usage

#### `nanochat/ui.html`
High-level role:
- browser UI for the web chat path
- frontend layer that pairs with `chat_web.py`

#### `nanochat/logo.svg`
- UI/docs asset used by the frontend/project identity

### Why this stage matters

This stage answers:

> Once the model exists, how do users actually talk to it?

So the repo’s runtime story is:
- train it
- evaluate it
- serve it through CLI or web

---

## 8. Experiment orchestration: repeatable research workflows

Nanochat is not just a training codebase — it is also an experimentation codebase.

That is what `runs/` is for.

### Files

#### `runs/speedrun.sh`
High-level role:
- the main end-to-end “fast path” recipe
- likely the most important experiment script to understand first

#### `runs/miniseries.sh`
High-level role:
- multi-model/depth series workflow
- useful for structured scaling comparisons

#### `runs/scaling_laws.sh`
High-level role:
- scaling-law style experiments
- focused on controlled comparisons across budgets/settings

#### `runs/runcpu.sh`
High-level role:
- small CPU/MPS-friendly path for local sanity checks
- useful when full multi-GPU setup is unavailable

### Why this matters

These scripts are not the core logic themselves.
They are the reproducible wiring of the core logic.

They tell you how the repo authors expect the pieces to be used together in practice.

---

## 9. Validation and correctness checks

### Files

#### `tests/test_attention_fallback.py`
High-level role:
- verifies correctness across attention backend paths
- important because flash/fallback divergence would poison both training and inference conclusions

#### `tests/test_engine.py`
High-level role:
- verifies inference engine behavior
- important because runtime bugs can make a trained model look worse or inconsistent

### Why this matters

Tests tell you what the repo authors considered risky enough to protect explicitly.
That is often as informative as the implementation itself.

---

## 10. Dev/research support material

These files are not the central runtime path, but they matter for understanding the research workflow around the repo.

### Files

#### `dev/LOG.md`
- running research/experiment notes
- helps explain why some code exists

#### `dev/LEADERBOARD.md`
- explains leaderboard/target metric framing
- useful for understanding what the project is optimizing for socially/research-wise

#### `dev/repackage_data_reference.py`
- data prep support utility

#### `dev/gen_synthetic_data.py`
- synthetic data support utility

#### `dev/scaling_analysis.ipynb`
- scaling-law/result analysis notebook

#### `dev/estimate_gpt3_core.ipynb`
- analysis notebook related to CORE/GPT-scale estimation

#### `dev/generate_logo.html`, `dev/nanochat.png`, `dev/scaling_laws_jan26.png`
- support assets and visual outputs

---

## 11. Reading order by build flow

A practical reading order is:

### Pass 1 — system flow
1. `README.md`
2. `scripts/base_train.py`
3. `nanochat/gpt.py`
4. `nanochat/tokenizer.py`
5. `nanochat/dataset.py`
6. `nanochat/dataloader.py`

### Pass 2 — evaluation
7. `scripts/base_eval.py`
8. `nanochat/core_eval.py`
9. `tasks/common.py`
10. task files under `tasks/`

### Pass 3 — chat stage
11. `scripts/chat_sft.py`
12. `scripts/chat_rl.py`
13. `scripts/chat_eval.py`
14. `nanochat/execution.py`

### Pass 4 — runtime
15. `nanochat/engine.py`
16. `scripts/chat_cli.py`
17. `scripts/chat_web.py`
18. `nanochat/ui.html`

### Pass 5 — experiment workflows
19. `runs/speedrun.sh`
20. `runs/miniseries.sh`
21. `runs/scaling_laws.sh`
22. `tests/*`
23. `dev/LOG.md` + `dev/LEADERBOARD.md`

---

## 12. Final repo picture

If you compress the entire repo into one sentence:

> Nanochat is a compact LLM research/training stack where `nanochat/` holds the implementation, `scripts/` runs the stages, `tasks/` defines what gets measured, `runs/` wires reproducible experiments, `tests/` guard risky behavior, and `dev/` records supporting research workflow and utilities.

That is the structure.
