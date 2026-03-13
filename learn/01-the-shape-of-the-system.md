# Chapter 1 — The Shape of the System

Nanochat is easiest to understand as a sequence of transformations.

A project starts as a directory with a pinned environment. That environment becomes a data pipeline. The data pipeline feeds a tokenizer. The tokenizer and corpus feed a transformer. The transformer is placed inside a training loop. The resulting checkpoints are evaluated, adapted into a chat model, and then exposed through an inference engine and serving layer. Around all of that sits an experiment harness that makes the system repeatable.

That sequence is the real structure of the repository. The folders only make sense once that flow is clear.

## 1. Project Bootstrap

Before any model code matters, the repository has to be runnable in a stable way.

`README.md` is the public face of the project. It tells you what nanochat is for, which workflows matter, and which scripts define the expected path through the codebase. In a project like this, the README is not just documentation; it is the top-level execution map.

`pyproject.toml` gives the repository its Python identity and dependency surface. `uv.lock` freezes that surface into a reproducible environment. `.python-version` pins the interpreter expected by local tooling. `.gitignore` defines the boundary between source and generated state: caches, environments, experiment artifacts, and machine-specific files.

At this stage, the system has no model behavior yet. What it has is a controlled runtime. That matters because later stages depend on kernel availability, tokenizer libraries, parquet tooling, and evaluation dependencies behaving consistently across runs.

## 2. Data and Tokenizer Preparation

Once the runtime exists, the next question is how the model will see language at all.

Nanochat answers that with a text pipeline that begins before the model and remains foundational after the model is introduced. The path is:

**corpus -> tokenizer artifacts -> token IDs -> training batches**

The tokenizer boundary lives in `nanochat/tokenizer.py`. This file is where raw text becomes an internal representation the rest of the stack can use. It is also where compatibility between tokenizer training artifacts and runtime encoding behavior has to hold.

The tokenizer has two dedicated entrypoints. `scripts/tok_train.py` creates tokenizer artifacts from text. `scripts/tok_eval.py` evaluates those artifacts, mostly through compression-oriented signals that help compare tokenizer choices before model training begins.

But the tokenizer does not operate over abstract text. It operates over a concrete corpus format, and that format is managed by the dataset path. `nanochat/dataset.py` describes how the pretraining corpus is stored, located, and streamed. `nanochat/dataloader.py` turns that stream into training-ready batches, handling chunking, packing, and distributed iteration.

The important structural point is that tokenization and batching are separate concerns. `tokenizer.py` defines how strings become token IDs. `dataset.py` and `dataloader.py` define how a large stored corpus becomes a stream of documents and, later, a stream of tensors.

Two support files also belong to this part of the system. `dev/repackage_data_reference.py` documents how the current corpus was converted into the parquet shard format used by runtime code. `dev/gen_synthetic_data.py` supports dataset generation workflows that feed into training or evaluation paths.

By the time this stage is finished, the system has crossed the first major boundary: it no longer sees text as arbitrary files on disk. It sees a managed corpus, a tokenizer, and a sequence of token batches.

## 3. Base Model Definition

Once token IDs exist, the next question is what network will consume them.

That answer begins in `nanochat/gpt.py`. This is the architectural center of the repository. It defines the transformer itself: embeddings, block structure, attention path, MLP path, residual flow, and the final projection to logits. If you want to know what is actually being trained, `gpt.py` is the first file that matters.

But the model in nanochat is not just an abstract transformer block stack. Its execution path is shaped by adjacent runtime files.

`nanochat/flash_attention.py` exists because attention is not just a mathematical operation; it is also a kernel-selection problem. This file manages the backend path used when the hardware and dtype configuration allow a faster implementation.

`nanochat/fp8.py` exists because precision is also not purely conceptual. The repo has a deliberate low-precision path, and that logic has to live somewhere explicit.

`nanochat/common.py` ties into this layer as the shared runtime substrate: dtype decisions, device/runtime detection, and utility functions used across both training and inference.

So the model layer in nanochat is not one file. It is a cluster:

- `gpt.py` defines the network
- `flash_attention.py` influences how attention executes
- `fp8.py` influences how low-precision execution is handled
- `common.py` provides the runtime assumptions the rest of that layer depends on

## 4. Base Model Training

A model definition on its own is static. Training is what turns it into a changing system.

The center of that stage is `scripts/base_train.py`. This is the base-model training entrypoint. It is where the repository stops being a collection of reusable modules and becomes a runnable optimization pipeline.

`base_train.py` pulls together the main ingredients already introduced:

- tokenized and batched data from `dataset.py` and `dataloader.py`
- model structure from `gpt.py`
- runtime helpers from `common.py`

Then it adds the remaining machinery required for optimization.

`nanochat/optim.py` defines how parameters are updated. `nanochat/checkpoint_manager.py` defines how model state, optimizer state, and run progress are saved and resumed. `nanochat/loss_eval.py` computes training-adjacent loss-side signals. `nanochat/report.py` packages outputs into experiment summaries that are easier to compare across runs.

The training stage therefore has a clear internal structure:

1. get a batch
2. run the model forward
3. compute a loss
4. backpropagate
5. update parameters
6. checkpoint state
7. record signals

That stage is the operational heart of the repository.

## 5. Base Model Evaluation

Once training produces checkpoints, the repository needs a way to measure what those checkpoints can do.

That path begins in `scripts/base_eval.py`. This is the base-model evaluation entrypoint. It loads trained checkpoints and runs them through the repository’s evaluation pipeline.

The evaluation core lives in `nanochat/core_eval.py`. This file is important because it captures part of the project’s philosophy about what “progress” means numerically.

But evaluation is not one score. The benchmark/task layer under `tasks/` defines how model capability is actually probed.

`tasks/common.py` is the shared abstraction layer. The individual task files specialize that interface for concrete benchmarks or data formats:

- `tasks/mmlu.py`
- `tasks/arc.py`
- `tasks/gsm8k.py`
- `tasks/humaneval.py`
- `tasks/spellingbee.py`
- `tasks/smoltalk.py`
- `tasks/customjson.py`

This is the point where the repository becomes more than a trainer. It becomes a measurement system with explicit benchmark interfaces.

## 6. Chat Post-Training

A pretrained next-token model is not yet the system most users would call a chatbot. Nanochat makes that transition in a separate post-training stack.

`scripts/chat_sft.py` is the supervised fine-tuning stage. It moves the model toward instruction-following and dialogue behavior by training on conversational or task-formatted data.

`scripts/chat_rl.py` adds a reinforcement-learning stage on top of that. This stage exists because the repo treats post-training as a multi-step process rather than as a single finetuning pass.

`scripts/chat_eval.py` then evaluates the resulting chat model. At this point, evaluation is no longer only about raw base-model competence. It also has to reflect assistant-like behavior.

`nanochat/execution.py` belongs here because some chat-oriented workflows require executing or validating model-produced code. It is not part of the base model path, but it becomes relevant once the model is expected to act more like an assistant than a pure next-token predictor.

## 7. A trained chat model still needs a runtime surface

After post-training, the system is ready to be used interactively.

That runtime begins in `nanochat/engine.py`. This file is the inference engine. It controls decode-time behavior, token generation, and the stateful mechanics that turn checkpoints into actual outputs.

On top of that engine, nanochat exposes two primary serving surfaces.

`scripts/chat_cli.py` is the command-line path. It is the simplest interactive runtime and often the most direct way to inspect inference behavior without web-layer overhead.

`scripts/chat_web.py` is the server-side path for browser-based use. `nanochat/ui.html` is the corresponding frontend. Together they form the web-facing surface of the project.

`nanochat/logo.svg` belongs to this layer only as a supporting UI asset; it is part of presentation, not of model execution.

At this point, the repository has completed the end-user path: model weights can now be loaded, run through an inference engine, and exposed through a usable interface.

## 8. Experiment scripts describe how the pieces are used in practice

The repo is not only a training stack. It is also an experimentation stack.

That is what `runs/` is for.

`runs/speedrun.sh` is the most direct end-to-end recipe. It is the quickest answer to the question, “how does the author expect these pieces to be used together?”

`runs/miniseries.sh` encodes a model-family workflow for structured comparisons. `runs/scaling_laws.sh` encodes scaling-law style experiments. `runs/runcpu.sh` gives a lighter-weight path for CPU or MPS systems that cannot run the full GPU-centric workflow.

These scripts are not implementation-heavy, but they are structurally important. They show the intended orchestration of the implementation.

## 9. Tests reveal what the repo considers risky

The `tests/` directory is small, but it is high signal.

`tests/test_attention_fallback.py` protects attention-path correctness across optimized and fallback implementations. `tests/test_engine.py` protects the inference engine.

That means two boundaries are clearly treated as correctness-critical:

- attention backend behavior
- inference runtime behavior

This is often one of the fastest ways to learn a codebase: look at what the authors chose to defend explicitly.

## 10. The dev directory is the surrounding research workflow

The `dev/` directory is not the primary runtime path, but it explains how the project is developed and evaluated over time.

`dev/LOG.md` is the running experiment log. `dev/LEADERBOARD.md` records leaderboard framing and target metrics. `dev/repackage_data_reference.py` and `dev/gen_synthetic_data.py` support data workflows. `dev/scaling_analysis.ipynb` and `dev/estimate_gpt3_core.ipynb` are analysis notebooks. `dev/generate_logo.html`, `dev/nanochat.png`, and `dev/scaling_laws_jan26.png` are support assets and visual outputs.

There is also `.claude/skills/read-arxiv-paper/SKILL.md`, which belongs to local tooling metadata rather than to the core nanochat model stack.

## 11. Reading Order

A practical reading order follows the same construction path:

1. `README.md`
2. `nanochat/tokenizer.py`
3. `nanochat/dataset.py`
4. `nanochat/dataloader.py`
5. `nanochat/gpt.py`
6. `scripts/base_train.py`
7. `scripts/base_eval.py`
8. `nanochat/core_eval.py`
9. `tasks/common.py`
10. task files under `tasks/`
11. `scripts/chat_sft.py`
12. `scripts/chat_rl.py`
13. `scripts/chat_eval.py`
14. `nanochat/execution.py`
15. `nanochat/engine.py`
16. `scripts/chat_cli.py`
17. `scripts/chat_web.py`
18. `nanochat/ui.html`
19. `runs/speedrun.sh`
20. `runs/miniseries.sh`
21. `runs/scaling_laws.sh`
22. `tests/test_attention_fallback.py`
23. `tests/test_engine.py`
24. `dev/LOG.md`
25. `dev/LEADERBOARD.md`

## 12. System Summary

Nanochat is a compact LLM training and experimentation stack.

The repository breaks down into:

- `nanochat/` for implementation
- `scripts/` for stage entrypoints
- `tasks/` for benchmark/task definitions
- `runs/` for experiment recipes
- `tests/` for correctness-critical checks
- `dev/` for the surrounding research workflow

That is the structural map for the rest of the book.
