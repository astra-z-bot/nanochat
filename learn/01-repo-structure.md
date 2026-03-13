# Nanochat Repository Structure

Nanochat makes the most sense if you read it as the story of building a chat model from the ground up.

A chat model does not begin with a web UI, or even with a transformer block. It begins with a runnable project, then a tokenization pipeline, then a base model, then a training loop, then evaluation, then post-training, and only after that does it become something you can chat with.

This chapter follows that order and places each file where it enters the build.

## Start at the outer boundary: the project has to run at all

Before the model exists, the repo needs a stable execution environment.

`README.md` is the human entrypoint. It tells you what nanochat is trying to optimize, what workflows the author expects you to run, and which scripts matter. If you want the shortest path to “how is this repo meant to be used?”, this is where it starts.

`pyproject.toml` defines the Python project itself. It gives the repo a package identity and declares the dependencies that make the rest of the code possible. `uv.lock` freezes that dependency graph into a reproducible environment. `.python-version` pins the interpreter version expected by local tooling. `.gitignore` marks off the non-source boundary: environments, caches, generated files, temporary outputs.

At this point nothing “AI” has happened yet. But this stage is still part of the system. Nanochat is a research-heavy codebase; if the runtime shifts, then performance, kernel paths, tokenizer behavior, and even evaluation results can shift with it.

So the repo begins here: not with learning, but with reproducibility.

## Then text has to become tokens

Once the environment exists, the first real modeling problem is representation. The model will never see raw text directly. It will see token IDs. So the next part of the repo answers a simple question:

**How does text become model input?**

That path starts in `nanochat/tokenizer.py`. This file is the token boundary of the whole system. It is where plain text enters the project and where encoded token sequences emerge. It also defines the compatibility contract between tokenizer artifacts created during training and tokenization behavior at runtime.

If `nanochat/tokenizer.py` is the interface, then `scripts/tok_train.py` is the builder. It exists to train a tokenizer from data. `scripts/tok_eval.py` is the judge for that tokenizer: it evaluates tokenizer quality, especially in terms of compression-oriented behavior, so the tokenizer can be compared before the model is trained on top of it.

But a tokenizer is only one half of the early data path. The repo also needs a storage and loading story for text data.

That is where `nanochat/dataset.py` and `nanochat/dataloader.py` enter.

`nanochat/dataset.py` is responsible for dataset access and preprocessing utilities. In this repo, the pretraining dataset is organized around parquet shards, so this file is part of the physical data contract of the system. `nanochat/dataloader.py` takes the next step: it turns tokenized examples into distributed, training-ready batches. This is where chunking, packing, ordering, and multi-process data flow become real.

There are also two support files that belong to this stage, even if they are not part of the main runtime path.

`dev/repackage_data_reference.py` exists because raw datasets are not always already shaped the way the training stack expects. It repackages data into the parquet shard format the rest of the repo is built around.

`dev/gen_synthetic_data.py` sits slightly off the main path, but still belongs to the data story. It generates synthetic examples, which makes it part of the broader question of how nanochat constructs or augments training/evaluation inputs.

By the end of this stage, the repo has crossed an important boundary. It is no longer just a Python project with scripts. It has a complete path from text to tokens to training batches.

## After tokens, the model itself appears

Once token IDs exist, the next question is: what network consumes them?

The answer begins in `nanochat/gpt.py`.

This is the center of the codebase. If someone asked, “Where is the model, really?”, `nanochat/gpt.py` is the first file you would hand them. It contains the transformer definition: embeddings, attention path, MLP path, residual flow, and the structure that turns a sequence of token IDs into logits.

But the transformer definition in this repo is not isolated. It is surrounded by implementation files that modify how that model actually runs.

`nanochat/flash_attention.py` is one of those files. It is not “the model” in the abstract sense, but it is part of how the model is executed. It chooses the attention backend path depending on runtime capability. In other words, the conceptual transformer lives in `gpt.py`, but the practical attention path may be rerouted through `flash_attention.py`.

`nanochat/fp8.py` plays a similar role for precision. It defines part of the low-precision execution story. Again, it is not the architecture definition itself, but it materially changes how that architecture is trained or run on modern hardware.

`nanochat/common.py` sits underneath all of this. It is not a model file, but it is a runtime substrate: shared helpers, dtype/runtime detection, logging and utility code that other parts of the training and inference stack lean on.

So if the tokenizer stage answers “how do we represent text?”, the model-definition stage answers two new questions:

- what network will learn over those tokens?
- what runtime/precision path will be used to execute it?

That is the model layer of nanochat.

## A model definition alone is not useful; it has to be trained

With token batches and a transformer definition in place, nanochat moves into optimization.

This is the stage where the abstract model becomes a learned model.

The center of that stage is `scripts/base_train.py`.

This file is the main base-model training entrypoint. It is the orchestration layer that brings together the model from `nanochat/gpt.py`, the tokenizer and data path from `nanochat/tokenizer.py`, `nanochat/dataset.py`, and `nanochat/dataloader.py`, and the optimization utilities below it.

Those optimization utilities live in a small cluster of files.

`nanochat/optim.py` defines how parameters are updated. It is the optimizer layer of the repo.

`nanochat/checkpoint_manager.py` defines how training state is saved and resumed. This is what lets long runs survive interruption and what gives experiments continuity instead of forcing every run to restart from zero.

`nanochat/loss_eval.py` sits near the train loop as the place where loss-side evaluation helpers live for the base model. It exists because training needs measurement, not just optimization.

`nanochat/report.py` turns run results into structured summaries and report-like outputs. It is part of the experiment lifecycle rather than the gradient path, but it belongs to this stage because training is not complete when the optimizer steps; it is complete when the run leaves behind interpretable evidence.

Put together, this stage is the full training runtime:

- batches come from the data path
- the model runs forward
- loss is computed
- gradients are applied
- state is checkpointed
- metrics are recorded

This is where the repo stops being a model implementation and becomes a trainer.

## Training is not enough; the base model has to be measured

After the base model has been trained, the next question is not “can it run?” but “how good is it?”

That stage starts with `scripts/base_eval.py`.

This file is the base-model evaluation entrypoint. It is the script that takes checkpoints and runs the repository’s evaluation logic over them.

That evaluation logic is anchored by `nanochat/core_eval.py`. This file implements the CORE metric path that the repo uses as one of its major comparison signals.

But base evaluation in nanochat is not only one metric. It depends on the task system under `tasks/`.

`tasks/common.py` is the shared abstraction layer. It defines the common contract that task files follow.

The individual task files each bind that abstraction to a concrete benchmark or data source:

- `tasks/mmlu.py` for MMLU
- `tasks/arc.py` for ARC
- `tasks/gsm8k.py` for GSM8K
- `tasks/humaneval.py` for HumanEval-style coding eval
- `tasks/spellingbee.py` for narrower spelling/counting capability
- `tasks/smoltalk.py` for conversational data/task usage
- `tasks/customjson.py` for user-supplied JSONL task data

This is the point where the repo becomes a benchmarking system, not just a training system.

The important shift here is conceptual: before this stage, the code is trying to make a model learn. In this stage, the code is trying to determine what the model has learned.

## The base model is not the final product; it must be turned into a chat model

A pretrained next-token predictor is not yet the assistant behavior people expect from a chat interface. Nanochat has a dedicated post-training stack for that transition.

`scripts/chat_sft.py` is the supervised fine-tuning stage. It teaches the model to behave more like a conversational assistant by training on conversation-style or task-style data.

`scripts/chat_rl.py` pushes further by running a reinforcement-learning stage on top of the chat model. That stage is where reward-driven behavior refinement enters the system.

`scripts/chat_eval.py` measures the result of those post-training stages. It is separate from `base_eval.py` because the target behavior is different: this is not just raw language-model competence anymore, but assistant-like interaction quality.

`nanochat/execution.py` belongs here because some chat/evaluation workflows require running or checking code emitted by the model. It is not part of the base transformer, but it becomes relevant once the model is being used in code-producing or tool-using settings.

Two task/data files connect strongly to this stage:

- `tasks/smoltalk.py`
- `tasks/customjson.py`

These matter here because post-training depends heavily on conversation-style or custom-structured data, not just generic pretraining text.

At this stage, the repo has moved from:

- training a base language model

into:

- shaping and evaluating a chat model

That is the post-training boundary.

## Once the chat model exists, it needs a runtime surface

After chat tuning, the repo shifts again. The question is no longer how to train the model, but how to use it.

This starts in `nanochat/engine.py`.

`nanochat/engine.py` is the inference engine. It is the runtime bridge between trained weights and actual token-by-token generation. If `gpt.py` is the model definition and `base_train.py` is the training runtime, then `engine.py` is the serving-time runtime.

On top of that engine, nanochat provides two user-facing surfaces.

`scripts/chat_cli.py` is the command-line interface. It is the simplest interactive path: no browser, no server indirection beyond what the script itself manages.

`scripts/chat_web.py` is the web-serving layer. It hosts the browser-facing experience and API behavior.

`nanochat/ui.html` is the frontend for that web path.

`nanochat/logo.svg` also belongs here, though only as a supporting asset. It is part of the UI/docs presentation layer, not of the inference logic itself.

So the repo now has a complete story from trained model to interactive system:

- load checkpoint
- run inference engine
- expose model through CLI or web

That is the serving stage.

## The repo is also an experiment machine

Nanochat is not just a single training program. It is a system for running and comparing experiments. That is the role of `runs/`.

The files here are not the core implementation. They are recipes that wire the implementation together.

`runs/speedrun.sh` is the main fast-path recipe. It is the most direct expression of how the author expects the end-to-end stack to be exercised.

`runs/miniseries.sh` is for structured series-style runs, useful when sweeping across a family of model configurations.

`runs/scaling_laws.sh` is for scaling-law style comparisons across budgets or model settings.

`runs/runcpu.sh` is a lighter-weight path for CPU or MPS machines, useful when you want to exercise codepaths locally without full multi-GPU infrastructure.

These files are important because they encode expected usage patterns. The implementation may tell you what the system can do; the run scripts tell you what the system is actually used for.

## The repo also tells you what it considers fragile

The `tests/` directory is small, but it is high signal.

`tests/test_attention_fallback.py` protects the attention backend boundary. That means the repo treats divergence between flash and fallback attention paths as a correctness-critical risk.

`tests/test_engine.py` protects the inference engine path. That means the repo treats generation/runtime behavior as another correctness-critical boundary.

A useful way to read tests in a repo like this is not just “what do they test?”, but “what did the author think was risky enough to need protection?”

In nanochat, at least two of those answers are clear:

- attention backend correctness
- inference engine correctness

## Finally, there is the surrounding research workflow

The `dev/` directory contains files that are not the main runtime, but they reveal how the project is developed and evaluated over time.

`dev/LOG.md` is the running experiment log. It records findings and shifts in understanding.

`dev/LEADERBOARD.md` defines the leaderboard framing and target metric context. It tells you what the project is trying to optimize toward in a comparative sense.

`dev/repackage_data_reference.py` and `dev/gen_synthetic_data.py` are utility scripts that support data-side workflows.

`dev/scaling_analysis.ipynb` and `dev/estimate_gpt3_core.ipynb` are analysis notebooks. They are not part of the main runtime path, but they are part of the intellectual workflow around the repo.

`dev/generate_logo.html`, `dev/nanochat.png`, and `dev/scaling_laws_jan26.png` are support assets and visual outputs.

There is also one metadata-oriented file outside the main runtime structure:

- `.claude/skills/read-arxiv-paper/SKILL.md`

That file belongs to local tooling/assistant metadata rather than to the nanochat model stack itself.

## Reading order

A useful reading order follows the same build sequence described above.

Start with the base stack:

1. `README.md`
2. `nanochat/tokenizer.py`
3. `nanochat/dataset.py`
4. `nanochat/dataloader.py`
5. `nanochat/gpt.py`
6. `scripts/base_train.py`

Then read evaluation:

7. `scripts/base_eval.py`
8. `nanochat/core_eval.py`
9. `tasks/common.py`
10. the individual task files in `tasks/`

Then post-training:

11. `scripts/chat_sft.py`
12. `scripts/chat_rl.py`
13. `scripts/chat_eval.py`
14. `nanochat/execution.py`

Then runtime:

15. `nanochat/engine.py`
16. `scripts/chat_cli.py`
17. `scripts/chat_web.py`
18. `nanochat/ui.html`

Then experiment wiring and support material:

19. `runs/speedrun.sh`
20. `runs/miniseries.sh`
21. `runs/scaling_laws.sh`
22. `tests/test_attention_fallback.py`
23. `tests/test_engine.py`
24. `dev/LOG.md`
25. `dev/LEADERBOARD.md`

## Condensed picture

Nanochat is a compact LLM training and experimentation stack.

- `nanochat/` contains the implementation.
- `scripts/` runs each stage.
- `tasks/` defines benchmark and task behavior.
- `runs/` encodes experiment recipes.
- `tests/` protects correctness-critical runtime paths.
- `dev/` records the surrounding research workflow and supporting utilities.

That is the structural map for the rest of the learning material.
