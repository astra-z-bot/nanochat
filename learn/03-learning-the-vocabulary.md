# Chapter 3 — Learning the Vocabulary

By the end of the previous stage, nanochat has a stable stream of text coming out of parquet shards. That solves the storage problem, but it does not yet solve the representation problem. The model cannot train directly on raw strings. It needs a discrete vocabulary and a deterministic way to move between text and token IDs.

That is the job of the tokenizer stage.

In nanochat, tokenizer training is treated as its own subsystem rather than as a one-time preprocessing footnote. That decision shows up in the repo layout immediately. There is a dedicated runtime interface in `nanochat/tokenizer.py`, a dedicated training entrypoint in `scripts/tok_train.py`, and a dedicated evaluation entrypoint in `scripts/tok_eval.py`. The tokenizer is not just a library object tucked inside training code. It is a first-class artifact that the rest of the stack depends on.

## 1. Why tokenizer training is a separate stage

The tokenizer sits between two very different worlds.

Upstream, there is plain text. That text can be long, messy, heterogeneous, and expensive to store or stream. Downstream, there is a transformer that expects fixed integer IDs and later expects those IDs to correspond to meaningful byte counts, sequence lengths, and runtime decode behavior.

Once that interface is wrong, everything that follows becomes distorted.

A weak tokenizer wastes context length. A mismatched tokenizer breaks checkpoint compatibility. A tokenizer whose runtime encoding path differs from its training path can quietly invalidate the entire experiment stack.

That is why nanochat separates tokenizer work into its own chapter of the pipeline. Before the model learns parameters, the system first learns the vocabulary through which all later learning will be expressed.

## 2. `nanochat/tokenizer.py`: the runtime boundary

The center of this stage is `nanochat/tokenizer.py`.

This file is the tokenizer boundary for the whole project. It is where nanochat defines what a tokenizer means operationally inside the repo. That includes three core behaviors:

1. turning text into token IDs
2. turning token IDs back into text
3. loading or saving the tokenizer artifacts that make those two operations stable across runs

Conceptually, this file does not exist only to “wrap a tokenizer library.” It exists to stabilize the contract between tokenizer training and the rest of the system.

That contract has to survive multiple later stages:

- base pretraining needs consistent token IDs
- evaluation needs the same encoding assumptions as training
- chat finetuning needs to remain compatible with the base model’s vocabulary
- inference needs a fast and reliable encode/decode path
- bits-per-byte metrics need the tokenizer to preserve byte accounting cleanly

So `nanochat/tokenizer.py` is really an artifact boundary. Once the tokenizer is trained, this file becomes the place where the rest of the system says, “give me the encoding behavior that corresponds to the saved vocabulary.”

This is also where a subtle design constraint appears: tokenizer training and tokenizer runtime are not identical problems. Training a tokenizer may use one backend or one set of utilities, while fast inference-time encoding may use a different one. Nanochat’s tokenizer layer exists in part to prevent that distinction from leaking into the rest of the codebase.

## 3. `scripts/tok_train.py`: how the vocabulary is learned

If `nanochat/tokenizer.py` is the stable runtime boundary, `scripts/tok_train.py` is the builder.

This script answers the practical question:

> given the prepared corpus stream from Chapter 2, how do we train the tokenizer artifacts that the rest of the system will rely on?

The first thing this script does is connect back to the dataset path from the previous chapter. It does not invent its own dataset interface. Instead, it consumes text through the iterator built on top of `parquets_iter_batched(...)` in `nanochat/dataset.py`.

That is an important design choice. The tokenizer stage and the later model-training stage are grounded in the same corpus storage format. The tokenizer is not trained from a separate ad hoc text dump. It is trained from the same managed shard pipeline that the repo uses for the rest of its data flow.

Inside `tok_train.py`, the dataset stream is narrowed into a tokenizer-training stream. The script caps document length and total character count before training begins. That turns tokenizer training into a budgeted process rather than an unbounded pass over the corpus.

This budget matters for two reasons.

First, tokenizer training does not need every last document to be useful. The purpose of the tokenizer stage is to learn a stable vocabulary over a large representative text sample, not to exhaustively consume the whole corpus every time.

Second, tokenizer experiments become much easier to repeat when the corpus budget is explicit. If vocabulary size changes or a training run is repeated, the tokenizer stage can still be compared under controlled input size constraints.

After building this text iterator, `tok_train.py` hands the stream to the tokenizer-training backend. This is the moment where the vocabulary is actually learned: merges are selected, token boundaries are induced, and a persistent tokenizer artifact is created.

The script then saves those artifacts into the repo’s configured base directory, where later stages can find them without needing to know how they were trained.

So the tokenizer training path has a clean structure:

1. get text from the prepared shard iterator
2. cap the stream into a bounded training budget
3. train tokenizer artifacts
4. save those artifacts to a stable location

That is the artifact-construction half of the vocabulary stage.

## 4. Tokenizer training is not finished when the tokenizer file is saved

Nanochat does one more thing in `tok_train.py` that is easy to miss but very important for the later training and evaluation pipeline.

It computes token-byte metadata.

This matters because the repo does not want to reason only in token counts. Token counts depend on the tokenizer. If the tokenizer changes, token-level losses and compression behavior can shift in ways that make raw cross-tokenizer comparisons misleading.

The repo addresses this by carrying byte-level accounting forward. In practice, that means the tokenizer stage is responsible not only for producing the vocabulary and encode/decode behavior, but also for producing auxiliary metadata that later metrics can use to normalize behavior in byte space.

This is where the tokenizer stage starts to influence later evaluation directly.

Without this metadata, the tokenizer would still work for training. But some of the repo’s later measurement conventions — especially byte-normalized ones — would be much weaker.

That is a characteristic nanochat design choice: tokenizer training is treated as part of the measurement pipeline, not only as part of the preprocessing pipeline.

## 5. `scripts/tok_eval.py`: the tokenizer has to be judged too

Nanochat does not assume that once a tokenizer has been trained, the job is done. There is a separate script for evaluating tokenizer quality: `scripts/tok_eval.py`.

That file exists because tokenizers encode tradeoffs.

A tokenizer can be judged along several axes:

- how efficiently it compresses text into tokens
- how stable its segmentation is across different text domains
- how well it preserves byte accounting for later metrics
- how it affects downstream sequence length and therefore training cost

The repo treats these as worth measuring explicitly rather than leaving them implicit.

This is structurally significant. The tokenizer is being treated the same way the model is treated: there is a training script and there is an evaluation script. That means the tokenizer is considered part of the experimental surface of the project, not merely a preprocessing artifact generated once and forgotten.

So the vocabulary stage in nanochat has the same high-level shape as later model stages:

- build the artifact
- evaluate the artifact
- feed the validated artifact into the next stage

## 6. The tokenizer stage is where byte space and token space meet

One of the deepest ideas in this part of the repo is that tokenization is not just about vocabulary quality. It is also about measurement discipline.

A model trains in token space, but language really exists in byte space or text space. If you only think in tokens, then changing the tokenizer can make the system appear better or worse for reasons that are partly representational rather than purely model-capability-related.

Nanochat’s tokenizer stage tries to reduce that confusion by preserving explicit byte-level information. This shows up in the token-byte artifact generation done during tokenizer training and in the way tokenizer evaluation is framed.

This is one of the most important conceptual points in the whole repository.

The tokenizer is not only a convenience layer between text and the model. It is also one of the variables that determines how expensive training is, how much context is available to the model, and how later metrics should be interpreted.

That is why this stage deserves to be separated out in the book.

## 7. How the tokenizer stage hands off to the next stage

Once the tokenizer has been trained and validated, the rest of the repo can begin treating token IDs as stable objects.

At that point:

- `nanochat/tokenizer.py` becomes the canonical runtime interface for encoding and decoding
- `nanochat/dataloader.py` can assume a concrete vocabulary and token stream format
- `nanochat/gpt.py` can be trained against a fixed token space
- base evaluation can interpret outputs under a stable tokenization scheme
- later chat finetuning and inference can reuse the same vocabulary boundary

This is the handoff that makes the next stage possible.

The corpus stage produced text shards. The tokenizer stage turns those shards into a vocabulary contract. The next stage will use that contract to build training batches and eventually optimize the transformer itself.

## 8. Key files in this stage

| File | Function in the system |
|---|---|
| `nanochat/tokenizer.py` | Runtime tokenizer interface and artifact boundary for encode/decode behavior. |
| `scripts/tok_train.py` | Tokenizer training entrypoint; learns tokenizer artifacts from the prepared corpus stream. |
| `scripts/tok_eval.py` | Tokenizer evaluation entrypoint; measures tokenizer quality before the model is trained on top of it. |
| `nanochat/dataset.py` | Supplies the corpus iterator that tokenizer training consumes. |
| `nanochat/common.py` | Provides shared runtime/base-directory helpers used to locate and persist tokenizer artifacts. |

## 9. Required concepts in this chapter

Three concepts matter here.

### 9.1 Vocabulary learning

The tokenizer stage is learning a discrete vocabulary over text, not model parameters over token IDs. That means the optimization target is representational efficiency and stability, not predictive accuracy over next tokens.

### 9.2 Compression as a tokenizer metric

Tokenizer quality is partly a compression problem. A tokenizer that expresses text more efficiently reduces average sequence length, which can improve effective context usage and reduce training cost.

### 9.3 Byte-normalized accounting

Nanochat’s metric design cares about byte-level interpretation, not only token counts. That is why tokenizer training emits additional token-byte information that later stages can reuse.

## 10. Non-obvious dependencies in this stage

Two non-obvious pieces matter here.

### 10.1 `rustbpe`

This is the tokenizer-training backend used for building BPE-style artifacts. It matters because tokenizer training and tokenizer runtime are not just generic string-processing utilities; they define the vocabulary contract of the whole system.

### 10.2 `tiktoken`

This enters the broader tokenizer story because the repo interacts with token/byte boundaries explicitly, and some earlier data preparation paths decode upstream tokenized corpora back into text. The exact train-time and runtime tokenization stack therefore matters more here than in a typical “just use a tokenizer” pipeline.

## 11. Reading order inside this stage

A clean order is:

1. `nanochat/tokenizer.py`
2. `scripts/tok_train.py`
3. `scripts/tok_eval.py`
4. revisit the relevant parts of `nanochat/dataset.py`
5. revisit `nanochat/common.py` only where tokenizer artifact paths are resolved

## 12. End-of-chapter synthesis

After this stage, the system has more than a corpus. It has a vocabulary.

That vocabulary is now a stable interface between text and model computation. The next stage can therefore stop thinking about strings and start thinking about training batches and model structure.

The natural next chapter is the one that follows from that handoff:

**from tokenized documents to training-ready batches.**
