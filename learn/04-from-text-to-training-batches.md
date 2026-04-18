# Chapter 4 — From Text to Training Batches

By the end of the tokenizer stage, nanochat has two things that matter enormously: a corpus stored as text-bearing parquet shards and a tokenizer that can deterministically turn those text documents into token IDs. That is enough to define the vocabulary boundary of the system, but it is still not enough to train a transformer.

Training wants something much more specific. It wants fixed-shape `torch.Tensor` objects on each device, already shifted into inputs and targets, already sharded across ranks, and delivered fast enough that the model does not sit idle waiting for Python and storage code.

That is the problem solved by the dataloader stage.

In nanochat, this stage is not implemented as a thin wrapper around `torch.utils.data.DataLoader`. The core logic lives in `nanochat/dataloader.py`, and it takes responsibility for almost the entire transformation from document stream to device-resident training batch. `scripts/base_train.py` then consumes that stream, scales it up through distributed training and gradient accumulation, and checkpoints enough loader state to resume later without restarting from the beginning.

## 1. The Batching Problem

The gap between “tokenizer exists” and “training can begin” is wider than it first appears.

A tokenizer can encode one document at a time, but a transformer training step wants dense rectangular tensors. That means the system has to solve several problems at once: it must choose which documents belong in the same micro-batch, decide how to pack variable-length token sequences into a fixed context length, preserve high GPU utilization, and keep the train and validation streams deterministic enough to checkpoint and resume.

Nanochat solves those problems in one place. Rather than pre-tokenizing the entire dataset into fixed training examples on disk, it keeps the dataset in text form and performs tokenization and packing inside the loader itself. That choice keeps the corpus representation simple, keeps tokenizer changes cheap, and lets the training stage rebuild batches directly from the current tokenizer artifact.

So the batching stage is really the point where three earlier decisions finally meet:

1. Chapter 2 stored the corpus as parquet shards of text documents.
2. Chapter 3 produced a tokenizer that can map text into stable token IDs.
3. This chapter turns those two ingredients into the `(x, y)` tensors that `GPT.forward(...)` can consume.

## 2. Distributed Document Ingress

The first important function in `nanochat/dataloader.py` is `_document_batches(...)`.

This function does not yield tensors. It yields small batches of raw documents, still as Python strings, along with enough position metadata to describe where those documents came from. That makes it the bridge between the corpus layer and the actual token-packing logic.

The first thing `_document_batches(...)` does is recover distributed-training context through `get_dist_info()` from `nanochat/common.py`. From there it can decide which rank is responsible for which part of the dataset. This is important because the loader is not allowed to behave like a single-process iterator if training is running under `torchrun` across multiple GPUs.

It then calls `list_parquet_files(...)` from `nanochat/dataset.py` to discover the prepared shard set. The split policy is deliberately simple and explicit:

- training uses all shards except the last one: `parquet_paths[:-1]`
- validation uses only the last shard: `parquet_paths[-1:]`

That means the loader inherits the same train/validation contract described in the corpus chapter instead of inventing a new split locally.

Within each parquet file, `_document_batches(...)` iterates by row group using `pyarrow.parquet.ParquetFile`. That is the crucial implementation choice for distribution. Each rank steps through row groups with stride `ddp_world_size`, so rank 0 sees row groups `0, world_size, 2*world_size, ...`, rank 1 sees `1, 1+world_size, ...`, and so on. The shard unit of work is therefore a parquet row group, not an individual row and not an entire file.

After reading a row group, the loader extracts the `text` column and converts it to a Python list. That list is then chunked into smaller document batches of size `tokenizer_batch_size` before being yielded onward. In other words, `_document_batches(...)` does just enough work to expose an infinite, distributed stream of text documents, but it stops short of tokenization.

The function also carries a small resume cursor: `(pq_idx, rg_idx, epoch)`. This cursor is not a perfect replay mechanism down to the exact document boundary, but it is good enough to restore the approximate point in the corpus without starting the run from scratch.

## 3. In-Loader Tokenization

The next responsibility belongs to `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)`.

This is the real training loader. It takes a tokenizer, the per-device batch size `B`, the maximum sequence length `T`, and the split name. It then pulls text batches from `_document_batches(...)`, tokenizes them, packs them, stages them through CPU memory, copies them onto the device, and finally yields training tensors.

The important design fact here is that tokenization still happens inside the loader. Nanochat does not persist a second dataset of tokenized examples for base pretraining. Instead, this function calls:

```python
doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
```

That means the loader is working from the current tokenizer artifact every time it runs. If the tokenizer changes, the training batches change automatically; there is no separate token-cache format that has to be regenerated and versioned independently.

The `prepend=bos_token` detail matters. Every document is explicitly prefixed with the beginning-of-sequence token before packing. That is what the rest of the loader’s “BOS-aligned” language is referring to. The packing stage is not assembling anonymous token fragments. It is assembling tokenized documents whose boundaries are marked.

This function also introduces `doc_buffer`, a working set of already tokenized documents. It is filled in batches through the nested `refill_buffer()` helper, which repeatedly tokenizes more text until the buffer reaches the requested `buffer_size`. This buffered design matters because the packer is not greedy in simple arrival order. It wants a menu of candidate documents so it can choose better fits for the remaining context space.

## 4. BOS-Aligned Best-Fit Packing

Once documents have become token lists, nanochat still has one more hard problem: how to turn variable-length documents into fixed-size rows that waste as little context as possible.

That logic is the core of the loader.

### 4.1 Row Capacity and Shifted Supervision

The packer does not build rows of length `T`. It builds rows of length `T + 1`:

```python
row_capacity = T + 1
row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
```

That extra token exists because a next-token language-modeling batch needs both inputs and shifted targets. Once a row of length `T + 1` is assembled, the loader slices it into:

- `inputs = row[:-1]`
- `targets = row[1:]`

So a single packed row becomes one `(x, y)` training example of shape `T`, where each target token is the next token after the corresponding input position.

This is a small implementation detail with big consequences. The packer is not merely filling context windows. It is directly constructing the supervision layout that the training loop expects.

### 4.2 Largest-Fit Selection from the Buffer

For each row, the loader repeatedly asks a simple question: given the remaining free space, which buffered document can fit entirely and uses the most space?

The code answers that by scanning the buffer and choosing the largest document whose length is less than or equal to the remaining capacity. When such a document exists, it is copied into the current row and removed from the buffer.

This is the “best-fit” part of the algorithm. The loader is not trying to preserve document order inside a row. It is trying to maximize utilization of the available context window while keeping document boundaries aligned to BOS-prefixed token streams.

The practical effect is that the loader produces dense batches without padding. Instead of letting a row trail off with unused positions, it tries to keep filling it with complete documents whenever possible.

### 4.3 Cropping When Nothing Fits

Eventually a row reaches a state where no buffered document fits into the remaining space.

At that point, nanochat does not pad. It crops.

More specifically, it chooses the shortest document in the buffer and truncates it to the exact remaining length needed to finish the row. This is the tradeoff exposed directly in the module docstring: the loader achieves effectively 100% utilization of the training window, but it does so by discarding some fraction of document tokens. The comment in the file gives the rough scale for the default long-context setting: around 35% of tokens may be lost to cropping at `T = 2048`.

That tradeoff is central to understanding this loader. Nanochat is choosing utilization over perfect preservation of every tokenized document boundary. The benefit is that no training capacity is spent on padding. The cost is that some document suffixes are never seen.

The loader is therefore optimizing for throughput and clean dense tensors, not for exact exhaustive replay of every token in the corpus.

## 5. Buffer Layout and Device Transfer

After the row packer fills `row_buffer`, the loader still does not yield it directly.

Instead, it stages the final tensors through a carefully chosen memory layout. This is where `nanochat/dataloader.py` becomes less like a textbook data pipeline and more like performance-sensitive systems code.

The function pre-allocates three main buffers once:

- `row_buffer`: shape `(B, T + 1)` on CPU, used while building packed rows
- `cpu_buffer`: shape `2 * B * T` on CPU, optionally pinned when training on CUDA
- `gpu_buffer`: shape `2 * B * T` on the target device

It then creates views into those flat buffers:

- `cpu_inputs` and `cpu_targets` as `(B, T)` views into `cpu_buffer`
- `inputs` and `targets` as `(B, T)` views into `gpu_buffer`

Once a row batch is ready, the loader copies `row_buffer[:, :-1]` into `cpu_inputs` and `row_buffer[:, 1:]` into `cpu_targets`. That produces the final shifted training pair without allocating new tensors for every batch.

Finally, it performs a single host-to-device copy:

```python
gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
```

This design does two useful things at once.

First, it avoids rebuilding many temporary Python lists or tiny tensors during every yield. Second, on CUDA it uses pinned host memory plus a single non-blocking copy into a persistent GPU buffer, which is a much better fit for high-throughput training than piecemeal transfers.

So the loader is not only solving “which tokens belong together?” It is also solving “how do we hand them to the GPU with minimal overhead?”

## 6. Training-Loop Consumption

Once `nanochat/dataloader.py` can emit `(inputs, targets, state_dict)`, `scripts/base_train.py` turns that stream into actual optimization steps.

The training script first loads the tokenizer and the token-byte metadata from the previous stage:

- `tokenizer = get_tokenizer()`
- `token_bytes = get_token_bytes(device=device)`

Then it constructs the loaders:

- training uses `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)`
- validation uses `tokenizing_distributed_data_loader_bos_bestfit(...)`

The distinction matters. Training needs resumable state. Validation only needs a fresh iterable that produces `(x, y)` tensors from the validation shard.

Immediately after construction, `base_train.py` pulls the first batch:

```python
x, y, dataloader_state_dict = next(train_loader)
```

That primes the pipeline before the main optimization loop begins.

From there, the script turns per-device micro-batches into a larger logical batch through distributed training and gradient accumulation. The per-rank token count for one forward/backward pass is:

- `device_batch_size * max_seq_len`

That is then multiplied by `ddp_world_size` to get tokens per micro-step across all ranks. Finally, `grad_accum_steps` is chosen so that repeated micro-steps add up to the requested `total_batch_size`.

So the loader itself is only responsible for micro-batches of shape `(B, T)`. The training script is responsible for combining many such micro-batches into the effective optimization batch seen by the optimizer.

Inside the training loop, consumption is intentionally pipelined. During each accumulation step, the script computes:

```python
loss = model(x, y)
```

and then immediately asks the loader for the next batch:

```python
x, y, dataloader_state_dict = next(train_loader)
```

The comment in the file says exactly why: it is prefetching the next batch while the GPU is busy with forward and backward work. That is the runtime handoff between the loader and the model stage.

## 7. Checkpoint Continuity

A loader this stateful would be awkward if the training script could not checkpoint it.

Nanochat addresses that directly. When `base_train.py` saves a checkpoint, the metadata JSON includes `dataloader_state_dict` alongside the model config, user config, batch-shape parameters, and other loop state.

That loader state comes directly from the train loader yield and carries three fields:

- `pq_idx`
- `rg_idx`
- `epoch`

On resume, `base_train.py` reads `meta_data["dataloader_state_dict"]` and passes it back into `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)` as `resume_state_dict`.

The resume behavior is intentionally approximate rather than exact. `_document_batches(...)` advances by row group after the recorded point so that resumed training does not simply repeat the same already-consumed row group. This means a resumed run may skip some documents within the partially consumed row group, but it preserves the more important property for long training runs: progress through the corpus continues forward instead of rewinding to an earlier region.

That is a sensible tradeoff for this pipeline. Exact replay would require much more fine-grained loader state, while the current design keeps the checkpoint metadata compact and robust.

## 8. Key Files in This Stage

| File | Function in the system | What comes in | What goes out |
|---|---|---|---|
| `nanochat/dataloader.py` | Converts text documents into packed `(x, y)` micro-batches. | Text shards, tokenizer, batch shape, DDP context. | Device-resident training tensors and loader resume state. |
| `scripts/base_train.py` | Consumes the loader during pretraining. | Micro-batches from the loader. | Forward/backward steps, gradient accumulation, checkpoints. |
| `nanochat/dataset.py` | Supplies the parquet shard list used by the loader. | Prepared corpus shards on disk. | Ordered train/validation shard discovery. |
| `nanochat/common.py` | Supplies distributed-rank context. | Runtime environment variables from `torchrun`. | Rank/world-size information used for row-group sharding. |
| `nanochat/checkpoint_manager.py` | Persists and restores training metadata around the loader. | Model state, optimizer state, loader cursor. | Resumable checkpoints. |

## 9. Required Concepts in This Chapter

### 9.1 Shifted Next-Token Targets

Language-model training does not supervise a row against itself. It supervises each token against the token that follows it. That is why the loader builds rows of length `T + 1` and then slices them into `(inputs, targets)` of length `T`.

Without that extra token, the loader could fill a context window, but it could not form the aligned prediction targets required by next-token training.

### 9.2 Packing, Cropping, and Utilization

Variable-length documents never line up cleanly with fixed context windows. There are only a few options: pad, drop leftovers, or crop. Nanochat chooses a best-fit packer that avoids padding and accepts some cropping.

That means the relevant tradeoff is not “is the loader lossless?” but “does the loader turn available context into useful training tokens efficiently enough?” In this design, utilization wins.

### 9.3 Distributed Micro-Batches and Logical Batch Size

The loader emits a per-rank micro-batch of shape `(device_batch_size, max_seq_len)`. That is not the optimizer batch yet. The true batch seen by training is built from:

- per-rank sequence count
- sequence length
- number of ranks
- gradient accumulation steps

So the dataloader stage defines the atomic training unit, while `base_train.py` defines how many such units make one optimization step.

## 10. Non-Obvious Dependencies in This Stage

### 10.1 `pyarrow.parquet`

`pyarrow.parquet` is what makes row-group-based iteration possible. The loader does not slurp whole parquet files into memory. It opens a `ParquetFile`, reads one row group at a time, and extracts the `text` column. That is why DDP sharding can happen at row-group granularity.

If the dataset were stored in a format without cheap row-group reads, this loader design would look very different.

### 10.2 CUDA Pinned Host Memory

When `device == "cuda"`, the loader allocates `cpu_buffer` with `pin_memory=True` and then performs a non-blocking copy into `gpu_buffer`. That is not cosmetic. Pinned host memory is what makes the host-to-device transfer path efficient enough to fit naturally inside the training loop.

Without it, the loader would still be correct, but the cost of getting batches onto the GPU would become more visible in end-to-end throughput.

## 11. Reading Order inside This Stage

A clean order is:

1. `_document_batches(...)` in `nanochat/dataloader.py`
2. `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)` in `nanochat/dataloader.py`
3. the loader initialization block in `scripts/base_train.py`
4. the gradient-accumulation training loop in `scripts/base_train.py`
5. the checkpoint save/load path in `scripts/base_train.py` and `nanochat/checkpoint_manager.py`

## 12. End-of-Chapter Synthesis

After this stage, the system has stopped dealing in documents and started dealing in tensors.

More specifically, it now has a reproducible path from text-bearing parquet shards to distributed, device-resident `(x, y)` micro-batches, plus enough loader state to resume training after interruption. The tokenizer is no longer just a learned artifact on disk. It is now embedded into the live batch-construction path.

That changes what the next necessary file must do. The remaining missing piece is the network that will consume those tensors and turn them into losses, gradients, and eventually a trained model.
