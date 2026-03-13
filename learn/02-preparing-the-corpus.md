# Chapter 2 — Preparing the Corpus

Before nanochat can train a tokenizer, it needs a large, stable stream of plain text.

That requirement sounds obvious, but it creates a real systems problem:

- the source corpus may not already be stored in the format the training code wants
- the dataset may be too large to keep as one monolithic file
- tokenizer training wants text, not pre-tokenized model batches
- later training wants the same corpus to be easy to stream and cache

So the dataset story in nanochat has two distinct parts:

1. **how the corpus is prepared into the repo’s shard format**
2. **how the runtime code downloads and iterates over those shards for tokenizer training**

This chapter follows that sequence.

## 1. Source Corpus and Reference Repackaging

The first relevant file is not in the main runtime package. It is `dev/repackage_data_reference.py`.

That is important because it tells you something about the repo design immediately: the training/runtime path is separated from the historical dataset-preparation path.

`dev/repackage_data_reference.py` is explicitly a reference script. It is not part of the normal runtime. It documents how the current pretraining corpus was assembled and repackaged before it was uploaded for later use.

The script supports two dataset histories:

- the old `fineweb_edu` path
- the newer `climbmix` path

The active branch in the file is `climbmix`.

That branch does something subtle and important.

It loads the source dataset from Hugging Face:

- `path="nvidia/Nemotron-ClimbMix"`
- `split="train"`

But the source column is not plain text. For ClimbMix, the script sets:

- `data_column_name = "tokens"`
- `tokenizer = tiktoken.encoding_for_model("gpt-2")`

That means the source dataset is already stored as GPT-2 token sequences. Before nanochat can train its **own** tokenizer, it has to decode those tokens back into text.

That decode happens here conceptually:

- read `doc[data_column_name]`
- if a tokenizer is present, decode tokens back into text
- append text into the current shard buffer

So even though the upstream source is tokenized, nanochat’s tokenizer-training path is built on recovered text, not on reusing the upstream tokenization.

That is the first key design choice.

## 2. Repackaging Objectives

The repackaging script is not just reformatting for convenience. It is creating a storage format that later runtime code can stream efficiently.

The script makes four important choices:

### 2.1 It shuffles the dataset

```python
# Source dataset
 ds = load_dataset(**dataset_kwargs)

# Shuffle to scramble the order
 ds = ds.shuffle(seed=42)
```

This means the stored shard order is already randomized once during preparation.

The point is not only statistical cleanliness. It also means the runtime dataloader can consume shards in order without inheriting a pathological source ordering.

### 2.2 It writes parquet shards

Each shard is written as a parquet file containing a single `text` column.

That matters because downstream runtime code in `nanochat/dataset.py` expects to read parquet row groups and extract `text` values directly. The repackaging script is therefore defining the schema that the runtime relies on.

The schema is simple on purpose:

- one parquet file per shard
- one `text` column
- no categorical tricks or fancy structure

### 2.3 It targets shard sizes, not just record counts

The script accumulates roughly `250_000_000` characters per shard before writing, with a `row_group_size` of `1024`.

That creates a compromise:

- shards are large enough to amortize IO overhead
- row groups are still small enough for incremental reading
- the runtime can stream row groups instead of materializing whole shards

The row-group detail matters later because nanochat’s runtime iterator works at row-group granularity.

### 2.4 It uploads the prepared shards for runtime download

At the end, the script uploads the repackaged parquet shard folder to Hugging Face using `HfApi.upload_large_folder`.

That is how the “offline” preparation path becomes the “online” runtime path.

The runtime no longer needs to know how ClimbMix was originally formed. It only needs to know where the prepared shards live and what format they have.

## 3. Runtime Dataset Contract

Once the repackaged dataset exists remotely, the runtime path becomes very small and direct.

At the top of `nanochat/dataset.py`, the file defines the concrete dataset contract used by the rest of the system:

- `BASE_URL`
- `MAX_SHARD`
- `index_to_filename`
- `DATA_DIR`

The values tell the whole story:

- the dataset is hosted remotely on Hugging Face
- shards are named `shard_00000.parquet`, `shard_00001.parquet`, ...
- the local cache directory is `base_data_climbmix`

This file is not a general dataset abstraction layer. It is the concrete runtime adapter for the project’s current pretraining corpus.

That is an important distinction.

## 4. Local Cache Layout

`nanochat/dataset.py` uses `get_base_dir()` from `nanochat/common.py`.

`get_base_dir()` does one simple thing:

- if `NANOCHAT_BASE_DIR` is set, use that
- otherwise use `~/.cache/nanochat`

So the dataset ends up, by default, under:

- `~/.cache/nanochat/base_data_climbmix`

That means the repo itself stays relatively clean. Large data artifacts live in a cache directory, not under the git checkout.

This is the storage convention that later tokenizer and training stages depend on.

## 5. Local Shard Discovery

The first real utility in `nanochat/dataset.py` is `list_parquet_files()`.

Its job is simple:

- look into the dataset directory
- collect all `.parquet` files
- ignore temporary `.tmp` files
- return the sorted list of full paths

That sorted ordering matters because later logic treats the **last shard** specially.

This function also contains legacy migration logic. If the new ClimbMix directory is missing, it can warn and fall back to the old `base_data` location from the previous FinewebEdu setup.

That fallback tells you the repo recently changed corpus, but the code still carries a transition bridge so old local setups do not fail immediately.

The warning message in this function is also revealing because it documents the expected migration sequence:

1. download the new ClimbMix shards
2. retrain the tokenizer on the new corpus

That is exactly the bridge between this chapter and the tokenizer-training chapter.

## 6. Row-Group Iteration

The central runtime function for this chapter is `parquets_iter_batched(split, start=0, step=1)`.

This function is the bridge from local parquet shards to an iterable text stream.

Its behavior is precise:

1. get all shard paths via `list_parquet_files()`
2. split them into:
   - `train`: all shards except the last one
   - `val`: only the last shard
3. open each shard with `pyarrow.parquet.ParquetFile`
4. iterate through row groups, not full files
5. extract the `text` column from each row group
6. yield each row group as a Python list of text documents

There are two structural choices worth noticing here.

### 6.1 Validation is pinned to the last shard

Nanochat does not create a separate validation dataset object here. It simply reserves the last shard as validation.

That is a very lightweight split policy, but it works because the repackaging stage already shuffled the global dataset before shard creation.

### 6.2 Iteration is row-group-based

The iterator yields batches at the parquet row-group level rather than yielding one document at a time or loading an entire shard at once.

That is the storage/runtime handshake created by the earlier repackaging step:

- `repackage_data_reference.py` writes row groups of size 1024
- `parquets_iter_batched()` reads those row groups back as natural iteration units

So the runtime loader is not arbitrary. It was designed together with the repackaging strategy.

## 7. Download and Retry Logic

The next important part of `nanochat/dataset.py` is `download_single_file(index)`.

This is the runtime download path.

Given a shard index, it:

1. computes the expected local file path
2. skips download if the file already exists
3. builds the remote URL from `BASE_URL` and `index_to_filename(index)`
4. downloads with `requests.get(..., stream=True)`
5. writes to a temporary `.tmp` path first
6. renames the temp file into place only after the download succeeds
7. retries with exponential backoff up to 5 times on failure

Three implementation details matter here.

### 7.1 Temporary-file write then atomic rename

This prevents partially downloaded shards from being mistaken for valid data files.

### 7.2 Exponential backoff

The dataset is large and fetched over the network. Retry logic is part of the normal operating design, not a rare edge-case add-on.

### 7.3 Runtime dataset code does its own download logic

Even though `nanochat/common.py` contains a generic `download_file_with_lock()` helper, `nanochat/dataset.py` does not use it here. Instead, it uses its own requests-based shard downloader and parallelizes at the shard level via multiprocessing.

That is a useful architectural detail: the dataset path is specialized rather than fully unified under a generic download abstraction.

## 8. Dataset Download CLI

The `if __name__ == "__main__":` block in `nanochat/dataset.py` turns the file into a runnable downloader.

Its CLI contract is simple:

- `-n / --num-files`: how many training shards to download
- `-w / --num-workers`: how many parallel download workers to use

The crucial policy is this:

- the user chooses how many **train** shards to fetch
- the **validation shard is always fetched**

That means even a partial local dataset still includes a fixed validation slice.

This is useful because tokenizer training and early experimentation do not need the full corpus, but evaluation consistency still benefits from always having the validation shard present.

The actual download work is distributed through `multiprocessing.Pool.map(download_single_file, ids_to_download)`.

So operationally, `nanochat.dataset` is both:

- a library module used by tokenizer/training code
- a direct CLI utility for assembling the local shard cache

## 9. Tokenizer-Training Ingress

Now the dataset management stage connects directly to `scripts/tok_train.py`.

This file answers a very specific question:

> Given the prepared local shard cache, how do we stream text into tokenizer training?

The answer is the `text_iterator()` function.

That function does three things:

1. it consumes batches from `parquets_iter_batched(split="train")`
2. it truncates each document to `args.doc_cap`
3. it stops after `args.max_chars` total characters

This tells you exactly how tokenizer training sees the dataset:

- training only, not validation
- row-group batches flattened into a document stream
- per-document length capped
- total training budget capped by character count

Those two caps are important.

### 9.1 Document cap: `doc_cap`

A small number of extremely long documents can otherwise dominate tokenizer merge learning. Cropping prevents a few giant examples from shaping the tokenizer disproportionately.

### 9.2 Character cap: `max_chars`

Tokenizer training is treated as a budgeted process. The tokenizer is not necessarily trained on the full corpus every time; it is trained on a controlled slice of the corpus stream.

This makes tokenizer experiments cheaper and more repeatable.

## 10. Tokenizer-Training Outputs

Once the text iterator exists, the rest of `tok_train.py` is straightforward:

- train a tokenizer with `RustBPETokenizer.train_from_iterator(...)`
- save tokenizer artifacts under the base directory
- run an encode/decode sanity check
- compute `token_bytes.pt`

The `token_bytes.pt` part is easy to overlook, but it matters.

Nanochat wants to report **bits per byte**, not just token loss, because bits-per-byte is less tied to a specific tokenizer vocabulary size.

So after tokenizer training, the script precomputes a mapping from token id to the number of bytes represented by that token. That artifact is later used to normalize evaluation metrics in a tokenizer-invariant way.

This means tokenizer training in nanochat is not just about producing vocab/merge files. It is also about producing the byte-accounting metadata needed by later evaluation.

## 11. End-to-End Corpus Flow

Putting the files together, the story is:

1. the original source corpus is loaded in `dev/repackage_data_reference.py`
2. if needed, tokenized upstream data is decoded back into text
3. the corpus is shuffled and written into parquet shards with fixed row-group structure
4. the shard set is uploaded to Hugging Face
5. `nanochat/dataset.py` defines where those shards live remotely and locally
6. missing shards are downloaded into `~/.cache/nanochat/base_data_climbmix`
7. `parquets_iter_batched()` streams row groups of text documents from those shards
8. `scripts/tok_train.py` flattens that stream, caps document length and total character budget, and feeds the text into tokenizer training
9. tokenizer artifacts and token-byte metadata are saved for later training and evaluation stages

That is the dataset-management pipeline for the tokenizer phase.

## 12. Reading Order

The natural next chapter is the tokenizer itself:

- `nanochat/tokenizer.py`
- `scripts/tok_train.py`
- `scripts/tok_eval.py`

This chapter stops at the point where text has been staged and streamed correctly. The next one should explain how nanochat actually learns and evaluates the tokenizer built on top of that dataset path.
