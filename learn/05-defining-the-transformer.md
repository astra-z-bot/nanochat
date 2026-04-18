# Chapter 5 — Defining the Transformer

By the end of the batching stage, nanochat can produce distributed `(x, y)` tensors of shape `(device_batch_size, max_seq_len)`. Those tensors are now ready for training, but they still have nowhere to go. The system has built data and representation; it has not yet built the network that will consume them.

That is the job of the model-definition stage.

In nanochat, this stage lives primarily in `nanochat/gpt.py`. That file does more than declare a few layers. It defines the architecture contract, the parameter layout, the initialization scheme, the forward pass, the parameter taxonomy used by the optimizer, and the structural state that checkpoints must later reconstruct. `scripts/base_train.py` and `nanochat/checkpoint_manager.py` sit around it, but `nanochat/gpt.py` is where the transformer becomes a concrete object instead of a plan.

## 1. Model Definition as the Next Bottleneck

Once batches exist, the next systems problem is defining the exact network those tokens will enter.

That question is more specific than it sounds. The model definition must settle the vocabulary size, the residual-stream width, the number of layers, the attention head layout, the per-layer context policy, the parameter initialization, and the output interface used by both training and inference. If any of those stay vague, the training loop cannot even allocate the model, let alone optimize it.

Nanochat chooses to keep all of that architecture state close together. Rather than scattering geometry in one file, initialization in another, and optimizer-aware parameter grouping in a third, the repo makes `nanochat/gpt.py` the place where the transformer’s internal contract is spelled out.

That means this chapter is about more than the forward pass. It is about the exact shape of the learnable object that Chapter 4’s batches are about to train.

## 2. `GPTConfig` and Build-Time Geometry

The first object that matters in `nanochat/gpt.py` is `GPTConfig`.

This dataclass is small, but it defines the persistent geometry of the model:

- `sequence_len`
- `vocab_size`
- `n_layer`
- `n_head`
- `n_kv_head`
- `n_embd`
- `window_pattern`

Those fields are enough to determine the full parameter layout of the transformer.

The important point is that nanochat does not treat this config as a cosmetic container. `scripts/base_train.py` constructs it from real upstream constraints. The sequence length comes from the batching stage. The vocabulary size comes from the tokenizer artifact. The model width and head count are derived from run arguments such as depth, aspect ratio, and head dimension.

That construction happens in `build_model_meta(...)`:

```python
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern=args.window_pattern,
)
```

This is where the earlier stages finally constrain the model.

The tokenizer fixes `vocab_size`. The dataloader fixes `sequence_len`. The training recipe fixes the scaling geometry. By the time `GPTConfig` is created, the model definition is no longer abstract; it is tied to the exact tokenizer, batch shape, and experiment configuration of the current run.

One more field deserves attention here even before the attention chapter: `window_pattern`. Nanochat does not treat attention span as a runtime afterthought. It stores the per-layer context policy directly inside model configuration, which means context geometry is part of the architecture itself.

## 3. Meta-Device Construction

Nanochat does not instantiate the model by immediately allocating and initializing full tensors on the target device.

Instead, `scripts/base_train.py` first builds it on the PyTorch meta device:

```python
with torch.device("meta"):
    model_meta = GPT(config)
```

That choice is important because `GPT.__init__` is written with that constraint in mind. The file even calls it out as a footgun: inside `__init__`, shapes and dtypes are real, but storage is not. So `__init__` is allowed to define modules, buffers, and parameter shapes, but not to rely on real initialized values.

The full construction path in `scripts/base_train.py` is therefore three-stage:

1. build the shape-only model on the meta device
2. call `to_empty(device=device)` to allocate storage on the real target device
3. call `init_weights()` to perform the actual parameter initialization

That separation is one of the more distinctive parts of nanochat’s model-definition path. The architecture is defined first, storage is allocated second, and actual data values appear only in the explicit initialization function.

This same pattern reappears later in `nanochat/checkpoint_manager.py` when a model is rebuilt from checkpoints. The ability to reconstruct a model cleanly from saved config depends on this architecture/data split being explicit.

## 4. `GPT.__init__`: The Model Skeleton

Once `GPT(config)` is entered, `nanochat/gpt.py` has to turn the config into a real module graph.

The highest-level structure is compact:

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),
    "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
})
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

This already tells you most of the architectural story. There is one token embedding table, one ordered stack of `Block` modules, and one output projection back into vocabulary space.

Two details matter immediately.

First, the embedding table and the output head are untied. Nanochat is not reusing the token embedding weights as the output projection. The file header calls this out explicitly. That means the model treats input representation and output decoding as separate learned objects.

Second, the vocabulary is padded upward to a multiple of `pad_vocab_size_to`, which defaults to `64`. This padding does not change the tokenizer vocabulary itself. It is an internal optimization so that embedding and output matrices have shapes that are friendlier to hardware and distributed execution. The padded logits are sliced back down to the true vocabulary size during the forward pass.

### 4.1 The Core Trunk

The trunk inside `self.transformer` is intentionally minimal.

- `wte` maps token IDs into residual-stream vectors of width `n_embd`
- `h` is the ordered stack of transformer blocks
- `lm_head` turns the final residual stream back into logits over the padded vocabulary

There are no learned positional embeddings anywhere in this core trunk. Nanochat uses rotary position encoding instead, so sequence position is handled later through precomputed cosine and sine buffers rather than through a second embedding table.

That is a meaningful design choice. The model’s persistent learned input state is dominated by token embeddings, blocks, and output head. Position handling is present, but it is not parameterized the same way as in older GPT variants with learned positional embeddings.

### 4.2 Extra Architecture State

`GPT.__init__` also defines several pieces of state that make this model more than a plain “stack of attention + MLP blocks” implementation.

The first is `self.window_sizes`, computed by `_compute_window_sizes(config)`. This turns `window_pattern` into one window-size tuple per layer. The attention implementation uses those values later, but the architectural decision about which layers see long versus short context is made here.

The second is the pair of per-layer scalar parameters:

- `self.resid_lambdas`
- `self.x0_lambdas`

These are learned residual-control parameters. `resid_lambdas` scales the running residual stream at each layer. `x0_lambdas` blends the original embedding back into the stream at each layer. In other words, nanochat makes residual routing itself part of the learnable architecture state.

The third is `self.value_embeds`, an embedding table for alternating layers determined by `has_ve(...)`. These value embeddings are not part of the base token embedding path. They are extra per-token learned vectors that feed into attention’s value path on selected layers.

Finally, `GPT.__init__` precomputes rotary cosine and sine tables and registers them as non-persistent buffers. They are real runtime state, but they are not checkpoint payload. That distinction matters: checkpoints save learned parameters, while rotary tables are regenerated from architecture shape when the model is rebuilt.

## 5. Local Building Blocks

The full model graph is assembled from a small number of local primitives inside `nanochat/gpt.py`.

These helpers matter because they show what nanochat considers part of the model definition proper versus what it considers an implementation detail delegated to later chapters.

### 5.1 `norm` and `Linear`

Nanochat’s normalization helper is extremely small:

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

This is RMSNorm with no learnable affine parameters. That means normalization is present throughout the architecture, but it does not add scale-and-shift parameters of its own. The model is therefore using normalization as a fixed functional transform rather than as another learned parameter family.

The custom `Linear` subclass is equally important. Its forward pass casts the weight matrix to the activation dtype before applying `F.linear(...)`. The surrounding comment explains why: master weights stay in high precision for optimizer stability, while actual matmuls run in the compute dtype, usually `bfloat16`.

So even the basic linear layer wrapper is encoding a model-wide policy about precision. Compute dtype is not just a training-script concern. It is built into the model definition boundary.

### 5.2 `MLP` and `Block`

`MLP` is the simpler of the two structural submodules.

It expands from `n_embd` to `4 * n_embd`, applies `relu(x).square()`, and projects back down. So the activation path is not GELU and not SwiGLU. It is explicitly the `relu^2` variant named in the file header.

`Block` then combines attention and MLP in a pre-norm residual shell:

```python
x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
x = x + self.mlp(norm(x))
```

This tells you exactly where the model-level structure ends and the attention-implementation chapter begins.

At this stage, the important point is that every block is built from two residual updates over a normalized stream: one through `CausalSelfAttention`, one through `MLP`. The block does not own positional state, cache policy, or context-window logic directly; those arrive as arguments.

That is a clean separation. `Block` defines the residual skeleton of the transformer. The exact mechanics of attention execution come later.

## 6. Initialization Is Part of the Architecture

In nanochat, `init_weights()` is not a minor cleanup step after module construction. It is part of the architecture contract.

Because the model was first built on the meta device, this function is the first place where real parameter values appear. It therefore has to define the initial operating regime of every parameter family.

Several choices are especially consequential:

- `wte` is initialized with a relatively large normal distribution
- `lm_head` is initialized with a much smaller normal distribution
- attention `q`, `k`, and `v` projections use uniform initialization scaled by `n_embd**-0.5`
- attention output projections and MLP output projections start at zero
- `resid_lambdas` start at `1.0`
- `x0_lambdas` start at `0.1`
- value embeddings follow the same scale as the value projection path
- value-embedding gates start as small positive weights

The zero initialization of `c_proj` and `mlp.c_proj` is particularly revealing. At initialization, each block’s additive update starts near neutral, which makes the residual stream conservative at the beginning of training. That is a structural stability choice, not just a random coding style.

`init_weights()` also regenerates the rotary buffers and casts embeddings into `COMPUTE_DTYPE` when the chosen precision policy allows it. So this function is simultaneously setting statistical initialization, reestablishing runtime buffers, and aligning the model with the current precision regime.

## 7. Forward Path from Token IDs to Loss

Once the model has been defined and initialized, `GPT.forward(...)` turns token IDs into either logits or a training loss.

The first step is positional preparation. The model slices the precomputed rotary buffers to the current sequence length, and if a KV cache is active it offsets those slices by the current cache position. That allows the same forward method to serve both training and streaming inference.

The actual trunk begins with token embedding:

```python
x = self.transformer.wte(idx)
x = x.to(COMPUTE_DTYPE)
x = norm(x)
x0 = x
```

The initial normalized embedding is saved as `x0`, because every layer later has the option to blend part of that original signal back into the residual stream.

The per-layer loop then applies two pieces of architecture-specific logic before each block runs. First, it reweights the running stream and the original embedding through `resid_lambdas[i]` and `x0_lambdas[i]`. Second, if the current layer owns a value embedding table, it looks up `ve` for the current token IDs and passes it into the block.

So the block input is not merely “the output of the previous layer.” It is a learned mixture of the current residual stream, the original embedded input, and possibly a separate value-embedding path.

After the block stack finishes, the model applies a final `norm(x)` and projects through `lm_head`. It then slices away any vocabulary padding and converts logits to `float32` before the final softcap:

```python
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
```

That softcap step is another example of nanochat preferring explicit runtime policy over invisible defaults. The logits are smoothly limited before loss computation or decoding.

If `targets` are provided, the method computes cross-entropy and returns a scalar loss. If not, it returns logits directly. That means `GPT.forward(...)` is the exact interface boundary between the architecture stage and the training stage: the training loop does not need to know the internals of the model, only that `(idx, targets)` can become a loss.

## 8. Optimizer Grouping as Architecture Metadata

One of the more unusual choices in `nanochat/gpt.py` is that the model defines `setup_optimizer(...)` itself.

This is not a convenience wrapper around a generic optimizer call. It is where the model partitions its own parameters into semantically different groups:

- transformer matrices
- token embeddings
- value embeddings
- output head
- residual scalars
- `x0` scalars

Those groups are then assigned different learning rates, optimizer kinds, betas, and weight-decay behavior. Matrix parameters are routed into Muon parameter groups, grouped again by shape. Embeddings, output head, and scalar parameters are routed into AdamW-style groups.

This tells you something important about nanochat’s architecture boundary. The repo does not believe the model can be defined independently of how its parameter families should be optimized. The model object itself owns the parameter taxonomy.

That does not replace the training chapter. `scripts/base_train.py` still owns the training horizon, scheduler policy, gradient accumulation, checkpoint cadence, and evaluation hooks. But the model file already decides which parameters are “matrix-like,” which are “embedding-like,” and which should be treated as special scalar controls.

## 9. Checkpoint Reconstruction

A model definition is only useful if the rest of the repo can rebuild it exactly.

That is why `nanochat/checkpoint_manager.py` matters in this chapter. Its `build_model(...)` function loads the saved checkpoint metadata, reconstructs `GPTConfig`, applies compatibility patches for older checkpoints when needed, and then rebuilds the model through the same meta-device path used during training.

The sequence is deliberate:

1. load checkpoint metadata and state tensors
2. reconstruct `GPTConfig` from saved `model_config`
3. instantiate `GPT` on the meta device
4. call `to_empty(device=device)`
5. call `init_weights()` so non-persistent runtime state exists
6. load the saved parameters with `load_state_dict(...)`

Two details are especially important.

First, the rebuild path strips `_orig_mod.` prefixes that can appear after `torch.compile`, so checkpoint structure is normalized back to the raw model form. Second, it asserts tokenizer compatibility by checking that `tokenizer.get_vocab_size()` matches the model config’s `vocab_size`.

That assert is the architectural handshake between Chapter 3 and this chapter. A trained checkpoint is not just “some weights.” It is weights that are only meaningful with the tokenizer vocabulary they were built against.

## 10. Key Files in This Stage

| File | Function in the system | What comes in | What goes out |
|---|---|---|---|
| `nanochat/gpt.py` | Defines the transformer architecture, initialization, forward path, and parameter grouping. | `GPTConfig`, token IDs, precision/runtime policy. | Losses, logits, parameter structure, optimizer groups. |
| `scripts/base_train.py` | Constructs the model for training from tokenizer and run configuration. | Tokenizer vocab size, run args, batch shape. | Concrete model instance and saved `model_config`. |
| `nanochat/checkpoint_manager.py` | Rebuilds the model from checkpoint metadata and state tensors. | Saved config, saved weights, tokenizer artifact. | Restored model consistent with the original training run. |

## 11. Required Concepts in This Chapter

### 11.1 Meta-Device Initialization

The PyTorch meta device lets the repo instantiate module structure without allocating real parameter storage yet. Nanochat uses that to separate architecture declaration from parameter initialization, which is why `init_weights()` becomes a first-class stage instead of an implementation footnote.

### 11.2 Padded Vocabulary and Untied Output Head

The embedding table and output head are not tied, and both are sized against a padded internal vocabulary for efficiency. The true tokenizer vocabulary is recovered later by slicing logits back to `config.vocab_size`.

That means the architecture’s internal matrix shapes are allowed to differ from the tokenizer’s exact external vocabulary size, as long as the forward path restores the true interface.

### 11.3 Architecture-Aware Parameter Groups

Nanochat does not let the optimizer discover parameter structure implicitly. The model itself declares that matrices, embeddings, value embeddings, and scalar residual controls are different kinds of parameters with different optimization needs.

That is why `setup_optimizer(...)` belongs here. Parameter grouping is part of the architecture contract, not just part of the training loop.

## 12. Non-Obvious Dependencies in This Stage

### 12.1 `torch.device("meta")`

This PyTorch subsystem is central to how nanochat builds large models. It allows `GPT(config)` to define module structure and buffer shapes without paying the allocation cost immediately. The downside is that `__init__` must be written carefully, because no real parameter values exist yet.

That is why the file treats `init_weights()` as the place where architecture becomes numerically real.

### 12.2 `nanochat.optim`

`nanochat/gpt.py` imports `MuonAdamW` and `DistMuonAdamW` from `nanochat.optim`. Those are not incidental training helpers. They are the reason `setup_optimizer(...)` can map architecture-defined parameter groups into the repo’s mixed Muon/AdamW optimization scheme.

So even though the training loop lives elsewhere, the model file is already coupled to a specific optimizer stack.

## 13. Reading Order inside This Stage

A clean order is:

1. `GPTConfig` and `GPT.__init__` in `nanochat/gpt.py`
2. `init_weights()` in `nanochat/gpt.py`
3. `forward()` in `nanochat/gpt.py`
4. `setup_optimizer()` in `nanochat/gpt.py`
5. `build_model_meta(...)` in `scripts/base_train.py`
6. `build_model(...)` in `nanochat/checkpoint_manager.py`

## 14. End-of-Chapter Synthesis

After this stage, the system has a concrete transformer object.

More specifically, it now has a saved configuration schema, a module graph, an initialization contract, a forward interface from token IDs to loss, and a rebuild path that can reconstitute the same architecture from checkpoints. The data pipeline no longer ends in anonymous tensors. It ends at a network that knows how to consume them.

The next stage is the attention path inside each block, because that is where this architecture turns residual streams into context-aware computation and where many of the repo’s most performance-sensitive implementation choices begin to matter.
