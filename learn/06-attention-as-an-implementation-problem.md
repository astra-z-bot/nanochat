# Chapter 6 — Attention as an Implementation Problem

By the end of the model-definition stage, nanochat has a concrete transformer object with blocks, residual paths, and a forward interface from token IDs to loss. But the most expensive and structurally delicate part of that model still sits behind one method call: attention.

That is where the architecture stops being a static graph and starts becoming an execution problem. Attention has to project the residual stream into queries, keys, and values, inject positional information, enforce causal structure, respect optional sliding-window limits, support group-query layouts, and switch cleanly between training-time full-sequence execution and inference-time KV-cache updates.

In nanochat, that responsibility is split across three files. `nanochat/gpt.py` defines the attention module and the per-layer window policy. `nanochat/flash_attention.py` provides the execution backend boundary that chooses between Flash Attention 3 and an SDPA fallback. `nanochat/engine.py` defines the KV cache structure used when attention runs incrementally during generation. Together, these files turn the abstract transformer from Chapter 5 into an actual context-processing system.

## 1. Why Attention Needs Its Own Stage

The model chapter could stop at “each block contains self-attention,” but that would hide most of the engineering substance of the repo.

In nanochat, attention is where several non-trivial design choices meet at once:

- head geometry is asymmetric because queries and key/value heads can differ
- position is injected through rotary embedding instead of learned positional parameters
- value paths can be modified by per-layer value embeddings and learned gates
- context length is not uniformly global because layers can use different window sizes
- training and inference take different execution branches because inference writes into a KV cache
- hardware support is not uniform, so execution must fall back from FA3 to SDPA without changing the model-facing API

That means attention is not a single formula in this repository. It is a bounded subsystem with its own input contract, layout constraints, and runtime branch structure.

## 2. `CausalSelfAttention` as the Structural Core

The center of this stage is `CausalSelfAttention` in `nanochat/gpt.py`.

This module receives a residual-stream tensor `x` of shape `(B, T, C)`, along with four additional pieces of state:

- `ve`, an optional value-embedding tensor for this layer
- `cos_sin`, the rotary position tensors for the current time range
- `window_size`, the left-context policy for this layer
- `kv_cache`, which is `None` during training and a live cache object during inference

That signature is already informative. Attention in nanochat is not just “input tensor in, output tensor out.” It depends on layer-specific context policy and on whether the model is executing in full-sequence or incremental mode.

Inside `__init__`, the module fixes the geometry of the attention path:

- `n_head` is the number of query heads
- `n_kv_head` is the number of key/value heads
- `head_dim` is `n_embd // n_head`

It then creates four projections:

- `c_q`
- `c_k`
- `c_v`
- `c_proj`

The first three map the residual stream into attention space. The last one returns the merged head output to the residual stream. Because `c_k` and `c_v` are sized by `n_kv_head` rather than `n_head`, the module is prepared for grouped-query attention from the start.

## 3. Queries, Keys, and Values Are Deliberately Asymmetric

The first thing `forward(...)` does is project the residual stream into Q, K, and V tensors:

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

The asymmetry here matters.

Queries are allocated one head per attention head, but keys and values can be shared across groups of query heads. That is what `n_kv_head <= n_head` and `n_head % n_kv_head == 0` are enforcing in the constructor. Nanochat therefore defines head layout in a way that can reduce KV-state cost without giving up a richer query decomposition.

This is not merely a future optimization hook. It affects both the live attention computation and the structure of the KV cache later in `engine.py`.

### 3.1 Group-Query Layout

Because queries and KV heads can differ, the attention backend has to know whether grouped-query attention is active. That logic shows up in `nanochat/flash_attention.py`, where the SDPA fallback computes:

```python
enable_gqa = q.size(1) != k.size(1)
```

and then passes that flag into `F.scaled_dot_product_attention(...)`.

So grouped-query attention is not just a config-time field. It is a live execution property carried all the way from `GPTConfig` to backend dispatch.

The practical consequence is most visible during inference. KV caches scale with `n_kv_head`, not `n_head`, which means the cached state can be significantly smaller than a symmetric multi-head design when grouped-query attention is used.

### 3.2 Value Embeddings and Learned Gates

Nanochat also modifies the value path in a non-standard way.

On selected layers, `GPT.forward(...)` looks up a value embedding tensor from `self.value_embeds` and passes it into the block. Inside `CausalSelfAttention.forward(...)`, that tensor is reshaped into `(B, T, n_kv_head, head_dim)` and mixed into `v` through a learned gate:

```python
gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
v = v + gate.unsqueeze(-1) * ve
```

This is one of the most distinctive parts of nanochat’s attention path.

The value embedding is not added blindly. The gate is input-dependent and computed per token, per KV head, from a small slice of the residual stream. That means the model can vary how strongly this extra value path enters attention as a function of the current hidden state.

So the `v` tensor in nanochat is not always just the output of `c_v(x)`. On layers that own value embeddings, it is a gated combination of the learned value projection and a separate embedding-derived contribution.

## 4. Positional Structure and Attention-Space Normalization

After Q, K, and V have been projected, nanochat still has not made them usable for attention. It next imposes the two transformations that define the model’s attention geometry: rotary position encoding and QK normalization.

### 4.1 Rotary Position Injection

`apply_rotary_emb(...)` is the helper that rotates pairs of channels using the cosine and sine tensors prepared earlier by the model.

The attention module applies it to queries and keys only:

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

This choice matches the architecture from Chapter 5. Position is not carried by a learned embedding table added to the residual stream. Instead, it is injected directly into the query/key geometry that governs attention scores.

The `cos_sin` argument itself is already a time-local slice. `GPT.forward(...)` computes it from the precomputed rotary buffers, offsetting by `kv_cache.get_pos()` during incremental inference when necessary. That means the attention module receives positional state that already corresponds to the current chunk or token range.

### 4.2 QK Norm and Explicit Sharpening

Immediately after rotary embedding, nanochat normalizes both `q` and `k`:

```python
q, k = norm(q), norm(k)
q = q * 1.15
k = k * 1.15
```

This is the QK-norm path mentioned in the file header.

The important detail is that normalization here happens in attention space, after projection and rotary application, not just once in the residual stream before the block. The result is that the score-producing vectors are explicitly normalized before the backend computes dot products.

Nanochat then scales both by `1.15`, which the inline comment describes as making attention sharper. The repo is therefore not treating the raw projection outputs as the final score geometry. It is applying an explicit post-processing policy to the Q/K path before any backend kernel sees them.

## 5. Layer-Local Context Windows

Attention in nanochat is not uniform across layers.

That policy is decided in `GPT._compute_window_sizes(...)`, which turns `config.window_pattern` into one `(left, right)` tuple per layer. The pattern string uses `L` for long and `S` for short context, tiles that pattern across the layer stack, and then forces the final layer to be long-context regardless of the repeating pattern.

The mapping is concrete:

- `L` becomes `(sequence_len, 0)`
- `S` becomes a reduced left window rounded up to the Flash Attention tile size

The rounding logic is not cosmetic. The code computes short windows with a ceiling step tied to `128`, because the attention backend wants tile-friendly sizes.

This means the model carries per-layer context policy as precomputed numeric window bounds, not as abstract labels. By the time `CausalSelfAttention.forward(...)` runs, each layer already knows the exact left-context limit it should enforce.

## 6. Training-Time Attention Execution

Once Q, K, V, positional state, and window size all exist, the attention module reaches its main runtime branch.

The first branch is the training path:

```python
if kv_cache is None:
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

This branch assumes the model is processing a full batch of tokens without an external cache. The attention kernel therefore receives only the current queries, keys, and values, plus two execution constraints:

- causal masking must be enforced
- the optional sliding-window bound must be respected

The attention module does not decide whether FA3 or SDPA is used. It delegates that to `nanochat/flash_attention.py`. That separation is important. `CausalSelfAttention` defines what has to be computed. The backend wrapper decides how to compute it on the current hardware.

The output `y` comes back in `(B, T, H, D)` layout, is flattened back into `(B, T, C)`, and is passed through `c_proj` to return to the residual stream.

## 7. Inference-Time Attention Execution

The second branch is the incremental inference path:

```python
else:
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
    )
```

This branch is where attention stops being purely local to the current input tensor. Instead of attending only over the current `k` and `v`, it writes the new keys and values into a persistent layer-local cache and then attends over the cache contents up to the current decode position.

The interface choice matters. `CausalSelfAttention` does not manipulate cache tensors directly beyond retrieving the layer views. It passes those views into the backend wrapper and expects the wrapper to handle the in-place update semantics.

The module then advances cache position only after the final layer has processed the current token chunk:

```python
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

That means cache position is treated as a model-wide decode cursor, not as independent per-layer bookkeeping. Each layer contributes its K/V slice for the current token span, and the position is advanced once after the full model pass has completed.

## 8. `nanochat/flash_attention.py` as the Backend Boundary

The attention module stays relatively clean because `nanochat/flash_attention.py` absorbs the backend-switching complexity.

This file exports a `flash_attn` namespace whose methods match the FA3 API shape, regardless of whether the actual implementation is Flash Attention 3 or the SDPA fallback. That means `gpt.py` can call:

- `flash_attn.flash_attn_func(...)`
- `flash_attn.flash_attn_with_kvcache(...)`

without caring which backend is currently active.

### 8.1 FA3 Selection

The detection path starts with `_load_flash_attention_3()`, which checks whether CUDA is available, whether the GPU major capability is `9` for Hopper, and whether the kernel package can be loaded. `_resolve_use_fa3()` then makes the final choice based on availability, override state, and compute dtype.

This is narrower than a generic “use flash attention when possible” rule. Nanochat is specifically targeting the FA3 Hopper path and otherwise falling back.

### 8.2 SDPA Fallback Semantics

When FA3 is unavailable, the wrapper converts the attention tensors from FA3’s preferred `(B, T, H, D)` layout into SDPA’s `(B, H, T, D)` layout, runs `_sdpa_attention(...)`, and transposes back.

The fallback helper is where masking semantics become explicit.

It has separate handling for three cases:

1. full-context equal-length attention, where `is_causal=True` is enough
2. single-token generation, where only a suffix of the cache may need to be visible under sliding-window limits
3. chunked or cache-offset inference, where an explicit boolean mask must be built because plain causal mode is no longer aligned to the current cache position

That last case is important. Once `Tq != Tk`, the backend cannot assume that the current query positions start at zero relative to the key sequence. The code therefore constructs row and column indices explicitly and builds the mask by hand.

This is the point where “causal attention” becomes an implementation problem instead of a conceptual one. The abstract rule is simple: no token should look rightward in time. The concrete mask depends on sequence offsets, cache position, and optional sliding-window constraints.

## 9. `KVCache` as the Inference Contract

The cache object that attention talks to is defined in `nanochat/engine.py`.

`KVCache` stores keys and values in a layout chosen to match FA3-style cache APIs:

- cache tensors are shaped `(n_layers, B, T, H, D)`
- per-layer views returned by `get_layer_cache(...)` are therefore `(B, T, H, D)`
- `cache_seqlens` tracks current position per batch element as `int32`

This layout choice matters because it keeps the cache tensors in the same head-major-last layout expected by the FA3 wrapper path. The fallback code in `flash_attention.py` can transpose for SDPA when needed, but the model-facing cache contract stays stable.

`Engine.generate(...)` shows how this contract is used in practice.

It first allocates a batch-1 prefill cache for the prompt, runs `model.forward(ids, kv_cache=kv_cache_prefill)`, and then clones that state into a larger decode cache for multiple sampled continuations. After that, each generation step passes only the newly selected token column back into `model.forward(...)`, along with the live decode cache.

So the inference path is not “rerun the full prompt every step.” It is “materialize prompt attention once, then extend cached attention state incrementally.” That is the runtime reason this cache contract exists at all.

## 10. Key Files in This Stage

| File | Function in the system | What comes in | What goes out |
|---|---|---|---|
| `nanochat/gpt.py` | Defines QKV projection, rotary/QK processing, layer window policy, and attention branching. | Residual stream, value embeddings, rotary slice, cache state. | Attention outputs returned to the block residual path. |
| `nanochat/flash_attention.py` | Provides a backend-stable API over FA3 and SDPA. | Q/K/V tensors, window limits, optional KV cache. | Attention results with consistent model-facing semantics. |
| `nanochat/engine.py` | Defines `KVCache` and generation-time cache usage. | Model config, prompt tokens, decode tokens. | Persistent K/V state for incremental inference. |

## 11. Required Concepts in This Chapter

### 11.1 Group-Query Attention

Nanochat allows `n_kv_head` to be smaller than `n_head`, which means queries can remain numerous while keys and values are shared across groups of them. That reduces the size of KV state and changes how the backend must interpret the head layout.

This is why attention geometry in nanochat cannot be described only by “number of heads.” Query heads and KV heads are distinct architectural quantities.

### 11.2 Rotary Position Encoding and QK Norm

Position enters attention through rotation of query and key channels, not through a learned position table added upstream. After that, nanochat normalizes Q and K directly in attention space.

Together, these steps define the score geometry seen by the backend kernel. They are not secondary decorations on top of attention. They are part of what the attention computation fundamentally is in this repo.

### 11.3 Causal Masking with Window Constraints and Cache Offsets

Causal attention is straightforward only when queries and keys refer to the same contiguous time range. Once the model supports sliding windows and cache-based incremental decoding, mask construction becomes a function of current position and visible left context.

That is why the wrapper layer has to handle full-sequence, single-token, and chunked-cache cases separately.

## 12. Non-Obvious Dependencies in This Stage

### 12.1 Flash Attention 3

FA3 is the preferred execution backend when Hopper hardware and compute dtype permit it. Nanochat shapes its model-facing attention API around FA3’s conventions, including `(B, T, H, D)` tensor layout and in-place KV-cache update semantics.

Even when FA3 is unavailable, its API shape still defines the contract the rest of the model code talks to.

### 12.2 PyTorch SDPA

`torch.nn.functional.scaled_dot_product_attention` is the fallback backend, but it is not a drop-in equivalent at the layout and masking level. Nanochat has to transpose tensor layouts, pass `enable_gqa` explicitly, and sometimes build masks manually to preserve the same semantics.

So the fallback is not just “slower attention.” It is an alternate execution path that must be carefully normalized to the same model-facing behavior.

## 13. Reading Order inside This Stage

A clean order is:

1. `CausalSelfAttention.__init__` and `forward()` in `nanochat/gpt.py`
2. `apply_rotary_emb(...)` and `_compute_window_sizes(...)` in `nanochat/gpt.py`
3. `flash_attn_func(...)` and `flash_attn_with_kvcache(...)` in `nanochat/flash_attention.py`
4. `_sdpa_attention(...)` in `nanochat/flash_attention.py`
5. `KVCache` and the prefill/decode path in `nanochat/engine.py`

## 14. End-of-Chapter Synthesis

After this stage, the transformer no longer has a generic “attention module.” It has a concrete attention execution path.

More specifically, queries, keys, and values now have a defined geometry; positional state and QK normalization have been injected; sliding-window and causal constraints are enforced per layer; and inference has a persistent KV-cache contract that extends the same attention logic across time.

That makes the next systems question narrower and more hardware-facing: once the attention path is structurally correct, how do dtype policy, FP8 conversion, kernel availability, and device-specific runtime behavior change the actual cost and behavior of executing it?
