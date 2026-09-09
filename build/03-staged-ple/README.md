# Old-preview staged FP8 PLE prototype

Status: the original 4,096-token/MTP-3 lane is GPU-qualified. The flexible
4,096/8,192-token and MTP-2/3 lanes have runtime evidence. MTP-4 is a new
source-tested extension and still needs startup, graph, memory, performance
and quality qualification before use.

## Scope

This overlay is pinned to:

- vLLM `0.1.dev20073+g8e685d198`;
- module `vllm.models.qwen3_8_flash_next.nvidia` (not `qwen4_exp`);
- the existing `vllm_ple_mmap.py.orig` implementation and its 44 GiB FP8 E4M3
  PLE table;
- one PLE layer, 16 n-gram heads, 160-byte head rows and 2,560 raw FP8 values
  per token;
- Model Runner V2, PP1, compile mode 0 and `FULL_DECODE_ONLY`;
- MTP-2, MTP-3 or MTP-4 and max sequences 8, with every corresponding target
  verification width and every draft request width explicitly captured;
  prefill chunks are restricted to 4,096 or 8,192.

It does not change the checkpoint, PLE representation, target weights,
quantization, KV dtype, MTP verifier, image processing, tool parser, reasoning
parser, or prefix caching. It does not include the unqualified PLE I/O hints.

## Files

- `model_state.py.orig`: exact installed old-preview source, SHA-pinned.
- `patch_model_state.py`: exact-anchor generator. It refuses any source hash
  other than the audited installed file.
- `model_state_patched.py`: generated overlay; do not edit by hand.
- `vllm_ple_staging.py`: opt-in staging companion for the existing mmap shim.
- `test_source.py`: local tests requiring only Python and NumPy; it does not
  import Torch or vLLM.
- `test_staging.py`: executes the real companion against a CPU NumPy/Torch
  facade; it imports neither Torch nor vLLM and never opens model shards.
- `test_graph_contract.py`: faithfully models the SHA-pinned MRV2 uniform
  descriptor/dispatch branch and proves target and draft graph inventories.
- `Dockerfile`: Python-only overlay with pre-install source hash gates. It has
  no enabling `ENV`, so installation alone is inert.
- `SOURCES.md`, `LEDGER.md`, `SHA256SUMS`: provenance and implementation record.

## Runtime contract

The adapter remains inactive unless `VLLM_PLE_STAGED=1`. Once requested, it
fails startup unless all of these are true:

1. `VLLM_PLE_MMAP=1` and explicit `VLLM_USE_V2_MODEL_RUNNER=1`.
2. Compilation mode is 0 and graph mode is exactly `FULL_DECODE_ONLY`, both at
   bind and again after vLLM has performed its final graph-mode resolution.
3. The scheduler uses chunk 4,096 or 8,192, sequence limit 8 and MTP-2,
   MTP-3 or MTP-4 with a fixed (non-dynamic) speculative configuration and one
   bonus token. The capture list must be exactly the deduplicated union of every
   `(1 + MTP depth) * sequence count` target width and every `1..8` draft width;
   its maximum must be `(1 + MTP depth) * 8`.
4. TP, PP and DP are all 1, and the live module inventory exactly matches
   `ple_layer_ids`.
5. Every layer is registered by the existing mmap shim and has its exact stock
   `forward_impl` saved.
6. Every model-tree layer is identity-equal to exactly one graph-visible inner
   PLE embedding in `static_forward_context`, and its real `forward()` resolves
   `self.forward_impl` dynamically rather than using a cached callable.
7. Its complete table and every configured mmap shard are already attached;
   the table is FP8 E4M3 with its scalar scale, 160 bytes per row, 16 heads and
   2,560 FP8 values per token.
8. Hash/packing/model-state workspaces cover the configured tokens and eight
   requests, and a stable buffer is initialized only once.

The Dockerfile separately refuses a parent whose old model state, PLE layer,
MTP implementation, model runner, graph/config/speculator implementation,
dummy-input implementation or mmap shim differs byte-for-byte from the audited
versions. The companion re-hashes every graph/speculator/MTP source and the mmap
source at startup. This intentionally means a future vLLM or PLE patch needs a
new compatibility review.

## Data flow

For a live step, the patched model state prepares the same query boundaries and
n-gram context as before. The companion then slices them to actual extents and
calls the mmap shim's saved stock `forward_impl`. That exact old-preview code
computes the hashes and invokes the existing mmap placeholder, returning raw
FP8 embeddings. The companion copies them on the current model stream into the
fixed module buffer and zeros the padded tail.

For CUDA-graph capture and runtime profile batches, the companion only zeros
the stable buffer. It never hashes or reads the table.

For an active staged layer, model forward ignores request metadata and returns
only `stable_buffer[:input_token_shape]`. The original PLE layer still performs
the same dequantization/global scaling, projections, gating and short conv.

## Local validation

Regenerate after any change to the exact input or generator:

```text
python3 patch_model_state.py model_state.py.orig model_state_patched.py
```

Then run:

```text
python3 -m py_compile *.py
python3 test_source.py
python3 test_graph_contract.py
python3 test_staging.py
```

The source test proves generator reproducibility and fail-closed hash behavior,
checks the patched call sites and host-free staged-forward boundary by AST,
exercises live/dummy ambiguity handling and confirms the 10/20 MiB
buffer/capture-width arithmetic. The graph-contract test reproduces the pinned
MRV2 rounding, descriptor construction and dispatch behavior for K2/K3/K4. The
CPU execution test loads the real adapter against a small NumPy-backed Torch
facade and covers env-off inertness, fixed-depth/bonus/max-width/final-mode and
topology gates, live ragged staging, dummy no-read behavior, wrong
dtype/capacity rejection, exact-byte copy, stale-tail zeroing and stable-address
reuse.

This is not a Torch, CUDA graph or model correctness test. Exact source review
confirms that the installed runner constructs runtime dummy batches with
zero-filled `prefill_len_np`, calls model-state input preparation before FULL
graph replay, and initializes model state only after model loading. Before
retention, all GPU and quality gates in `research-staged-ple.md` remain
mandatory.

## Unsupported operations

- Model Runner V1, PP greater than 1, compile modes other than 0, graph modes
  other than `FULL_DECODE_ONLY`, dynamic MTP depth, a non-one bonus-token count,
  or a graph list with missing, duplicate or extra widths.
- In-place checkpoint/weight reload.
- A second concurrent model-forward stream or async scheduling without a new
  lifetime proof.
- BF16/NVFP4 PLE rows, another table shape, multiple unaccounted PLE modules,
  another mmap shim revision, or another old-preview package build.
- Prefix caching. Staging does not address the existing hybrid-state bug.

The supported rollback is the unmodified parent image and PIECEWISE manifest.
