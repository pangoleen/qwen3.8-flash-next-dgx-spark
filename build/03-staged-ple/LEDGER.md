# Implementation ledger

## 2026-09-07 — scope and stop rule

Authorized a local-only old-preview staged-PLE prototype under this directory.
Explicit exclusions: no Spark mutation, image build, GPU work, service/config
change, benchmark/result access for implementation decisions, checkpoint or
quantization change. Stop if exact source compatibility cannot be proven.

## Exact-source gate

Read the active image's source as plain text only; no model/Python import and no
GPU request. Confirmed:

- model state is exactly `Qwen3_8FlashNextModelState`, using the old-preview
  V2 `InputBatch`/`MambaHybridModelState` interface;
- it exposes `prepare_inputs` and `prepare_dummy_inputs` at the expected seams;
- installed model-state SHA256 is `f127380f...8425`;
- installed PLE-layer SHA256 is `6690edcd...397e`;
- installed V2 model-runner SHA256 is `f08ba0f5...bce8`;
- installed V2 input-batch SHA256 is `3929c92e...4410`;
- the PLE class is `Qwen3_8FlashNextNGramEmbedding`, its stock lookup is
  `forward_impl(hidden_states, input_ids, query_start_loc, ngram_context,
  output_buffer=None)`, and the mmap hook is already applied at the file end;
- the qualified mmap shim SHA256 is `2c1de093...ef0e`, matching local
  `vllm_ple_mmap.py.orig`.

Compatibility was therefore proven before implementation. The Dockerfile
repeats all five hashes and refuses any different parent. At runtime the
companion also hashes the imported mmap shim's source file before binding.

## Design choice: companion, not shim fork

The research plan initially proposed a staged fork of the mmap shim. Review of
the exact source exposed a smaller and safer seam: the existing mmap hook has
already saved stock hashing as `_ple_mmap_orig_forward_impl` and registered
each live layer. `vllm_ple_staging.py` consumes those interfaces without
changing shard parsing, row gather, FP8 scale retention or the legacy PIECEWISE
fallback.

Benefits:

- env-off behavior retains the exact qualified mmap implementation;
- exact old-preview hashes are reused rather than reimplemented;
- no PLE-layer source replacement or duplicate custom-op registration;
- rollback is removal of two overlay files / use of the parent image;
- the first graph candidate cannot accidentally include the unqualified I/O
  hints from `vllm_ple_mmap.py`.

## Model-state patch

Stored the exact installed source as `model_state.py.orig`. The generator
checks its full SHA and three unique anchors, then adds only:

1. companion imports;
2. one bind/validation call after PLE context buffers exist;
3. one live/runtime-dummy staging call after the original n-gram context is
   prepared;
4. one capture-dummy zeroing call.

The generated output is `model_state_patched.py`. It is reproducible and must
not be edited manually.

## Fail-closed runtime invariants

When `VLLM_PLE_STAGED` is absent/false, imports occur but no class, instance,
buffer or forward is changed.

When requested, binding rejects:

- missing mmap or explicit V2 flags;
- any graph mode except `FULL_DECODE_ONLY` or compile mode other than 0;
- any scheduler shape other than chunk 4,096/8,192, sequence limit 8 and
  MTP-2/3/4, or a missing MTP verification or draft request width;
- any topology other than TP1/PP1/DP1;
- missing/extra model-tree PLE modules or mmap-registry disagreement;
- any mismatch between model-tree layers and graph-visible
  `static_forward_context` embeddings, or a real `forward()` that does not
  resolve `self.forward_impl` dynamically;
- absent/incomplete table, mmap shard, scalar FP8 scale, stock forward or hash
  workspace;
- a table other than FP8 E4M3, 160-byte rows, 16 heads and embedding 2,560;
- repeat buffer initialization.

The module buffer is registered once at `[max_num_tokens, 2560]` in the table's
raw FP8 dtype. It is exactly 10 MiB at 4,096 and 20 MiB at 8,192. Every preparation checks
capacity, contiguity and the saved data pointer.

## Live, runtime-dummy and capture-dummy behavior

The exact old preview has no separate runtime-dummy model-state method. Its V2
`InputBatch` contract supplies zero `prefill_len_np` for the synthetic empty
constructor, while every live request retains a positive original prefill
length during decode. The adapter uses that contract and rejects mixed or
missing data rather than guessing.

Live preparation slices every input to actual extents, calls the saved stock
old-preview implementation and copies the returned raw FP8 tensor into stable
storage on the current model stream. The padded tail is zeroed every step.

Both runtime dummy/profile and capture dummy paths zero only. A dummy path never
calls stock hashing or the table.

## Exact runner ordering proof

The installed `model_runner.py` was inspected as text and hash-pinned. Its load
sequence completes `model_loader.load_model(...)` before constructing model
state with `init_model_state(...)`, so mmap `load_weights` has attached the
table before staged binding validates it and allocates the stable buffer.

For every live or runtime-dummy execution, the same runner constructs
`model_inputs` by calling `self.model_state.prepare_inputs(...)`; only after
that call returns does it invoke `cudagraph_manager.run_fullgraph(...)`. Both
the copy/zero and replay are therefore enqueued in order on the current model
thread/stream in this exact runner. The installed dummy constructor is also
hash-pinned and sets every `prefill_len_np` entry to zero, while live request
state retains a positive original prefill length. This resolves the two source
questions recorded during initial implementation without adding the broader
upstream #54129 dispatch patch.

## Captured-forward boundary

At bind time, the class's existing mmap forward is preserved as the env-off
fallback and replaced with an instance-gated wrapper. For an active instance,
the executed branch ignores hidden state, query boundaries, n-gram context and
output buffer, and returns only a slice driven by the input tensor's symbolic
token dimension. It contains no `torch.ops`, allocation, hash, D2H, H2D, mmap
or CPU read.

## Local validation performed

- Generated the overlay from the exact source; output SHA256
  `01165b20...d0fa`.
- Compiled every Python file with `py_compile` on the local CPU-only Python.
- Ran `test_source.py` successfully. It verifies exact-source rejection,
  generator reproducibility, patched call sites, the host-free forward AST,
  live/dummy/mixed classification, exact 10/20 MiB and MTP-width arithmetic, raw
  byte equality, padded-tail clearing and stable-address reuse.
- Ran `test_staging.py` successfully against a CPU NumPy/Torch facade. It
  executes the real adapter's env, graph, topology, bind, live, runtime-dummy,
  capture-dummy, dtype, capacity, padding and pointer checks without importing
  Torch/vLLM or opening a model shard.
- Did not import Torch/vLLM locally because this workstation Python does not
  carry Torch. Did not substitute a remote runtime test because GPU/service
  work is outside this prototype's authorization.

## Review blockers before any image build

1. Confirm the proposed parent still has all five pinned source hashes; the
   Dockerfile enforces this again and now requires the parent to be named
   explicitly because it has no default.
2. Keep the source-proven single current-stream ordering. A future async or
   second model-forward stream is unsupported and requires a new lifetime
   proof even if source hashes are deliberately repinned.
3. Root must review and supply explicit activation and graph settings; the
   image has none.
4. Decide whether source-only validation is sufficient before authorizing an
   image build. No Torch, CUDA or remote runtime test was performed here.

## Required later qualification (not performed)

Startup contract evidence, stable pointer/counters, all eight FULL graph widths,
frozen deterministic replay and hashes, MTP acceptance, 1..8 concurrency,
multi-turn continuity, OpenCode, tool, image, Unicode and long-context gates,
memory/watchdog checks, then an independent reload. Any failure rolls back to
the unchanged PIECEWISE parent.

## 2026-09-08 — exact draft-graph retained-recipe hardening

Read-only review of the exact installed old preview proved that one global
`cudagraph_capture_sizes` list is consumed independently by three managers:
the target and speculator-prefill managers use uniform query length `K+1`,
while follow-on draft decode uses query length `1`. The manager rounds each raw
size to its own query length and deduplicates descriptors. Consequently the
exact raw union `1..8 U {(K+1)*s | s=1..8}` preserves the same eight target and
draft-prefill graphs while adding exact follow-on draft graphs for request
counts 1..8.

The previous target-only lists padded K3 single-stream follow-on drafts to four
requests and K4 to five; K4 request counts 6..8 had no eligible follow-on graph
and fell back to eager execution. The retained launcher now emits the exact
union. The runtime gate rejects missing, extra or duplicate widths, a maximum
other than `(K+1)*8`, dynamic/inconsistent speculation, or any target model
state that emits other than one bonus token per step.

Because model state binds before `resolve_cudagraph_mode_and_sizes`, the adapter
saves its exact config and repeats the graph contract once at the first capture
or live preparation. A final automatic downgrade can therefore no longer pass
the early bind check silently.

The behavior depends on four additional vLLM graph/config/speculator files and
the patched MTP implementation. Their exact installed hashes are now enforced
both by the Dockerfile and at runtime, and documented in `SOURCES.md`.
`test_graph_contract.py` faithfully models the pinned uniform descriptor and
dispatch branch and proves K2/K3/K4 target invariance, exact draft graphs,
historic K3/K4 padding and K4 eager fallback. The CPU facade additionally
covers exact-set, duplicate, extra, maximum-width, dynamic-depth, inconsistent
depth, bonus-token and post-resolution mode rejection.
