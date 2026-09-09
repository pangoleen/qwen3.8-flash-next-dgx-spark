# Source provenance

Recorded 2026-09-07. All authored artifacts remain local.

## Exact installed inputs

The following source identities were read without importing the model or
running GPU work from the active image based on vLLM
`0.1.dev20073+g8e685d198`:

| Installed path | SHA256 | Use |
|---|---|---|
| `vllm/models/qwen3_8_flash_next/nvidia/model_state.py` | `f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425` | Exact Apache-2.0 patch input, stored as `model_state.py.orig`. |
| `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` | `6690edcd2e2b2ae309b214991a060bffbdbbcdc42334e164db0aebcc0a5b397e` | Interface and import-hook contract; not copied or modified by this overlay. |
| `vllm/v1/worker/gpu/model_runner.py` | `f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8` | Proves model loading precedes state binding and `prepare_inputs` is enqueued before FULL graph replay. |
| `vllm/v1/worker/gpu/input_batch.py` | `3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410` | Exact runtime-dummy constructor contract: zero-filled `prefill_len_np`. |
| `vllm/config/compilation.py` | `189c1312f052341d8f19ad06f436ae853a885257a2d2e3d8626c3471892051f9` | Proves Model Runner V2 does not globally rewrite capture sizes to target multiples during final mode resolution. |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c` | Exact uniform FULL descriptor rounding, deduplication, request cap and dispatch contract. |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | `575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1` | Proves separate speculator managers use query lengths `K+1` for draft prefill and `1` for follow-on draft decode. |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` | `13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5` | Exact speculator graph-capture adapter and padded request/metadata contract. |
| `vllm/models/qwen3_8_flash_next/nvidia/mtp.py` | `8da66d9f48bd93c935d74e2635c45483dc999c87b94bc4ce828e727eb1713349` | Exact reduced-vocabulary MTP implementation used by the retained candidate. |
| `/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py` | `2c1de093f6f386f1fab88c2a44ccd27f620d05563a05c7a0682d7c16d4a9ef0e` | Existing mmap/hash/gather implementation, byte-identical to local `../vllm_ple_mmap.py.orig`; not copied or modified by this overlay. |

The exact installed commit-like package revision is not available as a public
Git commit. Hash pinning is therefore the compatibility authority. All ten
identities are checked before overlay installation, and the companion repeats
the graph/config/speculator/MTP and mmap source checks at runtime. The stored
model-state file retains its upstream Apache-2.0 header.

## Design references

No Tony or Trosfy file is mounted or copied into the old-preview package. The
adapter is an old-preview-specific implementation using the exact local APIs,
informed by these primary sources:

- Tony DeAngelo's independent staging introduction, Apache-2.0 model-state
  base: [`d27187f`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/commit/d27187fbce0676e29202b832620c3e94bd5e72e5).
- Tony's correction from one row to 16 rows per token:
  [`c1b3b93`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/commit/c1b3b93d4d5ae0533d545a101cdb7a8d7e02d93b).
- Tony's first measured staging result:
  [`8414105`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/commit/841410503fc5090d8099279ce1571ec5c0db9fc1).
- Pinned Tony combined result:
  [`9c9d8b6`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/commit/9c9d8b6faa69aeeaa9bdf5b568377e2504608f63).
- Trosfy's more defensive upstream implementation and validation:
  [vLLM PR #54129](https://github.com/vllm-project/vllm/pull/54129), especially
  Apache-2.0 commit
  [`5b581be`](https://github.com/Trosfy/vllm/commit/5b581bed9cc34143027636e8f577bd3a8f647725).
- The reason FULL prefill/piecewise capture is excluded:
  [local-inference-lab/vllm#600](https://github.com/local-inference-lab/vllm/pull/600).

The companion carries an Apache-2.0 SPDX identifier to match the vLLM model
state and the referenced implementations. The existing mmap shim retains its
own original attribution in `../vllm_ple_mmap.py.orig`.
