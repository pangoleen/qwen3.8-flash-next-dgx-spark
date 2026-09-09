# Deterministic top-k candidate overlay

This directory is a locally prepared, checksum-pinned overlay for the reviewed deterministic QSA `persistent_topk` implementation. Preparing it did not build an image, invoke CUDA, or alter any running container or service.

## Provenance

- Source: <https://github.com/jschmied/qwen38-flash-next-gb10/tree/e0ef69d4f5575dad00d34e05479eaf4c6547bace/patches/kernel-det>
- Pin: `e0ef69d4f5575dad00d34e05479eaf4c6547bace`
- Pin rationale: this commit synchronizes `topk_det.cu` with the tested build after the author found that the previously published source did not compile.
- License and notice from the same pin are included under `vendor/`.
- Full compatibility and numerical-order review: [`../../research-det-topk.md`](../../research-det-topk.md)

Run the non-GPU local verification before any build:

```sh
./verify-sources.sh
```

It validates every vendored file against `SHA256SUMS` and parses the three Python sources without importing vLLM or CUDA.

## Later image build

The default parent is the current draft-vocabulary image. The build must be run only when the active replay is complete and normal image-build resource use is acceptable:

```sh
docker build \
  --build-arg BASE_IMAGE=qwen38-flash-perf-vocab65:20260907 \
  --build-arg DET_ARCH=121a \
  --tag qwen38-flash-perf-vocab65-det-topk:20260907 \
  flash-experiments/build/det-topk
```

The Dockerfile verifies the bundle, compiles only `topk_det.cu` and `bindings_det.cpp` against the parent image's Torch/CUDA ABI, installs `_C_det.so`, and applies the env-gated QSA routing shim. It does not enable deterministic mode by default and does not rebuild vLLM.

Reported compile time ranges from about 15 seconds to one minute on GX10; reserve five minutes. The image build does not establish CUDA correctness because the extension test requires a GPU.

## GPU validation, only after an explicit idle check

Do not run this while a replay, model load, or other GPU workload is active. Validate the extension in a short-lived container before creating or changing a model-serving container:

```sh
docker run --rm --gpus all \
  --entrypoint python3 \
  qwen38-flash-perf-vocab65-det-topk:20260907 \
  /opt/llm/kernel-det/source/vendor/kernel-det/test_det.py \
  /opt/llm/kernel-det/_C_det.so
```

Require the final line `FAILS: 0`. The published synced-pin log passes 210 cases, including random values, heavy ties, short rows, all-equal rows, and threshold-boundary ties.

## Runtime candidate requirements

Use the same weights, FP8-hybrid target layers, BF16 KV cache, MTP setting, 262144 context, PIECEWISE graph mode, max sequences, 4096 chunk size, memory fraction, mounts, and networks as the selected parent candidate. Change only these runtime environment values:

```text
VLLM_QSA_EXACT_TOPK=0
VLLM_QSA_DET_TOPK=1
VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so
```

Require one `QSADET active: /opt/llm/kernel-det/_C_det.so` log line. `VLLM_QSA_EXACT_TOPK=1` bypasses the extension, so leaving exact mode enabled invalidates the candidate.

The deterministic kernel returns selected indices in ascending index order; the current `torch.topk(..., sorted=False)` fallback does not promise that order. Output hashes can therefore differ even when the selected value set is the same. Validate repeatability within the deterministic arm plus coherence, quality, token counts, and the fixed-work performance gates.

Rollback is runtime-only: set `VLLM_QSA_DET_TOPK=0` and `VLLM_QSA_EXACT_TOPK=1`, or return to the untouched parent image.
