# syntax=docker/dockerfile:1

# Overlay the deterministic QSA top-k extension on the current draft-vocabulary
# candidate. Override BASE_IMAGE explicitly if the selected parent image changes.
ARG BASE_IMAGE=qwen38-flash-perf-vocab65:20260907
FROM ${BASE_IMAGE}

ARG DET_ARCH=121a
ARG VLLM_SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages

LABEL org.opencontainers.image.title="Qwen3.8 Flash deterministic top-k overlay" \
      org.opencontainers.image.source="https://github.com/jschmied/qwen38-flash-next-gb10" \
      org.opencontainers.image.revision="e0ef69d4f5575dad00d34e05479eaf4c6547bace"

COPY SHA256SUMS /opt/llm/kernel-det/source/SHA256SUMS
COPY vendor/ /opt/llm/kernel-det/source/vendor/

# Verify the vendored source before compiling it against this image's installed
# Torch/CUDA ABI. This produces one local extension; it does not rebuild vLLM.
RUN cd /opt/llm/kernel-det/source \
 && sha256sum --check SHA256SUMS \
 && DET_ARCH=${DET_ARCH} \
    DET_BUILD_DIR=/opt/llm/kernel-det/build \
    python3 vendor/kernel-det/build_det.py \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so

# The shim is inert unless VLLM_QSA_DET_TOPK is enabled at container runtime.
# The existing exact-top-k path has precedence, so the candidate launcher must
# explicitly set VLLM_QSA_EXACT_TOPK=0 as well as VLLM_QSA_DET_TOPK=1.
RUN VLLM_QSA_PY=${VLLM_SITE_PACKAGES}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py \
    python3 /opt/llm/kernel-det/source/vendor/determinism/qsadet_patch.py \
 && python3 -c "import ast; p='${VLLM_SITE_PACKAGES}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py'; ast.parse(open(p).read()); print('det-topk QSA shim parsed OK')"

# Keep the extension path discoverable without enabling the candidate mode.
ENV VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so

