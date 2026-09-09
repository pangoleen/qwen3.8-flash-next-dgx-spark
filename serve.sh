#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a single DGX Spark / GB10 with the PLE table mmapped
# from disk. OpenAI-compatible API on $PORT.
#
# Drop this file (and scripts/weight_utils.nogds.py, if you use FAST_LOAD) into a
# clone of blazux/qwen3.8-Flash-DGX, overwriting its stock scripts/serve.sh. See
# the top-level README's Quickstart for the exact steps.
#
#   scripts/serve.sh                          # NVFP4 checkpoint as published, 262k ctx
#   MODE=hybrid scripts/serve.sh              # NVFP4 experts + fp8 side layers (scripts/prepare-hybrid.sh first)
#   YARN=1 CTX=500000 scripts/serve.sh        # 500k context via YaRN (validated)
#   docker logs -f qwen38-flash               # wait for "Application startup complete"
#
# Tunables (env):
#   MODE=nvfp4        nvfp4 = the checkpoint as published (side layers bf16)
#                     hybrid = side layers in blockwise fp8: +20% decode, +15-20% KV, same
#                     tournament score. Needs the one-time scripts/prepare-hybrid.sh
#   PREFIX_CACHE=1    Repeated prefixes skip the prefill. On a growing conversation this
#                     holds time-per-turn flat at ~18 s from 110k to 241k tokens; with it
#                     off the same turn tracks total context and costs ~145 s at 241k.
#                     Decode rate is unaffected either way (42.8 vs 41.4 tok/s) - this buys
#                     turn latency, not throughput. Below ~4k the two are equivalent.
#                     Caveat: this forces vLLM's Mamba cache into 'align' mode for this
#                     model (config.py:605), which produced wrong-topic completions once,
#                     on 2026-09-07, with no cause found. It has not reproduced across 16
#                     growing-suffix turns and 5 gate rounds on two images. Verify with
#                     bench/suffix_gate.py before depending on exact output; set 0 to avoid
#                     align mode entirely, at the cost of turn latency. RESULTS.md §11.
#   EXACT_TOPK=1      1 = exact, deterministic QSA top-k (identical output at temperature 0;
#                     costs ~10-40% on long prefills). 0 = stock kernel (faster, non-deterministic)
#   PORT=18300        host port for the API
#   CTX=262144        max context length (native). With YARN=1 up to ~500000 (see README)
#   YARN=0            1 = YaRN rope scaling (factor 4) for CTX > 262144
#   SEQS=8            max concurrent sequences. Do NOT leave this at 1-2 when measuring
#                     throughput: requests queue silently and aggregate tok/s flatlines
#   GPU_MEM=0.85      fraction of the 128 GB pool for weights+KV (0.875 got OOM-killed
#                     on a 300k prefill with MTP — keep the margin; 0.80 for long-running service)
#   MTP=2             speculative tokens from the model's MTP head (0 = off; RESULTS.md
#                     recommends 3, see the tuning ladder in the README)
#   KV_DTYPE=auto     keep auto (=bf16): fp8 is refused by the QSA layers
#   PREWARM=0         1 = stream the 48 GiB table once at boot to warm the page cache
#   WORKERS=32        threads for the mmap gather
#   REVISION=         pin the exact snapshot this recipe was measured on (see README,
#                     Weights). Empty = whatever snapshot is on disk, picked arbitrarily
#                     if more than one exists — fine for a single clean download, not
#                     for reproducing a specific number
#   EXTRA=            extra vllm flags passed verbatim
#   FAST_LOAD=0       1 = fastsafetensors loader, nogds forced (boot ~300 s instead of
#                     ~760 s). Experimental: OOM-killed 2 of 5 boots before the pass-2
#                     file filter below; that fix is deployed but not yet proven over
#                     many boots. Default off — see README, Traps.
#   IMAGE=qwen38-flash-dgx   MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODE="${MODE:-nvfp4}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
EXACT_TOPK="${EXACT_TOPK:-1}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
REVISION="${REVISION:-}"
EXTRA="${EXTRA:-}"

# FAST_LOAD=1: load weights with fastsafetensors, GDS forced off (this box has
# none), the PLE table files skipped, and the page cache + pinned host cache
# released as the loader goes (scripts/weight_utils.nogds.py, measured
# 2026-09-05: boot 763 s -> ~300 s, decode/acceptance unchanged). Default off:
# see README, Traps for the OOM history.
FAST_LOAD="${FAST_LOAD:-0}"
if [ "$FAST_LOAD" = 1 ]; then
  LOAD_FORMAT=fastsafetensors
  EXTRA_ENV="${EXTRA_ENV:-} -e VLLM_FASTSAFETENSORS_NOGDS=1 -e VLLM_FASTSAFETENSORS_SKIP_GLOB=model-plefp8-* -e VLLM_FASTSAFETENSORS_PASS2_GLOB=model-bf16-*"
  EXTRA_MOUNT="${EXTRA_MOUNT:-} -v $(cd "$(dirname "$0")" && pwd)/scripts/weight_utils.nogds.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py:ro"
fi

# Resolve the local snapshot directory and map it to the in-container mount.
# Two things a green docker-run does not prove: that the snapshot we picked is
# the one this recipe was measured on, and that it holds real weights rather
# than a partial download. Check both here, not 8-13 minutes into a boot.
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
if [ -n "$REVISION" ]; then
  SNAP_HOST="$REPO_DIR/snapshots/$REVISION"
  if [ ! -d "$SNAP_HOST" ]; then
    echo "!! REVISION=$REVISION has no snapshot dir under $REPO_DIR/snapshots"
    echo "   run scripts/download-weights.sh --revision $REVISION first."
    exit 1
  fi
else
  SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)"
  if [ -z "$SNAP_HOST" ]; then
    echo "!! checkpoint not found under $REPO_DIR"
    echo "   run scripts/download-weights.sh first."
    exit 1
  fi
  N=$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -vc -- '-fp8hybrid' || true)
  if [ "$N" -gt 1 ]; then
    echo "!! $N snapshots found under $REPO_DIR/snapshots and no REVISION set;"
    echo "   picked $(basename "$SNAP_HOST") arbitrarily. Set REVISION to pin one (README, Weights)."
  fi
fi
if ! compgen -G "$SNAP_HOST/*.safetensors" >/dev/null; then
  echo "!! $SNAP_HOST holds no .safetensors — a partial or wrong download."
  echo "   re-run scripts/download-weights.sh (README, Weights)."
  exit 1
fi
SNAP_NAME="$(basename "$SNAP_HOST")"
HYBRID_ENV=()
case "$MODE" in
  nvfp4) ;;
  hybrid)
    if [ ! -f "$REPO_DIR/snapshots/${SNAP_NAME}-fp8hybrid/.prepared" ]; then
      echo "!! hybrid checkpoint not prepared: run scripts/prepare-hybrid.sh first (one-time, ~10 min)"
      exit 1
    fi
    SNAP_NAME="${SNAP_NAME}-fp8hybrid"
    HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)
    ;;
  *) echo "!! MODE must be nvfp4 or hybrid"; exit 1 ;;
esac
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$SNAP_NAME"

# The PLE gather is a CPU op + a pageable host->device copy: it MUST run outside
# CUDA graphs. We declare it a splitting op and use PIECEWISE capture (never FULL*).
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# YaRN (Qwen's published recipe) to go past the native 262144.
OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; fi

# MTP + YaRN: dict hf_overrides are not propagated to the draft model, whose
# max_model_len then stays 262144 and vLLM aborts with
# "--mamba-block-size can only be set with --enable-prefix-caching". Forcing the
# draft's max_model_len through the speculative config fixes it.
SPEC=()
if [ "$MTP" != 0 ]; then
  if [ "$YARN" != 0 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    if [ -n "${SPEC_SCHEDULE:-}" ]; then
      SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"num_speculative_tokens_per_batch_size\":${SPEC_SCHEDULE}}")
    else
      SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
    fi
  fi
fi

PC_ARG=--no-enable-prefix-caching
[ "$PREFIX_CACHE" = 1 ] && PC_ARG=--enable-prefix-caching

# FAST_LOAD: CUDA on GB10 counts clean page cache as used memory (cudaFree == MemFree). After an
# hour of serving, the mmapped PLE table (47.7 GiB) sits in cache; the fast loader's staging then
# OOMs at ~1 GiB free (2026-09-05, 19:42). Drop every big file's pages first; needs no privilege.
if [ "$FAST_LOAD" = 1 ]; then
  python3 - "$HF_CACHE" <<'PY'
import os, sys
n = 0
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(root, f)
        try:
            if os.path.getsize(p) < (64 << 20): continue
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
        except OSError: pass
free = [l for l in open("/proc/meminfo") if l.startswith("MemFree")][0].split()[1]
print(f">> FAST_LOAD: evicted page cache of {n} files; MemFree {int(free)//1048576} GiB")
PY
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_QSA_EXACT_TOPK="$EXACT_TOPK" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  ${EXTRA_ENV:-} ${EXTRA_MOUNT:-} \
  "${HYBRID_ENV[@]}" \
  "$IMAGE" \
  "$SNAP_IN" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format ${LOAD_FORMAT:-safetensors} \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $PC_ARG --enable-chunked-prefill --max-num-batched-tokens ${BATCH_TOKENS:-8192} \
    $CC \
    ${AUTOTUNE_ARG:---no-enable-flashinfer-autotune} \
    --kv-cache-dtype "$KV_DTYPE" \
    "${OVR_ARGS[@]}" $EXTRA \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    "${SPEC[@]}"

echo ">> $NAME starting on :$PORT (model 'qwen3.8-flash-next', mode=$MODE, ctx $CTX, yarn=$YARN, mtp=$MTP, seqs=$SEQS, prefix_cache=$PREFIX_CACHE, exact_topk=$EXACT_TOPK)"
echo ">> first boot loads ~76 GiB of weights (~8-13 min, ~300 s with FAST_LOAD=1). Follow:  docker logs -f $NAME"
echo ">> ready when the log says 'Application startup complete'."
