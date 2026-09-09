#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the tuned image as four overlays on the stock one.
#
#   qwen38-flash-dgx                 <- you build this first, from blazux upstream
#     01 draft-vocab                 reduced MTP draft head
#       02 det-topk                  deterministic QSA top-k CUDA extension
#         03 staged-ple              stage the FP8 PLE read before forward
#           04 gdn-flashinfer        FlashInfer GDN backend on sm121
#
# Each overlay verifies the SHA256 of the file it patches and refuses anything
# else. That is deliberate: if your base image differs from the one these were
# anchored against, the build fails here rather than silently producing a
# different recipe under the same name. See README.md for how to re-anchor.
#
# Usage:  ./build.sh [--base qwen38-flash-dgx] [--tag qwen38-flash-tuned] [--stop-at N]
set -euo pipefail

BASE=qwen38-flash-dgx
TAG=qwen38-flash-tuned
STOP_AT=4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --base)    BASE="$2"; shift 2 ;;
    --tag)     TAG="$2"; shift 2 ;;
    --stop-at) STOP_AT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

docker image inspect "$BASE" >/dev/null 2>&1 || {
  echo "Base image '$BASE' not found." >&2
  echo "Build it first from https://github.com/blazux/qwen3.8-Flash-DGX -" >&2
  echo "this repository patches that image, it does not replace it." >&2
  exit 1
}

STAGES=(01-draft-vocab 02-det-topk 03-staged-ple 04-gdn-flashinfer)
prev="$BASE"

for i in "${!STAGES[@]}"; do
  n=$((i + 1))
  [ "$n" -gt "$STOP_AT" ] && break
  stage="${STAGES[$i]}"
  out="${TAG}-stage${n}"
  [ "$n" -eq "$STOP_AT" ] && out="$TAG"

  echo
  echo "=== stage $n/$STOP_AT: $stage ==="
  echo "    $prev -> $out"

  # Verify vendored sources against their checksums before compiling anything.
  if [ -x "$HERE/$stage/verify-sources.sh" ]; then
    ( cd "$HERE/$stage" && ./verify-sources.sh )
  fi

  docker build --build-arg "BASE_IMAGE=$prev" -t "$out" -f "$HERE/$stage/Dockerfile" "$HERE/$stage"
  prev="$out"
done

echo
echo "built: $prev"
echo
echo "Serve it with:"
echo "  IMAGE=$prev PREFIX_CACHE=0 MODE=hybrid MTP=3 GPU_MEM=0.80 ./serve.sh"
echo
echo "The overlays are inert until the launcher enables them; see README.md."
