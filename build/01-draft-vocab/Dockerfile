# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# Reduced draft vocabulary for the MTP proposal head.
#
# The drafter owns a BF16 head over the full ~248k vocabulary and reads it once
# per draft step. Draft sampling here is greedy, so the drafter only needs an
# argmax. Restricting that argmax to a frequency-ranked subset reads a fraction
# of the weight; a draft outside the subset is simply rejected by the target at
# verification, so this trades acceptance for bandwidth and cannot change what
# the server emits. compute_logits keeps the full head, so no other path moves.
#
# See NOTICE.md for credits. This is an independent implementation.

ARG BASE_IMAGE=qwen38-flash-dgx
FROM ${BASE_IMAGE}

ARG VLLM_SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages
ARG MTP_PATH=${VLLM_SITE_PACKAGES}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py

# SHA256 of the pristine upstream mtp.py this patch is anchored to. The patcher
# refuses any other file rather than applying to something it has not seen, so
# a changed base image fails the build loudly instead of producing a different
# recipe under the same name.
ARG MTP_SHA256=7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431

COPY draft_vocab_head.py apply_draft_vocab.py test_draft_vocab.py /opt/llm/draft-vocab/
COPY draft_vocab_65536.txt draft_vocab_65536.json /opt/llm/draft-vocab/
COPY NOTICE.md /opt/llm/draft-vocab/

# The unit tests are CPU-only and import neither torch nor vLLM, so they run
# here as a build-time gate rather than at serve time.
RUN cd /opt/llm/draft-vocab && python3 test_draft_vocab.py

RUN python3 /opt/llm/draft-vocab/apply_draft_vocab.py \
      --target "${MTP_PATH}" --sha256 "${MTP_SHA256}" \
 && python3 -c "import ast,sys; p='${MTP_PATH}'; t=ast.parse(open(p).read()); \
names=[n.name for c in t.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef)]; \
sys.exit(0 if 'get_top_tokens' in names else 'get_top_tokens missing after patch')" \
 && echo "draft-vocab overlay applied"

# Inert unless the launcher sets both of these. The vocabulary path is baked in
# so serve.sh only has to turn the feature on.
ENV VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft-vocab/draft_vocab_65536.txt
