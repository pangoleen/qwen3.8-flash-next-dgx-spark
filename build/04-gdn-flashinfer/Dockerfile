ARG BASE_IMAGE=qwen38-flash-perf-vocab65-det-staged-v2:20260908
FROM ${BASE_IMAGE}

ARG GDN_SOURCE=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
COPY gdn-flashinfer-sm121/patch_gdn.py /tmp/patch_gdn.py
RUN python3 /tmp/patch_gdn.py ${GDN_SOURCE} \
 && rm /tmp/patch_gdn.py

# No ENV, CMD, ENTRYPOINT, model, quantization, graph, or cache changes.
