# scripts/ overlay

`weight_utils.nogds.py` is vLLM's own `vllm/model_executor/model_loader/weight_utils.py`
(Apache License 2.0, copyright the vLLM project contributors), with one function
patched to fix the FAST_LOAD path (see README, Traps, and the comments inline
at the patched lines). `serve.sh` mounts it read-only over the stock file inside
the container when `FAST_LOAD=1`; it is never used otherwise.

Diff against the stock file at the pinned vLLM build (`0.1.dev20073+g8e685d198`),
in short: forces `nogds=True` (this box has no GPU Direct Storage), skips the
47.7 GiB PLE table files that fastsafetensors would otherwise stream through
device buffers for no reason, restricts the second load pass (the MTP draft) to
its own three shards instead of re-streaming the whole checkpoint, and runs a
background thread that returns each file's page-cache pages with `fadvise`
`DONTNEED` as it goes — GB10's unified memory does not reclaim clean page cache
for `cudaMalloc` on its own (see the top-level README, Traps).
