# Sources

- vLLM PR #55715: https://github.com/vllm-project/vllm/pull/55715
- Merged commit: https://github.com/vllm-project/vllm/commit/f6326f53bda46898a331c2d24500332c285d9a2b
- FlashInfer SM120 GDN implementation PR #3479: https://github.com/flashinfer-ai/flashinfer/pull/3479

The local backport is intentionally limited to the backend-selection hunk from
the merged vLLM commit. The parent source SHA is pinned in `patch_gdn.py`.
