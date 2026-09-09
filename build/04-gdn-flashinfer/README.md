# FlashInfer GDN prefill on SM121

This isolated child image backports only vLLM PR #55715 / merged commit
`f6326f53bda46898a331c2d24500332c285d9a2b` onto the immutable retained
Flash-Next image. It extends the existing FlashInfer GDN prefill backend gate to
SM12x when the key head dimension is 128 and CUDA is at least 13.

The change does not accelerate steady decode directly. It is intended to reduce
new-token prefill time in cold and mixed OpenCode workloads. Because the
FlashInfer implementation agrees with FLA within BF16 tolerance rather than
bit-for-bit, this candidate requires trajectory, multi-turn, mixed-load, tool,
image, code, prose, and math qualification before promotion.

The Dockerfile names the exact retained parent tag for readability. Before any
rebuild, verify that it still resolves to
`sha256:91161a77479af0cc59030452547165ec519c594fc03aface4dddc9d745d2e453`;
the built-image manifest records both the parent and resulting image digests.
