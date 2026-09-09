// Deterministic persistent_topk, registered under its own namespace so it can
// be built as a standalone extension next to vLLM's _C.
#include <torch/csrc/stable/library.h>
#include "torch_utils.h"

void persistent_topk_det(const torch::stable::Tensor& logits,
                         const torch::stable::Tensor& lengths,
                         torch::stable::Tensor& output,
                         torch::stable::Tensor& workspace, int64_t k,
                         int64_t max_seq_len);

STABLE_TORCH_LIBRARY(_C_det, ops) {
  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_det, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk_det));
}
