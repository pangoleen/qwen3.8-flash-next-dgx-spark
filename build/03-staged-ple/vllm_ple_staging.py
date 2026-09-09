# SPDX-License-Identifier: Apache-2.0
"""Old-preview Qwen3.8 staged PLE adapter for Model Runner V2.

This is deliberately a companion to the already-qualified ``vllm_ple_mmap``
shim. It does not replace shard discovery, hashing, row gathering, FP8 scale
handling, or PLE dequantization. With both VLLM_PLE_MMAP=1 and
VLLM_PLE_STAGED=1, model-state input preparation calls the saved stock
``forward_impl`` before model execution and copies its raw FP8 result into one
fixed module-owned GPU buffer. The model forward then reads only a symbolic
slice of that buffer, allowing FULL_DECODE_ONLY capture.

The adapter is fail-closed for the exact old preview only. Source hashes are
also checked by the Dockerfile before installation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_staging")

ENV_STAGED = "VLLM_PLE_STAGED"
EXPECTED_MMAP_MODULE_SHA256 = (
    "2c1de093f6f386f1fab88c2a44ccd27f620d05563a05c7a0682d7c16d4a9ef0e"
)
EXPECTED_RUNTIME_SOURCE_SHA256 = {
    "vllm.config.compilation": (
        "189c1312f052341d8f19ad06f436ae853a885257a2d2e3d8626c3471892051f9"
    ),
    "vllm.v1.worker.gpu.cudagraph_utils": (
        "ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c"
    ),
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator": (
        "575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1"
    ),
    "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils": (
        "13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5"
    ),
    "vllm.models.qwen3_8_flash_next.nvidia.mtp": (
        "8da66d9f48bd93c935d74e2635c45483dc999c87b94bc4ce828e727eb1713349"
    ),
}
_LAYERS_ATTR = "_ple_mmap_staged_layers"
_CONFIG_ATTR = "_ple_mmap_staged_vllm_config"
_FINAL_GRAPH_CHECK_ATTR = "_ple_mmap_staged_final_graph_checked"
EXPECTED_MAX_NUM_SEQS = 8
SUPPORTED_MAX_NUM_TOKENS = frozenset((4096, 8192))
SUPPORTED_SPECULATIVE_TOKENS = frozenset((2, 3, 4))


def requested() -> bool:
    return os.environ.get(ENV_STAGED, "0").lower() in ("1", "true", "yes")


def _require_runtime_flags() -> None:
    if not requested():
        return
    if os.environ.get("VLLM_PLE_MMAP", "0").lower() not in ("1", "true", "yes"):
        raise RuntimeError("VLLM_PLE_STAGED=1 requires VLLM_PLE_MMAP=1")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1":
        raise RuntimeError(
            "VLLM_PLE_STAGED=1 requires explicit VLLM_USE_V2_MODEL_RUNNER=1"
        )


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).rsplit(".", 1)[-1].upper()


def _source_path(module: Any, module_name: str) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"cannot verify {module_name}: imported module has no __file__")
    path = Path(module_file).resolve()
    if path.suffix == ".pyc":
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError as exc:
            raise RuntimeError(f"cannot resolve source for {module_name} from {path}") from exc
    return path


def _require_pinned_runtime_sources() -> None:
    for module_name, expected in EXPECTED_RUNTIME_SOURCE_SHA256.items():
        module = importlib.import_module(module_name)
        path = _source_path(module, module_name)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"cannot read pinned runtime source for {module_name}") from exc
        if actual != expected:
            raise RuntimeError(
                f"runtime source SHA256 mismatch for {module_name}: "
                f"{actual} != {expected}"
            )


def _require_graph_contract(vllm_config: Any) -> None:
    compilation = vllm_config.compilation_config
    graph_name = _enum_name(compilation.cudagraph_mode)
    if graph_name != "FULL_DECODE_ONLY":
        raise RuntimeError(
            "staged PLE requires cudagraph_mode=FULL_DECODE_ONLY, got "
            f"{compilation.cudagraph_mode!r}"
        )
    compile_mode = compilation.mode
    compile_name = _enum_name(compile_mode)
    compile_value = getattr(compile_mode, "value", compile_mode)
    if compile_name not in ("NONE", "0") and compile_value != 0:
        raise RuntimeError(
            f"staged PLE prototype requires compilation mode 0, got {compile_mode!r}"
        )

    k = int(getattr(vllm_config, "num_speculative_tokens", 0) or 0)
    speculative = getattr(vllm_config, "speculative_config", None)
    dynamic_check = getattr(speculative, "uses_dynamic_speculative_decoding", None)
    if speculative is None or not callable(dynamic_check):
        raise RuntimeError("staged PLE requires an explicit fixed speculative config")
    configured_k = int(getattr(speculative, "num_speculative_tokens", 0) or 0)
    is_dynamic = bool(dynamic_check())
    if configured_k != k or is_dynamic:
        raise RuntimeError(
            "staged PLE requires fixed speculative depth consistent across config; "
            f"vllm={k}, speculative={configured_k}, dynamic={is_dynamic}"
        )
    seqs = int(vllm_config.scheduler_config.max_num_seqs)
    tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    if (
        tokens not in SUPPORTED_MAX_NUM_TOKENS
        or seqs != EXPECTED_MAX_NUM_SEQS
        or k not in SUPPORTED_SPECULATIVE_TOKENS
    ):
        raise RuntimeError(
            "staged PLE supports max_num_batched_tokens in {4096,8192}, "
            "max_num_seqs=8 and MTP in {2,3,4}; got "
            f"tokens={tokens}, seqs={seqs}, speculative_tokens={k}"
        )
    target_widths = {(1 + k) * s for s in range(1, seqs + 1)}
    draft_widths = set(range(1, seqs + 1))
    expected = target_widths | draft_widths
    configured_list = [
        int(size) for size in (compilation.cudagraph_capture_sizes or ())
    ]
    configured = set(configured_list)
    missing = sorted(expected - configured)
    unexpected = sorted(configured - expected)
    duplicates = len(configured_list) != len(configured)
    if missing or unexpected or duplicates:
        raise RuntimeError(
            "staged PLE requires the exact deduplicated FULL graph set for MTP "
            "verify and draft request widths; "
            f"missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}, configured={configured_list}"
        )
    expected_max = (1 + k) * seqs
    configured_max = int(getattr(compilation, "max_cudagraph_capture_size", 0) or 0)
    if configured_max != expected_max:
        raise RuntimeError(
            "staged PLE max cudagraph width disagrees with the exact contract: "
            f"{configured_max} != {expected_max}"
        )


def _require_final_graph_contract(model_state: Any) -> None:
    if bool(getattr(model_state, _FINAL_GRAPH_CHECK_ATTR, False)):
        return
    vllm_config = getattr(model_state, _CONFIG_ATTR, None)
    if vllm_config is None:
        raise RuntimeError("staged PLE has no saved vLLM config for final graph check")
    _require_graph_contract(vllm_config)
    setattr(model_state, _FINAL_GRAPH_CHECK_ATTR, True)


def _classify_input_batch(input_batch: Any) -> bool:
    """Return True for live requests and False for a runtime dummy/profile batch.

    The exact preview InputBatch uses a zero-filled ``prefill_len_np`` only for
    its synthetic empty/profile constructor. Every live request retains its
    positive original prefill length during both prefill and decode. Mixed
    zero/positive rows are not a documented state and are rejected.
    """

    num_reqs = int(input_batch.num_reqs)
    if num_reqs == 0:
        return False
    values = getattr(input_batch, "prefill_len_np", None)
    if values is None or len(values) < num_reqs:
        raise RuntimeError(
            "cannot distinguish live from dummy PLE batch: prefill_len_np missing/short"
        )
    positive = np.asarray(values[:num_reqs], dtype=np.int64) > 0
    if bool(positive.all()):
        return True
    if not bool(positive.any()):
        return False
    raise RuntimeError(
        "ambiguous staged PLE batch: mixed zero and positive prefill lengths"
    )


def _staged_forward_impl(
    self: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    if getattr(self, "_ple_staged_active", False):
        # This branch is the only code captured for an active staged instance:
        # one stable module buffer, one symbolic token-dimension slice, no host
        # reads, custom ops, hashing, allocation, or request-metadata use.
        del hidden_states, query_start_loc, ngram_context, output_buffer
        staging = self._ple_staged_rows
        if staging is None:
            raise RuntimeError("staged PLE buffer was not initialized")
        return staging[: input_ids.reshape(-1).shape[0]]
    return self._ple_staged_legacy_forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        output_buffer=output_buffer,
    )


def _patch_layer_class(layer: Any) -> None:
    cls = layer.__class__
    if getattr(cls, "_ple_staged_class_patched", False):
        return
    if not hasattr(cls, "_ple_mmap_orig_forward_impl"):
        raise RuntimeError(
            "staged PLE requires the exact mmap shim to have saved stock forward_impl"
        )
    cls._ple_staged_legacy_forward_impl = cls.forward_impl
    cls.forward_impl = _staged_forward_impl
    cls._ple_staged_class_patched = True


def _require_dynamic_forward_dispatch(layer: Any) -> None:
    """Reject modules whose real ``forward`` cached an old implementation."""

    forward = getattr(layer, "forward", None)
    function = getattr(forward, "__func__", forward)
    code = getattr(function, "__code__", None)
    names = set(getattr(code, "co_names", ()))
    if code is None or "forward_impl" not in names:
        raise RuntimeError(
            "staged PLE requires forward() to resolve self.forward_impl dynamically"
        )
    if "_forward_method" in getattr(layer, "__dict__", {}):
        raise RuntimeError("staged PLE rejects a cached _forward_method dispatch")
    if bool(getattr(layer, "_is_cpu_offloaded", False)):
        raise RuntimeError("staged mmap PLE cannot use cross-process CPU offload")


def _initialize_layer(layer: Any, max_num_tokens: int, device: torch.device) -> None:
    table = getattr(getattr(layer, "ngram_embedding", None), "table", None)
    if table is None:
        raise RuntimeError(
            f"staged PLE table is not attached for {layer._ple_mmap_prefix!r}"
        )
    if table.torch_dtype != torch.float8_e4m3fn or int(table.row_bytes) != 160:
        raise RuntimeError(
            "staged PLE prototype is pinned to FP8 E4M3, 160-byte rows; got "
            f"dtype={table.torch_dtype}, row_bytes={table.row_bytes}"
        )
    if int(layer.ngram_heads) != 16 or int(layer.embedding_dim) != 2560:
        raise RuntimeError(
            "staged PLE prototype is pinned to 16 heads and embedding_dim=2560"
        )
    expected_rows = int(layer.ngram_embedding.org_vocab_size)
    if int(table.rows_total) != expected_rows:
        raise RuntimeError(
            "staged PLE table is incomplete or belongs to another checkpoint: "
            f"rows={table.rows_total}, expected={expected_rows}"
        )
    if len(table.mm) != int(layer.split_ngram_parts) or any(
        shard is None for shard in table.mm
    ):
        raise RuntimeError(
            "staged PLE requires every configured mmap shard to be attached"
        )
    scale = getattr(layer, "_offload_weight_scale", None)
    if scale is None or int(scale.numel()) != 1:
        raise RuntimeError("staged FP8 PLE requires one retained global weight scale")
    if int(layer.positions_buffer.numel()) < max_num_tokens:
        raise RuntimeError("PLE hash workspace is smaller than max_num_tokens")
    padded = getattr(layer, "padded_buffer", None)
    if (
        padded is None
        or len(padded.shape) != 2
        or int(padded.shape[0]) < EXPECTED_MAX_NUM_SEQS
        or int(padded.shape[1]) < max_num_tokens
    ):
        raise RuntimeError(
            "PLE packed workspace is smaller than the scheduler contract"
        )
    if hasattr(layer, "_ple_staged_rows"):
        raise RuntimeError("staged PLE layer initialized more than once")

    _require_dynamic_forward_dispatch(layer)
    _patch_layer_class(layer)
    staging = torch.zeros(
        (max_num_tokens, int(layer.embedding_dim)),
        dtype=table.torch_dtype,
        device=device,
    )
    layer.register_buffer("_ple_staged_rows", staging, persistent=False)
    layer._ple_staged_ptr = int(staging.data_ptr())
    layer._ple_staged_active = True
    layer._ple_staged_live_calls = 0
    layer._ple_staged_dummy_calls = 0


def bind_model_state(model_state: Any, vllm_config: Any, model: Any) -> None:
    """Discover and initialize every local mmap PLE module, or do nothing."""

    if not requested():
        return
    _require_runtime_flags()
    _require_pinned_runtime_sources()
    _require_graph_contract(vllm_config)
    bonus_tokens = int(
        getattr(model_state, "num_new_sampled_tokens_per_step", 0) or 0
    )
    if bonus_tokens != 1:
        raise RuntimeError(
            "staged PLE exact graph widths require one new sampled token per step; "
            f"got {bonus_tokens}"
        )
    parallel = vllm_config.parallel_config
    topology = (
        int(parallel.tensor_parallel_size),
        int(parallel.pipeline_parallel_size),
        int(getattr(parallel, "data_parallel_size", 1)),
    )
    if topology != (1, 1, 1):
        raise RuntimeError(
            "staged PLE prototype supports TP1/PP1/DP1 only; got "
            f"TP{topology[0]}/PP{topology[1]}/DP{topology[2]}"
        )
    scheduler_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    if int(model_state.max_num_tokens) != scheduler_tokens:
        raise RuntimeError(
            "model-state token capacity disagrees with scheduler contract: "
            f"{model_state.max_num_tokens} != {scheduler_tokens}"
        )
    ngram_context = getattr(model_state, "ngram_context", None)
    if ngram_context is None or int(ngram_context.shape[0]) < EXPECTED_MAX_NUM_SEQS:
        raise RuntimeError("model-state ngram context is smaller than max_num_seqs")
    query_starts = getattr(model_state, "ple_query_start_loc", None)
    if query_starts is None or int(query_starts.numel()) < EXPECTED_MAX_NUM_SEQS + 1:
        raise RuntimeError("model-state PLE query offsets are smaller than max_num_seqs+1")

    import vllm_ple_mmap as mmap_mod

    module_path = _source_path(mmap_mod, "vllm_ple_mmap")
    try:
        module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read mmap shim source at {module_path}") from exc
    if module_hash != EXPECTED_MMAP_MODULE_SHA256:
        raise RuntimeError(
            "mmap shim SHA256 does not match the audited implementation: "
            f"{module_hash} != {EXPECTED_MMAP_MODULE_SHA256}"
        )

    layers = []
    for module in model.modules():
        prefix = getattr(module, "_ple_mmap_prefix", None)
        if prefix is None:
            continue
        if not hasattr(module, "_ple_mmap_orig_forward_impl"):
            raise RuntimeError(f"PLE layer {prefix!r} has no saved stock forward")
        if getattr(mmap_mod, "_REGISTRY", {}).get(prefix) is not module:
            raise RuntimeError(f"PLE mmap registry disagrees for {prefix!r}")
        layers.append(module)

    expected_layers = len(model_state.model_config.hf_text_config.ple_layer_ids)
    if len(layers) != expected_layers or not layers:
        raise RuntimeError(
            "staged PLE module inventory mismatch: "
            f"found={len(layers)}, expected={expected_layers}"
        )

    # CUDA graph compilation addresses the outer PLE layer through the static
    # forward context, while model.modules() discovers its inner n-gram
    # embedding. Require a one-to-one identity mapping so no graph-visible
    # module can retain the legacy mmap forward path.
    static_context = getattr(
        vllm_config.compilation_config, "static_forward_context", None
    )
    if not isinstance(static_context, dict):
        raise RuntimeError("staged PLE requires compilation static_forward_context")
    graph_layers = []
    for outer in static_context.values():
        embedding = getattr(outer, "ple_embedding", None)
        if getattr(embedding, "_ple_mmap_prefix", None) is not None:
            graph_layers.append(embedding)
    if len(graph_layers) != len(layers) or {id(x) for x in graph_layers} != {
        id(x) for x in layers
    }:
        raise RuntimeError(
            "staged PLE graph-visible module inventory mismatch: "
            f"model={len(layers)}, graph={len(graph_layers)}"
        )
    for layer in layers:
        _initialize_layer(layer, int(model_state.max_num_tokens), model_state.device)
    setattr(model_state, _LAYERS_ATTR, tuple(layers))
    setattr(model_state, _CONFIG_ATTR, vllm_config)
    setattr(model_state, _FINAL_GRAPH_CHECK_ATTR, False)
    logger.info(
        "staged PLE active: %d layer(s), max_num_tokens=%d, %.2f MiB stable FP8",
        len(layers),
        int(model_state.max_num_tokens),
        sum(layer._ple_staged_rows.numel() for layer in layers) / 2**20,
    )


def _verify_destination(layer: Any, actual_tokens: int, padded_tokens: int) -> Any:
    staging = getattr(layer, "_ple_staged_rows", None)
    if staging is None or not getattr(layer, "_ple_staged_active", False):
        raise RuntimeError("staged PLE layer is inactive or missing its buffer")
    if int(staging.data_ptr()) != int(layer._ple_staged_ptr):
        raise RuntimeError("staged PLE buffer address changed after initialization")
    capacity = int(staging.shape[0])
    if not (0 <= actual_tokens <= padded_tokens <= capacity):
        raise RuntimeError(
            "invalid staged PLE extents: "
            f"actual={actual_tokens}, padded={padded_tokens}, capacity={capacity}"
        )
    if not staging.is_contiguous():
        raise RuntimeError("staged PLE buffer is not contiguous")
    return staging


def _zero_layers(model_state: Any, padded_tokens: int) -> None:
    for layer in getattr(model_state, _LAYERS_ATTR, ()):
        staging = _verify_destination(layer, 0, int(padded_tokens))
        staging[:padded_tokens].zero_()
        layer._ple_staged_dummy_calls += 1


def prepare_model_inputs(
    model_state: Any,
    input_batch: Any,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> None:
    """Stage one live V2 step, or zero a runtime dummy/profile step."""

    layers = getattr(model_state, _LAYERS_ATTR, ())
    if not layers:
        return
    _require_final_graph_contract(model_state)
    actual_tokens = int(input_batch.num_tokens)
    padded_tokens = int(input_batch.num_tokens_after_padding)
    num_reqs = int(input_batch.num_reqs)
    if not _classify_input_batch(input_batch):
        _zero_layers(model_state, padded_tokens)
        return
    if actual_tokens <= 0 or num_reqs <= 0:
        raise RuntimeError("live staged PLE batch has no requests or tokens")

    actual_input_ids = input_batch.input_ids[:actual_tokens]
    actual_query_start_loc = query_start_loc[: num_reqs + 1]
    actual_context = ngram_context[:num_reqs]
    for layer in layers:
        staging = _verify_destination(layer, actual_tokens, padded_tokens)
        result = layer._ple_mmap_orig_forward_impl(
            None,
            actual_input_ids,
            actual_query_start_loc,
            actual_context,
        )
        expected_shape = (actual_tokens, int(layer.embedding_dim))
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                f"staged PLE result shape {tuple(result.shape)} != {expected_shape}"
            )
        if result.dtype != staging.dtype or result.device != staging.device:
            raise RuntimeError(
                "staged PLE result dtype/device mismatch: "
                f"result={result.dtype}/{result.device}, "
                f"buffer={staging.dtype}/{staging.device}"
            )
        staging[:actual_tokens].copy_(result, non_blocking=True)
        if padded_tokens > actual_tokens:
            staging[actual_tokens:padded_tokens].zero_()
        layer._ple_staged_live_calls += 1


def prepare_dummy_inputs(model_state: Any, num_tokens: int) -> None:
    """Capture-time zero-only path: never hash or touch the mmap table."""

    if getattr(model_state, _LAYERS_ATTR, ()):
        _require_final_graph_contract(model_state)
    _zero_layers(model_state, int(num_tokens))


__all__ = [
    "bind_model_state",
    "prepare_dummy_inputs",
    "prepare_model_inputs",
    "requested",
]
