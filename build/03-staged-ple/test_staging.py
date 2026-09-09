#!/usr/bin/env python3
"""CPU-only execution tests for the staged-PLE adapter.

This loads the real companion against a deliberately tiny Torch facade. It
exercises the adapter's control/data path without importing vLLM, allocating a
CUDA tensor, opening a model shard, or contacting the serving host.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent


class FakeTensor:
    def __init__(self, array, *, dtype="fp8", device="cpu"):
        self.array = np.asarray(array)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return FakeTensor(self.array[key], dtype=self.dtype, device=self.device)

    def reshape(self, *shape):
        return FakeTensor(self.array.reshape(*shape), dtype=self.dtype, device=self.device)

    def numel(self):
        return self.array.size

    def data_ptr(self):
        return self.array.__array_interface__["data"][0]

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def copy_(self, other, non_blocking=False):
        del non_blocking
        self.array[...] = other.array
        return self

    def zero_(self):
        self.array.fill(0)
        return self


def load_adapter(fake_torch):
    spec = importlib.util.spec_from_file_location(
        "vllm_ple_staging_cpu_test", HERE / "vllm_ple_staging.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


def exact_config(capture_sizes=None, *, tokens=4096, speculative_tokens=3):
    expected_max = (speculative_tokens + 1) * 8
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode="FULL_DECODE_ONLY",
            mode=0,
            cudagraph_capture_sizes=(
                sorted(
                    set(range(1, 9))
                    | {
                        (speculative_tokens + 1) * seqs
                        for seqs in range(1, 9)
                    }
                )
                if capture_sizes is None
                else capture_sizes
            ),
            max_cudagraph_capture_size=expected_max,
            static_forward_context={},
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=tokens,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        num_speculative_tokens=speculative_tokens,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=speculative_tokens,
            uses_dynamic_speculative_decoding=lambda: False,
        ),
    )


def main() -> None:
    fp8 = object()
    fake_torch = types.ModuleType("torch")
    fake_torch.float8_e4m3fn = fp8
    fake_torch.zeros = lambda shape, dtype, device: FakeTensor(
        np.zeros(shape, dtype=np.uint8), dtype=dtype, device=device
    )
    adapter = load_adapter(fake_torch)

    # Installation is strictly inert without the explicit activation flag.
    inert_state = SimpleNamespace()
    with patch.dict(os.environ, {}, clear=True):
        adapter.bind_model_state(inert_state, None, None)
    assert not hasattr(inert_state, adapter._LAYERS_ATTR)

    # Every target dimension and graph width is fail-closed.
    for tokens in (4096, 8192):
        for speculative_tokens in (2, 3, 4):
            adapter._require_graph_contract(
                exact_config(tokens=tokens, speculative_tokens=speculative_tokens)
            )
    for bad in (
        exact_config([1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28]),
        exact_config(tokens=2048),
        exact_config(speculative_tokens=1),
        exact_config([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 21], speculative_tokens=2),
        exact_config([1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35], speculative_tokens=4),
        # Target-only widths reproduce the previous silent draft padding and
        # must now fail closed even though every verification graph exists.
        exact_config([4, 8, 12, 16, 20, 24, 28, 32]),
        # Extras and duplicates alter/obscure the retained graph inventory.
        exact_config(
            [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 33]
        ),
        exact_config(
            [1, 2, 3, 4, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
        ),
    ):
        try:
            adapter._require_graph_contract(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("an incompatible graph/token contract was accepted")

    wrong_max = exact_config()
    wrong_max.compilation_config.max_cudagraph_capture_size = 64
    dynamic = exact_config()
    dynamic.speculative_config.uses_dynamic_speculative_decoding = lambda: True
    inconsistent = exact_config()
    inconsistent.speculative_config.num_speculative_tokens = 2
    for bad in (wrong_max, dynamic, inconsistent):
        try:
            adapter._require_graph_contract(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("an incompatible fixed-depth/max-width contract was accepted")

    calls = []

    class Layer:
        def stock(self, hidden, input_ids, query_start_loc, ngram_context):
            del hidden
            calls.append(
                (input_ids.shape, query_start_loc.shape, ngram_context.shape)
            )
            rows = np.arange(input_ids.shape[0] * 2560, dtype=np.uint32)
            rows = (rows % 251).astype(np.uint8).reshape(input_ids.shape[0], 2560)
            return FakeTensor(rows, dtype=fp8, device="cpu")

        _ple_mmap_orig_forward_impl = stock

        def forward(self, *args, **kwargs):
            return self.forward_impl(*args, **kwargs)

        def forward_impl(self, *args, **kwargs):
            raise AssertionError("legacy mmap forward executed while staged")

        def register_buffer(self, name, value, persistent):
            assert persistent is False
            setattr(self, name, value)

    layer = Layer()
    layer._ple_mmap_prefix = "model.layers.0.ple.ple_embedding"
    layer.ngram_heads = 16
    layer.embedding_dim = 2560
    layer.split_ngram_parts = 2
    layer.positions_buffer = FakeTensor(np.zeros(4096, dtype=np.int64))
    layer.padded_buffer = FakeTensor(np.zeros((8, 4096), dtype=np.int64))
    layer._offload_weight_scale = FakeTensor(np.ones(1, dtype=np.uint8))
    layer.ngram_embedding = SimpleNamespace(
        org_vocab_size=10,
        table=SimpleNamespace(
            torch_dtype=fp8,
            row_bytes=160,
            rows_total=10,
            mm=[object(), object()],
        ),
    )
    model_state = SimpleNamespace(
        max_num_tokens=4096,
        num_new_sampled_tokens_per_step=1,
        device="cpu",
        ngram_context=FakeTensor(np.zeros((8, 4), dtype=np.int64)),
        ple_query_start_loc=FakeTensor(np.zeros(9, dtype=np.int64)),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_layer_ids=[0])
        ),
    )
    model = SimpleNamespace(modules=lambda: iter([layer]))
    mmap_module = types.ModuleType("vllm_ple_mmap")
    mmap_module.__file__ = str(HERE.parent / "vllm_ple_mmap.py.orig")
    mmap_module._REGISTRY = {layer._ple_mmap_prefix: layer}

    enabled = {
        "VLLM_PLE_STAGED": "1",
        "VLLM_PLE_MMAP": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    config = exact_config()
    config.compilation_config.static_forward_context = {
        "model.layers.0.ple": SimpleNamespace(ple_embedding=layer)
    }
    with patch.dict(os.environ, enabled, clear=True), patch.dict(
        sys.modules, {"vllm_ple_mmap": mmap_module}
    ), patch.object(adapter, "_require_pinned_runtime_sources"):
        adapter.bind_model_state(model_state, config, model)

    mismatched_state = SimpleNamespace(
        max_num_tokens=4096,
        num_new_sampled_tokens_per_step=1,
        device="cpu",
        ngram_context=model_state.ngram_context,
        ple_query_start_loc=model_state.ple_query_start_loc,
        model_config=model_state.model_config,
    )
    config_8192 = exact_config(tokens=8192)
    config_8192.compilation_config.static_forward_context = {
        "model.layers.0.ple": SimpleNamespace(ple_embedding=layer)
    }
    with patch.dict(os.environ, enabled, clear=True), patch.dict(
        sys.modules, {"vllm_ple_mmap": mmap_module}
    ), patch.object(adapter, "_require_pinned_runtime_sources"):
        try:
            adapter.bind_model_state(mismatched_state, config_8192, model)
        except RuntimeError as exc:
            assert "scheduler contract" in str(exc)
        else:
            raise AssertionError("mismatched model-state capacity was accepted")

    bad_bonus_state = SimpleNamespace(
        max_num_tokens=4096,
        num_new_sampled_tokens_per_step=2,
        device="cpu",
        ngram_context=model_state.ngram_context,
        ple_query_start_loc=model_state.ple_query_start_loc,
        model_config=model_state.model_config,
    )
    with patch.dict(os.environ, enabled, clear=True), patch.object(
        adapter, "_require_pinned_runtime_sources"
    ):
        try:
            adapter.bind_model_state(bad_bonus_state, exact_config(), model)
        except RuntimeError as exc:
            assert "one new sampled token" in str(exc)
        else:
            raise AssertionError("multi-bonus target width was accepted")

    stable = layer._ple_staged_rows
    pointer = stable.data_ptr()

    # Layout, table and capacity checks reject mutation after the pinned bind.
    original_dtype = layer.ngram_embedding.table.torch_dtype
    layer.ngram_embedding.table.torch_dtype = "bf16"
    try:
        adapter._initialize_layer(layer, 4096, "cpu")
    except RuntimeError as exc:
        assert "FP8 E4M3" in str(exc)
    else:
        raise AssertionError("wrong table dtype was accepted")
    layer.ngram_embedding.table.torch_dtype = original_dtype
    original_positions = layer.positions_buffer
    layer.positions_buffer = FakeTensor(np.zeros(1, dtype=np.int64))
    try:
        adapter._initialize_layer(layer, 4096, "cpu")
    except RuntimeError as exc:
        assert "workspace" in str(exc)
    else:
        raise AssertionError("undersized hash workspace was accepted")
    layer.positions_buffer = original_positions
    original_padded = layer.padded_buffer
    layer.padded_buffer = FakeTensor(np.zeros((8, 1), dtype=np.int64))
    try:
        adapter._initialize_layer(layer, 4096, "cpu")
    except RuntimeError as exc:
        assert "packed workspace" in str(exc)
    else:
        raise AssertionError("undersized packed workspace was accepted")
    layer.padded_buffer = original_padded
    try:
        adapter._verify_destination(layer, 4097, 4097)
    except RuntimeError as exc:
        assert "extents" in str(exc)
    else:
        raise AssertionError("oversized staging extent was accepted")

    stable.array[:16].fill(255)
    live = SimpleNamespace(
        num_reqs=2,
        num_tokens=7,
        num_tokens_after_padding=16,
        prefill_len_np=np.array([3, 4], dtype=np.int32),
        input_ids=FakeTensor(np.arange(16, dtype=np.int32), dtype="int32"),
    )
    query = FakeTensor(np.array([0, 3, 7, 16], dtype=np.int32), dtype="int32")
    context = FakeTensor(np.zeros((8, 3), dtype=np.int32), dtype="int32")

    # Binding occurs before vLLM resolves its final graph mode. The first
    # capture/live preparation must recheck that resolved state and fail closed.
    config.compilation_config.cudagraph_mode = "NONE"
    try:
        adapter.prepare_dummy_inputs(model_state, 16)
    except RuntimeError as exc:
        assert "FULL_DECODE_ONLY" in str(exc)
    else:
        raise AssertionError("post-bind graph-mode downgrade was accepted")
    config.compilation_config.cudagraph_mode = "FULL_DECODE_ONLY"
    adapter.prepare_model_inputs(model_state, live, query, context)
    assert getattr(model_state, adapter._FINAL_GRAPH_CHECK_ATTR) is True
    assert calls == [((7,), (3,), (2, 3))]
    expected = (np.arange(7 * 2560, dtype=np.uint32) % 251).astype(np.uint8)
    np.testing.assert_array_equal(stable.array[:7].reshape(-1), expected)
    assert not stable.array[7:16].any()
    assert layer._ple_staged_live_calls == 1

    # Runtime dummy and capture dummy paths zero and never touch mmap/hash code.
    stable.array[:32].fill(173)
    dummy = SimpleNamespace(
        num_reqs=2,
        num_tokens=16,
        num_tokens_after_padding=16,
        prefill_len_np=np.zeros(2, dtype=np.int32),
        input_ids=FakeTensor(np.zeros(16, dtype=np.int32), dtype="int32"),
    )
    adapter.prepare_model_inputs(model_state, dummy, query, context)
    adapter.prepare_dummy_inputs(model_state, 32)
    assert len(calls) == 1
    assert not stable.array[:32].any()
    assert layer._ple_staged_dummy_calls == 2

    # The active forward returns only a view of the same stable allocation.
    output = layer.forward(None, dummy.input_ids, query, context)
    assert output.shape == (16, 2560)
    assert stable.data_ptr() == pointer

    # The second qualified chunk size allocates exactly 20 MiB and exercises
    # the same live-copy, full padded-tail zero, dummy and stable-view paths.
    layer_8192 = Layer()
    layer_8192._ple_mmap_prefix = "model.layers.0.ple.ple_embedding"
    layer_8192.ngram_heads = 16
    layer_8192.embedding_dim = 2560
    layer_8192.split_ngram_parts = 2
    layer_8192.positions_buffer = FakeTensor(np.zeros(8192, dtype=np.int64))
    layer_8192.padded_buffer = FakeTensor(np.zeros((8, 8192), dtype=np.int64))
    layer_8192._offload_weight_scale = FakeTensor(np.ones(1, dtype=np.uint8))
    layer_8192.ngram_embedding = SimpleNamespace(
        org_vocab_size=10,
        table=SimpleNamespace(
            torch_dtype=fp8,
            row_bytes=160,
            rows_total=10,
            mm=[object(), object()],
        ),
    )
    adapter._initialize_layer(layer_8192, 8192, "cpu")
    stable_8192 = layer_8192._ple_staged_rows
    pointer_8192 = stable_8192.data_ptr()
    assert stable_8192.array.nbytes == 20 * 2**20
    stable_8192.array.fill(251)
    state_8192 = SimpleNamespace(
        **{
            adapter._LAYERS_ATTR: (layer_8192,),
            adapter._CONFIG_ATTR: exact_config(tokens=8192),
            adapter._FINAL_GRAPH_CHECK_ATTR: False,
        }
    )
    live_8192 = SimpleNamespace(
        num_reqs=2,
        num_tokens=7,
        num_tokens_after_padding=8192,
        prefill_len_np=np.array([3, 4], dtype=np.int32),
        input_ids=FakeTensor(np.arange(8192, dtype=np.int32), dtype="int32"),
    )
    adapter.prepare_model_inputs(state_8192, live_8192, query, context)
    assert not stable_8192.array[7:].any()
    assert stable_8192.data_ptr() == pointer_8192
    adapter.prepare_dummy_inputs(state_8192, 8192)
    assert not stable_8192.array.any()
    output_8192 = layer_8192.forward(None, live_8192.input_ids, query, context)
    assert output_8192.shape == (8192, 2560)
    assert stable_8192.data_ptr() == pointer_8192

    print(
        "PASS: env-off inertness, exact runtime gates, live ragged staging, "
        "dummy no-read, 4k/8k padding zero and stable address"
    )


if __name__ == "__main__":
    main()
