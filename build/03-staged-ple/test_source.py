#!/usr/bin/env python3
"""CPU/source-only contract tests; imports neither vLLM nor Torch."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "patch_model_state", HERE / "patch_model_state.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def extract_pure_function(tree: ast.AST, name: str, namespace: dict):
    node = find_function(tree, name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(HERE / "vllm_ple_staging.py"), "exec"), namespace)
    return namespace[name]


def main() -> None:
    generator = load_generator()
    original = (HERE / "model_state.py.orig").read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == (
        generator.EXPECTED_SOURCE_SHA256
    )
    generated = generator.generate(original)
    checked_in = (HERE / "model_state_patched.py").read_text()
    assert generated == checked_in

    with tempfile.TemporaryDirectory(prefix="staged-ple-source-") as td:
        bad = original.replace("Qwen3_8FlashNextModelState", "WrongState", 1)
        try:
            generator.generate(bad)
        except RuntimeError as exc:
            assert "SHA256" in str(exc)
        else:
            raise AssertionError("changed exact source was accepted")

    state_tree = ast.parse(generated)
    companion_text = (HERE / "vllm_ple_staging.py").read_text()
    companion_tree = ast.parse(companion_text)
    init = find_function(state_tree, "__init__")
    prepare = find_function(state_tree, "prepare_inputs")
    dummy = find_function(state_tree, "prepare_dummy_inputs")
    assert "_ple_staged_bind" in ast.unparse(init)
    assert "_ple_staged_prepare(" in ast.unparse(prepare)
    assert "_ple_staged_prepare_dummy" in ast.unparse(dummy)

    staged_forward = find_function(companion_tree, "_staged_forward_impl")
    forward_text = ast.unparse(staged_forward)
    assert "_ple_staged_rows" in forward_text
    assert "torch.ops" not in forward_text
    assert "query_start_loc" not in ast.unparse(staged_forward.body[0].body[-1])
    assert "ngram_context" not in ast.unparse(staged_forward.body[0].body[-1])

    bind_text = ast.unparse(find_function(companion_tree, "bind_model_state"))
    assert "sha256" in bind_text
    assert "EXPECTED_MMAP_MODULE_SHA256" in bind_text
    assert "module_path.read_bytes()" in bind_text
    assert "_require_pinned_runtime_sources" in bind_text
    assert "num_new_sampled_tokens_per_step" in bind_text
    assert "_CONFIG_ATTR" in bind_text

    live_prepare_text = ast.unparse(
        find_function(companion_tree, "prepare_model_inputs")
    )
    dummy_prepare_text = ast.unparse(
        find_function(companion_tree, "prepare_dummy_inputs")
    )
    assert "_require_final_graph_contract" in live_prepare_text
    assert "_require_final_graph_contract" in dummy_prepare_text

    dockerfile = (HERE / "Dockerfile").read_text()
    assert "ARG BASE_IMAGE\n" in dockerfile
    assert "ARG BASE_IMAGE=" not in dockerfile
    for digest in (
        "f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425",
        "6690edcd2e2b2ae309b214991a060bffbdbbcdc42334e164db0aebcc0a5b397e",
        "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8",
        "3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410",
        "189c1312f052341d8f19ad06f436ae853a885257a2d2e3d8626c3471892051f9",
        "ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c",
        "575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1",
        "13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5",
        "8da66d9f48bd93c935d74e2635c45483dc999c87b94bc4ce828e727eb1713349",
        "2c1de093f6f386f1fab88c2a44ccd27f620d05563a05c7a0682d7c16d4a9ef0e",
    ):
        assert digest in dockerfile

    classify = extract_pure_function(
        companion_tree,
        "_classify_input_batch",
        {"Any": object, "np": np, "RuntimeError": RuntimeError},
    )

    class Batch:
        def __init__(self, values):
            self.num_reqs = len(values)
            self.prefill_len_np = np.asarray(values, dtype=np.int32)

    assert classify(Batch([10, 20])) is True
    assert classify(Batch([0, 0])) is False
    assert classify(Batch([])) is False
    try:
        classify(Batch([10, 0]))
    except RuntimeError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("mixed live/dummy batch was accepted")

    # Exact target layouts: 10/20 MiB for the two qualified chunk sizes; every
    # MTP-2/3/4 sequence count 1..8 has a distinct verification graph.
    assert 4096 * 16 * 160 == 10 * 2**20
    assert 8192 * 16 * 160 == 20 * 2**20
    assert [(1 + 2) * s for s in range(1, 9)] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert [(1 + 3) * s for s in range(1, 9)] == [4, 8, 12, 16, 20, 24, 28, 32]
    assert [(1 + 4) * s for s in range(1, 9)] == [5, 10, 15, 20, 25, 30, 35, 40]
    assert sorted(
        set(range(1, 9)) | {(1 + 3) * s for s in range(1, 9)}
    ) == [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
    assert sorted(
        set(range(1, 9)) | {(1 + 4) * s for s in range(1, 9)}
    ) == [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35, 40]

    launcher = (HERE.parent / "manage_candidate.py").read_text()
    assert "set(range(1, 9))" in launcher
    assert "{(a.mtp + 1) * seqs for seqs in range(1, 9)}" in launcher

    # Pure byte-level staging contract: live bytes land exactly, and reuse by
    # a smaller padded batch cannot expose the prior tail.
    rng = np.random.default_rng(42)
    stable = np.zeros((4096, 2560), dtype=np.uint8)
    pointer = stable.__array_interface__["data"][0]
    first = rng.integers(0, 256, (32, 2560), dtype=np.uint8)
    stable[:32] = first
    np.testing.assert_array_equal(stable[:32], first)
    second = rng.integers(0, 256, (7, 2560), dtype=np.uint8)
    stable[:7] = second
    stable[7:16] = 0
    np.testing.assert_array_equal(stable[:7], second)
    assert not stable[7:16].any()
    assert stable.__array_interface__["data"][0] == pointer

    print(
        "PASS: exact-source generator, AST graph boundary, live/dummy gate, "
        "10/20 MiB layouts, exact draft and MTP-2/3/4 verify widths, "
        "byte/padding/address contract"
    )


if __name__ == "__main__":
    main()
