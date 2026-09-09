#!/usr/bin/env python3
"""Faithful CPU model of the pinned MRV2 CUDA-graph candidate contract.

This mirrors the relevant uniform-decode branch of the exact installed
``CudaGraphManager._init_candidates`` and ``dispatch`` methods. The runtime
source files implementing that behavior are SHA-pinned by the companion and
Dockerfile; this test deliberately imports neither Torch nor vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int
    uniform_token_count: int


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def build_candidates(
    capture_sizes: list[int], decode_query_len: int, max_num_reqs: int = 8
) -> tuple[list[Descriptor], dict[int, list[Descriptor]]]:
    """Mirror pinned uniform FULL_DECODE_ONLY descriptor construction."""

    max_decode_tokens = max_num_reqs * decode_query_len
    max_capture_size = max(capture_sizes)
    descriptors: list[Descriptor] = []
    by_tokens: dict[int, list[Descriptor]] = {}
    for requested in sorted(capture_sizes):
        num_tokens = round_up(requested, decode_query_len)
        num_reqs = num_tokens // decode_query_len
        if (
            num_tokens > max_decode_tokens
            or num_tokens > max_capture_size
            or num_reqs > max_num_reqs
        ):
            continue
        descriptor = Descriptor(num_tokens, num_reqs, decode_query_len)
        if descriptor not in descriptors:
            descriptors.append(descriptor)
            by_tokens.setdefault(num_tokens, []).append(descriptor)

    candidates: dict[int, list[Descriptor]] = {}
    range_start = 0
    for graph_tokens in sorted(by_tokens):
        for actual_tokens in range(range_start, graph_tokens + 1):
            candidates[actual_tokens] = by_tokens[graph_tokens]
        range_start = graph_tokens + 1
    return descriptors, candidates


def dispatch(
    candidates: dict[int, list[Descriptor]],
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int,
) -> Descriptor | None:
    """Mirror pinned compatibility selection; None is eager/NONE mode."""

    for descriptor in candidates.get(num_tokens, ()):
        if (
            descriptor.uniform_token_count == uniform_token_count
            and descriptor.num_reqs >= num_reqs
            and descriptor.num_tokens >= num_tokens
        ):
            return descriptor
    return None


def exact_union(k: int) -> list[int]:
    return sorted(set(range(1, 9)) | {(k + 1) * seqs for seqs in range(1, 9)})


def descriptor_tokens(sizes: list[int], query_len: int) -> list[int]:
    descriptors, _ = build_candidates(sizes, query_len)
    return [descriptor.num_tokens for descriptor in descriptors]


def main() -> None:
    for k in (2, 3, 4):
        query_len = k + 1
        target_only = [query_len * seqs for seqs in range(1, 9)]
        union = exact_union(k)

        # Target and speculator-prefill managers both use K+1. Raw draft sizes
        # round and deduplicate into the same eight valid verification graphs.
        expected_target = target_only
        assert descriptor_tokens(target_only, query_len) == expected_target
        assert descriptor_tokens(union, query_len) == expected_target

        # The follow-on speculator manager uses query_len=1 and therefore gains
        # exact request-count graphs rather than inheriting target padding.
        assert descriptor_tokens(union, 1) == list(range(1, 9))
        _, draft_candidates = build_candidates(union, 1)
        for seqs in range(1, 9):
            chosen = dispatch(draft_candidates, seqs, seqs, 1)
            assert chosen == Descriptor(seqs, seqs, 1)

        assert max(union) == query_len * 8

    # Reproduce the retained K3 and candidate K4 behavior before the union.
    k3_old = [4 * seqs for seqs in range(1, 9)]
    k4_old = [5 * seqs for seqs in range(1, 9)]
    _, k3_candidates = build_candidates(k3_old, 1)
    _, k4_candidates = build_candidates(k4_old, 1)
    assert descriptor_tokens(k3_old, 1) == [4, 8]
    assert dispatch(k3_candidates, 1, 1, 1) == Descriptor(4, 4, 1)
    assert descriptor_tokens(k4_old, 1) == [5]
    assert dispatch(k4_candidates, 1, 1, 1) == Descriptor(5, 5, 1)
    assert dispatch(k4_candidates, 6, 6, 1) is None

    # Full union changes only draft graph inventory: +6 graphs for K3 and +7
    # for K4. Target and speculator-prefill remain at eight graphs each.
    assert 8 + 8 + len(descriptor_tokens(k3_old, 1)) == 18
    assert 8 + 8 + len(descriptor_tokens(exact_union(3), 1)) == 24
    assert 8 + 8 + len(descriptor_tokens(k4_old, 1)) == 17
    assert 8 + 8 + len(descriptor_tokens(exact_union(4), 1)) == 24

    print(
        "PASS: pinned MRV2 descriptor/dispatch model, unchanged target graphs, "
        "exact draft S1..8 graphs, K3/K4 old-padding reproduction"
    )


if __name__ == "__main__":
    main()
