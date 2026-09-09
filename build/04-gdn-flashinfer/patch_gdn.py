#!/usr/bin/env python3
"""Backport vLLM PR #55715 onto the pinned Flash-Next image."""

from __future__ import annotations

import hashlib
import py_compile
import sys
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0"
)
PATCHED_SOURCE_SHA256 = (
    "dec15cb5f2e8fd7d4b9e1d0387b1da86206a68ed18ef06d5ad95fb0b30bfcdc4"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact anchor, found {count}")
    return source.replace(old, new)


def patch(source: str) -> str:
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"qwen_gdn_linear_attn.py SHA256 {actual} != pinned "
            f"{EXPECTED_SOURCE_SHA256}"
        )

    source = replace_once(
        source,
        "      - Hopper (SM90) — no further constraints;\n"
        "      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.\n",
        "      - Hopper (SM90) — no further constraints;\n"
        "      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``;\n"
        "      - Blackwell (SM12.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.\n",
        "FlashInfer support documentation",
    )
    source = replace_once(
        source,
        "        supports_flashinfer = True\n"
        "        supports_cutedsl = True\n\n"
        "    if backend in [\"flashinfer\", \"auto\"] and supports_flashinfer:\n",
        "        supports_flashinfer = True\n"
        "        supports_cutedsl = True\n"
        "    elif (\n"
        "        current_platform.is_device_capability_family(120)\n"
        "        and head_k_dim == 128\n"
        "        and current_platform.get_cuda_runtime_major() >= 13\n"
        "    ):\n"
        "        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.\n"
        "        supports_flashinfer = True\n\n"
        "    if backend in [\"flashinfer\", \"auto\"] and supports_flashinfer:\n",
        "SM12x FlashInfer backend gate",
    )
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gdn.py QWEN_GDN_LINEAR_ATTN_PATH")
    path = Path(sys.argv[1])
    updated = patch(path.read_text())
    updated_sha = hashlib.sha256(updated.encode()).hexdigest()
    if updated_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            f"patched qwen_gdn_linear_attn.py SHA256 {updated_sha} != expected "
            f"{PATCHED_SOURCE_SHA256}"
        )
    path.write_text(updated)
    py_compile.compile(str(path), doraise=True)
    print(f"qwen_gdn_linear_attn={updated_sha}")


if __name__ == "__main__":
    main()
