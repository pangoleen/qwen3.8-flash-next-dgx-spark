#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Insert the reduced draft vocabulary code into an installed vLLM ``mtp.py``.

The script refuses to touch a file whose SHA256 does not match the value you
pass. Run it twice and the second run does nothing. The result must parse as
Python, or the script fails and writes nothing.

Usage::

    python3 apply_draft_vocab.py \\
        --target /usr/lib/python3/.../vllm/models/qwen3_8_flash_next/nvidia/mtp.py \\
        --sha256 7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431

Add ``--dry-run`` to print the patched file instead of writing it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import shutil
import sys

MARKER = "vllm-draft-vocab:"
TARGET_CLASS = "Qwen3_8FlashNextMTP"
ANCHOR_METHOD = "compute_logits"
NEW_METHOD = "get_top_tokens"

MODULE_BEGIN = "---8<--- BEGIN DRAFT VOCAB MODULE BLOCK ---8<---"
MODULE_END = "---8<--- END DRAFT VOCAB MODULE BLOCK ---8<---"
METHOD_BEGIN = "---8<--- BEGIN DRAFT VOCAB METHOD BLOCK ---8<---"
METHOD_END = "---8<--- END DRAFT VOCAB METHOD BLOCK ---8<---"


class PatchError(Exception):
    """Any reason the patch must not be applied."""


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_block(source: str, begin: str, end: str) -> str:
    """Return the lines between two marker comments, markers excluded."""

    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if begin in line]
    ends = [i for i, line in enumerate(lines) if end in line]
    if len(starts) != 1 or len(ends) != 1:
        raise PatchError("the source must hold exactly one %r / %r pair" % (begin, end))
    if ends[0] <= starts[0]:
        raise PatchError("the %r marker comes before %r" % (end, begin))
    block = "\n".join(lines[starts[0] + 1 : ends[0]]).strip("\n")
    if not block.strip():
        raise PatchError("the block between %r and %r is empty" % (begin, end))
    return block


def indent_block(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else "" for line in block.split("\n"))


def find_module_insert_line(tree: ast.Module) -> int:
    """The line after the last top-level import. 1-based, insert before it."""

    last = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = node
    if last is None:
        raise PatchError("the target holds no top-level imports")
    return last.end_lineno + 1


def check_torch_import(tree: ast.Module) -> None:
    """The inserted code uses the module-global ``torch``."""

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" and alias.asname in (None, "torch"):
                    return
    raise PatchError("the target does not `import torch` at module level")


def find_method_insert_line(tree: ast.Module) -> tuple[int, int]:
    """Return ``(line to insert before, body indent)`` for the new method."""

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS:
            klass = node
            break
    else:
        raise PatchError("the target holds no class %s" % TARGET_CLASS)

    for item in klass.body:
        if isinstance(item, ast.FunctionDef) and item.name == NEW_METHOD:
            raise PatchError("%s.%s already exists" % (TARGET_CLASS, NEW_METHOD))

    for item in klass.body:
        if isinstance(item, ast.FunctionDef) and item.name == ANCHOR_METHOD:
            return item.end_lineno + 1, item.col_offset

    raise PatchError(
        "the class %s holds no %s method to anchor on" % (TARGET_CLASS, ANCHOR_METHOD)
    )


def build_patched(target_text: str, source_text: str) -> str:
    tree = ast.parse(target_text)
    check_torch_import(tree)

    module_block = extract_block(source_text, MODULE_BEGIN, MODULE_END)
    method_block = extract_block(source_text, METHOD_BEGIN, METHOD_END)
    if MARKER not in module_block:
        raise PatchError("the module block holds no %r marker" % MARKER)
    if ("def %s(" % NEW_METHOD) not in method_block:
        raise PatchError("the method block does not define %s" % NEW_METHOD)

    module_line = find_module_insert_line(tree)
    method_line, indent = find_method_insert_line(tree)

    edits = [
        (module_line, "\n" + module_block + "\n"),
        (method_line, "\n" + indent_block(method_block, indent) + "\n"),
    ]
    # Apply from the bottom up so earlier line numbers stay valid.
    edits.sort(key=lambda edit: edit[0], reverse=True)

    lines = target_text.splitlines(keepends=True)
    for lineno, text in edits:
        index = min(lineno - 1, len(lines))
        if index > 0 and not lines[index - 1].endswith("\n"):
            lines[index - 1] += "\n"
        lines.insert(index, text if text.endswith("\n") else text + "\n")
    return "".join(lines)


def verify(patched_text: str) -> None:
    """Fail loudly when the patched file does not parse or lacks the method."""

    try:
        tree = ast.parse(patched_text)
    except SyntaxError as exc:
        raise PatchError("the patched file does not parse: %s" % exc) from exc

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS:
            names = [
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            ]
            if NEW_METHOD not in names:
                raise PatchError(
                    "%s.%s is missing after patching" % (TARGET_CLASS, NEW_METHOD)
                )
            return
    raise PatchError("class %s is missing after patching" % TARGET_CLASS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="the installed mtp.py to patch")
    parser.add_argument(
        "--sha256",
        required=True,
        help="the expected SHA256 of the unpatched target",
    )
    parser.add_argument(
        "--source",
        default=str(pathlib.Path(__file__).with_name("draft_vocab_head.py")),
        help="the file that holds the marked code blocks",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    parser.add_argument("--no-backup", action="store_true", help="skip the .orig copy")
    args = parser.parse_args(argv)

    target = pathlib.Path(args.target)
    source = pathlib.Path(args.source)

    try:
        target_text = target.read_text(encoding="utf-8")
        source_text = source.read_text(encoding="utf-8")
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if MARKER in target_text:
        try:
            verify(target_text)
        except PatchError as exc:
            print("error: %s is already patched but broken: %s" % (target, exc),
                  file=sys.stderr)
            return 1
        print("%s is already patched, nothing to do." % target)
        return 0

    actual = sha256_of(target_text)
    expected = args.sha256.strip().lower()
    if actual != expected:
        print(
            "error: refusing to patch %s\n  expected sha256 %s\n  actual   sha256 %s"
            % (target, expected, actual),
            file=sys.stderr,
        )
        return 2

    try:
        patched_text = build_patched(target_text, source_text)
        verify(patched_text)
    except PatchError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    if args.dry_run:
        sys.stdout.write(patched_text)
        return 0

    if not args.no_backup:
        backup = target.with_suffix(target.suffix + ".orig-" + actual[:8])
        if not backup.exists():
            shutil.copy2(target, backup)
            print("backed up the original to %s" % backup)

    target.write_text(patched_text, encoding="utf-8")
    print(
        "patched %s\n  new sha256 %s" % (target, sha256_of(patched_text))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
