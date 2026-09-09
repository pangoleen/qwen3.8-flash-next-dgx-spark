#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``draft_vocab_head``.

Neither torch nor vLLM is installed in this environment, so the test installs a
small numpy-backed stand-in for the handful of torch operations the module uses
(``tensor``, ``index_select``, ``contiguous``, ``detach``, ``to``, ``reshape``,
``argmax``, ``is_floating_point`` and ``nn.functional.linear``). The stand-in is
registered as ``torch`` in ``sys.modules`` before ``draft_vocab_head`` is
imported. Under a real torch install the same code runs unchanged.

Run it with::

    python3 test_draft_vocab.py
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import tempfile
import types

import numpy as np

# --------------------------------------------------------------------------
# The torch stand-in.
# --------------------------------------------------------------------------


class _Dtype:
    def __init__(self, name: str, np_dtype) -> None:
        self.name = name
        self.np = np_dtype

    def __repr__(self) -> str:
        return "torch." + self.name


_DTYPES: dict = {}


def _dtype_for(np_dtype) -> _Dtype:
    key = np.dtype(np_dtype).name
    if key not in _DTYPES:
        _DTYPES[key] = _Dtype(key, np.dtype(np_dtype))
    return _DTYPES[key]


class Tensor:
    """A numpy array with the slice of the torch API this module needs."""

    def __init__(self, array) -> None:
        self.a = np.asarray(array)

    @property
    def shape(self):
        return tuple(self.a.shape)

    @property
    def ndim(self) -> int:
        return int(self.a.ndim)

    @property
    def dtype(self) -> _Dtype:
        return _dtype_for(self.a.dtype)

    @property
    def device(self) -> str:
        return "cpu"

    def detach(self) -> "Tensor":
        return self

    def contiguous(self) -> "Tensor":
        return Tensor(np.ascontiguousarray(self.a))

    def to(self, dtype: _Dtype) -> "Tensor":
        return Tensor(self.a.astype(dtype.np))

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        assert dim == 0, "the stand-in only selects along dim 0"
        return Tensor(self.a[index.a])

    def argmax(self, dim: int = -1) -> "Tensor":
        return Tensor(np.argmax(self.a, axis=dim).astype(np.int64))

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.a.reshape(shape))

    def tolist(self):
        return self.a.tolist()


def _make_torch_module() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.long = _dtype_for(np.int64)
    torch.int64 = torch.long
    torch.float32 = _dtype_for(np.float32)
    torch.uint8 = _dtype_for(np.uint8)

    def tensor(data, dtype=None, device=None):
        array = np.array(data, dtype=dtype.np if dtype is not None else None)
        return Tensor(array)

    def is_floating_point(value: Tensor) -> bool:
        return bool(np.issubdtype(value.a.dtype, np.floating))

    torch.tensor = tensor
    torch.is_floating_point = is_floating_point

    functional = types.ModuleType("torch.nn.functional")

    def linear(inputs: Tensor, weight: Tensor) -> Tensor:
        return Tensor(inputs.a @ weight.a.T)

    functional.linear = linear
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    torch.nn = nn
    return torch


sys.modules.setdefault("torch", _make_torch_module())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import draft_vocab_head as dv  # noqa: E402

torch = sys.modules["torch"]


# --------------------------------------------------------------------------
# Fakes for the pieces of the model the hook touches.
# --------------------------------------------------------------------------


class FakeHead:
    def __init__(self, weight, bias=None, tp_size: int = 1) -> None:
        self.weight = weight
        self.bias = bias
        self.tp_size = tp_size


class FakeMissingHead:
    """Stands in for PPMissingLayer: a head object with no weight."""


class FakeConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class FakeMTP:
    """The part of Qwen3_8FlashNextMTP that ``get_top_tokens`` reads."""

    def __init__(self, weight, vocab_size=None, bias=None, tp_size=1, head=None):
        if head is None:
            head = FakeHead(weight, bias=bias, tp_size=tp_size)
        self.lm_head = head
        self.config = FakeConfig(
            vocab_size if vocab_size is not None else int(weight.shape[0])
        )
        self.compute_logits_calls = 0

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        self.compute_logits_calls += 1
        return torch.nn.functional.linear(hidden_states, self.lm_head.weight)

    def full_argmax(self, hidden_states):
        return self.compute_logits(hidden_states).argmax(dim=-1)


VOCAB = 64
HIDDEN = 8
TOKENS = 5


def make_weight(seed: int = 0, rows: int = VOCAB, hidden: int = HIDDEN):
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal((rows, hidden), dtype=np.float32))


def make_hidden(seed: int = 1, tokens: int = TOKENS, hidden: int = HIDDEN):
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal((tokens, hidden), dtype=np.float32))


def reset_module_state() -> None:
    """Clear the process-wide caches so each case starts clean."""

    dv._draft_vocab_file_cache = None
    dv._draft_vocab_logged.clear()


@contextlib.contextmanager
def vocab_env(path=None, size=None):
    reset_module_state()
    saved = {key: os.environ.get(key) for key in (dv._DRAFT_VOCAB_ENV,
                                                  dv._DRAFT_VOCAB_SIZE_ENV)}
    for key in saved:
        os.environ.pop(key, None)
    if path is not None:
        os.environ[dv._DRAFT_VOCAB_ENV] = str(path)
    if size is not None:
        os.environ[dv._DRAFT_VOCAB_SIZE_ENV] = str(size)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_module_state()


@contextlib.contextmanager
def id_file(ids, text=None):
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "draft_vocab.txt"
        if text is None:
            text = "".join("%d\n" % token_id for token_id in ids)
        path.write_text(text, encoding="utf-8")
        yield path


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError("%s: %r != %r" % (message, actual, expected))


def assert_true(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def strict_argmax_rows(logits) -> None:
    """Fail when any row has a tied maximum, which would make a test ambiguous."""

    for row in logits.a:
        top = np.max(row)
        assert_true(int(np.sum(row == top)) == 1, "the test data has a tied argmax")


# --------------------------------------------------------------------------
# (a) The argmax token is inside the subset.
# --------------------------------------------------------------------------


def test_hit_matches_full_argmax():
    weight = make_weight()
    hidden = make_hidden()
    model = FakeMTP(weight)
    full_logits = model.compute_logits(hidden)
    strict_argmax_rows(full_logits)
    expected = full_logits.argmax(dim=-1).tolist()

    # A frequency-ranked subset that happens to hold every winning row.
    subset = sorted(set(expected) | {0, 1, 2, 3})
    with id_file(subset) as path, vocab_env(path=path):
        model.compute_logits_calls = 0
        out = dv.get_top_tokens(model, hidden)
        logged = sorted(dv._draft_vocab_logged)

    assert_equal(len(logged), 1, "one INFO line per outcome")
    assert_true(logged[0].startswith("on, "), "the log must say it is on")
    assert_true(
        "%d of %d rows" % (len(subset), VOCAB) in logged[0],
        "the log must name the subset size, got %r" % logged[0],
    )
    assert_equal(out.tolist(), expected, "the subset path must match the full argmax")
    assert_equal(out.dtype.name, "int64", "the ids must keep the argmax dtype")
    assert_equal(out.shape, (TOKENS,), "the ids must keep the argmax shape")
    assert_equal(model.compute_logits_calls, 0, "the full head must not be read")


# --------------------------------------------------------------------------
# (b) The argmax token is outside the subset.
# --------------------------------------------------------------------------


def test_miss_returns_a_subset_member():
    weight = make_weight()
    hidden = make_hidden()
    model = FakeMTP(weight)
    full_logits = model.compute_logits(hidden)
    strict_argmax_rows(full_logits)
    winners = set(full_logits.argmax(dim=-1).tolist())

    # Every winning row is excluded, so every draft must be wrong but valid.
    subset = sorted(set(range(VOCAB)) - winners)[:16]
    assert_true(bool(subset), "the test needs a non-empty subset")

    with id_file(subset) as path, vocab_env(path=path):
        out = dv.get_top_tokens(model, hidden).tolist()

    reference = np.array(subset)[np.argmax(full_logits.a[:, subset], axis=-1)].tolist()
    for token_id in out:
        assert_true(token_id in subset, "the draft id must be a subset member")
        assert_true(token_id not in winners, "the draft must differ from the full argmax")
    assert_equal(out, reference, "the draft must be the argmax over the subset")


# --------------------------------------------------------------------------
# (c) Every guard falls back to the full head.
# --------------------------------------------------------------------------


def check_falls_back(model, hidden, path=None, size=None, label="", expect=None):
    """Assert the hook falls back, and that it fell back for the stated reason."""

    expected = model.full_argmax(hidden).tolist()
    with vocab_env(path=path, size=size):
        model.compute_logits_calls = 0
        out = dv.get_top_tokens(model, hidden)
        logged = sorted(dv._draft_vocab_logged)
    assert_equal(out.tolist(), expected, "fallback must match the full argmax: " + label)
    assert_equal(model.compute_logits_calls, 1, "the full head must run: " + label)
    assert_equal(
        model.__dict__[dv._DRAFT_VOCAB_CACHE_KEY], (None, None),
        "the fallback must be cached: " + label,
    )
    assert_equal(len(logged), 1, "one INFO line per outcome: " + label)
    assert_true(logged[0].startswith("off, "), "the log must say it is off: " + label)
    if expect is not None:
        assert_true(
            expect in logged[0],
            "the log must name the guard (%s), got %r" % (expect, logged[0]),
        )
    model.__dict__.pop(dv._DRAFT_VOCAB_CACHE_KEY, None)


def test_guard_env_unset():
    model = FakeMTP(make_weight())
    check_falls_back(model, make_hidden(), path=None, label="env unset",
                     expect="VLLM_MTP_DRAFT_VOCAB is not set")


def test_guard_missing_file():
    model = FakeMTP(make_weight())
    with tempfile.TemporaryDirectory() as directory:
        missing = pathlib.Path(directory) / "not-there.txt"
        check_falls_back(model, make_hidden(), path=missing, label="missing file",
                         expect="cannot read")
        check_falls_back(model, make_hidden(), path=directory, label="path is a dir",
                         expect="cannot read")


def test_guard_unreadable_file():
    model = FakeMTP(make_weight())
    with id_file([1, 2, 3]) as path:
        os.chmod(path, 0o000)
        try:
            with open(path, encoding="utf-8"):
                print("    skipped the unreadable-file case (running as root?)")
                return
        except OSError:
            pass
        try:
            check_falls_back(model, make_hidden(), path=path, label="unreadable file",
                             expect="cannot read")
        finally:
            os.chmod(path, 0o600)


def test_guard_empty_file():
    model = FakeMTP(make_weight())
    with id_file([], text="") as path:
        check_falls_back(model, make_hidden(), path=path, label="empty file",
                         expect="holds no token ids")
    with id_file([], text="\n\n# only a comment\n") as path:
        check_falls_back(model, make_hidden(), path=path, label="no ids in file",
                         expect="holds no token ids")


def test_guard_non_integer_line():
    model = FakeMTP(make_weight())
    with id_file([], text="12\nnot-a-number\n7\n") as path:
        check_falls_back(model, make_hidden(), path=path, label="non-integer line",
                         expect="line 2: 'not-a-number' is not an integer")


def test_guard_id_out_of_range():
    model = FakeMTP(make_weight())
    with id_file([1, 2, VOCAB]) as path:
        check_falls_back(model, make_hidden(), path=path, label="id == vocab_size",
                         expect="token id 64 outside [0, 64)")
    with id_file([1, 2, -3]) as path:
        check_falls_back(model, make_hidden(), path=path, label="negative id",
                         expect="token id -3 outside [0, 64)")


def test_guard_tensor_parallel():
    model = FakeMTP(make_weight(), tp_size=2)
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, label="tp_size 2",
                         expect="tensor parallel size is 2")


def test_guard_sharded_weight():
    # The head holds fewer rows than the vocabulary: a vocab-parallel shard.
    model = FakeMTP(make_weight(rows=VOCAB // 2), vocab_size=VOCAB)
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, label="sharded head",
                         expect="it is sharded")


def test_guard_placeholder_head():
    model = FakeMTP(make_weight(), head=FakeMissingHead())
    model.lm_head = FakeMissingHead()
    model.lm_head.weight = None  # PPMissingLayer has no usable weight
    full_weight = make_weight()

    # compute_logits still has to work, so give the fake a usable full head.
    def compute_logits(hidden_states, spec_step_idx: int = 0):
        model.compute_logits_calls += 1
        return torch.nn.functional.linear(hidden_states, full_weight)

    model.compute_logits = compute_logits
    model.full_argmax = lambda hidden: compute_logits(hidden).argmax(dim=-1)
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, label="placeholder head",
                         expect="no 2-D weight")


def test_guard_head_with_bias():
    weight = make_weight()
    model = FakeMTP(weight, bias=Tensor(np.zeros(VOCAB, dtype=np.float32)))
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, label="head with bias",
                         expect="has a bias")


def test_guard_non_float_weight():
    # A packed or quantized head is not a plain matmul, so fall back.
    weight = Tensor(np.arange(VOCAB * HIDDEN, dtype=np.uint8).reshape(VOCAB, HIDDEN))
    model = FakeMTP(weight)
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, label="non-float head",
                         expect="is not floating point")


def test_guard_full_vocabulary_subset():
    model = FakeMTP(make_weight())
    with id_file(list(range(VOCAB))) as path:
        check_falls_back(model, make_hidden(), path=path, label="subset is everything",
                         expect="covers the whole vocabulary")


def test_guard_bad_size_env():
    model = FakeMTP(make_weight())
    with id_file([1, 2, 3, 4]) as path:
        check_falls_back(model, make_hidden(), path=path, size="many", label="size text",
                         expect="is not an integer: 'many'")
        check_falls_back(model, make_hidden(), path=path, size="0", label="size zero",
                         expect="must be positive")


# --------------------------------------------------------------------------
# Behaviour around the cache and the size knob.
# --------------------------------------------------------------------------


def test_rows_are_gathered_once():
    weight = make_weight()
    hidden = make_hidden()
    model = FakeMTP(weight)
    subset = [3, 9, 17, 25, 40]
    with id_file(subset) as path, vocab_env(path=path):
        first = dv.get_top_tokens(model, hidden).tolist()
        # Change the head under the hook. A cached buffer cannot see this.
        weight.a[:] = 0.0
        second = dv.get_top_tokens(model, hidden).tolist()
    assert_equal(second, first, "the subset rows must be gathered once")
    assert_equal(model.compute_logits_calls, 0, "the full head must stay unread")


def test_size_env_truncates():
    weight = make_weight()
    hidden = make_hidden()
    model = FakeMTP(weight)
    subset = [3, 9, 17, 25, 40]
    with id_file(subset) as path, vocab_env(path=path, size=2):
        out = dv.get_top_tokens(model, hidden).tolist()
    for token_id in out:
        assert_true(token_id in subset[:2], "the size knob must truncate the subset")


def test_duplicate_ids_are_dropped():
    weight = make_weight()
    hidden = make_hidden()
    plain = FakeMTP(weight)
    doubled = FakeMTP(weight)
    subset = [3, 9, 17, 25, 40]
    with id_file(subset) as path, vocab_env(path=path):
        expected = dv.get_top_tokens(plain, hidden).tolist()
    with id_file(subset + subset, text="".join(
        "%d\n" % i for i in subset + subset
    )) as path, vocab_env(path=path):
        actual = dv.get_top_tokens(doubled, hidden).tolist()
    assert_equal(actual, expected, "repeated ids must not change the result")


def test_comments_and_blank_lines():
    weight = make_weight()
    hidden = make_hidden()
    model = FakeMTP(weight)
    text = "# frequency ranked\n\n3\n 9 \n\n17\n25\n40\n"
    reference = FakeMTP(weight)
    with id_file([3, 9, 17, 25, 40]) as path, vocab_env(path=path):
        expected = dv.get_top_tokens(reference, hidden).tolist()
    with id_file([], text=text) as path, vocab_env(path=path):
        actual = dv.get_top_tokens(model, hidden).tolist()
    assert_equal(actual, expected, "comments and blanks must be skipped")


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - the runner reports everything
            failures.append((test.__name__, exc))
            print("FAIL %s: %s" % (test.__name__, exc))
        else:
            print("ok   %s" % test.__name__)
    print("\n%d passed, %d failed, out of %d" % (
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
