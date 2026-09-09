# SPDX-License-Identifier: Apache-2.0
"""Reduced draft vocabulary for the Qwen3.8-Flash-Next MTP drafter.

vLLM's speculative config exposes ``use_local_argmax_reduction``. When it is on,
``llm_base_proposer`` calls ``model.get_top_tokens(hidden_states)`` instead of
``model.compute_logits(hidden_states).argmax(dim=-1)``. The two calls must return
the same thing: greedy draft token ids in the target vocabulary id space.

At TP=1 that hook buys memory bandwidth. The drafter owns a ``ParallelLMHead``
over the full vocabulary (~248k rows). Every draft step reads all of it, and the
drafter only needs the argmax, not calibrated logits. So this module scores the
hidden state against a fixed, frequency-ranked subset of head rows and reads a
fraction of the weight.

When the true argmax sits outside the subset the draft is wrong. A wrong draft is
rejected by the target model at verification, exactly like any other bad draft,
so this trades acceptance rate for bandwidth and cannot change server output.
``compute_logits`` stays on the full head, so every non-draft path is untouched.

Layout of this file
-------------------
Everything between the ``BEGIN``/``END`` markers is the code that
``apply_draft_vocab.py`` copies into an installed ``mtp.py``:

* the MODULE block goes in after the last top-level import;
* the METHOD block goes into ``Qwen3_8FlashNextMTP`` after ``compute_logits``,
  re-indented by four spaces.

The blocks deliberately have no module-level imports. They use the module-global
``torch`` that both this file and ``mtp.py`` already import, and they import
``os`` and ``logging`` inside the functions that need them.
"""

import torch  # noqa: F401  (the blocks below use the module-global `torch`)

# ---8<--- BEGIN DRAFT VOCAB MODULE BLOCK ---8<---
# vllm-draft-vocab: reduced draft vocabulary for the MTP greedy head.
# SPDX-License-Identifier: Apache-2.0
# Independent implementation. See NOTICE.md for prior art and credits.

_DRAFT_VOCAB_ENV = "VLLM_MTP_DRAFT_VOCAB"
_DRAFT_VOCAB_SIZE_ENV = "VLLM_MTP_DRAFT_VOCAB_SIZE"
_DRAFT_VOCAB_CACHE_KEY = "_draft_vocab_subset_cache"

# Module-level caches. The token id file is read once per process. The messages
# already written are remembered so each outcome is logged one time only.
_draft_vocab_file_cache: tuple | None = None
_draft_vocab_logged: set = set()
_draft_vocab_logger_obj = None


def _draft_vocab_logger():
    """The vLLM logger when it is importable, else a stdlib logger."""

    global _draft_vocab_logger_obj
    if _draft_vocab_logger_obj is None:
        try:
            from vllm.logger import init_logger

            _draft_vocab_logger_obj = init_logger(__name__)
        except Exception:  # pragma: no cover - only outside a vLLM process
            import logging

            _draft_vocab_logger_obj = logging.getLogger(__name__)
    return _draft_vocab_logger_obj


def _draft_vocab_log_once(message: str) -> None:
    """Write one INFO line per distinct outcome."""

    if message in _draft_vocab_logged:
        return
    _draft_vocab_logged.add(message)
    _draft_vocab_logger().info("MTP draft vocabulary: %s", message)


def _draft_vocab_read_file(path: str):
    """Read one integer token id per line.

    Blank lines and ``#`` comments are skipped. Repeated ids are dropped, since
    a repeated row adds bytes and changes nothing.

    Returns ``(ids, None)`` or ``(None, reason)``.
    """

    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return None, "cannot read %r (%s)" % (path, exc)
    except UnicodeDecodeError as exc:
        return None, "cannot decode %r (%s)" % (path, exc)

    ids: list = []
    seen: set = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        try:
            token_id = int(entry, 10)
        except ValueError:
            return None, "%r line %d: %r is not an integer" % (path, lineno, entry)
        if token_id in seen:
            continue
        seen.add(token_id)
        ids.append(token_id)

    if not ids:
        return None, "%r holds no token ids" % (path,)
    return ids, None


def _draft_vocab_ids_for_path(path: str):
    """``_draft_vocab_read_file`` with a one-entry process cache."""

    global _draft_vocab_file_cache
    if _draft_vocab_file_cache is not None and _draft_vocab_file_cache[0] == path:
        return _draft_vocab_file_cache[1], _draft_vocab_file_cache[2]
    ids, reason = _draft_vocab_read_file(path)
    _draft_vocab_file_cache = (path, ids, reason)
    return ids, reason


def _draft_vocab_tp_size(lm_head) -> int:
    """Tensor parallel size of the head. Returns 1 when it cannot be found."""

    tp_size = getattr(lm_head, "tp_size", None)
    if isinstance(tp_size, int) and tp_size > 0:
        return tp_size
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _draft_vocab_build(model):
    """Gather the subset rows once.

    Returns ``(rows, ids, reason)``. ``rows`` is ``None`` when any guard fires;
    ``reason`` always explains the outcome.
    """

    import os

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return None, None, "the drafter has no lm_head"

    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None, None, "the lm_head has no 2-D weight (a placeholder head?)"
    if getattr(lm_head, "bias", None) is not None:
        return None, None, "the lm_head has a bias, which a plain matmul drops"
    if not torch.is_floating_point(weight):
        return None, None, "the lm_head weight dtype %s is not floating point" % (
            weight.dtype,
        )

    tp_size = _draft_vocab_tp_size(lm_head)
    if tp_size != 1:
        return None, None, (
            "tensor parallel size is %d, so the head is vocab-parallel" % tp_size
        )

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab_size = int(weight.shape[0])
    if int(weight.shape[0]) < vocab_size:
        return None, None, (
            "the lm_head holds %d rows for a vocabulary of %d, so it is sharded"
            % (int(weight.shape[0]), vocab_size)
        )

    path = os.environ.get(_DRAFT_VOCAB_ENV, "").strip()
    if not path:
        return None, None, "%s is not set" % (_DRAFT_VOCAB_ENV,)

    ids, reason = _draft_vocab_ids_for_path(path)
    if ids is None:
        return None, None, reason

    limit_text = os.environ.get(_DRAFT_VOCAB_SIZE_ENV, "").strip()
    if limit_text:
        try:
            limit = int(limit_text, 10)
        except ValueError:
            return None, None, "%s is not an integer: %r" % (
                _DRAFT_VOCAB_SIZE_ENV,
                limit_text,
            )
        if limit <= 0:
            return None, None, "%s must be positive, got %d" % (
                _DRAFT_VOCAB_SIZE_ENV,
                limit,
            )
        ids = ids[:limit]

    for token_id in ids:
        if token_id < 0 or token_id >= vocab_size:
            return None, None, "%r holds token id %d outside [0, %d)" % (
                path,
                token_id,
                vocab_size,
            )

    if len(ids) >= vocab_size:
        return None, None, "the subset covers the whole vocabulary, so it saves nothing"

    ids_tensor = torch.tensor(ids, dtype=torch.long, device=weight.device)
    rows = weight.detach().index_select(0, ids_tensor).contiguous()
    return rows, ids_tensor, "%d of %d rows from %r" % (len(ids), vocab_size, path)


def _draft_vocab_subset(model):
    """Return the cached ``(rows, ids)`` pair, building it on the first call.

    ``(None, None)`` means the caller must use the full head.
    """

    cache = model.__dict__.get(_DRAFT_VOCAB_CACHE_KEY)
    if cache is not None:
        return cache

    try:
        rows, ids, reason = _draft_vocab_build(model)
    except Exception as exc:  # keep the drafter alive on any setup failure
        rows, ids, reason = None, None, "setup failed (%r)" % (exc,)

    if rows is None:
        _draft_vocab_log_once("off, the drafter reads the full lm_head: %s" % reason)
        cache = (None, None)
    else:
        _draft_vocab_log_once("on, the drafter reads %s" % reason)
        cache = (rows, ids)

    model.__dict__[_DRAFT_VOCAB_CACHE_KEY] = cache
    return cache


# ---8<--- END DRAFT VOCAB MODULE BLOCK ---8<---


# ---8<--- BEGIN DRAFT VOCAB METHOD BLOCK ---8<---
def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Greedy draft token ids, read from a reduced slice of the LM head.

    This is the ``use_local_argmax_reduction`` hook. It returns what
    ``self.compute_logits(hidden_states).argmax(dim=-1)`` returns: token ids in
    the target vocabulary id space, same shape, same dtype.

    With ``VLLM_MTP_DRAFT_VOCAB`` set and every guard passed, the argmax runs
    over a fixed subset of head rows, so the step reads a fraction of the
    weight. A hit outside the subset yields a wrong draft, and the target model
    rejects it at verification. Otherwise the full head runs, unchanged.
    """

    rows, ids = _draft_vocab_subset(self)
    if rows is None:
        return self.compute_logits(hidden_states).argmax(dim=-1)

    if hidden_states.dtype != rows.dtype:
        hidden_states = hidden_states.to(rows.dtype)
    # rows keeps the [subset, hidden] layout of the head itself, so this is the
    # same access pattern as the full projection over fewer rows.
    subset_logits = torch.nn.functional.linear(hidden_states, rows)
    local = subset_logits.argmax(dim=-1)
    return ids.index_select(0, local.reshape(-1)).reshape(local.shape)


# ---8<--- END DRAFT VOCAB METHOD BLOCK ---8<---
