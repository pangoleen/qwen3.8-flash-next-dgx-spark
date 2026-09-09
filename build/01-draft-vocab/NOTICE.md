# NOTICE

SPDX-License-Identifier: Apache-2.0

This directory holds a reduced draft vocabulary for the Qwen3.8-Flash-Next MTP
drafter in vLLM: `draft_vocab_head.py`, `apply_draft_vocab.py`,
`test_draft_vocab.py` and this file.

## Credits

**MiaAI Lab** (<https://x.com/MiaAI_lab>) — prior art for the technique of
running the MTP draft argmax over a reduced, frequency-ranked slice of the LM
head instead of the full vocabulary. MiaAI Lab published an implementation of
this idea under AGPL-3.0-or-later. The credit here is for the idea.

**The FR-Spec paper** — the underlying idea: a speculative drafter can restrict
its head to a frequency-ranked subset of the vocabulary, because a draft outside
that subset is only a rejected draft, not a wrong output.

**The vLLM project** — the `use_local_argmax_reduction` speculative-decoding
option, the `get_top_tokens(hidden_states)` hook in
`vllm/v1/spec_decode/llm_base_proposer.py`, and the Apache-2.0 source file
`vllm/models/qwen3_8_flash_next/nvidia/mtp.py` that this code extends. vLLM is
Apache-2.0, Copyright contributors to the vLLM project.

## Independence

This is an independent implementation. It was written from two inputs only:

1. the vLLM interface contract — the documented `use_local_argmax_reduction`
   flag, the `get_top_tokens` call site, and the init-time check that the draft
   model provides the method;
2. the pristine Apache-2.0 vLLM source file it extends.

The author did not read, copy, adapt or consult MiaAI Lab's AGPL-3.0-or-later
implementation, or any other existing FR-Spec reference code. No file from that
implementation was opened or searched. Copyright covers expression, not ideas;
the idea is credited above, and the expression here is original.

Nothing in this directory is a derivative work of any AGPL-3.0-or-later code.
It is offered under Apache-2.0.
