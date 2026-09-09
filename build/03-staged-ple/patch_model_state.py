#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the staged-PLE old-preview model_state overlay.

The input must be byte-identical to the file extracted from vLLM
0.1.dev20073+g8e685d198. Anchor counts and the source SHA make this fail closed
instead of applying to a merely similar qwen4_exp revision.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

EXPECTED_SOURCE_SHA256 = (
    "f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact anchor, found {count}")
    return source.replace(old, new)


def generate(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"model_state source SHA256 {digest} != pinned {EXPECTED_SOURCE_SHA256}"
        )

    source = replace_once(
        source,
        "from vllm.v1.worker.gpu.states import RequestState\n",
        "from vllm.v1.worker.gpu.states import RequestState\n\n"
        "from vllm_ple_staging import (\n"
        "    bind_model_state as _ple_staged_bind,\n"
        "    prepare_dummy_inputs as _ple_staged_prepare_dummy,\n"
        "    prepare_model_inputs as _ple_staged_prepare,\n"
        ")\n",
        "staging imports",
    )
    source = replace_once(
        source,
        "        self.ple_query_start_loc = torch.zeros(\n"
        "            self.max_num_reqs + 1,\n"
        "            dtype=torch.int32,\n"
        "            device=self.device,\n"
        "        )\n",
        "        self.ple_query_start_loc = torch.zeros(\n"
        "            self.max_num_reqs + 1,\n"
        "            dtype=torch.int32,\n"
        "            device=self.device,\n"
        "        )\n"
        "        # Opt-in old-preview FP8 PLE staging. The helper is a no-op\n"
        "        # unless VLLM_PLE_STAGED=1 and otherwise validates the full\n"
        "        # V2/FULL_DECODE_ONLY/capture-width contract before allocation.\n"
        "        _ple_staged_bind(self, vllm_config, model)\n",
        "model-state bind",
    )
    source = replace_once(
        source,
        "        model_inputs.update(\n"
        "            query_start_loc=query_start_loc,\n"
        "            ngram_context=self._prepare_ngram_context(input_batch, req_states),\n"
        "        )\n"
        "        return model_inputs\n",
        "        ngram_context = self._prepare_ngram_context(input_batch, req_states)\n"
        "        model_inputs.update(\n"
        "            query_start_loc=query_start_loc,\n"
        "            ngram_context=ngram_context,\n"
        "        )\n"
        "        _ple_staged_prepare(\n"
        "            self, input_batch, query_start_loc, ngram_context\n"
        "        )\n"
        "        return model_inputs\n",
        "live input preparation",
    )
    source = replace_once(
        source,
        "        model_inputs.update(\n"
        "            query_start_loc=query_start_loc,\n"
        "            ngram_context=ngram_context,\n"
        "        )\n"
        "        return model_inputs\n\n\n"
        "__all__ = [\"Qwen3_8FlashNextModelState\"]\n",
        "        model_inputs.update(\n"
        "            query_start_loc=query_start_loc,\n"
        "            ngram_context=ngram_context,\n"
        "        )\n"
        "        _ple_staged_prepare_dummy(self, num_tokens)\n"
        "        return model_inputs\n\n\n"
        "__all__ = [\"Qwen3_8FlashNextModelState\"]\n",
        "capture dummy preparation",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    output = generate(source)
    args.output.write_text(output)
    print(f"wrote {args.output} ({hashlib.sha256(output.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
