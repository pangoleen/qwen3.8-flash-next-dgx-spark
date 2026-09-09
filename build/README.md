# Rebuilding the tuned image

Four overlays on top of the stock image. Together they are the `tuned` arm in
[RESULTS.md §10](../RESULTS.md): **+22.0% decode on code, +28.9% on thinking,
+16.9% on prose**, and four real tasks in 102.3 s against 138.3 s.

```
qwen38-flash-dgx                 you build this first, from blazux upstream
  01-draft-vocab                 reduced MTP draft head
    02-det-topk                  deterministic QSA top-k CUDA extension
      03-staged-ple              stage the FP8 PLE read before forward
        04-gdn-flashinfer        FlashInfer GDN backend on sm121
```

```bash
./build.sh                       # -> qwen38-flash-tuned
./build.sh --stop-at 3           # stop after staged-ple
```

## What each overlay does

| Overlay | Change | Why it pays |
|---|---|---|
| `01-draft-vocab` | MTP draft argmax runs over a 65,536-row slice of the LM head | The drafter reads a 1.27 GB BF16 head once per draft step. Greedy drafting only needs an argmax, and a draft outside the slice is rejected by the target like any other bad draft. Bandwidth, not accuracy. |
| `02-det-topk` | Deterministic QSA `persistent_topk` CUDA extension | Order-stable top-k. Vendored, Apache-2.0. |
| `03-staged-ple` | The FP8 PLE mmap read is staged into a fixed buffer **before** `forward()` | This is the one that unlocks the rest. A synchronous host-to-device copy inside `forward()` cannot be captured into a CUDA graph, which is why the stock image is limited to PIECEWISE. Move the read out and `FULL_DECODE_ONLY` capture becomes legal. |
| `04-gdn-flashinfer` | FlashInfer GDN backend selection on sm121 | Backport of the merged vLLM PR. |

Accepted tokens per pass is unchanged across all of these (3.64 -> 3.57 on
code, RESULTS.md §10). The gain is not better drafting — each speculative pass
costs less. That is why it survives on prose, where drafting is weak.

## Requirements

- **A GB10.** `02-det-topk` compiles for `sm_121a` at image build time. This is
  not portable to other hardware without changing that.
- **The stock image**, built from
  [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX).
  These overlays patch that image; they do not replace it.
- Docker with the NVIDIA Container Toolkit.

## Why the build fails loudly, and what to do about it

Every patcher verifies the SHA256 of the file it is about to modify and
**refuses anything else**. `03-staged-ple` pins `f127380f…8425` for
`model_state.py`; `01-draft-vocab` pins `7735cee4…af431` for `mtp.py`.

That cuts both ways, and it is worth understanding before you hit it:

- **It is why this is reproducible.** If a patch applies, your input bytes are
  identical to the ones every number in RESULTS.md was measured on. There is no
  silent drift between your build and ours.
- **It means the recipe is pinned to one upstream build.** If blazux's image
  moves and vLLM changes underneath, the patchers stop. The build fails with the
  expected and actual hash rather than producing something different under the
  same name.

If that happens, the honest fix is to re-anchor: diff your installed file
against the `.orig` shipped beside each patcher, confirm the anchor points still
exist, update the expected SHA, and **re-measure**. Do not force a patch past a
hash mismatch and assume the published numbers still hold.

## Turning the overlays on

They are inert until the launcher enables them, so the built image behaves
exactly like the stock one until you ask for the tuned path:

```
VLLM_MTP_DRAFT_VOCAB   baked in by 01; the head is only reduced when
                       --speculative-config sets use_local_argmax_reduction
VLLM_QSA_DET_TOPK=1    with VLLM_QSA_EXACT_TOPK=0 (the exact path has priority)
VLLM_PLE_STAGED=1      enables the staged read; without it 03 is a no-op
```

and on the vLLM command line:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
--mamba-ssm-cache-dtype bfloat16
--max-num-batched-tokens 4096
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}'
```

`FULL_DECODE_ONLY` only captures with `VLLM_PLE_STAGED=1`. Without it the
capture fails on the synchronous copy, which is the whole point of overlay 03.

## What is not here

**The `tuned v2` arm.** RESULTS.md §10's strongest column (+31.7% / +34.2% /
+27.6%) adds four further overlays — compacted QSA, FP8 MTP experts, an FP8
draft head, and async metadata. They are not in this directory yet. What builds
here is `tuned`, not `tuned v2`.

**BF16 recurrent state changes output hashes.** `--mamba-ssm-cache-dtype
bfloat16` is a precision change, not a lossless one. Generations differ from the
stock recipe. No task benchmark in this repository proves quality is unchanged,
and RESULTS.md §10 says so.

## Licences

Everything here is Apache-2.0.

- `01-draft-vocab` is an independent implementation. See its `NOTICE.md`:
  **MiaAI Lab** is credited for the reduced-draft-vocabulary technique as prior
  art, and the **FR-Spec** paper for the underlying idea. MiaAI Lab's own
  implementation is AGPL-3.0-or-later and is **not** used, copied, or adapted
  here; this code was written from vLLM's `use_local_argmax_reduction` interface
  and its Apache-2.0 source.
- `02-det-topk` vendors work by **Jürgen Schmied** under Apache-2.0. Keep
  `vendor/LICENSE` and `vendor/NOTICE` with it.
- `03-staged-ple` and `04-gdn-flashinfer` extend Apache-2.0 vLLM source;
  `04` backports a merged vLLM pull request credited in its `SOURCES.md`.
