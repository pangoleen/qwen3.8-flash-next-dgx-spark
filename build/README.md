# Building the tuned image

## What this does, in short

The stock image works, but it leaves speed on the table in four places. This
directory fixes all four by adding layers on top of that image — it does not
replace or rebuild it. You run one script and get a faster image out.

**Result:** +22.0% on code, +28.9% on code with thinking on, +16.9% on prose,
and four real coding tasks in 102.3 s instead of 138.3 s.
([RESULTS.md §10](../RESULTS.md).)

## How to build it

```bash
./build.sh                       # -> qwen38-flash-tuned
```

Then serve that image instead of the stock one:

```bash
IMAGE=qwen38-flash-tuned MODE=hybrid MTP=3 GPU_MEM=0.80 ./serve.sh
```

You need the stock `qwen38-flash-dgx` image first (README Quickstart), a GB10,
and about 20 minutes — one overlay compiles a CUDA kernel.

## The four changes

Each is one Docker layer on the one before:

**1. `01-draft-vocab` — stop reading the whole vocabulary to guess one token.**
Speculative decoding uses a small "draft" model to guess the next few tokens,
which the real model then checks. That drafter reads a 1.27 GB table every
guess, to pick from 248,000 possible tokens. It only ever needs the single
best one, and in practice that is almost always a common token. So this
restricts the guess to the 65,536 most frequent, and reads a quarter of the
table. If the right token was outside that set, the guess is wrong and the real
model rejects it — exactly as it rejects any other bad guess. **The output the
server produces cannot change; only the guessing gets cheaper.**

**2. `02-det-topk` — make one attention kernel order-stable.** Vendored,
Apache-2.0, so the same input gives the same output.

**3. `03-staged-ple` — the one that unlocks the rest.** This model keeps a
44 GiB lookup table on disk and reads from it mid-calculation. A read from disk
in the middle of a calculation cannot be recorded into a CUDA graph — a
pre-recorded replay of the GPU work that removes most per-step overhead. So the
stock image can only record fragments. Move that read to *before* the
calculation starts, into a fixed buffer, and the whole decode step can be
recorded. Every token generated after that is cheaper.

**4. `04-gdn-flashinfer` — use the faster attention backend on this chip.**
A backport of a merged vLLM pull request.

## Why the gain shows up on prose too

The drafter's hit rate does not change — 3.64 accepted guesses per pass before,
3.57 after. So this is not better guessing, which would only help predictable
text like code. Each pass just costs less, which helps everything. That is why
prose improves as well, and why thinking improves most: a thinking turn is
thousands of small decode steps, and the per-step overhead is what got removed.

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

- `01-draft-vocab` was written from vLLM's `use_local_argmax_reduction`
  interface and its Apache-2.0 source. Credit for the reduced-draft-vocabulary
  technique goes to **MiaAI Lab** as prior art, and to the **FR-Spec** paper for
  the underlying idea. See its `NOTICE.md`.
- `02-det-topk` vendors work by **Jürgen Schmied** under Apache-2.0. Keep
  `vendor/LICENSE` and `vendor/NOTICE` with it.
- `03-staged-ple` and `04-gdn-flashinfer` extend Apache-2.0 vLLM source;
  `04` backports a merged vLLM pull request credited in its `SOURCES.md`.
