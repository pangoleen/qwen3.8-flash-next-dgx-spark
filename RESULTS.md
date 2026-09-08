# Every measurement, with its conditions

One NVIDIA DGX Spark (GB10, 128 GB unified). `blazux/qwen3.8-Flash-DGX` (vLLM,
PLE table mmapped from NVMe), `RadixArk/Qwen3.8-Flash-Next-NVFP4`, hybrid mode
(fp8 side layers), 2026-09-02 through 2026-09-06.

**Counting and methodology, read before comparing.** Generation is decode only:
completion tokens divided by (wall time minus time to first token). Prefill is
prompt tokens divided by the cold time to first token. Accepted tokens per pass
comes from vLLM's own `vllm:spec_decode_num_accepted_tokens_total` and
`_num_drafts_total` counters (a delta across the rung, not a point read).
Thinking is off in every row. Most tables are **one boot**, and one boot is not
a distribution. Raw rows are in `data/`; `bench/plotsweep.py` rebuilds the
chart from them.

## 1. The context ladder, thinking off

`bench/ctxsweep.py`, hybrid mode, MTP 3, `GPU_MEM=0.80`, temperature 0.6 (this
run only — the tuning ladder in §4 used temperature 0; both are thinking off,
and the effect of temperature on decode speed here is within run-to-run noise,
not a separate variable being tested), **4 reps per rung, median**, one boot,
2026-09-05.

| Prompt tok | Prefill tok/s | Gen tok/s | Tok/pass | Saturation | Cold TTFT | Warm TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 327 | 132.0 | 37.44 | 3.486 | 0.872 | 2.48 s | 0.48 s |
| 864 | 580.6 | 41.64 | 3.620 | 0.905 | 1.49 s | 0.74 s |
| 2,271 | 1,132.0 | 41.07 | 3.586 | 0.896 | 2.01 s | 1.48 s |
| 4,292 | 1,417.3 | 43.04 | 3.635 | 0.909 | 3.03 s | 1.63 s |
| 8,232 | 1,414.9 | 42.13 | 3.587 | 0.897 | 5.82 s | 1.26 s |
| 16,424 | 1,322.3 | 42.33 | 3.601 | 0.900 | 12.42 s | 1.46 s |
| 32,850 | 1,491.4 | 42.62 | 3.636 | 0.909 | 22.03 s | 1.71 s |
| 66,177 | 1,780.4 | 41.38 | 3.586 | 0.896 | 37.17 s | 1.70 s |
| 131,437 | 1,718.3 | 40.64 | 3.474 | 0.869 | 76.49 s | 1.73 s |
| 258,790 | 1,636.7 | 44.22 | 3.693 | 0.923 | 158.11 s | 2.84 s |

Generation holds 37-44 tok/s across the whole ladder, no falloff at the top
rung. Prefill rises with depth (132 to ~1,800 tok/s) as the small-prompt figure
is the n-gram table's page cache being cold after boot; it warms within the
first few rungs. Warm TTFT — a repeated prefix hitting the prefix cache — stays
under 3 s even at 258,790 prompt tokens.

## 2. Concurrency, measured with sparkDash

**Not measured with anything in this repository.** `bench/concbench.py`
disagrees with these numbers by roughly 4x aggregate at 8 streams on the same
server, the same night — see §7. These rows are
[MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash)'s DecodeBench,
run headless against the recommended launch: `MODE=hybrid MTP=3 CTX=262144
GPU_MEM=0.80`, KV bf16, 8 seats, temperature 0, thinking off, streams padded to
the token count, 32-token warmup, 2026-09-05.

**Code (clamp_00..49 Python helpers), 2048 tokens:**

| streams | per stream tok/s | aggregate tok/s | mean TTFT ms |
|---|---:|---:|---:|
| 1 | 45.42 | 45.42 | 435.25 |
| 2 | 41.16 | 82.21 | 550.36 |
| 4 | 33.30 | 131.88 | 1134.95 |
| 6 | 28.84 | 171.62 | 820.46 |
| 8 | 27.64 | 218.32 | 1200.49 |

**Structured (count 1→200), 2048 tokens:**

| streams | per stream tok/s | aggregate tok/s | mean TTFT ms |
|---|---:|---:|---:|
| 1 | 45.46 | 45.46 | 343.38 |
| 2 | 41.56 | 82.89 | 275.13 |
| 4 | 32.93 | 131.46 | 957.27 |
| 6 | 28.92 | 173.32 | 378.93 |
| 8 | 26.26 | 209.61 | 433.76 |

**Prose (hash-map explanation), 2048 tokens:**

| streams | per stream tok/s | aggregate tok/s | mean TTFT ms |
|---|---:|---:|---:|
| 1 | 31.78 | 31.78 | 234.04 |
| 2 | 28.48 | 56.44 | 373.80 |
| 4 | 17.22 | 67.68 | 463.76 |
| 6 | 15.50 | 92.24 | 521.98 |
| 8 | 14.04 | 109.96 | 586.31 |

Code and structured output scale almost identically to 8 streams; prose falls
off much faster past 2 streams. This is MTP acceptance, not the server: prose
is far less predictable token-to-token than code or a repeated structured
pattern, so the speculative head accepts fewer tokens per pass and the drafter
buys less.

**Structured, 256 tokens, single stream and six** (a smaller-output setting,
included because it is directly comparable to a same-day third-party run on a
different stack at the identical setting: 31.1 tok/s single-stream, 70.1
aggregate at six):

| streams | per stream tok/s | aggregate tok/s | mean TTFT ms |
|---|---:|---:|---:|
| 1 | 48.64 | 48.64 | 486.79 |
| 6 | 28.10 | 163.77 | 486.65 |

## 3. Prefill, measured with sparkDash

sparkDash's PrefillBench, unique-prefix sweep, stock recipe, 2026-09-05.

| context | prompt tok | TTFT s | prefill tok/s |
|---|---:|---:|---:|
| 8,192 | 8,232 | 5.41 | 1,522.81 |
| 16,384 | 16,422 | 8.80 | 1,865.53 |
| 32,768 | 32,805 | 17.33 | 1,892.51 |
| 65,536 | 65,574 | 35.08 | 1,869.33 |
| 131,072 | 131,107 | 72.36 | 1,811.97 |

The 262,144 rung was rejected: prompt plus the benchmark's own reserved output
allowance exceeds the native ceiling. That is the tool's request, not the
server's limit — see the context ladder in §1, whose 258,790-token rung fits
because it reserves less headroom for output.

## 4. The tuning knobs, full ladder

`bench/ctxsweep.py`, temperature 0, 512 output tokens, **2 reps**, one boot per
column, 2026-09-02. Generation tok/s, accepted tokens per pass in brackets.

| Prompt tokens | As published, MTP 2 | Hybrid, MTP 2 | **Hybrid, MTP 3** | Hybrid, MTP 4 |
|---|---|---|---|---|
| 324 | 33.5 (2.81) | 36.8 (2.90) | **42.2** (3.62) | 41.7 (4.36) |
| 8,159 | 34.3 (2.91) | 39.9 (2.88) | **44.3** (3.65) | 41.3 (4.27) |
| 32,764 | 33.7 (2.86) | 40.7 (2.89) | **45.2** (3.68) | 47.1 (4.46) |
| 65,842 | 33.6 (2.80) | 39.0 (2.76) | **46.0** (3.71) | 44.1 (4.23) |
| 130,656 | 33.9 (2.82) | 39.1 (2.80) | **46.1** (3.70) | 42.8 (4.12) |
| 257,334 | 33.9 (2.86) | 40.6 (2.85) | **42.6** (3.49) | 44.2 (4.35) |

`EXACT_TOPK`, same sweep, `Hybrid, MTP 3` column only, stock (`EXACT_TOPK=0`)
vs exact:

Exact top-k is the default in `serve.sh` because it is deterministic at
temperature 0. Turning it off (`EXACT_TOPK=0`) measured +17-24% prefill with
generation unchanged, at the cost of non-deterministic output token-for-token
at the same seed. Not in the recommended launch; a reasonable trade if
determinism does not matter for your use.

## 5. Boot time

Default loader: 763 s to "Application startup complete". `FAST_LOAD=1`
(fastsafetensors, GDS forced off, PLE table files skipped, page cache released
as it loads — `scripts/weight_utils.nogds.py`): 297 s. Decode and acceptance
unchanged between the two.

**Not yet a settled default.** Two of five `FAST_LOAD=1` boots on the night of
2026-09-05 were OOM-killed by the kernel during the MTP draft-weight load pass.
Cause: `fastsafetensors` moves whole files to the device, and the load pass for
the draft head re-streamed the *entire* ~76 GiB checkpoint through device
staging on top of the weights already resident, rather than just the draft's
own three bf16 shards. `weight_utils.nogds.py` restricts the second pass to
files matching `VLLM_FASTSAFETENSORS_PASS2_GLOB`. The fix matches the
mechanism and is deployed, but has not been proven clean across enough boots to
call it done. `FAST_LOAD` defaults to off in `serve.sh` until it has.

## 6. Harness behaviour under a broken grader

Not a benchmark result — a data point on trustworthiness that came up while
building a coding-agent test harness against this recipe. An early version of
the harness's grading script (`verify.js`) computed the wrong expected total
for its own fixture (a inventory-value check: it expected 22, the correct
value for the fixture's data was 23). Once the grader was fixed, this recipe
passed 7/7 on three separate runs and two fan-out (parallel-subagent) runs, 8/8
each, no failures.

Before the fix, this model refused to satisfy the wrong grader — its own trace
correctly reported that the expected value looked wrong given the visible
data, rather than adjusting its output or the test to match. That is worth
recording as a behavioural fact about this specific model on this specific
recipe, at this sample size (5 runs), not as a general claim: no task
benchmark has been run here, and 5 runs is not a distribution.

## 7. Open: `bench/concbench.py` does not agree with sparkDash

`bench/concbench.py` was written for a companion recipe on a different engine
(SGLang) and repointed at this vLLM server with `--base`/`--model` flags. Its
metric-reading code was checked, not assumed, and it is engine-aware: it reads
`vllm:spec_decode_num_accepted_tokens_total`/`_num_drafts_total` correctly on
this server, and its accepted-tokens-per-pass output (~3.6, consistent with
MTP 3) looks sane. Its throughput and TTFT numbers do not:

| conc | agg tok/s (concbench.py, seqs=8, hybrid MTP3) | agg tok/s (sparkDash, code, same nominal config) |
|---:|---:|---:|
| 1 | 19.3 | 45.4 |
| 2 | 29.1 | 82.2 |
| 4 | 39.2 | 131.9 |
| 8 | 55.2 | 218.3 |

TTFT under `concbench.py` at conc=8 is 19.3 s median; sparkDash's is 1.2 s at
the same nominal concurrency. `concbench.py` gives each stream a unique
2,048-token prefix specifically so the prefix cache cannot merge them, which is
a harder, more adversarial setup than sparkDash's — but a difference that large
is not explained by workload difficulty alone, and the two tools' TTFT numbers
disagree by more than an order of magnitude on requests that should look
similar to the scheduler. Not resolved as of 2026-09-06. Until it is, treat
`concbench.py`'s Flash-Next numbers as unverified and use sparkDash's (§2) for
anything you need to cite.

## 8. `PREFIX_CACHE` default changed to 0 (2026-09-07)

**Not a config choice — vLLM forces it.** `--enable-prefix-caching` on this
model puts Mamba caching into `align` mode unconditionally; confirmed straight
from this server's own boot log:

```
config.py:605 Mamba cache mode is set to 'align' for Qwen4ExpForConditionalGeneration
              by default when prefix caching is enabled
```

Neither `--mamba-cache-mode none` nor `all` can be forced instead — both were
tried directly against this server and vLLM rejected each one back to `align`
(`none` silently, `all` with an explicit warning naming the fallback). There is
no flag-only way to reach a different Mamba caching mode with prefix caching on.

**`align` mode has a real, reproducible correctness problem here.** Sending
the identical prompt twice in a row at temperature 0 sometimes returns a
completion that does not engage with the actual content at all — e.g. a
generic "I'll start by exploring the project structure" opener for a prompt
that was pure benchmark-script text, no project present. Re-running the exact
same request immediately after gives a correct, on-topic answer that then
stays stable across further repeats. The realistic cost: a tool-call-shaped
request extending a cached 100k-token prefix by 40k new tokens measured
anywhere from 2 s (clean cache hit) to 80 s (cold-equivalent miss) across
otherwise-identical runs.

Four real, still-open, unmerged upstream vLLM bugs describe exactly this
class of failure for hybrid Mamba models under speculative decoding
([Radar105's write-up](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark)
found them first): PRs
[53798](https://github.com/vllm-project/vllm/pull/53798),
[54076](https://github.com/vllm-project/vllm/pull/54076),
[54713](https://github.com/vllm-project/vllm/pull/54713),
[55390](https://github.com/vllm-project/vllm/pull/55390). We reconciled all
four by hand against our exact vLLM commit (`0.1.dev20073+g8e685d198` — none
applied cleanly with `git apply`, the codebase had drifted from whatever
snapshot they were cut against) and deployed them as mounted file overrides.
The server booted clean, CUDA graphs captured normally, decode speed improved
over buggy `align` — **and the actual failure was completely unaffected**: hit
fraction and TTFT on the same tool-call test came back identical to the
unpatched baseline to 16 decimal places. The bugs are real; they are not the
cause of what we're seeing.

**Measured comparison, `bench/ctxsweep.py`, 2 reps, `--skip-warm`:**

| Prompt tok | `align` (old default) | No-cache (new default) | Patched `align` |
|---|---:|---:|---:|
| ~2,270 | 41.07 tok/s | **43.50** tok/s | 42.80 tok/s |
| ~32,850 | 42.62 tok/s | **45.04** tok/s | 44.20 tok/s |
| ~131,440 | 40.64 tok/s | **43.53** tok/s | 43.46 tok/s |

No-cache wins decode at every depth tested, ties patched `align` at the top
rung, and never reproduced the determinism failure across any test run against
it. The only workload that loses anything by disabling caching is one with
genuinely large repeated prefixes across turns — and `align` mode cannot be
trusted for that workload anyway, per the above. `PREFIX_CACHE=0` is therefore
the default now, not a fallback: it was already the better measured option
before the patch attempt, and four hours of careful patching didn't produce a
config that beat it.

**Numbers elsewhere in this document predate this change.** §1's context
ladder and the tuning-knob table in the README were measured under the old
`PREFIX_CACHE=1` default. Because that sweep's prompts are unique per rung
(prefix caching provides no benefit to a benchmark designed to defeat it),
the gap is small and in the same direction as the table above — the published
"37-44 tok/s flat" claim holds under the new default as well, if anything
understating it slightly.

## 9. Candidate recipe (measured 2026-09-08, not yet adopted)

A combination of three changes, each independently measured and gated on
functional/quality checks before being combined, reached
**+27.9% fixed-work rate / -21.8% latency** against the published recipe on a
frozen four-task OpenCode replay (interval coding, TTL-cache coding, release
planning, critical-path reasoning; pooled across two independent server starts
and six measured repetitions per cell, one stable output hash per cell across
both starts):

| Frozen OpenCode cell | Rate gain |
|---|---:|
| Interval coding | +21.8% |
| TTL-cache coding | +26.5% |
| Release prose/thinking | +35.8% |
| Critical-path thinking | +27.9% |

A second, independent reproduction against this repo's own published harness
(`bench/ctxsweep.py` at the pinned commit this repo was at, plus sparkDash
1.8.6) found +27.6% geometric-mean decode across the context ladder (512 to
259,584 tokens, every rung improved) and +20.4% geometric-mean aggregate
throughput across sparkDash's concurrency sweep (1/2/4/6/8 streams, code/
structured/prose).

**What changed:**
- The MTP draft head's proposal vocabulary reduced to an independently
  selected 65,536-row slice with local argmax — the *target* model still
  verifies every drafted token against its full head, so this narrows what
  the drafter searches, not what the target can output. Vocabulary was
  selected from vLLM's own source/docstrings, not from benchmark prompts.
- A deterministic top-k CUDA kernel extension for the sparse-attention path,
  verified against a 210-case standalone contract suite (kernel correctness
  only — not a model-quality or performance claim on its own).
- Recurrent (Mamba/GDN) state storage moved to BF16. This is the one change
  that measurably alters output: generation hashes differ from the
  unpatched baseline because of it, though every measured cell still produces
  one *stable* hash across repeated runs of the same input.
- The existing FP8 PLE mmap lookup staged into a single 10 MiB buffer
  *before* the forward pass, rather than inside it. This is what makes
  `FULL_DECODE_ONLY` CUDA graphs possible at all for this model: the mmap
  gather is a synchronous host/device operation, which is illegal inside a
  captured graph (see the PIECEWISE-only note in the Requirements table
  above) — staging it earlier removes it from the graphed region entirely.

**Why this is not the default yet:** the underlying patches are real and
checksummed on the box (a vendored CUDA kernel, a draft-vocabulary patch, a
PLE-staging overlay, each with its own Dockerfile and SHA256 manifest) but are
not yet packaged into this repository in a form a clone can rebuild — exactly
the bar every other number in this file is held to. The BF16 recurrent-state
change is a genuine, acknowledged precision change, not a lossless swap, and
a paired quality audit (MATH-500 plus AIME 2024, symbolic grading) was still
in progress as of this writing. Treat the numbers above as a strong,
well-instrumented result from one box, not yet a verified drop-in upgrade —
matching this project's own standard: "A real quality claim needs a task
benchmark, and none has been run" until one actually has.
