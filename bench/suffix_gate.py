#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Growing-suffix correctness gate: the shape that actually failed on 2026-09-07.

cache_gate.py exercises identical repeats and alternating whole prompts. Neither
is what an agent produces, and neither is what failed. The 09-07 report was a
shared prefix of ~100k tokens extended by ~40k NEW tokens, returning completions
that ignored the content. That is the agent shape: a fixed system/context
prefix, a conversation growing on the end, so every turn is a partial cache hit
whose boundary moves.

The test builds one long prefix, then appends numbered blocks to it turn after
turn. Each turn plants a fresh marker in the newest block and asks only about
that marker. A stale-state failure shows up as an answer naming an older marker,
or a generic answer naming none.

Two conversations are interleaved so the cache has to evict and restore, which
a single growing conversation would never force.

Exit 0 only if every turn answers with its own current marker.
"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18300/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
PREFIX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
GROW_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
TURNS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
CONTEXT_LIMIT = int(os.environ.get("CTX_LIMIT", "262144"))


def _key():
    for var in ("OPENAI_API_KEY", "SPARK_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = os.environ.get("SPARK_API_KEY_FILE")
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return ""


HDR = {"Content-Type": "application/json", "Authorization": "Bearer " + _key()}


def ntok(text):
    req = urllib.request.Request(
        BASE.rsplit("/v1", 1)[0] + "/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(), headers=HDR)
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return len(d.get("tokens") or d.get("input_ids") or [])


def chat(prompt, max_tokens=120):
    body = {"model": MODEL, "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return (d["choices"][0]["message"]["content"] or "").strip(), time.perf_counter() - t0


FILLER = ("The scheduler log records a queue depth, a batch identifier and a "
          "wall-clock stamp for each admitted request. ")


def grow_to(text, target):
    """Extend text with filler until the server's tokenizer says it is long enough."""
    while ntok(text) < target:
        short = max(1, (target - ntok(text)) // 12)
        text += FILLER * min(short, 4000)
    return text


print(f"building a shared ~{PREFIX_TOKENS}-token prefix ...", flush=True)
PREFIX = grow_to("[SHARED CONTEXT]\n" + FILLER, PREFIX_TOKENS)
print(f"  prefix = {ntok(PREFIX)} tokens", flush=True)

# Two conversations over the SAME prefix, so they compete for the same blocks.
convos = {
    "X": {"body": PREFIX, "marker": None},
    "Y": {"body": PREFIX, "marker": None},
}
fails = []
rows = []

stop = False
for turn in range(1, TURNS + 1):
    if stop:
        break
    for name in ("X", "Y"):
        marker = f"{name}{turn}-{7919 * turn + ord(name)}"
        block = (f"\n\n[BLOCK {name} {turn}] The current verification code in this "
                 f"conversation is {marker}. Ignore any earlier codes; they are "
                 f"superseded. ")
        c = convos[name]
        c["body"] = grow_to(c["body"] + block + FILLER,
                            ntok(c["body"]) + GROW_TOKENS)
        c["marker"] = marker
        q = (c["body"] + f"\n\n---\nWhat is the current verification code for "
             f"conversation {name}? Reply with exactly that code and nothing else.")
        n = ntok(q)
        if n > CONTEXT_LIMIT - 512:
            print(f"  stopping: turn {turn} {name} would need {n} tokens, "
                  f"over the {CONTEXT_LIMIT} window", flush=True)
            stop = True
            break
        out, dt = chat(q)
        hit = marker.lower() in out.lower()
        rows.append((turn, name, n, dt, marker, out[:60], hit))
        print(f"  turn {turn} {name}  {n:>7} tok  {dt:7.2f}s  "
              f"want={marker:<12} got={out[:40]!r}  {'ok' if hit else 'FAIL'}",
              flush=True)
        if not hit:
            fails.append(f"turn {turn} {name}: expected {marker}, got {out[:120]!r}")

print()
print("=" * 68)
warm = [r[3] for r in rows[2:]]
if warm:
    print(f"turns after the first pair: median {sorted(warm)[len(warm)//2]:.2f}s "
          f"(min {min(warm):.2f}s, max {max(warm):.2f}s)")
if fails:
    print("SUFFIX GATE FAILED")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"SUFFIX GATE PASSED - {len(rows)} growing-suffix turns, every one answered "
      f"with its own current marker")
sys.exit(0)
