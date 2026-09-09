#!/usr/bin/env python3
"""Correctness gate for prefix caching on the hybrid Mamba model.

The 2026-09-07 failure was: with caching on, vLLM forces Mamba 'align' mode,
and an identical prompt repeated at temperature 0 sometimes returned a
completion that ignored the prompt entirely. Decode speed is irrelevant if
that still happens, so this runs BEFORE any timing work.

Three checks, hardest last:

  A repeat      same prompt N times. Every completion must be identical.
  B interleave  A B A B A - an agent returning to an earlier prefix. This is
                where stale recurrent state shows up and a plain repeat loop
                does not.
  C on-topic    each completion must contain the marker planted in its own
                prompt, so we catch "fluent but about the wrong thing".

Exit code 0 only if every check passes.
"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18300/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
def _key():
    """OPENAI_API_KEY, else SPARK_API_KEY_FILE, else no key (server may not need one)."""
    for var in ("OPENAI_API_KEY", "SPARK_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = os.environ.get("SPARK_API_KEY_FILE")
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return ""


KEY = _key()
HDR = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def ntok(text):
    req = urllib.request.Request(BASE.rsplit("/v1", 1)[0] + "/tokenize",
                                 data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers=HDR)
    d = json.load(urllib.request.urlopen(req, timeout=120))
    return len(d.get("tokens") or d.get("input_ids") or [])


def chat(prompt, max_tokens=160):
    body = {"model": MODEL, "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d["choices"][0]["message"]["content"] or "", time.perf_counter() - t0


def build(marker):
    """A long, distinctive prompt carrying its own verifiable marker."""
    filler = ("The deployment log records routine scheduler activity. "
              "Each entry lists a queue depth and a batch identifier. ")
    body = filler
    while ntok(body) < TARGET:
        body += filler * 200
    return (f"[SESSION {marker}]\n" + body +
            f"\n\n---\nThe session identifier above is {marker}. "
            f"Reply with exactly one line: 'session {marker} acknowledged'. "
            "Do not add anything else.")


print(f"building two ~{TARGET}-token prompts ...", flush=True)
pa, pb = build("ALPHA7"), build("BRAVO3")
print(f"  A={ntok(pa)} tokens   B={ntok(pb)} tokens", flush=True)

fails = []

print("\n[A] repeat: same prompt 5x at temperature 0", flush=True)
outs = []
for i in range(5):
    t, dt = chat(pa)
    outs.append(t.strip())
    print(f"   {i}  {dt:6.2f}s  {t.strip()[:70]!r}", flush=True)
if len(set(outs)) != 1:
    fails.append(f"A: {len(set(outs))} distinct completions for an identical prompt")

print("\n[B] interleave: A B A B A (agent returning to an earlier prefix)", flush=True)
seq = []
for i, (p, m) in enumerate([(pa, "ALPHA7"), (pb, "BRAVO3")] * 2 + [(pa, "ALPHA7")]):
    t, dt = chat(p)
    seq.append((m, t.strip()))
    print(f"   {i} {m}  {dt:6.2f}s  {t.strip()[:70]!r}", flush=True)
for m, t in seq:
    if m.lower() not in t.lower():
        fails.append(f"B: completion for {m} does not mention {m}: {t[:90]!r}")
a_outs = {t for m, t in seq if m == "ALPHA7"}
b_outs = {t for m, t in seq if m == "BRAVO3"}
if len(a_outs) != 1:
    fails.append(f"B: ALPHA7 gave {len(a_outs)} different answers across interleaving")
if len(b_outs) != 1:
    fails.append(f"B: BRAVO3 gave {len(b_outs)} different answers across interleaving")

print("\n[C] on-topic: marker present in the repeat outputs", flush=True)
if "alpha7" not in outs[0].lower():
    fails.append(f"C: repeat completion never mentions ALPHA7: {outs[0][:90]!r}")

print("\n" + "=" * 60)
if fails:
    print("GATE FAILED")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("GATE PASSED - caching did not corrupt output on any check")
sys.exit(0)
