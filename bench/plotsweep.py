#!/usr/bin/env python3
"""
plotsweep — chart a ctxsweep run.

Produces the six-panel grid the Apple-silicon benchmarks use (prompt tok/s,
generation tok/s, total time, TTFT/TPOT, and the two tokenizer-free throughput
panels), plus three panels that format does not have and this project's data
argues for:

  cold vs warm TTFT   the prefix cache made visible per rung. Ours goes from
                      1x at 258 tokens to 131x at 132k, and for agentic work
                      that ratio matters more than tok/s.
  accepted tokens     speculative decoding only pays when output is guessable;
  and saturation      plotting it explains the generation curve instead of
                      leaving it mysterious.

Usage:
    python3 plotsweep.py results/ctxsweep-qwen38-27b-v2-*.jsonl
    python3 plotsweep.py a.jsonl b.jsonl --out compare.png   # overlay runs
"""
import argparse, glob, json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["prompt_tokens"])
    return rows


def label_of(path):
    stem = pathlib.Path(path).stem
    return stem.replace("ctxsweep-", "").rsplit("-", 2)[0]


def fmt_ctx(n):
    return f"{n/1000:.0f}k" if n >= 1000 else str(n)


def annotate(ax, xs, ys, fmt="{:.0f}"):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Qwen3.8-27B NVFP4 + DFlash2 · one DGX Spark (GB10)")
    ap.add_argument("--subtitle", default="SGLang, draft 16, KV bf16, mem 0.65, tuned tactic cache")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) or [f])
    runs = [(label_of(p), load(p)) for p in paths]
    runs = [(n, r) for n, r in runs if r]
    if not runs:
        sys.exit("no rows found")

    fig, axes = plt.subplots(3, 3, figsize=(19, 14))
    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.955, args.subtitle, ha="center", fontsize=10, color="#555")
    single = len(runs) == 1

    panels = [
        ("Prompt processing  [tok/s]",      "prefill_tok_s",  "{:.0f}",  "#1f9be0"),
        ("Generation  [tok/s]",             "gen_tok_s",      "{:.1f}",  "#22c1c3"),
        ("Cold TTFT  [s]  — log scale",     "cold_ttft_s",    "{:.1f}",  "#e0245e"),
        ("Warm TTFT (prefix cache)  [s]",   "warm_ttft_s",    "{:.2f}",  "#2ca02c"),
        ("Time per output token  [ms]",     "tpot_ms",        "{:.0f}",  "#9467bd"),
        ("Accepted tokens per pass",        "tok_per_pass",   "{:.1f}",  "#ff7f0e"),
        ("Prompt throughput  [KB/s]",       "prefill_bytes_s","{:.0f}",  "#4c72b0"),
        ("Generation throughput  [B/s]",    "gen_bytes_s",    "{:.0f}",  "#55a868"),
        ("Cold ÷ warm TTFT  [×]",           None,             "{:.0f}",  "#d62728"),
    ]

    for ax, (title, field, fmt, colour) in zip(axes.flat, panels):
        for i, (name, rows) in enumerate(runs):
            xs = [r["prompt_tokens"] for r in rows]
            if field is None:
                ys = [(r["cold_ttft_s"] / r["warm_ttft_s"]) if r.get("warm_ttft_s") else None
                      for r in rows]
            else:
                ys = [r.get(field) for r in rows]
            if field == "prefill_bytes_s":
                ys = [y / 1000 if y else None for y in ys]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.plot(px, py, marker="o", markersize=4, linewidth=1.8,
                    color=colour if single else None, label=name)
            if single:
                annotate(ax, px, py, fmt)
        ax.set_xscale("log", base=2)
        if title.startswith("Cold TTFT"):
            ax.set_yscale("log")
        elif field in ("gen_tok_s", "prefill_tok_s", "tpot_ms", "tok_per_pass",
                       "gen_bytes_s", "prefill_bytes_s"):
            # Rates start at zero: an autoscaled 37-44 window turns a flat decode
            # line into a zigzag (2026-09-05 Flash ladder). Keep 8 % headroom.
            top = max(y for _, rows in runs for y in
                      ([(r.get(field) or 0) / (1000 if field == "prefill_bytes_s" else 1) for r in rows]))
            ax.set_ylim(0, top * 1.08)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("prompt tokens", fontsize=8)
        ticks = [r["prompt_tokens"] for r in runs[0][1]]
        ax.set_xticks(ticks); ax.set_xticklabels([fmt_ctx(t) for t in ticks], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if not single:
            ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    out = args.out or f"results/ctxsweep-{runs[0][0]}.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
