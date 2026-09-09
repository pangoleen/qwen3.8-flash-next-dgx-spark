# Correctness + determinism tests for _C_det.persistent_topk (v2, after review).
#   argv[1] = path to _C_det .so
# Reference: exact top-k = value desc, ties by index asc; the row returned sorted ascending.
# Adversarial: all-equal rows (result must be [0..k)), boundary tie populations around every
# buffer size the original kernels used (k-1, k, k+1, 2048, 2049, 3708, 3709, 4096, 16384, 16385),
# 100 repeats on the all-equal case, shapes forcing every path (rows 1 -> persistent single-CTA
# up to 32768 then multi-CTA; rows 64 -> filtered kernel).
import sys, itertools, torch
torch.ops.load_library(sys.argv[1])
import vllm  # noqa: F401  (loads the stable-ABI ops incl. _C.persistent_topk)
dev = "cuda"
def ref_topk(logits, lengths, k):
    rows, cols = logits.shape; out = torch.full((rows, k), -1, dtype=torch.int32, device=dev)
    col = torch.arange(cols, device=dev)
    for r in range(rows):
        n = int(lengths[r]); x = logits[r, :n]
        if n <= k: out[r, :n] = col[:n]; continue
        order = torch.argsort(x, descending=True, stable=True)[:k]
        out[r] = torch.sort(order.to(torch.int32)).values
    return out
def run(op, logits, lengths, k, ws):
    out = torch.empty((logits.shape[0], k), dtype=torch.int32, device=dev)
    op(logits, lengths, out, ws, k, logits.shape[1]); torch.cuda.synchronize(); return out
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
det = torch.ops._C_det.persistent_topk; stock = torch.ops._C.persistent_topk
fails = 0
def check(name, logits, lengths, k, repeats=6, expect=None):
    global fails
    r = ref_topk(logits, lengths, k) if expect is None else expect
    outs = [run(det, logits, lengths, k, ws) for _ in range(repeats)]
    same = all(torch.equal(o, outs[0]) for o in outs); exact = torch.equal(outs[0], r)
    st = [run(stock, logits, lengths, k, ws) for _ in range(3)]
    st_same = all(torch.equal(o, st[0]) for o in st); st_set = torch.equal(torch.sort(st[0], dim=1).values, torch.sort(r, dim=1).values)
    tag = "OK  " if (same and exact) else "FAIL"; fails += (tag == "FAIL")
    print(f"{tag} {name}: det identical x{repeats}={same} exact={exact} | stock identical x3={st_same} stock set==ref={st_set}", flush=True)
    if tag == "FAIL":
        bad = (outs[0] != r).nonzero()[:3].tolist(); print("      first diffs (row,pos):", bad, "det:", outs[0].flatten()[:6].tolist(), "ref:", r.flatten()[:6].tolist())
# 1. random and tie-heavy across paths
for rows, cols, k, kind in itertools.product((1, 8, 64), (1024, 4096, 8192, 20000, 40000), (512, 1024, 2048), ("rand", "ties")):
    if k >= cols: continue
    g = torch.Generator(device=dev).manual_seed(rows * 7 + cols + k)
    logits = torch.randn(rows, cols, generator=g, device=dev) if kind == "rand" else torch.randint(0, 5, (rows, cols), generator=g, device=dev).float()
    lengths = torch.full((rows,), cols, dtype=torch.int32, device=dev); lengths[0] = cols - 3
    check(f"rows={rows:2d} cols={cols:5d} k={k:4d} {kind}", logits, lengths, k)
# 1b. rows SHORTER than or equal to k (v2.4 regression): the main build's block-level indexer calls
#     TopK 512 with 256-block rows at warm-up; the old host guard rejected every such call.
for rows, cols, k in itertools.product((1, 8, 64), (256, 512, 700, 1024, 2048), (512, 1024, 2048)):
    if cols > k: continue
    g = torch.Generator(device=dev).manual_seed(rows * 11 + cols + k)
    logits = torch.randn(rows, cols, generator=g, device=dev)
    lengths = torch.full((rows,), cols, dtype=torch.int32, device=dev); lengths[0] = max(cols - 3, 1)
    check(f"short rows={rows:2d} cols={cols:5d} k={k:4d}", logits, lengths, k)
# 2. all values equal -> must be exactly [0..k) every time (100 repeats), every path
for rows, cols in ((1, 8192), (1, 20000), (1, 40000), (64, 8192), (64, 40000)):
    for k in (512, 1024, 2048):
        logits = torch.zeros((rows, cols), device=dev); lengths = torch.full((rows,), cols, dtype=torch.int32, device=dev)
        exp = torch.arange(k, dtype=torch.int32, device=dev).expand(rows, k).contiguous()
        check(f"ALL-EQUAL rows={rows:2d} cols={cols:5d} k={k}", logits, lengths, k, repeats=100, expect=exp)
# 3. boundary tie populations at the pivot: m elements share the pivot value, k - m//2 above it
for rows, cols in ((1, 8192), (1, 40000), (64, 20000)):
    for k in (512, 1024, 2048):
        for m in (k - 1, k, k + 1, 2048, 2049, 3708, 3709, 4096, 16384, 16385):
            if m + 8 > cols: continue
            g = torch.Generator(device=dev).manual_seed(m + cols)
            logits = torch.full((rows, cols), -1.0, device=dev)
            above = max(k - m // 2, 0); above = min(above, k - 1)      # k-above ties needed from the pivot group
            perm = torch.randperm(cols, generator=g, device=dev)
            logits[:, perm[:above]] = 2.0                                # strictly above the pivot
            logits[:, perm[above:above + m]] = 1.0                       # exactly the pivot, m of them
            lengths = torch.full((rows,), cols, dtype=torch.int32, device=dev)
            check(f"PIVOT-TIES rows={rows:2d} cols={cols:5d} k={k} above={above} ties={m}", logits, lengths, k, repeats=20)
print("FAILS:", fails)
