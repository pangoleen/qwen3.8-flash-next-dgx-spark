#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent code/docstring vocabulary, supplemented by base BPE rank.

Uses installed vLLM source, NOT the benchmark prompts or outputs. This is
an initial generic-code vocabulary, not a model-output frequency estimate.
The target vocabulary is never changed. Run CPU-only before benchmarks.
"""
import argparse
import hashlib
import json
import pathlib
from collections import Counter
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--size', type=int, default=65536)
a = p.parse_args()
tokenizer_file = pathlib.Path(a.model) / 'tokenizer.json'
tok = Tokenizer.from_file(str(tokenizer_file))
data = json.loads(tokenizer_file.read_text())
added = {t['id'] for t in data.get('added_tokens', [])}
counts = Counter()
corpus_hash = hashlib.sha256()
files = sorted(pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm').rglob('*.py'))
for path in files:
    raw = path.read_bytes()
    corpus_hash.update(str(path).encode())
    corpus_hash.update(raw)
    counts.update(tok.encode(raw.decode(errors='replace'), add_special_tokens=False).ids)
    if len(counts) > tok.get_vocab_size():
        raise RuntimeError('Invalid tokenizer')
keep = set(added)
for tid, _ in counts.most_common():
    if len(keep) >= a.size:
        break
    keep.add(tid)
# A modest source corpus can have fewer distinct IDs than the budget. Fill
# unused slots in tokenizer/BPE order, preserving every added/special token.
for tid in sorted(tok.get_vocab().values()):
    if len(keep) >= a.size:
        break
    keep.add(tid)
assert len(keep) == a.size and added <= keep
out = pathlib.Path(a.output)
with out.open('x') as handle:
    handle.write(''.join(str(t) + '\n' for t in sorted(keep)))
report = {'strategy': 'vllm-source-frequency-then-low-token-id-fill', 'files': len(files),
          'corpus_sha256': corpus_hash.hexdigest(), 'tokenizer_sha256': hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
          'source_distinct_tokens': len(counts), 'source_tokens': sum(counts.values()),
          'selected_tokens': len(keep), 'added_tokens': len(added),
          'training_coverage': sum(counts[i] for i in keep) / sum(counts.values()),
          'vocab_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
with out.with_suffix('.json').open('x') as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report))
