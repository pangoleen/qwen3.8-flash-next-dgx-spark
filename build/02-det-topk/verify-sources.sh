#!/bin/sh
set -eu

artifact_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$artifact_dir"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check SHA256SUMS
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 --check SHA256SUMS
else
  echo "need sha256sum or shasum to verify vendored sources" >&2
  exit 1
fi

python3 -c "import ast, pathlib; files=('vendor/kernel-det/build_det.py','vendor/kernel-det/test_det.py','vendor/determinism/qsadet_patch.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]; print('Python source parse: OK')"

echo "det-topk source bundle verified at e0ef69d4f5575dad00d34e05479eaf4c6547bace"

