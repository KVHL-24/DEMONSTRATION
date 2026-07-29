#!/usr/bin/env bash
# Run the full profiling suite and render the figures.
#
# BLAS threading is pinned and CUDA disabled so timings are comparable
# between runs — see README.md for why the GPU is not used here.
set -euo pipefail

cd "$(dirname "$0")"
PY=../.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "error: $PY not found — run 'uv sync' in the project root first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== 1/4  pipeline stages, internals, hotspots ==="
$PY profile_pipeline.py "$@"

echo
echo "=== 2/4  fixed overhead vs O(N^2) ==="
$PY profile_scaling.py "$@"

echo
echo "=== 3/4  optimization A/B ==="
$PY bench_optimizations.py "$@"

echo
echo "=== 4/4  figures ==="
$PY plot_results.py

echo
echo "Outputs in runtime_profile/outputs/"
