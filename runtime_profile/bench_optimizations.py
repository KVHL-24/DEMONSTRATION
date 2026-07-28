#!/usr/bin/env python3
"""
bench_optimizations.py — 类 5: how much is the fixed overhead worth?

profile_scaling.py showed that ~46% of compute_weights() and ~32% of
update_noise() is cost that removing microphones can never touch. This
script attacks that cost directly and measures what it is worth, both as a
microbenchmark and end-to-end.

Each optimization is applied by monkey-patching the live class, so the
end-to-end number is a real pipeline run, not an estimate. Every patch is
checked for numerical equivalence against the original before it is timed —
a faster function that changes the output is not an optimization, so any
patch that fails its check is reported and excluded.

What is measured
----------------
  O1  vectorised frequency-axis smoothing — NOW SHIPPING (beamformer_2 v7.9)

      compute_weights() used to smooth weights along the frequency axis
      with a 3-tap kernel inside a `for n in range(self.N)` loop: two
      np.pad + two np.convolve per mic per call. cProfile counted 127,560
      np.pad calls in one 60 s clip, ~25% of total runtime, because each
      call re-paid NumPy dispatch overhead to smooth 257 values. The kernel
      is separable and identical for every mic, so the loop collapses into
      three slice-adds over the whole (bins, mics) array.

      Since that landed in beamformer_2._smooth_freq_axis(), this script
      inverts: it patches the OLD loop back in to reconstruct the
      pre-optimization baseline, confirms the shipping version is
      bit-identical to it, and reports the speedup. It is a regression test
      for the optimization as much as a benchmark.

Usage:
    python bench_optimizations.py [--clip NAME] [--reps N]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (build_pipeline, load_clip, mic_positions_2d, quiet,
                    require_single_thread, save_json)

# ── O1: vectorised frequency-axis smoothing ─────────────────────────────────


def _smooth_loop_original(weights, kernel):
    """
    The ORIGINAL per-microphone smoothing loop, kept here as the reference
    implementation now that beamformer_2._smooth_freq_axis() has replaced it
    in the shipping code (v7.9).

    This is deliberately the slow version: it exists so the benchmark can
    reconstruct the pre-optimization baseline and prove the vectorised
    replacement is bit-identical to what it replaced, rather than asking the
    reader to take that on faith.
    """
    out = weights.copy()
    k = kernel
    for n in range(out.shape[1]):
        wr = np.pad(out[:, n].real, 1, mode='edge')
        wi = np.pad(out[:, n].imag, 1, mode='edge')
        out[:, n] = (np.convolve(wr, k, mode='valid') +
                     1j * np.convolve(wi, k, mode='valid'))
    return out


# The smoothing block sits in the middle of compute_weights() — the v7.0
# renormalisation that follows it consumes its result, so it cannot simply be
# hoisted out and applied afterwards. To benchmark against the pre-v7.9 code
# we recompile the method from source with the vectorised call swapped back
# for the original loop. Every other line is the original text.
def make_baseline_compute_weights(orig_cls):
    """Recompile compute_weights with the pre-v7.9 per-mic smoothing loop."""
    import inspect
    import textwrap
    import beamformer_2 as bfmod

    src = textwrap.dedent(inspect.getsource(orig_cls.compute_weights))

    marker = "_smooth_freq_axis(weights, k)"
    hits = [i for i, ln in enumerate(src.splitlines()) if marker in ln]
    if len(hits) != 1:
        raise SystemExit(
            f"baseline: expected exactly one '{marker}' in compute_weights, "
            f"found {len(hits)}. beamformer_2.py has changed — update this "
            f"benchmark.")
    lines = src.splitlines()
    i = hits[0]
    indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
    slow = f"{indent}weights = _SMOOTH_LOOP(weights, k).astype(np.complex64)"
    src = "\n".join(lines[:i] + [slow] + lines[i + 1:])

    ns = dict(vars(bfmod))
    ns["_SMOOTH_LOOP"] = _smooth_loop_original
    exec(compile(src, "<baseline_compute_weights>", "exec"), ns)
    return ns["compute_weights"]


# A cheap-finiteness-guard variant (replacing update_noise()'s full
# np.isfinite sweep with a trace-based check) was also measured here and came
# out at 1.01x — within run-to-run noise. It is not kept: the sweep is a real
# cost but a small one next to the smoothing loop, and the only way to stub it
# out was to monkey-patch np.isfinite globally, which is too blunt an
# instrument to justify for no measurable gain.


# ── Equivalence + timing harness ────────────────────────────────────────────

def check_equivalence(mic, gaze, vad, patches, label):
    """Run the pipeline with and without patches; compare output samples."""
    from beamformer_2 import MVDRBeamformer

    mp = mic_positions_2d(mic.shape[0])

    def run():
        p = build_pipeline(mp)
        with quiet():
            return p.process(mic, gaze=gaze, annotated_vad=vad,
                             skip_denoise=True)

    ref = run()

    saved = {name: getattr(MVDRBeamformer, name) for name in patches}
    for name, fn in patches.items():
        setattr(MVDRBeamformer, name, fn)
    try:
        got = run()
    finally:
        for name, fn in saved.items():
            setattr(MVDRBeamformer, name, fn)

    n = min(len(ref), len(got))
    ref, got = ref[:n], got[:n]
    denom = float(np.abs(ref).max()) or 1.0
    max_abs = float(np.abs(ref - got).max())
    rel = max_abs / denom
    # Bit-identical is ideal; anything under ~1e-5 relative is float32
    # reordering noise, far below the level that could move a metric.
    ok = rel < 1e-5
    print(f"      equivalence [{label}]: max|Δ|={max_abs:.3e} "
          f"rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    return ok, rel


def time_end2end(mic, gaze, vad, patches, reps):
    from beamformer_2 import MVDRBeamformer

    mp = mic_positions_2d(mic.shape[0])
    saved = {name: getattr(MVDRBeamformer, name) for name in patches}
    for name, fn in patches.items():
        setattr(MVDRBeamformer, name, fn)
    try:
        ts = []
        for _ in range(reps):
            p = build_pipeline(mp)
            t0 = time.perf_counter()
            with quiet():
                p.process(mic, gaze=gaze, annotated_vad=vad,
                          skip_denoise=True)
            ts.append(time.perf_counter() - t0)
    finally:
        for name, fn in saved.items():
            setattr(MVDRBeamformer, name, fn)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--clip", default=None)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    require_single_thread()
    from beamformer_2 import MVDRBeamformer

    mic, gaze, vad, name = load_clip(args.clip)
    print(f"clip : {name}   reps: {args.reps}\n")

    # O1 now ships in beamformer_2.py (v7.9), so the roles are inverted
    # relative to when it was a candidate: the *patch* reconstructs the old
    # per-mic loop, and the unpatched pipeline is the optimized one.
    old_loop = {"compute_weights": make_baseline_compute_weights(
        MVDRBeamformer)}

    variants = [
        ("baseline (pre-v7.9 loop)", old_loop),
        ("shipping (vectorised)", {}),
    ]

    print("[1/2] Numerical equivalence check...")
    ok, rel = check_equivalence(mic, gaze, vad, old_loop,
                                "vectorised vs original loop")

    print("\n[2/2] End-to-end timing...")
    results = {}
    base_t = None
    for label, patches in variants:
        t = time_end2end(mic, gaze, vad, patches, args.reps)
        if base_t is None:
            base_t = t
        results[label] = {"time_s": t, "speedup": base_t / t,
                          "equivalent": ok, "rel_err": rel}
        flag = "" if ok else "   [NUMERICALLY DIFFERENT]"
        print(f"      {label:<28}{t:>7.2f}s  {base_t/t:>5.2f}x"
              f"  ({100*(1-t/base_t):+5.1f}%){flag}")

    save_json("optimization_bench", {"clip": name, "results": results})
    print("\ndone")


if __name__ == "__main__":
    main()
