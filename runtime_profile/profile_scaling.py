#!/usr/bin/env python3
"""
profile_scaling.py — 类 3: fixed overhead vs O(N^2) compute

The central question left over from the mic-count experiment: dropping from
6 mics to 2 gave only a 1.6x end-to-end speedup where O(N^2) predicts 9x.
This script quantifies why, by fitting

    t(N) = c0 + c2 * N^2

to the measured per-call cost of the MVDR core at N = 2..6. `c0` is the
part that does not shrink when you remove microphones (Python dispatch, the
257-bin sweep, the isfinite/Hermitian guards, the per-mic smoothing loop);
`c2 * N^2` is the part that does.

The ratio c0 / t(6) is the answer to "what fraction of the cost can removing
microphones never touch" — and therefore the ceiling on that whole strategy.

Usage:
    python profile_scaling.py [--iters N] [--clip NAME]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (MIC_COUNTS, build_pipeline, load_clip, mic_positions_2d,
                    quiet, require_single_thread, save_json)


def fit_c0_c2(ns, ts):
    """
    Least-squares fit of t = c0 + c2*N^2.

    Returns (c0, c2, r2). Uses the closed-form 2-parameter linear fit on the
    design matrix [1, N^2]; no iterative solver needed.
    """
    ns = np.asarray(ns, dtype=float)
    ts = np.asarray(ts, dtype=float)
    A = np.stack([np.ones_like(ns), ns ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    c0, c2 = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((ts - pred) ** 2).sum())
    ss_tot = float(((ts - ts.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c0, c2, r2


def micro_scaling(iters):
    """Per-call cost of the MVDR core at each N, on synthetic frames."""
    from beamformer_2 import MVDRBeamformer
    from stft import F_WIN

    n_bins = F_WIN // 2 + 1
    rng = np.random.default_rng(0)
    out = {}

    for N in MIC_COUNTS:
        with quiet():
            bf = MVDRBeamformer(N, n_fft=F_WIN, alpha=0.97)
            bf.init_isotropic()

        Xk = (rng.standard_normal((N, n_bins))
              + 1j * rng.standard_normal((N, n_bins))).astype(np.complex64)
        d = np.ones((N, n_bins), dtype=np.complex64)

        with quiet():
            for _ in range(30):                  # warm-up
                bf.update_noise(Xk)
                bf.compute_weights(d)

            def micro(fn, *a):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn(*a)
                return (time.perf_counter() - t0) / iters * 1e6   # us

            out[N] = {
                "update_noise": micro(bf.update_noise, Xk),
                "compute_weights": micro(bf.compute_weights, d),
                "apply": micro(bf.apply, Xk),
            }
    return out


def _time_one(mic, gaze, vad, N, reps):
    sub = np.ascontiguousarray(mic[:N])
    mp = mic_positions_2d(N)
    ts = []
    for _ in range(reps):
        p = build_pipeline(mp)
        t0 = time.perf_counter()
        with quiet():
            p.process(sub, gaze=gaze, annotated_vad=vad, skip_denoise=True)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def end2end_scaling(mic, gaze, vad, reps):
    """
    Full pipeline.process() wall time at each N, for the shipping code and
    for the pre-v7.9 per-microphone smoothing loop.

    Both variants are measured in the SAME process, interleaved per N, so a
    drift in machine load cannot masquerade as a difference between them —
    which it could if the "before" numbers came from a separate earlier run.

    The old loop is restored by reusing bench_optimizations' patcher rather
    than duplicating it, so there is one definition of what "before" means.
    """
    from beamformer_2 import MVDRBeamformer
    from bench_optimizations import make_baseline_compute_weights

    old_loop = make_baseline_compute_weights(MVDRBeamformer)
    shipping = MVDRBeamformer.compute_weights

    after, before = {}, {}
    for N in MIC_COUNTS:
        # Interleaved: same N, both variants, back to back.
        MVDRBeamformer.compute_weights = shipping
        after[N] = _time_one(mic, gaze, vad, N, reps)
        MVDRBeamformer.compute_weights = old_loop
        try:
            before[N] = _time_one(mic, gaze, vad, N, reps)
        finally:
            MVDRBeamformer.compute_weights = shipping
    return after, before


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--clip", default=None)
    ap.add_argument("--iters", type=int, default=500,
                    help="microbenchmark iterations per N")
    ap.add_argument("--reps", type=int, default=3,
                    help="end-to-end repetitions per N")
    args = ap.parse_args()

    require_single_thread()
    mic, gaze, vad, name = load_clip(args.clip)
    print(f"clip : {name}   N sweep: {MIC_COUNTS}\n")

    print("[1/2] MVDR core microbenchmarks...")
    micro = micro_scaling(args.iters)

    fits = {}
    print(f"      {'op':<16}{'c0 (fixed)':>12}{'c2*36 (N=6)':>13}"
          f"{'fixed@N=6':>11}{'r2':>7}")
    print("      " + "-" * 59)
    for op in ("update_noise", "compute_weights", "apply"):
        ts = [micro[N][op] for N in MIC_COUNTS]
        c0, c2, r2 = fit_c0_c2(MIC_COUNTS, ts)
        t6 = micro[6][op]
        frac = c0 / t6 if t6 else float("nan")
        fits[op] = {"c0_us": c0, "c2_us": c2, "r2": r2,
                    "fixed_frac_at_6": frac}
        print(f"      {op:<16}{c0:>10.1f}us{c2*36:>11.1f}us"
              f"{100*frac:>10.1f}%{r2:>7.3f}")

    print("\n[2/2] End-to-end scaling (both variants, interleaved)...")
    e2e, e2e_before = end2end_scaling(mic, gaze, vad, args.reps)
    nmax = max(MIC_COUNTS)
    base_a, base_b = e2e[nmax], e2e_before[nmax]

    print(f"      {'N':>3}{'before':>9}{'after':>9}{'saved':>9}"
          f"{'spd(before)':>13}{'spd(after)':>12}")
    print("      " + "-" * 55)
    for N in MIC_COUNTS:
        a, b = e2e[N], e2e_before[N]
        print(f"      {N:>3}{b:>8.2f}s{a:>8.2f}s{b-a:>8.2f}s"
              f"{base_b/b:>12.2f}x{base_a/a:>11.2f}x")

    # The headline of the comparison: vectorising removed cost that itself
    # scaled with N, so the remaining head-room for dropping microphones is
    # smaller in both absolute and relative terms.
    head_b = e2e_before[nmax] - e2e_before[min(MIC_COUNTS)]
    head_a = e2e[nmax] - e2e[min(MIC_COUNTS)]
    print(f"\n      head-room for {nmax}→{min(MIC_COUNTS)} mics:"
          f"  before {head_b:.2f}s ({base_b/e2e_before[min(MIC_COUNTS)]:.2f}x)"
          f"   after {head_a:.2f}s ({base_a/e2e[min(MIC_COUNTS)]:.2f}x)")

    save_json("scaling_profile", {
        "clip": name, "mic_counts": MIC_COUNTS,
        "micro_us": micro, "fits": fits,
        "end2end_s": e2e, "end2end_before_s": e2e_before,
    })
    print("\ndone")


if __name__ == "__main__":
    main()
