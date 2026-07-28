#!/usr/bin/env python3
"""
profile_pipeline.py — 类 1 / 2 / 4: where does the time actually go?

  Class 1  Pipeline stages      — STFT / DOA+steering / beamformer
  Class 2  Beamformer internals — update_noise / compute_weights / apply
  Class 4  Hotspot ranking      — cProfile top-N by tottime and cumtime

Stage timings are measured by re-running each stage in isolation (wall
clock, no profiler attached) rather than by reading cProfile numbers —
cProfile's per-call overhead is large relative to these functions and would
distort the split. cProfile is used only for the hotspot *ranking*, where
relative order is what matters.

Usage:
    python profile_pipeline.py [--clip NAME] [--reps N]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from common import (PROJECT_ROOT, build_pipeline, load_clip, mic_positions_2d,
                    quiet, require_single_thread, save_json)


def profile_stages(mic, gaze, vad, reps):
    """Class 1 + 2: isolate each stage and time it on the wall clock."""
    from stft import make_window, stft_multichannel, istft, F_WIN, HOP

    win = make_window(F_WIN)
    mic_pos = mic_positions_2d(mic.shape[0])

    # ── Stage 1: STFT ───────────────────────────────────────────────────
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        X = stft_multichannel(mic, window=win, n_fft=F_WIN, hop=HOP)
        ts.append(time.perf_counter() - t0)
    t_stft = float(np.median(ts))
    n_frames = X.shape[2]
    n_bins = X.shape[1]

    # ── Stage 2: DOA + steering (per frame) ─────────────────────────────
    ga = gaze[:n_frames]
    ts = []
    for _ in range(reps):
        p = build_pipeline(mic_pos)
        t0 = time.perf_counter()
        with quiet():
            for k in range(n_frames):
                p._get_steering(X[:, :, k], ga[k], True, bool(vad[k]))
        ts.append(time.perf_counter() - t0)
    t_doa = float(np.median(ts))

    # Cache steering vectors so the beamformer stage does not re-pay for DOA.
    p = build_pipeline(mic_pos)
    with quiet():
        steer = [p._get_steering(X[:, :, k], ga[k], True, bool(vad[k]))
                 for k in range(n_frames)]

    # ── Stage 3: beamformer (per frame) ─────────────────────────────────
    ts = []
    for _ in range(reps):
        p2 = build_pipeline(mic_pos)
        t0 = time.perf_counter()
        with quiet():
            for k in range(n_frames):
                d, th, ph = steer[k]
                p2.beamformer.process_frame(X[:, :, k], d,
                                            is_noise=not bool(vad[k]),
                                            doa_reliable=False,
                                            theta=th, phi=ph)
        ts.append(time.perf_counter() - t0)
    t_bf = float(np.median(ts))

    # ── Stage 4: ISTFT ──────────────────────────────────────────────────
    Y = X[0]
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        istft(Y, window=win, n_fft=F_WIN, hop=HOP, length=mic.shape[1])
        ts.append(time.perf_counter() - t0)
    t_istft = float(np.median(ts))

    # ── Class 2: beamformer internals ───────────────────────────────────
    # Timed as microbenchmarks on a warmed-up beamformer: the per-frame
    # cost of each entry point, times the number of frames that actually
    # reach it in a real run (weights are not recomputed on every frame).
    bf = build_pipeline(mic_pos).beamformer
    with quiet():
        bf.init_isotropic()
    Xk = X[:, :, n_frames // 2]
    d0 = steer[n_frames // 2][0]

    with quiet():
        for _ in range(30):                     # warm-up
            bf.update_noise(Xk)
            bf.compute_weights(d0)

    def micro(fn, *a, iters=400):
        with quiet():
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(*a)
            return (time.perf_counter() - t0) / iters

    per_update = micro(bf.update_noise, Xk)
    per_weights = micro(bf.compute_weights, d0)
    per_apply = micro(bf.apply, Xk)

    n_noise = int((~vad[:n_frames]).sum())
    n_speech = n_frames - n_noise

    return {
        "n_frames": n_frames,
        "n_bins": n_bins,
        "n_mics": int(mic.shape[0]),
        "n_noise_frames": n_noise,
        "n_speech_frames": n_speech,
        "stages": {
            "STFT": t_stft,
            "DOA+steering": t_doa,
            "beamformer": t_bf,
            "ISTFT": t_istft,
        },
        "beamformer_internals_per_call_us": {
            "update_noise": per_update * 1e6,
            "compute_weights": per_weights * 1e6,
            "apply": per_apply * 1e6,
        },
        # Extrapolated share of the beamformer stage. update_noise runs on
        # noise frames, apply on every frame; compute_weights runs once per
        # frame that changes the weights, which in practice is ~every frame.
        "beamformer_internals_total_s": {
            "update_noise": per_update * n_noise,
            "compute_weights": per_weights * n_frames,
            "apply": per_apply * n_frames,
        },
    }


def profile_hotspots(mic, gaze, vad, top_n=25):
    """Class 4: cProfile ranking of the full end-to-end run."""
    mic_pos = mic_positions_2d(mic.shape[0])
    p = build_pipeline(mic_pos)

    pr = cProfile.Profile()
    with quiet():
        pr.enable()
        p.process(mic, gaze=gaze, annotated_vad=vad, skip_denoise=True)
        pr.disable()

    st = pstats.Stats(pr, stream=io.StringIO())
    rows = []
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        fname, lineno, funcname = func
        # site-packages lives *under* PROJECT_ROOT (.venv/), so test for it
        # first — otherwise every numpy frame is misattributed to the project.
        is_lib = "site-packages" in fname or fname.startswith("<")
        short = fname
        if "site-packages/" in short:
            short = short.split("site-packages/")[-1]
        elif str(PROJECT_ROOT) in short:
            short = short.split(str(PROJECT_ROOT))[-1].lstrip("/")
        rows.append({
            "func": funcname,
            "file": short,
            "line": lineno,
            "ncalls": nc,
            "tottime": tt,
            "cumtime": ct,
            "is_project": (not is_lib) and str(PROJECT_ROOT) in fname,
        })
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return {"total_s": st.total_tt, "rows": rows[:top_n]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--clip", default=None)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    require_single_thread()
    mic, gaze, vad, name = load_clip(args.clip)
    print(f"clip   : {name}")
    print(f"audio  : {mic.shape}   frames: {len(vad)}   reps: {args.reps}\n")

    print("[1/2] Stage + beamformer-internal timings...")
    stages = profile_stages(mic, gaze, vad, args.reps)
    tot = sum(stages["stages"].values())
    for k, v in stages["stages"].items():
        print(f"      {k:<16} {v:7.3f}s  ({100*v/tot:5.1f}%)")

    print("\n[2/2] cProfile hotspot ranking...")
    hot = profile_hotspots(mic, gaze, vad)
    print(f"      profiled total: {hot['total_s']:.2f}s "
          f"(includes profiler overhead)")
    for r in hot["rows"][:8]:
        tag = "*" if r["is_project"] else " "
        print(f"      {tag} {r['tottime']:6.3f}s  {r['ncalls']:>8,}x  "
              f"{r['func']}")

    save_json("pipeline_profile", {
        "clip": name, "reps": args.reps, **stages, "hotspots": hot,
    })
    print("\ndone")


if __name__ == "__main__":
    main()
