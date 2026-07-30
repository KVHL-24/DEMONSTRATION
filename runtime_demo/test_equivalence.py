"""Phase 1 merge gate — equivalence + observer overhead.

Two claims are enforced, mirroring the bench_optimizations.py house rule:

1. **Bit-identical**: with every new knob at its default (weight stride 1,
   bypass gate off, observer None) the pipeline output must equal the
   pre-change output SAMPLE FOR SAMPLE (max |Δ| == 0.0).
2. **Observer overhead < 3%**: attaching a recording observer must not
   slow the pipeline by more than 3%.

Usage:
    ../.venv/bin/python test_equivalence.py --capture   # BEFORE editing (HEAD)
    ../.venv/bin/python test_equivalence.py             # after editing

`--capture` stores baseline outputs to baseline_outputs/. The plain run
recomputes with knob defaults and compares, then times observer on/off.
Three clips are used: an easy static one, a hard dynamic one, and a
white-noise one — different code paths (VAD mix, saccade activity).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_2 import AriaDenoisingPipeline            # noqa: E402
from eval_synthetic_2 import SYNTH_MIC_POSITIONS_2D     # noqa: E402

CLIPS = [
    "cocktail_taz-001_iaz+101_snr-15_rep00",
    "cocktail_dynamic_taz+030_iaz-048_snr+10_rep00",
    "white_noise_taz+035_iaz+000_snr+0_rep00",
]
SECONDS = 10.0
FS, HOP = 48000, 256
BASE_DIR = pathlib.Path(__file__).parent / "baseline_outputs"


def load(clip_name: str):
    clip = ROOT / "synthetic_dataset" / clip_name
    audio, fs = sf.read(str(clip / "array_audio.wav"), dtype="float32")
    audio = np.ascontiguousarray(audio.T[:, : int(SECONDS * FS)])
    n_frames = audio.shape[1] // HOP + 1
    gaze = np.load(clip / "gaze.npy")[: n_frames + 64]
    vad = np.load(clip / "vad.npy").astype(bool)[:n_frames]
    return audio, gaze, vad


def run_pipeline(audio, gaze, vad, **kw):
    pipe = AriaDenoisingPipeline(
        use_gaze=True, mic_pos=SYNTH_MIC_POSITIONS_2D, alpha=0.97,
        vad_thr_db=3.0, rt60_s=0.15, doa_reliable=False, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        out = pipe.process(audio, gaze=gaze, annotated_vad=vad,
                          skip_denoise=True)
    return np.asarray(out)


def capture() -> None:
    BASE_DIR.mkdir(exist_ok=True)
    for name in CLIPS:
        out = run_pipeline(*load(name))
        np.save(BASE_DIR / f"{name}.npy", out)
        print(f"captured {name}: {out.shape} rms={np.sqrt((out**2).mean()):.6f}")
    print(f"\nbaselines → {BASE_DIR}")


def verify() -> int:
    failed = False

    # ── 1. bit-identical with knobs at defaults ───────────────────────────
    for name in CLIPS:
        ref = np.load(BASE_DIR / f"{name}.npy")
        out = run_pipeline(*load(name))
        d = float(np.abs(out - ref).max()) if out.shape == ref.shape else np.inf
        ok = d == 0.0
        failed |= not ok
        print(f"[bit-identical] {name}: max|Δ|={d}  {'OK' if ok else 'FAIL'}")

    # ── 2. observer overhead ──────────────────────────────────────────────
    # A recording observer, the worst realistic case: keeps every frame's
    # payload alive (the demo's probe layer will subsample, not keep all).
    audio, gaze, vad = load(CLIPS[0])
    rec = []

    def observer(frame_idx: int, data: dict) -> None:
        rec.append((frame_idx, data))

    def best_of(n, **kw):
        ts = []
        for _ in range(n):
            rec.clear()
            t0 = time.perf_counter()
            run_pipeline(audio, gaze, vad, **kw)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    best_of(1)                                    # warm-up
    t_plain = best_of(3)
    try:
        t_obs = best_of(3, observer=observer)
    except TypeError:
        print("[observer] pipeline does not accept observer= yet — SKIP")
        return 1
    ovh = (t_obs - t_plain) / t_plain * 100
    ok = ovh < 3.0
    failed |= not ok
    print(f"[observer] plain={t_plain:.3f}s  observed={t_obs:.3f}s  "
          f"overhead={ovh:+.1f}%  frames_recorded={len(rec)}  "
          f"{'OK' if ok else 'FAIL'}")

    print("\n" + ("EQUIVALENCE GATE: FAIL" if failed else "EQUIVALENCE GATE: PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()
    if args.capture:
        capture()
    else:
        sys.exit(verify())
