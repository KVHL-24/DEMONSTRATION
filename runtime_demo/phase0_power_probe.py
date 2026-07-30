"""Phase 0 — power feasibility probe.

Question: can the VDD_CPU_SOC_MSS rail, minus idle baseline, tell apart
pipeline configurations running at a realistic 1x duty cycle?

Method: alternate measured phases —

    idle(20 s) → [ N=6 @1x (30 s) → idle → N=2 @1x (30 s) → idle
                   → N=6 flat-out (30 s) → idle ] × REPEATS → idle(20 s)

"1x duty cycle" means: process a 5 s chunk of audio, then sleep until 5 s
of wall clock has passed — the duty cycle a live demo paced at 1x
produces. The flat-out phase is the SWEEP-mode burst, included so we learn
whether the degraded plan (power only during full-speed sweeps) works even
if the 1x deltas drown in sensor noise.

Run this WHILE `thorprof daemon` is sampling.

Trace alignment: thorprof's CSV records a `phase` column per sample, but
only accepts the four values pre/run/post/idle, and markers do NOT persist
to the CSV. So this script sets phase="run" for the duration of each
workload block and phase="idle" between them; contiguous run-blocks in the
CSV are then matched IN ORDER against the workload sequence recorded in
phase0_workload_summary.json. Wall-clock timestamps are also printed as a
manual fallback.

No sudo needed here — the daemon (started by the user) owns the sensors.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        ../.venv/bin/python phase0_power_probe.py \
        [--daemon-url http://127.0.0.1:8080] [--token TOKEN]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import time
import urllib.request

# Pin the compute environment BEFORE numpy/torch import: the phase-0 rerun
# showed that mixing a GPU-STFT run with a CPU-only run inflates the
# apparent spread ~8x and flips the verdict. The live demo will run CPU-only
# single-thread, so measure exactly that.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent   # DEMONSTRATION/
sys.path.insert(0, str(ROOT))

from pipeline_2 import AriaDenoisingPipeline            # noqa: E402
from eval_synthetic_2 import SYNTH_MIC_POSITIONS_2D     # noqa: E402

CLIP = ROOT / "synthetic_dataset/cocktail_taz-001_iaz+101_snr-15_rep00"
CHUNK_S = 5.0          # audio seconds per processing burst
PHASE_S = 30.0         # duration of each workload phase
IDLE_EDGE_S = 20.0     # idle bracket at start/end (baseline)
IDLE_MID_S = 15.0      # idle gap between workload phases
REPEATS = 2

FS = 48000
HOP = 256


def load_chunk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, fs = sf.read(str(CLIP / "array_audio.wav"), dtype="float32")
    assert fs == FS
    audio = audio.T[:, : int(CHUNK_S * FS)]
    n_frames = audio.shape[1] // HOP + 1
    gaze = np.load(CLIP / "gaze.npy")[: n_frames + 64]
    vad = np.load(CLIP / "vad.npy").astype(bool)[:n_frames]
    return audio, gaze, vad


def make_pipeline(n_mic: int) -> AriaDenoisingPipeline:
    return AriaDenoisingPipeline(
        use_gaze=True, mic_pos=SYNTH_MIC_POSITIONS_2D[:n_mic], alpha=0.97,
        vad_thr_db=3.0, rt60_s=0.15, doa_reliable=False)


class DaemonCtl:
    """Best-effort phase/marker control of a running thorprof daemon."""

    def __init__(self, url: str | None, token: str):
        self.url, self.token, self.ok = url, token, False
        if url:
            try:
                self._post({"action": "marker", "label": "phase0_probe_hello"})
                self.ok = True
            except OSError as e:
                print(f"[daemon] not reachable at {url} ({e}) — "
                      "CSV phase column will stay as-is; use printed "
                      "wall-clock timestamps for alignment")

    def _post(self, obj: dict) -> None:
        if self.token:
            obj = {**obj, "token": self.token}
        req = urllib.request.Request(
            f"{self.url}/control", data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2) as r:
            resp = json.loads(r.read() or b"{}")
        if not resp.get("ok", True):
            print(f"[daemon] control refused: {resp}")

    def set_phase(self, phase: str) -> None:       # 'run' | 'idle'
        if self.ok:
            with contextlib.suppress(OSError):
                self._post({"action": "phase", "phase": phase})

    def mark(self, label: str) -> None:
        print(f"[{time.time():.3f}] === {label} ===", flush=True)
        if self.ok:
            with contextlib.suppress(OSError):
                self._post({"action": "marker", "label": label})


def run_block(pipe_factory, audio, gaze, vad, duration_s: float,
              pace_1x: bool) -> dict:
    """Process CHUNK_S-second chunks for duration_s; sleep out the chunk
    budget when pacing at 1x, back-to-back when flat-out."""
    t_start = time.monotonic()
    t_end = t_start + duration_s
    busy, chunks = 0.0, 0
    while time.monotonic() < t_end:
        pipe = pipe_factory()               # fresh state per chunk, like live
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            pipe.process(audio, gaze=gaze, annotated_vad=vad,
                         skip_denoise=True)
        dt = time.monotonic() - t0
        busy += dt
        chunks += 1
        if pace_1x and (leftover := CHUNK_S - dt) > 0:
            time.sleep(min(leftover, max(0.0, t_end - time.monotonic())))
    wall = time.monotonic() - t_start
    return {"chunks": chunks, "busy_s": round(busy, 3),
            "audio_s": chunks * CHUNK_S, "wall_s": round(wall, 3),
            "duty": round(busy / wall, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon-url", default="http://127.0.0.1:8080")
    ap.add_argument("--token", default="")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    audio, gaze, vad = load_chunk()
    ctl = DaemonCtl(args.daemon_url, args.token)
    summary: list[dict] = []

    # Warm-up outside any measured phase (first-touch alloc, BLAS init).
    with contextlib.redirect_stdout(io.StringIO()):
        make_pipeline(6).process(audio, gaze=gaze, annotated_vad=vad,
                                 skip_denoise=True)

    ctl.set_phase("idle")
    ctl.mark("idle_start")
    time.sleep(IDLE_EDGE_S)

    for rep in range(args.repeats):
        for label, n_mic, pace_1x in (
            (f"n6_1x_rep{rep}", 6, True),
            (f"n2_1x_rep{rep}", 2, True),
            (f"n6_flat_rep{rep}", 6, False),
        ):
            au = np.ascontiguousarray(audio[:n_mic])
            ctl.mark(label)
            ctl.set_phase("run")
            stats = run_block(lambda: make_pipeline(n_mic), au, gaze, vad,
                              PHASE_S, pace_1x)
            ctl.set_phase("idle")
            stats["label"] = label
            summary.append(stats)
            print(f"    {stats}", flush=True)
            ctl.mark(f"idle_after_{label}")
            time.sleep(IDLE_MID_S)

    ctl.mark("idle_end")
    time.sleep(IDLE_EDGE_S)
    ctl.mark("done")

    out = pathlib.Path(__file__).parent / "phase0_workload_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nworkload summary → {out}")
    print("Next: stop the daemon, then run "
          "phase0_analyze.py <daemon_trace.csv>")


if __name__ == "__main__":
    main()
