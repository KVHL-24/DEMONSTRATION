"""
common.py — shared helpers for the runtime profiling suite.

All profiling scripts import from here so clip loading, timing methodology,
and output paths stay consistent across them.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

# Default clip. white_noise @ 0 dB is a middle-of-the-road case: the VAD
# behaves, the beamformer neither diverges nor sits in DAS fallback, so the
# timing reflects the normal code path rather than an error branch.
DEFAULT_CLIP = "white_noise_taz+035_iaz+000_snr+0_rep00"

MIC_COUNTS = [2, 3, 4, 5, 6]


def require_single_thread() -> None:
    """
    Timing is only comparable if BLAS threading is pinned. numpy reads these
    at import time, so this must run before numpy is imported by anything
    that matters — the runner scripts set them in the shell too, and this is
    a defensive check rather than the primary mechanism.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(var) != "1":
            print(f"  [WARN] {var} is not 1 — timings may be noisy. "
                  f"Run via the provided shell wrapper.", file=sys.stderr)
            break


@contextlib.contextmanager
def quiet():
    """Suppress the pipeline's very chatty stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def load_clip(clip: str | None = None, dataset: str = "synthetic_dataset"):
    """
    Load one clip's array audio, gaze, and VAD.

    Returns (mic (N,T) float32, gaze (F,3) float32, vad (F,) bool, name).
    """
    import soundfile as sf

    name = clip or DEFAULT_CLIP
    root = PROJECT_ROOT / dataset / name
    if not root.is_dir():
        raise SystemExit(
            f"Clip not found: {root}\n"
            f"Generate the dataset first (see README.md), or pass --clip.")

    mic, sr = sf.read(str(root / "array_audio.wav"),
                      dtype="float32", always_2d=True)
    gaze = np.load(str(root / "gaze.npy"))
    vad = np.load(str(root / "vad.npy")).astype(bool)
    return mic.T, gaze, vad, name


def mic_positions_2d(n: int | None = None) -> np.ndarray:
    """Array geometry in the (x, z) plane, optionally truncated to n mics."""
    from generate_synthetic_dataset import MIC_POSITIONS
    mp = MIC_POSITIONS[:, [0, 2]].astype(np.float32)
    return mp if n is None else mp[:n]


def build_pipeline(mic_pos: np.ndarray, **overrides):
    """Construct an AriaDenoisingPipeline with the standard eval settings."""
    from pipeline_2 import AriaDenoisingPipeline
    kwargs = dict(use_gaze=True, mic_pos=mic_pos, alpha=0.97,
                  vad_thr_db=3.0, rt60_s=0.15, doa_reliable=False)
    kwargs.update(overrides)
    with quiet():
        return AriaDenoisingPipeline(**kwargs)


def timed(fn, *args, reps: int = 3, **kwargs) -> tuple[float, list[float]]:
    """
    Run fn(*args) `reps` times, return (median, all_times).

    Median rather than min: we want the typical cost, not the best case a
    warm cache can produce.
    """
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with quiet():
            fn(*args, **kwargs)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2], ts


def save_json(name: str, payload: dict) -> Path:
    """Write payload to outputs/<name>.json and return the path."""
    path = OUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  → {path.relative_to(PROJECT_ROOT)}")
    return path


def load_json(name: str) -> dict:
    path = OUT_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the corresponding profile "
                         f"script first.")
    with open(path) as f:
        return json.load(f)
