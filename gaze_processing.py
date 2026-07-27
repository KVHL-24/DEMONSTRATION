"""
gaze_processing.py — Gaze-signal robustness  (v1.1)
====================================================

Motivation
----------
`doa_2.DOA_Gaze` already protects against one specific failure mode:
large, discrete saccades (5–25°, ~3 Hz) — it holds the previous committed
direction for a few frames so the beamformer's null doesn't chase the eye
mid-saccade. What it does *not* protect against is smaller, continuous
frame-to-frame jitter in the raw gaze signal itself (eye-tracker /
vergence-depth noise, occasional low-confidence or degenerate samples)
that never crosses the saccade threshold but still perturbs the steering
vector every single frame. The eval sweep's SACCADE COST and GAZE VALUE
tables show oracle_gaze losing meaningfully to oracle_target_dir / SRP in
several scenarios even outside the `_dynamic` (moving-source) subset —
consistent with this kind of continuous steering noise, not just discrete
saccades.

`GazeStabilizer` sits *upstream* of `DOA_Gaze`/steering-vector computation
and addresses that: instead of trusting each incoming gaze sample
verbatim, it keeps a short rolling window of recent samples and reports a
recency- and confidence-weighted average, with a light-touch outlier gate
that down-weights (rather than discards outright) any single sample that
deviates sharply from the rest of the window — e.g. a blink, a vergence
glitch, or a momentary tracking failure. Real, sustained eye motion is
never suppressed: a genuine saccade quickly dominates the window as
several consecutive samples agree with each other and disagree with the
stale average, and outright large jumps are still handled downstream by
DOA_Gaze's saccade hold. This class only smooths noise that the saccade
detector is deliberately too coarse to catch.

Averaging is done in unit-vector space (2-D for scalar azimuth angles, 3-D
for full gaze vectors), never by directly averaging angles, to avoid the
usual atan2 wrap-around problem when directions cross ±180°.

This is the "take a window, don't just trust the raw sample" mechanism
requested. As of doa_2.py v7.1 it is wired directly into
`DOA_Gaze.steering_vector_from_gaze()` — enabled by default via
`use_gaze_stabilizer=True` in `DOA_Gaze.__init__` — operating on the raw
3-D gaze vector once per STFT frame, *before* DOA_Gaze's saccade-hold
logic ever sees it, exactly the ordering this module was designed for.
If your pipeline instead parses/resamples gaze upstream of DOA_Gaze (e.g.
a separate `_parse_gaze()` step producing scalar azimuth angles), you can
still use this class standalone there — the interface (`update()` taking
either representation) is identical either way.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from stft import FS, HOP

# ── Defaults ────────────────────────────────────────────────────────────────
GAZE_WINDOW_SECONDS       = 0.25   # ~3 saccade cycles; preserves smooth pursuit
GAZE_MIN_WINDOW_FRAMES    = 5      # frames before the outlier gate activates
GAZE_OUTLIER_THRESH_DEG   = 20.0   # angular deviation from the window's robust
                                   # mean beyond which a sample is down-weighted
GAZE_OUTLIER_DOWNWEIGHT   = 0.15   # multiply confidence by this for outliers
                                   # (down-weighted, not dropped — a real fast
                                   # saccade should still be able to pull the
                                   # window if several frames in a row agree)
GAZE_RECENCY_MIN_WEIGHT   = 0.2    # oldest sample in the window gets this
                                   # weight, newest always gets 1.0


class GazeStabilizer:
    """
    Rolling, confidence- and recency-weighted gaze smoother with a light
    outlier gate, operating directly on the per-STFT-frame gaze stream
    already used by AriaDenoisingPipeline (scalar azimuth angles OR 3-D
    unit gaze vectors — whichever `_parse_gaze()` decided the input was).

    Parameters
    ----------
    fs                : audio sample rate (for converting window_seconds
                         to a frame count at the STFT frame rate).
    hop                : STFT hop size in samples.
    window_seconds     : rolling window length in seconds (default 0.25 s
                          ≈ 47 frames at 187.5 fps — covers a few saccade
                          cycles while still tracking smooth pursuit).
    min_window_frames   : minimum window length in frames regardless of
                          window_seconds (safety floor).
    outlier_thresh_deg  : angular deviation (degrees) from the window's
                          current robust-weighted mean beyond which an
                          incoming sample is treated as suspect.
    outlier_downweight  : confidence multiplier applied to suspect samples
                          (not dropped outright — see module docstring).
    recency_min_weight  : weight assigned to the oldest sample in the
                          window; weight ramps linearly to 1.0 for the
                          newest sample.
    """

    def __init__(self,
                 fs:                 int   = FS,
                 hop:                int   = HOP,
                 window_seconds:     float = GAZE_WINDOW_SECONDS,
                 min_window_frames:  int   = GAZE_MIN_WINDOW_FRAMES,
                 outlier_thresh_deg: float = GAZE_OUTLIER_THRESH_DEG,
                 outlier_downweight: float = GAZE_OUTLIER_DOWNWEIGHT,
                 recency_min_weight: float = GAZE_RECENCY_MIN_WEIGHT):
        frame_rate = float(fs) / float(hop)
        self.window_frames = max(min_window_frames,
                                 int(round(window_seconds * frame_rate)))
        self.min_window_frames  = int(min_window_frames)
        self.outlier_thresh_deg = float(outlier_thresh_deg)
        self.outlier_downweight = float(outlier_downweight)
        self.recency_min_weight = float(recency_min_weight)

        # Each entry: (unit_vector, confidence) — unit_vector is (2,) for
        # scalar-angle input, (3,) for full gaze-vector input.
        self._history: deque = deque(maxlen=self.window_frames)
        self._is_vec: Optional[bool] = None   # locked in on first update()

        # Diagnostics
        self._n_updates:  int = 0
        self._n_outliers: int = 0

    # ── Encoding helpers ────────────────────────────────────────────────────

    @staticmethod
    def _angle_to_unit(theta: float) -> np.ndarray:
        # Matches the (sin θ, cos θ) = (X, Z) convention used throughout
        # doa_2.py's 2-D steering-vector path.
        return np.array([np.sin(theta), np.cos(theta)], dtype=np.float64)

    @staticmethod
    def _unit_to_angle(v: np.ndarray) -> float:
        return float(np.arctan2(v[0], v[1]))

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, gaze_entry, is_vec: bool,
              confidence: float = 1.0):
        """
        Feed one raw gaze sample for the current STFT frame and get back the
        stabilised value in the SAME representation (scalar angle if
        is_vec=False, else a (3,) unit vector) so it's a drop-in replacement
        at the call site.

        Parameters
        ----------
        gaze_entry : float azimuth (radians) if is_vec=False,
                     else a (3,) array-like unit gaze vector.
        is_vec     : representation flag, as produced by pipeline_2._parse_gaze.
        confidence : optional external confidence in [0, 1] for this sample
                     (defaults to 1.0 — trust it, subject to the outlier gate).

        Returns
        -------
        Smoothed gaze entry in the same representation as the input.
        """
        self._n_updates += 1
        if self._is_vec is None:
            self._is_vec = is_vec

        if is_vec:
            v = np.asarray(gaze_entry, dtype=np.float64).ravel()
            n = np.linalg.norm(v)
            v = v / n if n > 1e-9 else v
        else:
            v = self._angle_to_unit(float(gaze_entry))

        conf = float(np.clip(confidence, 0.0, 1.0))

        # ── Outlier gate: compare against the window's CURRENT weighted
        # mean (i.e. before this sample is added) — down-weight, don't drop,
        # so a real fast saccade can still pull the window if it persists.
        if len(self._history) >= self.min_window_frames:
            prev_mean = self._weighted_mean()
            pn = np.linalg.norm(prev_mean)
            if pn > 1e-9:
                cos_sim = float(np.dot(v, prev_mean) / pn) / max(np.linalg.norm(v), 1e-9)
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                dev_deg = float(np.degrees(np.arccos(cos_sim)))
                if dev_deg > self.outlier_thresh_deg:
                    conf *= self.outlier_downweight
                    self._n_outliers += 1

        self._history.append((v, conf))

        avg = self._weighted_mean()
        an  = np.linalg.norm(avg)
        if an > 1e-9:
            avg = avg / an
        else:
            avg = v   # degenerate window (e.g. all-zero confidence) → passthrough

        if is_vec:
            return avg.astype(np.float32)
        return self._unit_to_angle(avg)

    def _weighted_mean(self) -> np.ndarray:
        n = len(self._history)
        vectors     = np.stack([h[0] for h in self._history])       # (n, d)
        confidences = np.array([h[1] for h in self._history])       # (n,)
        recency     = np.linspace(self.recency_min_weight, 1.0, n)  # (n,)
        weights     = recency * confidences
        wsum        = weights.sum()
        if wsum < 1e-12:
            weights = recency
        return np.average(vectors, axis=0, weights=weights)

    # ── Reset / diagnostics ────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear rolling window and diagnostics. Call between clips."""
        self._history.clear()
        self._is_vec = None
        self._n_updates  = 0
        self._n_outliers = 0

    def diagnostics(self) -> dict:
        return {
            'gaze_stab_updates':      self._n_updates,
            'gaze_stab_outliers':     self._n_outliers,
            'gaze_stab_outlier_rate': self._n_outliers / max(self._n_updates, 1),
            'gaze_stab_window_frames': self.window_frames,
        }

    def print_diagnostics(self, prefix: str = '') -> None:
        d   = self.diagnostics()
        tag = f'[GazeStab {prefix}] ' if prefix else '[GazeStab] '
        print(f'{tag}updates={d["gaze_stab_updates"]}  '
              f'outliers={d["gaze_stab_outliers"]} '
              f'({100*d["gaze_stab_outlier_rate"]:.1f}%)  '
              f'window={d["gaze_stab_window_frames"]} frames')


# ── Optional: raw-sensor-level smoother (Aria SDK EyeGaze objects) ──────────
#
# GazeStabilizer above operates on the already-extracted per-STFT-frame gaze
# stream used by AriaDenoisingPipeline (this repo's actual data path — e.g.
# gaze.npy in the synthetic eval sets). If you also want windowed smoothing
# further upstream, directly on live Aria `mps.EyeGaze` samples (before
# they're ever resampled to the STFT frame rate), the class below is kept
# for that use case. It mirrors GazeStabilizer's weighting scheme but reads
# Aria's native yaw/pitch/vergence fields instead of pre-computed vectors.
# It is NOT used by pipeline_2.py — wire it in wherever gaze.npy is produced
# from a raw Aria recording, if you want smoothing that early.

class StableEyeGazeEstimator:
    """
    Stable gaze estimator for Meta Aria glasses (raw-sensor-level).

    - Estimates 3-D gaze vector in CPF (Central Pupil Frame).
    - Estimates depth from vergence when available; falls back to a
      configurable default depth otherwise.
    - Smooths over a rolling time window to suppress saccade transients,
      using the same recency+confidence weighting as GazeStabilizer.

    Parameters
    ----------
    window_seconds : rolling window for temporal smoothing (seconds).
                     0.25 s covers ~3 saccade cycles while preserving
                     smooth-pursuit tracking.
    default_depth  : fallback depth when vergence is unavailable (metres).
    max_depth      : depth clip ceiling (metres). Vergence is unreliable
                     beyond ~6 m; 10 m is a safe hard limit.
    min_depth      : depth clip floor (metres). Sub-15 cm is inside the
                     near-field singularity for typical IPDs.
    """

    def __init__(
        self,
        window_seconds: float = 0.25,
        default_depth:  float = 1.0,
        max_depth:      float = 10.0,
        min_depth:      float = 0.15,
    ):
        self.window_seconds = window_seconds
        self.default_depth  = default_depth
        self.max_depth      = max_depth
        self.min_depth      = min_depth
        self.history: deque = deque()   # (timestamp_s, vector_3d, depth_m, confidence)

    def yaw_pitch_to_vector(self, yaw: float, pitch: float) -> np.ndarray:
        """
        Convert CPF yaw / pitch (radians) to a unit gaze vector.

        Aria CPF convention: +X right, +Y down (pitch > 0 = looking down),
        +Z forward.
        """
        x = np.cos(pitch) * np.sin(yaw)
        y = np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        v = np.array([x, y, z], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def estimate_depth(self, gaze) -> tuple[float, Optional[float], Optional[float]]:
        """Estimate depth + combined (vergence) yaw/pitch; falls back safely."""
        import mps  # local import: only required if this raw-sensor path is used

        pitch_raw = getattr(gaze, "pitch_rads_cpf", getattr(gaze, "pitch", 0.0))
        try:
            depth, combined_yaw, combined_pitch = (
                mps.compute_depth_and_combined_gaze_direction(
                    gaze.vergence.left_yaw,
                    gaze.vergence.right_yaw,
                    pitch_raw,
                )
            )
            depth = float(depth)
            if not np.isfinite(depth):
                return self.default_depth, None, None
            clipped = float(np.clip(depth, self.min_depth, self.max_depth))
            return clipped, float(combined_yaw), float(combined_pitch)
        except Exception:
            return self.default_depth, None, None

    def update(self, gaze) -> Optional[dict]:
        """Ingest one Aria gaze sample and return the current stable estimate."""
        timestamp = float(gaze.tracking_timestamp.total_seconds())
        depth, combined_yaw, combined_pitch = self.estimate_depth(gaze)

        if combined_yaw is not None:
            vector = self.yaw_pitch_to_vector(combined_yaw, combined_pitch)
        else:
            yaw   = getattr(gaze, "yaw_rads_cpf",   getattr(gaze, "yaw",   0.0))
            pitch = getattr(gaze, "pitch_rads_cpf", getattr(gaze, "pitch", 0.0))
            vector = self.yaw_pitch_to_vector(yaw, pitch)

        confidence = float(np.clip(getattr(gaze, "confidence", 1.0), 0.0, 1.0))
        self.history.append((timestamp, vector, depth, confidence))

        while (self.history
               and timestamp - self.history[0][0] > self.window_seconds):
            self.history.popleft()

        return self.get_stable_gaze()

    def get_stable_gaze(self) -> Optional[dict]:
        """Temporally smoothed gaze estimate over the current window."""
        if not self.history:
            return None

        n = len(self.history)
        vectors     = np.array([e[1] for e in self.history], dtype=np.float64)
        depths      = np.array([e[2] for e in self.history], dtype=np.float64)
        confidences = np.array([e[3] for e in self.history], dtype=np.float64)

        recency = np.linspace(0.2, 1.0, n)
        weights = recency * confidences
        if weights.sum() < 1e-12:
            weights = recency

        avg_vector = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(avg_vector)
        avg_vector = avg_vector / norm if norm > 1e-9 else vectors[-1]

        avg_depth = float(np.average(depths, weights=weights))
        point_3d  = avg_vector * avg_depth

        return {
            "gaze_vector_cpf": avg_vector,
            "depth_m":         avg_depth,
            "point_cpf":       point_3d,
        }