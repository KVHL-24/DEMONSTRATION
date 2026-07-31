"""Dual-pipeline live engine.

Streams one clip through TWO independently-configured pipelines frame by
frame, paced to 1x wall-clock (the glasses' real duty cycle — see PLAN.md
phase-0 result for why the duty numbers only mean something at 1x), and
publishes ~23 Hz telemetry packets to a Hub for the SSE server.

Stage 1 (STFT) is computed once per clip up front: it is 8% of runtime,
identical for every configuration, and precomputing it lets the per-frame
loop time exactly the stages the knobs actually affect (DOA + beamformer).
The displayed duty cycle therefore covers stages 2–3 only, and says so.
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stft import stft_multichannel, istft, make_window   # noqa: E402
from doa_2 import gaze_vector_to_theta                   # noqa: E402
from configs import PipeConfig, build_pipeline           # noqa: E402
from probes import BeamPatternProbe, spec_column         # noqa: E402

FS       = 48000
N_FFT    = 512
HOP      = 256
FRAME_S  = HOP / FS                    # 5.333 ms
PKT_FRAMES = 8                         # ~23 Hz telemetry
SISDR_EVERY_PKTS = 23                  # refresh live SI-SDR ~1 Hz
SISDR_WIN_FRAMES = 375                 # 2 s sliding window


def si_sdr_db(est: np.ndarray, ref: np.ndarray) -> 'float | None':
    """Scale-invariant SDR of est against ref (display metric)."""
    n = min(len(est), len(ref))
    if n < FS // 4:
        return None
    e, r = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    rr = float(np.dot(r, r))
    if rr < 1e-12:
        return None
    a = float(np.dot(e, r)) / rr
    err = e - a * r
    num = a * a * rr
    den = float(np.dot(err, err)) + 1e-12
    return float(10.0 * np.log10(num / den + 1e-12))

DATASET = ROOT / "synthetic_dataset"


# ── Clip loading ─────────────────────────────────────────────────────────────

def list_clips() -> list[dict]:
    out = []
    for p in sorted(DATASET.iterdir()):
        meta_p = p / "metadata.json"
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text())
        out.append({
            "name": p.name,
            "scenario": m.get("scenario", "?"),
            "snr_db": m.get("snr_db"),
            "sep_deg": round(m.get("target_interferer_sep_deg", 0) or 0, 1),
            "dynamic": bool(m.get("is_dynamic")),
            "duration_s": m.get("duration_s"),
        })
    return out


class Clip:
    def __init__(self, name: str):
        p = DATASET / name
        if not p.is_dir():
            raise FileNotFoundError(name)
        self.name = name
        self.meta = json.loads((p / "metadata.json").read_text())
        audio, fs = sf.read(str(p / "array_audio.wav"), dtype="float32")
        assert fs == FS
        self.audio = np.ascontiguousarray(audio.T)             # (6, T)
        self.ref, _ = sf.read(str(p / "reverberant_reference.wav"),
                              dtype="float32")                  # (T,)
        self.window = make_window(N_FFT)
        self.X = stft_multichannel(self.audio, window=self.window,
                                   n_fft=N_FFT, hop=HOP)        # (6, B, K)
        self.n_frames = self.X.shape[2]
        gaze = np.load(p / "gaze.npy")
        if len(gaze) < self.n_frames:
            gaze = np.pad(gaze, ((0, self.n_frames - len(gaze)), (0, 0)),
                          mode="edge")
        self.gaze = gaze[: self.n_frames]                       # (K, 3)
        vad = np.load(p / "vad.npy").astype(bool)
        if len(vad) < self.n_frames:
            vad = np.pad(vad, (0, self.n_frames - len(vad)), mode="edge")
        self.vad = vad[: self.n_frames]

    @property
    def target_az(self) -> float:
        return float(self.meta.get("target_azimuth_deg", 0.0))

    @property
    def interf_az(self) -> 'float | None':
        v = self.meta.get("interferer_azimuth_deg")
        return None if v is None else float(v)


# ── One configured side (A or B) ─────────────────────────────────────────────

class Side:
    def __init__(self, label: str, cfg: PipeConfig):
        self.label = label
        self.cfg = cfg
        self.pipe, self.tap = build_pipeline(cfg)
        self.probe = BeamPatternProbe(cfg.n_mics)
        self.Y: list[np.ndarray] = []      # produced spectra, for listening
        self.busy_s = 0.0                  # total stage-2/3 compute time
        self.busy_recent: list[float] = [] # per-frame times, rolling window
        self.gated_frames = 0
        self.si_sdr: 'float | None' = None       # sliding-window, ~1 Hz
        self.si_sdr_delta: 'float | None' = None  # vs raw mic 0, same window
        # operator-set azimuth, radians; only read when steering == 'manual'.
        # Set as an attribute (not a process() parameter) so subclasses that
        # override process() with the original signature keep working.
        self.manual_az = 0.0
        # cumulative (whole clip so far) — the number that settles and lets
        # A/B be compared at a glance; built incrementally, see _extend_out()
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._out_frames = 0
        # engine thread (SI-SDR refresh) and HTTP threads (audio download)
        # both extend/read the buffer
        self._out_lock = threading.Lock()
        self.si_sdr_cum: 'float | None' = None
        self.si_sdr_cum_delta: 'float | None' = None

    def process(self, Xk: np.ndarray, gaze_vec: np.ndarray,
                speech: bool) -> None:
        x = Xk[: self.cfg.n_mics]
        if self.cfg.steering == "gaze":
            g = gaze_vec
        elif self.cfg.steering == "manual":
            g = float(self.manual_az)          # scalar azimuth, radians
        else:                                  # srp
            g = None
        t0 = time.perf_counter()
        y = self.pipe.process_frame(x, gaze=g, speech_override=speech)
        dt = time.perf_counter() - t0
        self.busy_s += dt
        self.busy_recent.append(dt)
        if len(self.busy_recent) > 187:            # ~1 s window
            self.busy_recent.pop(0)
        if self.tap.last is not None and self.tap.last.get("gated"):
            self.gated_frames += 1
        self.Y.append(y)

    def duty(self) -> float:
        if not self.busy_recent:
            return 0.0
        return float(sum(self.busy_recent) / (len(self.busy_recent) * FRAME_S))

    def telemetry(self) -> dict:
        tap = self.tap.last or {}
        gated = bool(tap.get("gated", False))
        w = None if gated else tap.get("w")
        theta = tap.get("theta")
        return {
            "cfg": self.cfg.to_dict(),
            "short": self.cfg.short(),
            "beam": self.probe.pattern_db(w),
            "theta_deg": (None if theta is None
                          else round(float(np.degrees(theta)), 1)),
            "gated": gated,
            "mics": tap.get("mics"),
            "gate_frac": round(self.gated_frames / max(1, len(self.Y)), 3),
            "duty": round(self.duty(), 4),
            "stage_us": {
                "doa": round(float(tap.get("t_doa_us", 0.0)), 1),
                "bf":  round(float(tap.get("t_bf_us", 0.0)), 1),
            },
            "si_sdr": None if self.si_sdr is None else round(self.si_sdr, 1),
            "si_sdr_delta": (None if self.si_sdr_delta is None
                             else round(self.si_sdr_delta, 1)),
            "si_sdr_cum": (None if self.si_sdr_cum is None
                           else round(self.si_sdr_cum, 2)),
            "si_sdr_cum_delta": (None if self.si_sdr_cum_delta is None
                                 else round(self.si_sdr_cum_delta, 2)),
        }

    def _extend_out(self, clip: 'Clip') -> None:
        """Incrementally extend the time-domain output buffer to the current
        frame. ISTFT is run on [prev−1 .. n) with one context frame so the
        overlap-add seam is interior-exact (win = 2·hop → every sample is
        covered by exactly two frames); only the fully-covered new samples
        are appended. Display-only, so a sub-sample seam imperfection at the
        very first chunk is irrelevant."""
        with self._out_lock:
            n = len(self.Y)
            if n <= self._out_frames:
                return
            a = max(0, self._out_frames - 1)
            Yc = np.stack(self.Y[a:n], axis=1)
            chunk = istft(Yc, window=clip.window, n_fft=N_FFT, hop=HOP)
            skip = (self._out_frames - a) * HOP
            keep = (n - self._out_frames) * HOP
            new = chunk[skip: skip + keep]
            self._out_buf = np.concatenate([self._out_buf, new])
            self._out_frames = n

    def refresh_si_sdr(self, clip: 'Clip') -> None:
        """Sliding-window + cumulative SI-SDR vs the reverberant reference
        (display only, runs outside the timed sections)."""
        n = len(self.Y)
        if n < SISDR_WIN_FRAMES:
            return
        a = n - SISDR_WIN_FRAMES
        Yw = np.stack(self.Y[a:n], axis=1)                  # (B, W)
        out = istft(Yw, window=clip.window, n_fft=N_FFT, hop=HOP)
        s0, s1 = a * HOP, a * HOP + len(out)
        ref = clip.ref[s0:s1]
        raw = clip.audio[0][s0:s1]
        self.si_sdr = si_sdr_db(out, ref)
        base = si_sdr_db(raw, ref)
        self.si_sdr_delta = (None if self.si_sdr is None or base is None
                             else self.si_sdr - base)

        # cumulative over everything played so far — settles with time
        self._extend_out(clip)
        m = len(self._out_buf)
        if m >= FS:
            ref_c = clip.ref[:m]
            raw_c = clip.audio[0][:m]
            self.si_sdr_cum = si_sdr_db(self._out_buf, ref_c)
            base_c = si_sdr_db(raw_c, ref_c)
            self.si_sdr_cum_delta = (
                None if self.si_sdr_cum is None or base_c is None
                else self.si_sdr_cum - base_c)

    def audio_out(self, clip: 'Clip',
                  upto_frame: 'int | None' = None) -> np.ndarray:
        """Time-domain output up to upto_frame, served from the incremental
        buffer. Only the frames produced since the last call are ISTFT'd —
        the old full-history recompute made every listen request cost 1–2 s
        by the end of a clip (twice, counting the seek's Range request)."""
        self._extend_out(clip)
        with self._out_lock:
            n = (self._out_frames if upto_frame is None
                 else min(upto_frame, self._out_frames))
            return self._out_buf[: n * HOP].copy()


# ── Engine ───────────────────────────────────────────────────────────────────

class Engine:
    """Owns the playback thread; thread-safe control via a lock + flags."""

    def __init__(self, hub):
        self.hub = hub
        self._lock = threading.RLock()
        self._clip: Clip | None = None
        self._cfgs = {"A": PipeConfig(steering="gaze"),
                      "B": PipeConfig(steering="srp")}
        self._sides: dict[str, Side] | None = None
        self._frame = 0
        self._playing = False
        self._speed = 1.0                  # 1.0 = real time; 0 = flat out
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._generation = 0               # bumped on every rebuild
        # operator azimuths (radians) — engine-level so they survive
        # rebuilds/restarts: your aim is not part of the causal clip state
        self._manual_az = {"A": 0.0, "B": 0.0}

    # ── control (called from HTTP threads) ────────────────────────────────

    def load_clip(self, name: str) -> dict:
        with self._lock:
            self._playing = False
            self._clip = Clip(name)
            self._rebuild()
            return self.status()

    def set_config(self, side: str, cfg_dict: dict) -> dict:
        with self._lock:
            if side not in ("A", "B"):
                raise ValueError("side must be A or B")
            self._cfgs[side] = PipeConfig.from_dict(cfg_dict)
            self._rebuild()                 # restart: state is causal
            return self.status()

    def play(self) -> dict:
        with self._lock:
            if self._clip is None:
                raise RuntimeError("no clip loaded")
            if self._sides is None or self._frame >= self._clip.n_frames:
                self._rebuild()
            self._playing = True
            self._ensure_thread()
            return self.status()

    def pause(self) -> dict:
        with self._lock:
            self._playing = False
            return self.status()

    def restart(self) -> dict:
        with self._lock:
            self._rebuild()
            return self.status()

    def set_speed(self, speed: float) -> dict:
        with self._lock:
            self._speed = max(0.0, float(speed))
            return self.status()

    def set_manual_az(self, side: str, az_deg: float) -> dict:
        """Live steering input — deliberately NO rebuild/restart: direction
        is per-frame data (like real gaze), not configuration."""
        with self._lock:
            if side not in ("A", "B"):
                raise ValueError("side must be A or B")
            az = float(az_deg)
            if not -180.0 <= az <= 180.0:
                raise ValueError("az_deg out of range")
            self._manual_az[side] = float(np.deg2rad(az))
            if self._sides is not None and side in self._sides:
                self._sides[side].manual_az = self._manual_az[side]
            return self.status()

    def status(self) -> dict:
        clip = self._clip
        return {
            "type": "status",
            "clip": None if clip is None else {
                "name": clip.name,
                "n_frames": clip.n_frames,
                "duration_s": clip.n_frames * FRAME_S,
                "target_az": clip.target_az,
                "interf_az": clip.interf_az,
                "snr_db": clip.meta.get("snr_db"),
                "scenario": clip.meta.get("scenario"),
                "sep_deg": clip.meta.get("target_interferer_sep_deg"),
            },
            "configs": {s: c.to_dict() for s, c in self._cfgs.items()},
            "shorts": {s: c.short() for s, c in self._cfgs.items()},
            "playing": self._playing,
            "frame": self._frame,
            "speed": self._speed,
            "manual_az_deg": {s: round(float(np.degrees(v)), 1)
                              for s, v in self._manual_az.items()},
        }

    def audio_wav(self, which: str) -> bytes:
        """WAV bytes of input mic-0 / side A / side B, up to current frame."""
        import io as _io
        with self._lock:
            clip, sides, upto = self._clip, self._sides, self._frame
        if clip is None:
            raise RuntimeError("no clip loaded")
        if which == "in":
            x = clip.audio[0][: upto * HOP]
        else:
            if sides is None or which not in sides:
                raise RuntimeError(f"no side {which}")
            x = sides[which].audio_out(clip, upto)
        buf = _io.BytesIO()
        sf.write(buf, np.asarray(x, dtype=np.float32), FS, format="WAV")
        return buf.getvalue()

    # ── internals ─────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._sides = {s: Side(s, c) for s, c in self._cfgs.items()}
        for s, side in self._sides.items():
            side.manual_az = self._manual_az[s]
        self._frame = 0
        self._generation += 1
        self.hub.broadcast(self.status())

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="demo-engine")
            self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                playing = self._playing
                clip, sides = self._clip, self._sides
                k, gen = self._frame, self._generation
            if not playing or clip is None or sides is None:
                time.sleep(0.05)
                continue
            if k >= clip.n_frames:
                with self._lock:
                    self._playing = False
                self.hub.broadcast(self.status())
                continue

            t_pkt0 = time.perf_counter()
            end = min(k + PKT_FRAMES, clip.n_frames)
            for i in range(k, end):
                Xk = clip.X[:, :, i]
                for side in sides.values():
                    side.process(Xk, clip.gaze[i], bool(clip.vad[i]))

            with self._lock:
                # A config change mid-packet rebuilt the sides; drop this
                # packet's bookkeeping, the loop re-reads fresh state next.
                if gen != self._generation:
                    continue
                self._frame = end

            # ~1 Hz: refresh the sliding-window SI-SDR (display-only work,
            # deliberately outside the per-frame timed sections)
            if (end // PKT_FRAMES) % SISDR_EVERY_PKTS == 0:
                for side in sides.values():
                    side.refresh_si_sdr(clip)

            pkt = {
                "type": "telemetry",
                "frame": end,
                "t": round(end * FRAME_S, 3),
                "shared": {
                    "vad": bool(clip.vad[end - 1]),
                    "in_spec": spec_column(clip.X[:, :, end - 1]),
                    # where the RECORDED eyes point right now — the ghost
                    # ray / hand-vs-eyes reference for manual steering
                    "gaze_az_deg": round(float(np.degrees(
                        gaze_vector_to_theta(clip.gaze[end - 1], None))), 1),
                },
                "sides": {s: side.telemetry()
                          for s, side in sides.items()},
            }
            self.hub.broadcast(pkt)

            # ── pacing ────────────────────────────────────────────────────
            if self._speed > 0:
                budget = (end - k) * FRAME_S / self._speed
                leftover = budget - (time.perf_counter() - t_pkt0)
                if leftover > 0:
                    time.sleep(leftover)
