"""SWEEP mode — serial full-speed config sweep on one sample.

Runs a representative subset of the config space over the current clip,
each config as a full-speed burst (phase-0 result: burst energy is
measurable to ~4% within a session; 1x-duty energy is not), measuring:

    ΔSI-SDR   vs raw mic 0, against the reverberant reference
    RTF       stages 2–3 compute seconds per audio second
    mJ/s      VDD_CPU_SOC_MSS rail energy above idle baseline,
              normalized per audio second (the house rule)

Power is sampled directly from the INA sysfs nodes (via
thor_profile/sensors.py) in a background thread — no daemon needed.
A/B-comparable because every config runs back-to-back in one session.

Results are cached per (clip, config-set, seconds) in sweep_cache/.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT.parent / "thor_profile")):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs import PipeConfig, build_pipeline           # noqa: E402
from engine import Clip, si_sdr_db, HOP, FS              # noqa: E402

CACHE_DIR = HERE / "sweep_cache"
SWEEP_SECONDS = 30.0        # per-config audio length (keeps 10 configs < 90 s)
IDLE_PRE_S    = 4.0
IDLE_GAP_S    = 2.0

# The representative subset. (chao-v2: the weight_stride/gate runtime
# knobs were removed with the re-alignment to main's algorithm code, so
# the sweep now spans steering mode and mic count only.)
SWEEP_CONFIGS: list[dict] = [
    {"steering": "gaze", "n_mics": 6},
    {"steering": "gaze", "n_mics": 4},
    {"steering": "gaze", "n_mics": 3},
    {"steering": "gaze", "n_mics": 2},
    {"steering": "srp",  "n_mics": 6},
    {"steering": "srp",  "n_mics": 3},
]


class PowerSampler(threading.Thread):
    """20 Hz sampler of the VDD_CPU_SOC_MSS rail. Falls back gracefully
    (enabled=False) when the INA nodes are missing (non-Jetson host)."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="sweep-power")
        self.samples: list[tuple[float, float]] = []   # (t, W)
        self._stop = threading.Event()
        self.enabled = False
        self._v = self._c = None
        try:
            from sensors import discover
            self._sensors = discover()
            chans = {c.name: c for c in self._sensors.power}
            self._v = chans.get("VDD_CPU_SOC_MSS_volt_mV")
            self._c = chans.get("VDD_CPU_SOC_MSS_curr_mA")
            self.enabled = self._v is not None and self._c is not None
        except Exception:                              # noqa: BLE001
            self._sensors = None

    def run(self) -> None:
        if not self.enabled:
            return
        while not self._stop.is_set():
            try:
                v = self._v.fd.read_int()
                c = self._c.fd.read_int()
                if v is not None and c is not None:
                    self.samples.append((time.monotonic(), v * c / 1e6))
            except OSError:
                pass
            self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._sensors is not None:
            with contextlib.suppress(Exception):
                self._sensors.close()

    def window_j(self, t0: float, t1: float, baseline_w: float) -> 'float | None':
        """∫(P − baseline) dt over [t0, t1], trapezoidal."""
        pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(pts) < 4:
            return None
        t = np.array([p[0] for p in pts])
        w = np.array([p[1] for p in pts]) - baseline_w
        return float(np.trapz(w, t))

    def median_w(self, t0: float, t1: float) -> 'float | None':
        pts = [w for t, w in self.samples if t0 <= t <= t1]
        return float(np.median(pts)) if len(pts) >= 4 else None


def _cache_key(clip_name: str, cfgs: list[dict], seconds: float) -> pathlib.Path:
    payload = json.dumps({"clip": clip_name, "cfgs": cfgs, "s": seconds},
                         sort_keys=True).encode()
    h = hashlib.sha1(payload).hexdigest()[:12]
    return CACHE_DIR / f"{clip_name}__{h}.json"


def run_sweep(clip_name: str,
              progress=lambda i, n, label: None,
              configs: 'list[dict] | None' = None,
              seconds: float = SWEEP_SECONDS,
              use_cache: bool = True) -> dict:
    cfgs = configs if configs is not None else SWEEP_CONFIGS
    cache = _cache_key(clip_name, cfgs, seconds)
    if use_cache and cache.exists():
        res = json.loads(cache.read_text())
        res["cached"] = True
        return res

    clip = Clip(clip_name)
    n_frames = min(int(seconds / (HOP / FS)), clip.n_frames)
    n_samp = n_frames * HOP
    audio = clip.audio[:, :n_samp]
    gaze = clip.gaze[:n_frames]
    vad = clip.vad[:n_frames]
    ref = clip.ref[:n_samp]
    raw = clip.audio[0][:n_samp]
    audio_s = n_samp / FS

    sampler = PowerSampler()
    sampler.start()
    time.sleep(IDLE_PRE_S)                       # idle baseline window
    t_idle0, t_idle1 = time.monotonic() - IDLE_PRE_S, time.monotonic()

    base_sisdr = si_sdr_db(raw, ref)
    results = []
    n = len(cfgs)
    for i, cd in enumerate(cfgs):
        cfg = PipeConfig.from_dict(cd)
        progress(i, n, cfg.short())
        pipe, _tap = build_pipeline(cfg)
        pipe.observer = None                      # sweep wants raw speed
        g = gaze if cfg.steering == "gaze" else None
        x = np.ascontiguousarray(audio[: cfg.n_mics])
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pipe.process(x, gaze=g, annotated_vad=vad,
                               skip_denoise=True)
        t1 = time.monotonic()
        sisdr = si_sdr_db(np.asarray(out), ref)
        results.append({
            "cfg": cfg.to_dict(),
            "short": cfg.short(),
            "si_sdr": None if sisdr is None else round(sisdr, 2),
            "delta_db": (None if sisdr is None or base_sisdr is None
                         else round(sisdr - base_sisdr, 2)),
            "wall_s": round(t1 - t0, 3),
            "rtf": round((t1 - t0) / audio_s, 4),
            "gate_frac": round(pipe._gate_frames
                               / max(1, n_frames), 3),
            "_t0": t0, "_t1": t1,
        })
        time.sleep(IDLE_GAP_S)                    # settle + inter-config idle

    # trailing idle for baseline stability check
    time.sleep(IDLE_PRE_S)
    t_end = time.monotonic()
    baseline = sampler.median_w(t_idle0, t_idle1)
    baseline_end = sampler.median_w(t_end - IDLE_PRE_S, t_end)
    sampler.stop()

    if baseline is not None and baseline_end is not None:
        baseline = min(baseline, baseline_end)    # conservative
    for r in results:
        e = (None if baseline is None
             else sampler.window_j(r["_t0"], r["_t1"], baseline))
        r["mj_per_s"] = None if e is None else round(e * 1e3 / audio_s, 1)
        del r["_t0"], r["_t1"]

    full_rtf = results[0]["rtf"]                  # first config = full gaze
    for r in results:
        r["speedup"] = round(full_rtf / r["rtf"], 2) if r["rtf"] > 0 else None

    out = {
        "type": "sweep_result",
        "clip": clip_name,
        "seconds": audio_s,
        "raw_si_sdr": None if base_sisdr is None else round(base_sisdr, 2),
        "baseline_w": None if baseline is None else round(baseline, 2),
        "power_available": sampler.enabled and baseline is not None,
        "results": results,
        "cached": False,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--seconds", type=float, default=SWEEP_SECONDS)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    def prog(i, n, label):
        print(f"[{i+1}/{n}] {label}", flush=True)

    res = run_sweep(args.clip, prog, seconds=args.seconds,
                    use_cache=not args.no_cache)
    print(json.dumps(res, indent=2))
