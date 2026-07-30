"""Phase 0 — analyze the thorprof daemon trace against the workload log.

Reads the daemon CSV (t, phase, per-rail mV/mA columns) and
phase0_workload_summary.json (written by phase0_power_probe.py), matches
contiguous phase=="run" blocks in order to the workload sequence, and
answers the phase-0 question:

    Is the config delta (N=6 vs N=2 at 1x) in mJ per second of audio
    larger than 3x the repeat-to-repeat spread?

Per the house rule, energy is normalized per second of audio processed —
raw joules differ ~15% between identical runs.

Usage:
    ../.venv/bin/python phase0_analyze.py daemon-YYYYmmdd-HHMMSS.csv
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

RAILS = {
    "cpu_soc": ("VDD_CPU_SOC_MSS_volt_mV", "VDD_CPU_SOC_MSS_curr_mA"),
    "gpu":     ("VDD_GPU_volt_mV",         "VDD_GPU_curr_mA"),
    "board":   ("VIN_volt_mV",             "VIN_curr_mA"),
}


def load_trace(path: pathlib.Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    phase = np.array([r["phase"] for r in rows])
    power = {}
    for name, (vcol, ccol) in RAILS.items():
        if vcol not in rows[0]:
            continue
        v = np.array([float(r[vcol] or 0) for r in rows])
        c = np.array([float(r[ccol] or 0) for r in rows])
        power[name] = v * c / 1e6            # mV * mA / 1e6 = W
    return t, phase, power


def contiguous_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[::2], idx[1::2]))


def integrate_j(t: np.ndarray, w: np.ndarray) -> float:
    return float(np.trapz(w, t))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    trace_path = pathlib.Path(sys.argv[1])
    here = pathlib.Path(__file__).parent
    workloads = json.loads((here / "phase0_workload_summary.json").read_text())

    t, phase, power = load_trace(trace_path)
    run_blocks = contiguous_blocks(phase == "run")
    idle_blocks = contiguous_blocks(phase == "idle")

    if len(run_blocks) != len(workloads):
        print(f"WARNING: {len(run_blocks)} run-blocks in trace vs "
              f"{len(workloads)} workload phases in the JSON — matching "
              "the first min() of both, check alignment manually.")
    n = min(len(run_blocks), len(workloads))

    # Idle baseline per rail: median power over all idle samples, which is
    # robust to the settle-in transients at phase edges.
    idle_mask = phase == "idle"
    baselines = {name: float(np.median(w[idle_mask]))
                 for name, w in power.items()}
    print("idle baselines (median W): "
          + "  ".join(f"{k}={v:.2f}" for k, v in baselines.items()))
    print()

    results = []
    hdr = f"{'label':<16}{'wall s':>8}{'audio s':>9}"
    for name in power:
        hdr += f"{name + ' mJ/s':>16}"
    print(hdr)
    for (i0, i1), wl in zip(run_blocks[:n], workloads[:n]):
        tt = t[i0:i1]
        row = {"label": wl["label"], "audio_s": wl["audio_s"]}
        line = f"{wl['label']:<16}{tt[-1] - tt[0]:>8.1f}{wl['audio_s']:>9.0f}"
        for name, w in power.items():
            e_j = integrate_j(tt, w[i0:i1] - baselines[name])
            mj_per_audio_s = e_j * 1e3 / wl["audio_s"]
            row[name] = mj_per_audio_s
            line += f"{mj_per_audio_s:>16.1f}"
        results.append(row)
        print(line)

    # ── The phase-0 verdict ────────────────────────────────────────────────
    print()
    for name in power:
        by_cfg: dict[str, list[float]] = {}
        for r in results:
            cfg = r["label"].rsplit("_rep", 1)[0]
            by_cfg.setdefault(cfg, []).append(r[name])
        if not {"n6_1x", "n2_1x"} <= by_cfg.keys():
            continue
        m6 = float(np.mean(by_cfg["n6_1x"]))
        m2 = float(np.mean(by_cfg["n2_1x"]))
        delta = abs(m6 - m2)
        spread = max(
            (max(v) - min(v)) for k, v in by_cfg.items()
            if k in ("n6_1x", "n2_1x") and len(v) > 1) if any(
            len(v) > 1 for v in by_cfg.values()) else float("nan")
        verdict = ("PASS" if np.isfinite(spread) and delta > 3 * spread
                   else "FAIL")
        print(f"[{name:>7}] n6_1x={m6:.1f}  n2_1x={m2:.1f} mJ/s  "
              f"delta={delta:.1f}  spread={spread:.1f}  "
              f"criterion delta>3*spread → {verdict}")
        if "n6_flat" in by_cfg:
            print(f"          n6_flat={np.mean(by_cfg['n6_flat']):.1f} mJ/s "
                  "(SWEEP-burst reference)")

    out = here / "phase0_analysis.json"
    out.write_text(json.dumps(
        {"baselines_w": baselines, "results": results}, indent=2))
    print(f"\nfull numbers → {out}")


if __name__ == "__main__":
    main()
