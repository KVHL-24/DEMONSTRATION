#!/usr/bin/env python3
"""
plot_results.py — render the profiling results to outputs/*.png

Reads the JSON written by profile_pipeline.py, profile_scaling.py and
bench_optimizations.py. Run those first.

Usage:
    python plot_results.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, load_json

# Categorical slots from the validated reference palette (light mode).
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e2de"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _style(ax, ylabel=None, title=None, xlabel=None):
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left", pad=10)


def _save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → outputs/{name}.png")


# ── 1. Stage breakdown ──────────────────────────────────────────────────────

def plot_stages(prof):
    stages = prof["stages"]
    names = list(stages)
    vals = [stages[k] for k in names]
    total = sum(vals)
    colors = [C_AQUA, C_YELLOW, C_ORANGE, C_MAGENTA]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # Horizontal bars, largest first.
    order = np.argsort(vals)
    y = np.arange(len(names))
    ax1.barh(y, [vals[i] for i in order], height=0.62,
             color=[colors[i] for i in order], zorder=2)
    ax1.set_yticks(y, [names[i] for i in order])
    for i, idx in enumerate(order):
        ax1.text(vals[idx] + total * 0.012, i,
                 f"{vals[idx]:.2f}s  ({100*vals[idx]/total:.0f}%)",
                 va="center", fontsize=8.5, color=INK_2)
    ax1.set_xlim(0, max(vals) * 1.34)
    ax1.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("time (s)")
    ax1.set_title(f"Pipeline stages — {total:.2f}s total, "
                  f"{prof['n_mics']} mics, {prof['n_frames']:,} frames",
                  loc="left", pad=10)

    # Beamformer internals.
    ints = prof["beamformer_internals_total_s"]
    inames = list(ints)
    ivals = [ints[k] for k in inames]
    x = np.arange(len(inames))
    ax2.bar(x, ivals, width=0.55, color=C_BLUE, zorder=2)
    ax2.set_xticks(x, inames, fontsize=8.5)
    per = prof["beamformer_internals_per_call_us"]
    for i, k in enumerate(inames):
        ax2.text(i, ivals[i] + max(ivals) * 0.03,
                 f"{ivals[i]:.2f}s\n{per[k]:.0f}µs/call",
                 ha="center", fontsize=8, color=INK_2)
    ax2.set_ylim(0, max(ivals) * 1.30)
    _style(ax2, ylabel="time (s)",
           title="Beamformer internals (extrapolated)")

    _save(fig, "fig1_stage_breakdown")


# ── 2. Hotspots ─────────────────────────────────────────────────────────────

def plot_hotspots(prof, top=12):
    rows = prof["hotspots"]["rows"][:top][::-1]

    def _label(r):
        # cProfile spells builtins two ways:
        #   "<built-in method numpy.core._multiarray_umath.c_einsum>"
        #   "<method 'reduce' of 'numpy.ufunc' objects>"
        # Keep the identifier that actually names the operation; naively
        # taking the last token turns both of the latter into "objects".
        f = r["func"]
        if not f.startswith("<"):
            return f
        body = f.strip("<>")
        if body.startswith("method '"):
            name = body.split("'")[1]
            owner = body.split("of '")[-1].split("'")[0].split(".")[-1]
            return f"{owner}.{name}"
        return body.split(".")[-1].split(" ")[-1]

    labels = [_label(r) for r in rows]
    vals = [r["tottime"] for r in rows]
    calls = [r["ncalls"] for r in rows]
    # Project code vs library code — identity by color, restated in the legend.
    colors = [C_ORANGE if r["is_project"] else C_BLUE for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, vals, height=0.66, color=colors, zorder=2)
    ax.set_yticks(y, labels, fontsize=8.5)
    for i, (v, c) in enumerate(zip(vals, calls)):
        ax.text(v + max(vals) * 0.015, i, f"{v:.3f}s   {c:,}×",
                va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("self time, cProfile (s)")
    ax.set_title("Hotspots by self time — call counts annotated",
                 loc="left", pad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=C_BLUE)]
    ax.legend(handles, ["project code", "numpy / library"],
              frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, "fig2_hotspots")


# ── 3. Fixed overhead vs O(N^2) ─────────────────────────────────────────────

def plot_scaling(sc):
    ns = [int(n) for n in sc["mic_counts"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    fig.subplots_adjust(wspace=0.30)

    for ax, op in zip(axes, ("update_noise", "compute_weights")):
        meas = [sc["micro_us"][str(n)][op] for n in ns]
        f = sc["fits"][op]
        c0, c2 = f["c0_us"], f["c2_us"]
        fixed = [c0] * len(ns)
        var = [c2 * n * n for n in ns]

        ax.bar(ns, fixed, width=0.6, color=C_ORANGE, zorder=2,
               label="fixed (N-independent)")
        ax.bar(ns, var, width=0.6, bottom=fixed, color=C_BLUE, zorder=2,
               label="O(N²) compute")
        ax.plot(ns, meas, "o-", color=INK, lw=1.6, ms=5, zorder=3,
                label="measured")
        _style(ax, ylabel="µs per call" if op == "update_noise" else None,
               xlabel="microphones (N)",
               title=f"{op}  —  {100*f['fixed_frac_at_6']:.0f}% fixed at N=6")
        ax.set_xticks(ns)
        if op == "update_noise":
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    # End-to-end speedup vs the O(N^2) ideal.
    ax = axes[2]
    e2e = [sc["end2end_s"][str(n)] for n in ns]
    base = sc["end2end_s"][str(max(ns))]
    actual = [base / t for t in e2e]
    ideal = [(max(ns) / n) ** 2 for n in ns]
    ax.plot(ns, ideal, "--", color=INK_MUTED, lw=1.6, label="ideal O(N²)")
    ax.plot(ns, actual, "o-", color=C_ORANGE, lw=2, ms=6, label="measured")
    for n, a in zip(ns, actual):
        ax.annotate(f"{a:.2f}×", (n, a), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=INK_2)
    _style(ax, ylabel="speedup vs N=6", xlabel="microphones (N)",
           title="Removing mics: ideal vs actual")
    ax.set_xticks(ns)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    _save(fig, "fig3_scaling")


# ── 4. Optimization results ─────────────────────────────────────────────────

def plot_optimizations(opt, sc):
    res = opt["results"]
    labels = list(res)
    times = [res[k]["time_s"] for k in labels]
    base = times[0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})

    short = {"baseline (pre-v7.9 loop)": "baseline\nper-mic loop\n(pre-v7.9)",
             "shipping (vectorised)": "shipping\nvectorised\n(v7.9)"}
    short = [short.get(l, l) for l in labels]
    colors = [INK_MUTED if l.startswith("baseline") else
              (C_AQUA if res[l]["equivalent"] else C_MAGENTA) for l in labels]
    x = np.arange(len(labels))
    ax1.bar(x, times, width=0.56, color=colors, zorder=2)
    ax1.axhline(base, color=INK_MUTED, lw=1, ls="--", zorder=1)
    for i, k in enumerate(labels):
        sp = res[k]["speedup"]
        tag = f"{times[i]:.2f}s"
        if i:
            tag += f"\n{sp:.2f}×"
        ax1.text(i, times[i] + base * 0.03, tag, ha="center",
                 fontsize=8.5, color=INK_2)
    ax1.set_xticks(x, short, fontsize=8.5)
    ax1.set_ylim(0, base * 1.28)
    _style(ax1, ylabel="end-to-end time (s)",
           title="Optimizations — all outputs bit-identical to baseline")

    # Compare against what removing microphones buys.
    ns = [int(n) for n in sc["mic_counts"]]
    e2e_base = sc["end2end_s"][str(max(ns))]
    mic_best = e2e_base / sc["end2end_s"][str(min(ns))]
    opt_best = max(res[k]["speedup"] for k in labels if res[k]["equivalent"])

    ax2.bar([0, 1], [mic_best, opt_best], width=0.5,
            color=[C_BLUE, C_AQUA], zorder=2)
    ax2.set_xticks([0, 1],
                   [f"6→{min(ns)} mics\n(halves spatial DoF)",
                    "vectorised smoothing\n(bit-identical output)"],
                   fontsize=8.5)
    for i, v in enumerate([mic_best, opt_best]):
        ax2.text(i, v + 0.04, f"{v:.2f}×", ha="center", fontsize=9.5,
                 color=INK)
    ax2.set_ylim(0, max(mic_best, opt_best) * 1.22)
    _style(ax2, ylabel="speedup", title="Where the wins actually are")

    _save(fig, "fig4_optimizations")


def plot_before_after(sc):
    """
    Why vectorising the smoothing loop SHRANK the pay-off from dropping
    microphones — the point that is easy to misread as a paradox.

    Both curves are faster after the change. But the loop that was removed
    ran once per microphone, so it cost more at N=6 than at N=2; removing it
    therefore saved more at N=6 (right panel, rising bars). The gap between
    N=6 and N=2 — the head-room that dropping mics can recover — is what
    shrinks, in absolute seconds and as a ratio.
    """
    if "end2end_before_s" not in sc:
        print("  (skipping fig5 — re-run profile_scaling.py)")
        return

    ns = [int(n) for n in sc["mic_counts"]]
    after = [sc["end2end_s"][str(n)] for n in ns]
    before = [sc["end2end_before_s"][str(n)] for n in ns]
    saved = [b - a for b, a in zip(before, after)]
    nmax, nmin = max(ns), min(ns)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})

    # ── Left: the two curves, with the head-room each leaves ────────────
    ax1.plot(ns, before, "o-", color=INK_MUTED, lw=2, ms=6,
             label="before (per-mic loop)", zorder=3)
    ax1.plot(ns, after, "o-", color=C_AQUA, lw=2, ms=6,
             label="after (vectorised)", zorder=3)
    ax1.fill_between(ns, after, before, color=C_AQUA, alpha=0.10, zorder=1)

    # Annotate the N=6 → N=2 head-room on each curve: a horizontal guide at
    # the N=2 level makes clear the span is measured against N=6, not a gap
    # local to N=2.
    for series, color, xoff, halign in ((before, INK_MUTED, -0.34, "right"),
                                        (after, C_AQUA, 0.16, "left")):
        hi, lo = series[ns.index(nmax)], series[ns.index(nmin)]
        ax1.plot([nmin, nmax], [lo, lo], ":", color=color, lw=1.1, zorder=2)
        xa = nmax + xoff
        ax1.annotate("", xy=(xa, hi), xytext=(xa, lo),
                     arrowprops=dict(arrowstyle="<->", color=color, lw=1.4))
        dx = -0.08 if halign == "right" else 0.08
        ax1.text(xa + dx, (hi + lo) / 2,
                 f"{hi - lo:.2f}s\n{hi/lo:.2f}×", color=color,
                 fontsize=8.5, va="center", ha=halign)

    _style(ax1, ylabel="end-to-end time (s)", xlabel="microphones (N)",
           title="Dropping mics buys less once the loop is gone")
    ax1.set_xticks(ns)
    ax1.set_ylim(0, max(before) * 1.16)
    ax1.set_xlim(nmin - 0.25, nmax + 0.72)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    # ── Right: what the optimization saved, per N ───────────────────────
    ax2.bar(ns, saved, width=0.58, color=C_ORANGE, zorder=2)
    for n, s in zip(ns, saved):
        ax2.text(n, s + max(saved) * 0.035, f"{s:.2f}s", ha="center",
                 fontsize=8.5, color=INK_2)
    _style(ax2, ylabel="time saved by vectorising (s)",
           xlabel="microphones (N)",
           title="The removed loop ran once per mic")
    ax2.set_xticks(ns)
    ax2.set_ylim(0, max(saved) * 1.22)

    _save(fig, "fig5_before_after_scaling")


def main():
    prof = load_json("pipeline_profile")
    sc = load_json("scaling_profile")
    print("Rendering figures...")
    plot_stages(prof)
    plot_hotspots(prof)
    plot_scaling(sc)
    plot_before_after(sc)
    try:
        opt = load_json("optimization_bench")
        plot_optimizations(opt, sc)
    except SystemExit:
        print("  (skipping fig4 — run bench_optimizations.py first)")
    print("done")


if __name__ == "__main__":
    main()
