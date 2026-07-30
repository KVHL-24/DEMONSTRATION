# runtime_demo — live A/B demo on Jetson Thor

Shows, on real hardware at 1x real time, that **no fixed pipeline
configuration is optimal for every sample** — and what each knob buys or
costs in accuracy (ΔSI-SDR) and energy (mJ per second of audio).

Two independently-configured pipelines process the same clip side by side;
the page shows their beam patterns overlaid, per-stage compute, duty
cycle, live SI-SDR, and lets you *listen* to input/A/B. A SWEEP button
runs ~9 configurations over the current clip full-speed and renders the
accuracy↔energy Pareto plot, with rail power measured on the board.

## Run

```bash
cd DEMONSTRATION/runtime_demo
../.venv/bin/python server.py            # http://<thor-ip>:8085
```

No dependencies beyond the project `.venv`; the server pins itself to
CPU-only single-thread BLAS (the measured configuration). Energy numbers
require the Jetson INA sensors (present on Thor; the sweep degrades to
RTF-only elsewhere).

## The three preset buttons (the demo script)

All numbers below are from the cached 30 s sweeps in `sweep_cache/`.

| preset | clip | A vs B | the argument |
|---|---|---|---|
| ① eyes beat ears | babble, **17° separation**, −10 dB | gaze vs SRP, rest equal | SRP's steering ray wanders (watch it vs the pinned blue one) — and SRP *costs more* (216 vs 161 mJ/s; its correlation search is the most expensive DOA) for less accuracy (+0.35 vs +0.53 dB). |
| ② cheap wins | white noise, +10 dB (easy) | full vs 3 mic + stride 4 + gate | Gate engages on **68%** of frames, duty collapses, energy **179 → 15 mJ/s (11.7x)** — for −0.9 dB on a clip already at ~10 dB SI-SDR. A fixed full config burns power for nothing here. |
| ③ …but not always | back to the ① clip | **same cheap B as ②** | On 17° separation the cheap config lands at **−0.42 dB — worse than doing nothing** — while full holds +0.53. Nulling a nearby interferer needs the whole aperture. Configuration must follow the sample — that is the product. |

| ④ you steer | directional mid, **moving target** (~27° span), +10 dB | gaze vs **manual** (a slider), rest equal | The free-play closer, not part of the argument chain: set a side's steering to *manual (you)* and chase the moving target with the azimuth slider. The gray **ghost ray** shows where the recorded eyes point; the *hand vs eyes* readout accumulates your average pointing error. The point partners take away: any scheme that needs the direction *told* to it (buttons, apps, remotes) trails the conversation — the eyes do it for free. |

Manual steering notes: the direction is per-frame *data*, not
configuration — dragging the slider never restarts the clip (switching a
side *into* manual mode does, like any config change). Your aim also
survives ⟲/preset reloads. Dragging onto the interferer makes the MVDR
treat the true talker as noise — you will hear the interferer enhanced
and the talker suppressed; that is the system working, not a bug. Nulls
take ~1–2 s to migrate after a large jump (recursive R_nn), so fast
scrubbing shows the beam lagging the ray — also real physics.

Close with **⚡ SWEEP**: the Pareto plot shows the whole trade space on
hardware-measured energy. Bonus talking point from the sweeps: on the
*hard dynamic* clip (cocktail −15) the cheap config actually **wins**
(+0.76 dB, best of all nine) — the optimal configuration is not even
monotone in sample difficulty, which is exactly why a fixed choice loses.

## Reading the page (operator's manual)

**Transport.** Pick a clip (or press a preset — it sets clip + both
configs and starts playback). Any config change restarts the clip from
0 s: the pipeline state is causal, so mid-clip switching would produce
numbers that belong to no single configuration.

**Beam pattern (center).** Blue = A, orange = B, drawn over the truth
rays (green = target, red = interferer). A side's curve appears only
once its MVDR weights are valid — expect **a few seconds of empty plot
after every (re)start** while the beamformer accumulates noise frames;
until then the output is a mic-0 passthrough. A healthy gaze config
shows ~0 dB toward the target and a deep notch pinned on the red ray.

**GATED chip (yellow, on a config card).** Lights while that side's
bypass gate is actively skipping stages 2–3. Requires `gate=on`, plus
the online SNR estimate to be warmed up (~1–2 s) and above +5 dB; it
releases below +2 dB (hysteresis). While a side is gated its beam curve
disappears — nothing is beamforming, so there is nothing to draw. Gate
lit = that side is (correctly) being lazy; curve visible = it's working.

**Compute duty.** Compute seconds per audio second at 1x playback,
stages 2–3 only, 1 s rolling window. Full config ≈ 19%, cheap config
≈ 7%; 100% would mean "can't keep up with real time". Display-only work
(beam pattern, spectrogram, SI-SDR) is deliberately excluded, as is the
config-independent STFT. This is the live proxy for energy — see below
for why watts are not shown live.

**Which side is winning.** The badge under the beam pattern says it
outright ("A leads by 0.53 dB (clip so far)"), colored by the leader.
It compares the *cumulative* SI-SDR (whole playback so far) — the big
number on each card — which settles as the clip plays; judge only after
30+ s. The small print holds the 2 s sliding window value (bouncy, shows
what's happening *now*), the delta vs raw mic 0 (the beamformer's net
contribution), and the gated-frame percentage. Expect A≈B for the first
~20 s; separation accumulates. SI-SDR gaps here are ~1 dB-scale — the
visual (null behavior) and the listening buttons carry the demo; the
numbers are for the rigorous.

**Listening.** The three 🎧 buttons fetch what has been *played so far*
(input mic / A output / B output) and play it in your browser — audio
comes out of the machine running the browser, the Thor needs no audio
hardware.

**SWEEP.** Runs 9 configs × 30 s full-speed on the current clip
(~75 s; cached clips return instantly) and draws the accuracy↔energy
Pareto scatter. The engine pauses during a sweep — a live pipeline
racing it would corrupt the power measurement.

## What each number means

- **duty** — stages 2–3 compute time / wall time at 1x pacing. Phase-0
  finding: live *power* at 1x is not measurable above rail noise
  (delta ~0.2 W vs ±0.3 W ambient wander), so the live view shows duty
  instead; duty is rock-stable (<0.4% rep spread).
- **SI-SDR** — sliding 2 s window against the reverberant reference,
  ~1 Hz refresh; Δ is against raw mic 0 over the same window.
- **SWEEP mJ/s** — `VDD_CPU_SOC_MSS` rail energy above idle baseline
  during a full-speed burst, divided by audio seconds (per the
  normalize-per-work rule; within-session spread ≈4%). Configs are
  measured back-to-back in one session — cross-session absolute values
  drift ~15%, comparisons within one sweep do not.
- **beam pattern** — `|Σₙ wₙ dₙ(θ)|²`, 0.8–3 kHz average. The band is
  bounded by the array's physics (min spacing 4.9 cm → spatial Nyquist
  3.5 kHz; aperture 15 cm → resolution floor ~1.1 kHz). The response
  convention matches the beamformer's constraint dᵀw = 1 — see
  `probes.py` for why `wᴴd` is wrong here.

## Files

| file | role |
|---|---|
| `server.py` | stdlib HTTP + SSE (port 8085), control endpoint, audio WAVs |
| `engine.py` | dual-pipeline frame driver, 1x pacing, telemetry packets |
| `configs.py` | `PipeConfig` knob space + pipeline factory |
| `probes.py` | display-only compute: beam pattern, spectrogram columns |
| `sweep.py` | serial full-speed config sweep + INA power sampling + cache |
| `web/index.html` | the whole frontend (single file, no build step) |
| `test_equivalence.py` | merge gate for `pipeline_2.py`/`beamformer_2.py` edits |
| `phase0_*` | power-feasibility study (kept as the record of why duty, not watts, is shown live) |
| `PLAN.md` | design decisions + phase results log |

## Invariants to keep

- New pipeline knobs must keep `test_equivalence.py` passing: defaults
  bit-identical to pre-change output, observer overhead < 3%.
- Anything computed for display (beam pattern, spectrogram, SI-SDR)
  stays OUT of the per-frame timed sections.
- SWEEP measures configs back-to-back in one session; never compare
  mJ/s across sessions.
