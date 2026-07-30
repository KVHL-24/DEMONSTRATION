# runtime_demo — implementation plan

Status: **COMPLETE — all phases done**
Last updated: 2026-07-29

## Phase 5 result (2026-07-29) — preset narratives, measured

The 30 s sweeps (cached in `sweep_cache/`) validated presets ① and ② and
**falsified the scripted ③**: on cocktail_dynamic −15 the cheap config
(gaze+3mic+k4+gate) did not collapse — it was the best of all nine
configs (+0.76 dB). Cheap actually breaks on the SMALL-SEPARATION clip
(babble 17°): −0.42 dB, below raw mic, while full holds +0.53 dB —
nulling a nearby interferer needs the full aperture. Preset ③ therefore
reuses the ① clip with ②'s cheap B config; the cocktail_dynamic result
became the closing talking point (optimal config is not monotone in
difficulty). Headline sweep numbers: white +10: gate 68% of frames,
179 → 15 mJ/s (**11.7x**) for −0.9 dB; SRP costs 216 vs 161 mJ/s and
gains less on the sep-17° clip. Final acceptance: all checklist items
pass except live-power (consciously degraded to duty-cycle per phase 0);
SWEEP completes 9 configs × 30 s ≈ 75 s including idle brackets.

## Phase 2–4 results (2026-07-29)

- **Beam-pattern physics verified** (phase-2 acceptance): on the cocktail
  −15 dB clip, gaze config shows ≈0 dB at the target and a **−11.8 dB null
  pinned at the true interferer azimuth**; SRP's steering ray visibly
  wanders (captured at −86° vs target −1°). Two probe lessons encoded in
  `probes.py`: (1) display band must stay inside the array physics
  (0.8–3 kHz; min spacing 4.9 cm → Nyquist 3.5 kHz, aperture 15 cm →
  resolution floor 1.1 kHz), (2) the response convention is Σ w·d (the
  code's constraint is dᵀw=1, a=d*), not wᴴd — the latter washes the
  pattern out.
- **Live view**: dual pipeline at 1x, duty meters (A full ≈19%, SRP ≈24%),
  per-frame stage split (doa/bf µs), sliding 2 s SI-SDR + Δ vs raw mic at
  ~1 Hz, input spectrogram strip, WebAudio listen (in/A/B), 3 preset
  buttons, per-side config panels. Server: stdlib SSE on port 8085.
- **SWEEP works with real energy numbers** (first run, babble −10 dB,
  sep 17°, 20 s window): full gaze 147.6 mJ/s → gaze+2mic+k4+gate
  41.5 mJ/s (**3.6x energy span**); gaze+6mic+k4 ties full accuracy at
  1.44x speedup; **SRP costs more than gaze** (0.82x, 193 mJ/s) *and*
  gains less — SRP's correlation search is the single most expensive DOA.
  Direct INA sysfs sampling (no daemon dependency), cache in
  `sweep_cache/`.
- **Preset clips found in the existing dataset** (no regeneration needed):
  ① `babble_taz-007_iaz-025_snr-10_rep00` (sep 17.2°!) ② `white_noise_
  taz-008_iaz+000_snr+10_rep00` ③ `cocktail_dynamic_taz+016_iaz-144_
  snr-15_rep00`. The 35° floor only applied to some scenarios.

## Phase 1 result (2026-07-29)

All three knobs landed in `pipeline_2.py` (v8.0 blocks) / `beamformer_2.py`
(v8.0 blocks), equivalence gate **PASS** (`test_equivalence.py`):
- knobs at defaults → output bit-identical to pre-change HEAD on 3 clips
  (max|Δ| = 0.0); observer overhead +1.2% (< 3% budget).
- `weight_stride` k: 1.21x / 1.34x / 1.41x end-to-end at k=2/4/8
  (10 s dynamic +10 dB clip, incl. STFT).
- `use_bypass_gate`: 62% frames bypassed on the +10 dB clip
  (test thresholds on=5/off=2 dB), 1 hysteresis transition (no chatter),
  **2.32x** end-to-end; observer reports gated frames correctly.
- Baselines for future re-checks: `baseline_outputs/*.npy`
  (10 s prefixes, 3 clips). Re-run `test_equivalence.py` after any
  pipeline/beamformer change.

## Phase 0 result (2026-07-29, two independent runs)

**Live 1x power: FAIL — degraded path taken.** The 1x-duty signal
(~0.1–0.35 W on VDD_CPU_SOC_MSS) is the same magnitude as ambient rail
wander from background OS activity (±0.2–0.4 W); delta>3×spread failed in
both a mixed run and a clean CPU-only pinned run. The live view therefore
drops the live power strip and shows **measured duty cycle** instead —
free, rock-stable (n6=8.1%, n2=4.1%, rep spread <0.4%), same story.

**Full-speed burst (SWEEP): PASS.** n6 flat-out = 250.4 mJ/s audio,
within-session spread 3.7%. Cross-session it drifted 214→250 mJ/s (~15%),
re-confirming the normalize-per-work rule — SWEEP must measure A and B
back-to-back in the same session, which the design already does.

Byproduct: fixed a phase-mark race in `thor_profile/hub.py` (daemon-mode
`push()` clobbered operator-set phases ~10-30% per post; see the comment
at the `if not self.daemon` guard). Probe + analyzer live in
`phase0_power_probe.py` / `phase0_analyze.py`; traces in
`phase0_trace_v2.csv`, verdicts in `phase0_analysis.json`.

## Goal

An online A/B demo running on Jetson Thor: one sample clip streams through
**two configurable pipelines in parallel at 1x real time**, visualizing the
intermediate computation (beam pattern, DOA, VAD, per-stage timing, power)
alongside accuracy/energy metrics, plus a per-sample configuration sweep
producing a Pareto plot. Audience: **project partners** — the narrative is
carried by three preset scenarios, not by free-form knob turning.

## Why these design choices

Measured on Thor (5 s clip, warm-up discarded, 3 reps, spread ≤ 4%):

| N mics | time | realtime factor | speedup vs N=6 |
|---|---|---|---|
| 6 | 0.383 s | 13.0x | 1.00x |
| 5 | 0.328 s | 15.2x | 1.17x |
| 4 | 0.280 s | 17.9x | 1.37x |
| 3 | 0.232 s | 21.6x | 1.65x |
| 2 | 0.195 s | 25.7x | 1.97x |

- **Live is feasible**: N=6 runs 13x realtime → two parallel pipelines +
  probes at 1x pace has ample headroom (est. < 0.4x RTF combined).
- **Pace deliberately at 1x**, not flat-out: the glasses' duty cycle is 1x,
  so live power numbers only mean something at 1x. The 13x headroom is a
  smoothness guarantee, not a speed to show off.
- **Primary cost metric: mJ per second of audio** (rail power minus idle
  baseline, normalized by audio seconds processed) — raw joules differ ~15%
  between identical runs; per-work normalization is the house rule.
- **Rail choice matters**: board VIN idles at ~22 W, our ~1 W single-thread
  workload drowns in it. Use `VDD_CPU_SOC_MSS` (idles ~6.7 W) minus idle
  baseline. GPU rail plotted flat as a feature ("no GPU needed").
  → must be validated in phase 0 before anything is built on it.

## UI concept (one page, A/B throughout, blue=A / orange=B)

```
┌──────────────────────────────────────────────────────────────────┐
│ sample: [cocktail ▾][SNR -15 ▾][sep 102°]   ▶ ‖ ⟲     [SWEEP]   │
│ presets:  [① eyes beat ears] [② cheap wins] [③ but not always]  │
├───────────────────────┬──────────────────────────────────────────┤
│ config A ●blue        │ config B ●orange                         │
│  steering [gaze ▾]    │  steering [SRP ▾]                        │
│  mics [6 ▾][□adapt]   │  mics [3 ▾][□adapt]                      │
│  weight stride [1 ▾]  │  weight stride [4 ▾]                     │
│  bypass gate [off ▾]  │  bypass gate [on ▾]                      │
├───────────────────────┴──────────────────────────────────────────┤
│        ★ beam pattern, polar, A/B OVERLAID, live ★               │
│    main lobe / nulls / gaze ray / true target / true interferer  │
├──────────────────────────┬───────────────────────────────────────┤
│ array A ●●●●●●           │ array B ●●○○○●                        │
│ stage-time bar A ▇▇▇▇▇▇  │ stage-time bar B ▇▇▇                  │
│ SI-SDR +8.2 dB           │ SI-SDR +3.1 dB                        │
│ power 3.4 W · 47 mJ/s    │ power 1.6 W · 21 mJ/s                 │
│ output spectrogram A     │ output spectrogram B                  │
├──────────────────────────┴───────────────────────────────────────┤
│ shared timeline (single x-axis):                                 │
│   input spectrogram / VAD strip (R_nn update ticks) /            │
│   DOA traces (truth · gaze · SRP) / gate state / power curve     │
├──────────────────────────────────────────────────────────────────┤
│ 🎧 listen:  [ input ] [ A ] [ B ]                                │
└──────────────────────────────────────────────────────────────────┘
```

Deliberate choices:
- Beam patterns **overlaid**, not side by side — the difference must be
  seen, not reconstructed by eye across two plots.
- All timeline strips share one x-axis — "gate opens here → power drops
  here" is seen, not narrated.
- Listening is three buttons; partners press buttons, they don't tune.

## The three preset scenarios (the actual demo content)

1. **"Eyes see what mics can't hear"** (accuracy argument)
   Sample: small target/interferer separation (< 30°), SNR −10.
   A = gaze/6 mics vs B = SRP/6 mics, rest identical.
   Watch B's null get pulled onto the target; A unaffected. Big audible gap.
2. **"Cheap performance is free"** (efficiency argument)
   Sample: white_noise/+10 dB (easy). A = full config vs
   B = 3 mics/stride 4/gate on. B's stage bar collapses, energy → ~¼,
   SI-SDR ~unchanged. Fixed full config is burning power for nothing.
3. **"But you can't always be cheap"** (adaptivity argument)
   Sample: cocktail_dynamic/−15 (hard). **Same A/B as #2, unchanged.**
   The same cheap config now loses 6–8 dB and sounds broken.
   → configuration must follow the sample — that's the product.

Then SWEEP on the current sample pulls up the full Pareto plot as the close.

## Telemetry contract

STFT is 187.5 fps → aggregate 8 frames per packet (~23 Hz), SSE:

```json
{"t": 12.345, "frame": 2315,
 "shared": {"vad": 1, "doa_true": -1.0, "doa_gaze": -3.2, "doa_srp": 44.1,
            "in_spec": [64]},
 "A": {"beam": [72], "mics": [1,1,1,1,1,1], "gated": 0,
       "stage_us": {"stft": 41, "doa": 88, "weights": 512, "smooth": 96},
       "si_sdr": 8.2, "out_spec": [64]},
 "B": {"...": "same shape"},
 "power": {"cpu_soc_w": 5.1, "gpu_w": 2.8, "board_w": 24.3}}
```

~60 KB/s — trivial for local SSE.

Beam pattern = `|wᴴd(θ)|²`, 72 azimuths, averaged over ~12 bins in a
500–4000 Hz speech-critical subband. This is visualization-only compute and
is **excluded from the pipeline stage-timing bars** (otherwise the timing
numbers lie).

## File layout

```
runtime_demo/
  PLAN.md       # this file
  configs.py    # config space definition, config IDs / short names
  probes.py     # observer-data → beam pattern, DOA, mic mask, stage timing
  engine.py     # clip loader, 1x pacing, dual-pipeline frame driver
  power.py      # rail power via thor_profile sampler/hub
  sweep.py      # serial full-speed config sweep on one sample (mode 2)
  server.py     # stdlib HTTP + SSE, modeled on thor_profile/server.py
  web/
    index.html  # single-file frontend, dark theme, blue/orange A-B
```

Reuses the `thor_profile` architecture (stdlib-only HTTP + SSE + hub
pub/sub) — no new dependencies.

## Changes to existing code

| change | file | risk |
|---|---|---|
| `observer` callback emitting per-frame internals (`w, d, theta, phi, speech, mic_mask`, stage timings) | `pipeline_2.py`, `beamformer_2.py` | low — default `None`, zero behavior change |
| weight-update stride `k` (recompute MVDR weights every k frames, reuse between) | `beamformer_2.py` hot loop | **medium** |
| SNR bypass gate (est. SNR above threshold → bypass beamformer to mic 0, with hysteresis) | `pipeline_2.py` | low |
| fixed mic subset | already exists | — |

**Merge gate** (per the `bench_optimizations.py` house rule): with `k=1`,
gate off, observer `None`, output must be **bit-identical** to current
HEAD; with observer attached, overhead < 3%.

## Phases

### Phase 0 — power feasibility (half day, FIRST — may change the design)
- `thorprof daemon` sampling; manually run 30 s N=6 vs N=2 workloads at 1x
  pace; check `VDD_CPU_SOC_MSS` minus idle baseline separates the configs.
- **Criterion**: config delta in mJ/s > 3× repeat-to-repeat spread → power
  axis as planned. Otherwise: power measured only during full-speed SWEEP
  bursts; live view drops the live power strip.
- sudo steps proposed as commands for the user to run (house rule).

### Phase 1 — probes + new knobs (1–2 days)
Observer callback; stride k; bypass gate; equivalence tests as merge gate.

### Phase 2 — engine + beam-pattern minimal page (1 day, the first cut)
`engine.py` + `probes.py` + `server.py` + minimal page drawing ONLY the
overlaid polar beam pattern + transport controls.
**Acceptance**: on a cocktail sample, gaze config's null pins to the true
interferer azimuth; SRP config visibly drifts on small-separation samples.
If the physics looks wrong, stop and fix the probes before building more.

### Phase 3 — live view, MINIMAL SET (1–1.5 days)
Demo-critical panels only:
- top bar: sample select + 3 preset buttons + transport
- A/B config panels
- overlaid beam pattern (from phase 2)
- per-config stage-time bars + live SI-SDR number
- 🎧 listen: input / A / B (WebAudio)
- duty-cycle meter per config (phase 0 outcome: replaces the live power strip)

Deferred to post-demo backlog: array mic plots, scrolling spectrograms,
shared timeline strips (VAD/DOA/gate), PESQ/STOI live windows.

### Phase 4 — SWEEP mode (1–2 days)
Serial full-speed run of ~10 representative configs on the current sample
(sampled along gaze/SRP × N × k × gate); per config ΔSI-SDR, mJ/s
(full-speed burst measurement), RTF. Pareto scatter + progress bar;
results cached on disk.

### Phase 5 — scenario tuning + polish (1–2 days)
Pick concrete clips per preset (preset 1 needs sep < 30° — verify the
dataset has them, generate a few if not); one-click preset loading; visual
polish to partner-demo grade; full 15-min dry run, timed.

Critical path: 0 → 1 → 2; phases 3/4 can proceed in parallel after 2.

## Acceptance checklist

- [x] Equivalence: knobs all-off output bit-identical; observer overhead < 3% (+1.2%)
- [x] Live dual pipeline at 1x (duty A≈19% + B≈24% ≪ 100%, ample margin)
- [x] Beam pattern physically correct (−11.8 dB null on true interferer azimuth)
- [x] Power axis: degraded to SWEEP-only per phase-0 criterion; live view shows duty
- [x] Each preset's argument is measured, not scripted (see phase-5 result)
- [x] SWEEP: 9 configs × 30 s window ≈ 75 s to Pareto plot (cached: instant)

## Risks

| risk | mitigation |
|---|---|
| CPU_SOC rail can't separate configs | phase 0 validates first; degradation path defined |
| stride k>1 collapses accuracy on dynamic scenes | that IS preset 3's content — a feature; k range set by sweep |
| SRP not "broken enough" in preset 1 | pick smaller-separation sample; generate new clips if needed |
| dual pipeline + probes + server exceed realtime budget | measured 0.16x RTF single; est. < 0.4x dual; confirm in phase 2 |

## Open questions

- Exact SNR-gate threshold + hysteresis window (tune in phase 1 against
  eval data, not hand-picked).
- Representative config subset for SWEEP (~10 of the 240-point grid) —
  sample along the expected Pareto frontier, finalize in phase 4.
- Whether preset clips need regeneration with smaller `--interferer-az-min-sep`
  (current dataset floor is 35°, preset 1 wants < 30°).
