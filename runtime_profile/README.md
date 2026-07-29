# Runtime profiling

Where the pipeline's time actually goes, and what it is worth optimizing.

This exists to answer a concrete question: **can we save compute by turning
microphones off?** The short answer is *not much* — and the profiling below
shows why, and what to do instead.

## Running it

All scripts must run with BLAS threading pinned, or the timings are noise:

```bash
cd runtime_profile
./run_all.sh                    # everything, then renders the figures
```

Or individually:

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../.venv/bin/python profile_pipeline.py      # classes 1, 2, 4
../.venv/bin/python profile_scaling.py       # class 3
../.venv/bin/python bench_optimizations.py   # class 5
../.venv/bin/python plot_results.py          # figures
```

GPU is disabled deliberately — it makes no difference here (measured: 22.18 s
GPU vs 21.90 s CPU per clip). Only Stage 1's STFT is batched; Stages 2–3 are
inherently sequential, and they dominate.

## What each script measures

| Script | Class | Question |
|---|---|---|
| `profile_pipeline.py` | 1, 2, 4 | Which stage is expensive? Which function? |
| `profile_scaling.py` | 3 | How much cost is *fixed* vs O(N²) in the mic count? |
| `bench_optimizations.py` | 5 | Is the shipped optimization still correct, and what is it worth? |
| `plot_results.py` | — | Renders `outputs/*.png` from the JSON |

Each writes JSON to `outputs/`; `plot_results.py` reads it back, so figures
can be re-rendered without re-measuring.

## Findings

**The beamformer is 78% of runtime**, and `compute_weights` is most of it.
STFT and ISTFT together are 8% — irrelevant.

**Removing microphones does not pay.** 6→2 mics gives 1.61x where O(N²)
predicts 9x. Fitting `t(N) = c0 + c2·N²` to the MVDR core explains it: at
N=6, **46% of `compute_weights` and 32% of `update_noise` is fixed cost**
that no mic reduction can touch — Python dispatch, the 257-bin sweep, the
v7.2 `isfinite` guard, and a per-mic smoothing loop.

**Vectorizing one loop beat the entire mic-reduction strategy — and has
landed.** [`beamformer_2.py`'s](../beamformer_2.py) frequency-axis smoothing
used to run `np.pad` + `np.convolve` inside `for n in range(self.N)` —
**127,560 `np.pad` calls per 60 s clip**, ~25% of runtime, because each call
re-paid NumPy dispatch overhead to smooth 257 values. The 3-tap kernel is
separable and identical per mic, so the loop collapses into three slice-adds.
Now `_smooth_freq_axis()` (v7.9):

| | speedup | accuracy cost |
|---|---:|---|
| 6→2 microphones | 1.62x | halves spatial DoF |
| **vectorized smoothing (shipped)** | **1.51x** | **none — bit-identical** |

After the change the beamformer is 68% of runtime (was 78%), 2.38 s (was
4.03 s) on the reference clip.

A cheap-finiteness-guard variant was also measured and came out at 1.01x —
noise. It was not kept; see the note in `bench_optimizations.py`.

## Verification

The optimization is **bit-identical**, not approximately equal, and this is
checked three ways:

1. **Unit** — against the original loop over N = 1..8 × B = 2..257:
   max\|Δ\| = 0.0.
2. **End-to-end** — `bench_optimizations.py` patches the old per-mic loop
   back in, runs both, and compares output samples before reporting any
   timing: max\|Δ\| = 0.0.
3. **Metrics** — a re-run of `eval_synthetic_2.py` reproduced all 72
   SI-SDR/PESQ/STOI values across 8 clips exactly.

`bench_optimizations.py` therefore doubles as a regression test: if someone
changes the smoothing and breaks equivalence, it fails loudly. Note that
`_smooth_freq_axis()` is specialised for a 3-tap kernel and raises if
`SMOOTH_KERNEL` ever changes length.

## Figures

| File | Shows |
|---|---|
| `fig1_stage_breakdown.png` | Stage split + beamformer internals |
| `fig2_hotspots.png` | Top functions by self time, with call counts |
| `fig3_scaling.png` | Fixed vs O(N²) split; ideal vs actual mic speedup |
| `fig4_optimizations.png` | A/B results; optimization vs mic reduction |
| `fig5_before_after_scaling.png` | The N curve before vs after vectorizing |

### Why vectorizing *shrank* the pay-off from dropping mics

`fig5` addresses something that looks like a paradox: after the
optimization, 6→2 mics gives **1.39x** where it used to give **1.64x**. Both
curves got faster — so why did the ratio fall?

Because the loop that was removed **ran once per microphone**. It cost more
at N=6 than at N=2, so removing it saved more at N=6 (1.81 s) than at N=2
(0.71 s) — the right panel of `fig5`. The N=6→N=2 gap is exactly the
head-room that dropping microphones can recover, and it shrank from 2.15 s
to 1.04 s.

This is not a measurement artifact of comparing against a stale baseline:
`profile_scaling.py` measures both variants in the same process, interleaved
per N, so machine-load drift cannot masquerade as a difference between them.

The practical reading: dropping microphones is now an even worse deal.
6→2 buys 1.39x for half the spatial degrees of freedom, and the realistic
6→4 buys only ~1.22x. This holds **for this Python/NumPy implementation** —
on a C/DSP/hardware target the fixed overhead collapses, O(N²) reasserts
itself, and mic count matters again. That distinction matters for the
glasses' hardware design.
