# Gaze-Steered Multi-Microphone Speech Enhancement

Gaze-guided MVDR beamforming for AR-glasses microphone arrays, with a synthetic
dataset generator and an evaluation harness.

The research question: **the user's eyes point at whoever they want to hear** —
can gaze drive a beamformer better than acoustic DOA estimation can?

## Pipeline

```
6-channel array audio (48 kHz)
  │
  ├─ Stage 1  STFT                      stft.py           512 window / 256 hop / 257 bins / 187.5 fps
  ├─ Stage 2  DOA → steering vector     doa_2.py          gaze vectors  OR  GCC-PHAT+SRP
  ├─ Stage 3  MVDR beamforming          beamformer_2.py   Woodbury recursive R_nn⁻¹ update
  └─ Stage 4  DeepFilterNet post-filter denoiser.py
  │
  └─ enhanced mono audio
```

| Module | Role |
|---|---|
| `stft.py` | STFT/ISTFT. Batched via torch (GPU when available); falls back to NumPy if torch is absent. |
| `vad.py` | Energy VAD. Broadband threshold OR 500–4000 Hz spectral ratio — the latter is what detects speech at negative SNR. |
| `gaze_processing.py` | `GazeStabilizer` — rolling confidence/recency-weighted gaze smoothing in unit-vector space. |
| `doa_2.py` | `DOA_GCCSRP`, `DOA_Gaze`, `FreeFieldSteering`, `ATFSteering`, `GEVDRTFEstimator`. |
| `beamformer_2.py` | `MVDRBeamformer` with SNR-adaptive blending, self-nulling detection, DAS fallback. |
| `mic_selection.py` | `AdaptiveMicSelector` — picks the K most informative mics, K adapting to SNR. |
| `pipeline_2.py` | `AriaDenoisingPipeline` — end-to-end orchestration (`process()` batch, `process_frame()` streaming). |
| `generate_synthetic_dataset.py` | Builds the synthetic multi-channel dataset. |
| `eval_synthetic_2.py` | Evaluation, ablation sweeps, summary plots. |

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                              # creates .venv and installs everything
python scripts/fetch_data.py         # downloads LibriSpeech + EasyCom ATF into ./data
```

`fetch_data.py` is idempotent — re-running it skips anything already complete.
It prints ready-to-paste generate/eval commands when it finishes.

### External data

| Dataset | Size | Required? |
|---|---|---|
| [LibriSpeech test-clean](https://www.openslr.org/12/) | 346 MB | **Yes** — target and interferer speech |
| [EasyCom `Device_ATFs.h5`](https://github.com/facebookresearch/EasyComDataset) | 37 MB | Optional — measured array transfer functions. Without it, pyroomacoustics ISM simulation is used instead. |

> **Note on `numpy`**: the resolved environment pins numpy to 1.26.x because
> `pesq` / `pyroomacoustics` / `deepfilternet` are not yet numpy-2.x clean.
> This is why the project uses its own `.venv` rather than a shared conda env.

> **VSCode**: point the interpreter at `.venv/bin/python` (Ctrl+Shift+P →
> "Python: Select Interpreter"), otherwise the editor reports every dependency
> as missing while the code runs fine from the terminal.

### Verified platforms

| Host | Arch | GPU | Stages 1–3 | Notes |
|---|---|---|---|---|
| x86_64 workstation | x86_64 | H100 (sm_90) | GPU | The original development host. |
| NVIDIA Jetson Thor | **aarch64** | Thor (sm_110) | GPU | JetPack 7 / L4T R39, Ubuntu 24.04, CUDA 13.2. Verified 2026-07. |

**On aarch64 / Jetson Thor, `uv sync` just works — no special index needed.**
This is worth stating explicitly because the usual assumption is the opposite:

- Every locked dependency ships a `manylinux_2_28_aarch64` wheel (torch,
  torchaudio, triton, numpy, scipy, soundfile, pyroomacoustics, numba,
  deepfilterlib). The one exception is `pesq`, which is sdist-only and compiles
  from source — `gcc`/`g++` must be present, and on a stock Ubuntu 24.04 image
  they are.
- The PyPI aarch64 torch wheel is built for SBSA, not Tegra, so it is fair to
  expect it to fail on a Jetson iGPU. It does not: `torch 2.13.0+cu130` ships
  `sm_110` in its arch list, and real kernels execute. **Verify this rather than
  trusting `torch.cuda.is_available()`** — the characteristic Tegra failure is a
  successful CUDA init followed by "no kernel image is available for execution
  on the device" the moment a kernel actually runs:

  ```bash
  .venv/bin/python -c "
  import torch
  print(torch.cuda.get_arch_list(), torch.cuda.get_device_capability(0))
  x = torch.randn(512, 512, device='cuda') @ torch.randn(512, 512, device='cuda')
  torch.cuda.synchronize(); print('kernel OK', x.abs().sum().item())"
  ```

  If that ever fails on a future JetPack, the fallback is NVIDIA's Jetson torch
  wheels — but that leaves `uv.lock` behind, so keep the locked `.venv` intact
  as a reproducible reference and build the Jetson env alongside it.
- `stft.py` resolves the device automatically and prints
  `[stft] torch device: cuda` on the first STFT. If you see `cpu` there, torch
  is not seeing the GPU and Stages 1–3 have silently fallen back to NumPy.

Python 3.11 does not need to be installed system-wide — `uv sync` downloads a
standalone aarch64 CPython 3.11 to satisfy `requires-python = "==3.11.*"`.

### Stage 4 (DeepFilterNet) is unavailable in this environment

Two independent things are described below. The first applies **everywhere**;
the second is specific to the x86_64/H100 host and does **not** apply on Thor,
where modern torch runs on the GPU natively.

**1 — The API incompatibility (all platforms).** `deepfilternet` 0.5.6 uses
`torchaudio.backend.common.AudioMetaData` and `torchaudio.info()`, both removed
in torchaudio 2.1. It therefore requires `torchaudio<=2.0.2`, which pins
`torch==2.0.1`. Since the locked environment carries modern torch, importing
Stage 4 fails — `denoiser.py` detects this and raises a clear, actionable error
rather than dying obscurely mid-run.

**2 — Why we don't just pin the old pair (x86_64/H100 host only).**
`torch==2.0.1` is a CUDA 11.7 build supporting at most **sm_86**. That host has
an **H100 (sm_90)**, so that torch cannot execute any CUDA kernel, and the
failure hits `stft.py` in Stage 1, killing the whole run. Downgrading is
therefore not an option there. On Jetson Thor (sm_110) the same reasoning
applies even more strongly.

Either way: run with `--no-denoise` (all beamformer research is in Stages 1–3),
or build a separate CPU environment with `torch==2.0.1, torchaudio==2.0.2` for
Stage 4.

Verified on CPU with the pinned pair, Stage 4 does work and helps substantially:
SI-SDR +2.5 → **+16.5 dB**, PESQ 1.12 → 2.32, STOI 0.871 → 0.924.

## Usage

### 1. Generate a dataset

```bash
.venv/bin/python generate_synthetic_dataset.py \
    --librispeech ./data/LibriSpeech/test-clean \
    --out ./synthetic_dataset \
    --scenarios white_noise directional_mid \
    --snrs 0 10 \
    --duration 10
```

Add `--atf-path './data/EasyCom/Calibration/Array Transfer Functions/Device_ATFs.h5'`
to use measured ATFs instead of ISM simulation.

Omitting `--scenarios`/`--snrs`/`--duration` generates the **full** sweep:
14 scenarios × 8 SNRs × 60 s = 112 clips. That takes a while — start small.

**Scenarios** (7 base × {static, `_dynamic`}): `white_noise`, `pink_noise`,
`directional_near|mid|far`, `babble`, `cocktail`.
**SNRs**: −20 −15 −10 −5 0 5 10 20 dB.

Each clip directory holds `array_audio.wav` (6ch), `reverberant_reference.wav`
(the SI-SDR reference), `close_mic.wav`, `gaze.npy`, `vad.npy`,
`noise_envelope.npy`, `metadata.json`.

### 2. Evaluate

```bash
.venv/bin/python eval_synthetic_2.py \
    --dataset ./synthetic_dataset \
    --no-denoise \
    --modes raw_mic oracle_gaze srp \
    --jobs 4 --plot
```

`--no-denoise` skips Stage 4, which isolates the beamformer — the right setting
when you are debugging spatial processing. Drop it to run the full chain.

If the dataset was generated with `--atf-path`, pass the matching `--atf` here,
or the wrong reference signal gets selected.

**Modes**

| Mode | What it tests |
|---|---|
| `raw_mic` | Baseline — mic 0, unprocessed |
| `oracle_gaze` | The actual proposal: real gaze vectors, saccade noise included |
| `oracle_target_dir` | Upper bound — true target azimuth, no gaze noise |
| `energy_vad` | Internal energy VAD instead of annotated VAD (closer to deployment) |
| `srp` | Traditional acoustic-only GCC-PHAT+SRP DOA |

**Metrics**: SI-SDR, PESQ (wideband), STOI.

### 3. Ablations

```bash
.venv/bin/python eval_synthetic_2.py \
    --dataset ./synthetic_dataset \
    --no-denoise --ablate oracle_gaze srp --jobs 8
```

Runs each base mode in all 4 combinations of gaze-stabilizer × mic-selection
and prints each feature's individual dB contribution.

## Project status

This is active research code, and the module docstrings double as a debugging
log. Worth reading before changing anything:

- `beamformer_2.py` v7.2 fixed single-precision drift in `R_nn⁻¹` that caused
  weight blowups and run-to-run non-reproducibility on identical inputs.
- `doa_2.py` v7.2 fixed a steering-vector phase-reference bug where `d[0] ≠ 1`,
  making the beamformer/mic-0 blend destructive rather than additive.
- `pipeline_2.py` notes that `GazeStabilizer`, saccade-hold, and mic-selection
  were dead code in every evaluation run prior to v7.1 — they are now behind
  explicit constructor flags so each can be measured.

**Open issue**: the beamformer still loses to `raw_mic` by 3–6 dB at +20 dB
input SNR across every mode and scenario. See the `mic_selection.py` module
docstring for the current hypothesis.
