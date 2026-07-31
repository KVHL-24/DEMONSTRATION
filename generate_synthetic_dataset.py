"""
generate_synthetic_dataset.py — Synthetic multi-channel audio dataset for pipeline testing
============================================================================================

v6 changes vs v5
-----------------
Three bugs in the ISM simulation branch have been fixed.

FIX 1 — close_mic.wav now contains reverberant channel 0, not dry speech (ISM path).
    Root cause:
        sf.write(...'close_mic.wav'..., sc['target_audio'], ...) was always
        writing the dry, resampled 48 kHz target speech — regardless of whether
        the ATF or ISM path was active.

        eval_synthetic.py uses close_mic.wav in two ways:
          (a) As the ultimate fallback reference when reverberant_reference.wav
              is missing.  The eval script explicitly warns "Do NOT use
              close_mic.wav (dry speech) — that gives a ~0 dB SI-SDR ceiling".
          (b) For OLD 'atf+reverb' datasets: close_mic ⊗ ATF_0 is computed
              on-the-fly to recover the correct reference.  This convolution is
              ONLY correct when close_mic.wav really is dry speech.

        For new ISM datasets simulation_mode='ism', eval_synthetic.py
        uses reverberant_reference.wav directly and only falls back to
        close_mic.wav if that file is missing.  So the practical risk is
        low when reverberant_reference.wav is present, but it is still
        semantically wrong and dangerous if that file ever goes missing.

    Fix:
        close_mic.wav now stores reverb_ref[:T]  (= channel 0 of the
        reverberant target-only simulation, exactly as reverberant_reference.wav
        stores it) when simulation_mode='ism'.

        For simulation_mode='atf' close_mic.wav still stores the DRY speech
        sc['target_audio'] because the legacy 'atf+reverb' re-convolution path
        in eval_synthetic.py requires a dry signal to convolve with ATF_0.

        A 'close_mic_is_dry' boolean flag is written to metadata.json so
        eval_synthetic.py can distinguish the two cases without inspecting the
        audio content.

FIX 2 — effective_rt60 is corrected to 0.0 when inverse_sabine falls back to
        the anechoic direct-path simulation.
    Root cause:
        simulate_room() calls pra.inverse_sabine() which can raise ValueError
        for extreme RT60 values (e.g. very short rooms, very high absorption).
        On failure it falls back to simulate_room_direct() but the caller
        (generate_one) never updated effective_rt60, so metadata.json recorded
        the sampled RT60 while the audio had no reverberation at all.  This
        caused compute_vad_stft() to add a spurious hangover extension and
        eval_synthetic.py to select a mismatched reference.

    Fix:
        simulate_room() now returns a second value: the RT60 actually used
        (0.0 on any fallback to simulate_room_direct).  generate_one unpacks
        this and uses it as effective_rt60.

FIX 3 — simulation_mode tag confirmed correct; no change needed.
    sim_mode = 'atf' if atfs is not None else 'ism' was already correct
    and is written to metadata.json under 'simulation_mode'.  Confirmed.

v5 changes vs v4
-----------------
1. Schroeder late-reverb tail removed from the ATF simulation path.

   Root cause of the original bug:
     simulate_room_atf() was adding a Schroeder exponential-decay tail on top
     of the ATF convolution and saving that composite signal as
     reverberant_reference.wav.  The MVDR distortionless constraint passes
     target ⊗ ATF_0 exactly, but does NOT pass the Schroeder tail at the same
     amplitude (because Σ_m w_m* ≠ 1 for unequal MVDR weights).  At high SNR
     (where the interferer is negligible) the only distortion term is
     (Σ w_m − 1) × late_reverb, which caused the ~17 dB SI-SDR collapse
     visible in the improvement tables.

   Fix:
     The EasyCom ATF already captures the true device + room acoustics for the
     measured environment.  Adding a synthetic Schroeder tail on top creates a
     model mismatch with no physical benefit.  It has been removed entirely.

     reverberant_reference.wav now contains target ⊗ ATF_0 only, which is
     exactly what the MVDR distortionless constraint outputs.

     Consequence: effective_rt60 is always 0.0 for ATF clips, so VAD hangover
     extension is also 0 (correct — there is no reverb tail to hang over).

     simulation_mode tag in metadata.json is now 'atf' (was 'atf+reverb').
     eval_synthetic.py uses this tag to select the correct reference.

   Removed functions (no longer needed):
     _make_late_reverb(), _late_reverb_schroeder(), _late_reverb_pra()

2. All v4 fixes are preserved unchanged (ATF loading, gaze model, etc.).

v4 changes vs v3
-----------------
1. EasyCom ATF integration (primary change).
   simulate_room_atf() replaces simulate_room() as the default path when
   --atf-path is supplied.  It convolves dry signals with the measured
   Array Transfer Functions from Device_ATFs.h5.

2. load_easycom_atfs() caches the ATFs after first load and prints the HDF5
   tree on first call.

3. --atf-path CLI flag (default: None → falls back to pyroomacoustics ISM).

4. All v3 fixes are preserved unchanged.

Known ATF H5 layout assumption (EasyCom public release)
---------------------------------------------------------
  /IR              (768, 1020, 6)  — RIRs, axes: (rir_len, n_dirs, n_mics)
  /Phi             (1,   1020)     — azimuth  in degrees
  /Theta           (1,   1020)     — elevation in degrees
  /SamplingFreq_Hz (1,   1)        — measurement sample rate
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import warnings
from math import gcd
from pathlib import Path
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt, fftconvolve, resample_poly

# ── Constants ──────────────────────────────────────────────────────────────────
FS              = 48_000
LIBRI_FS        = 16_000
F_WIN           = 512
HOP             = 256
VIDEO_FPS       = 20
AUDIO_HOP_VIDEO = FS // VIDEO_FPS      # 2400 samples per video frame
C_SOUND         = 343.0

MIC_POSITIONS = np.array([
    [-0.030,  0.000,  0.045],
    [ 0.030,  0.000,  0.045],
    [ 0.030,  0.000, -0.030],
    [-0.030,  0.000, -0.030],
    [-0.075, -0.010, -0.010],
    [ 0.075, -0.010, -0.010],
], dtype=np.float64)
N_MICS = MIC_POSITIONS.shape[0]

ROOM_DIM     = [6.0, 3.0, 5.0]
TARGET_DIST  = 1.5
TARGET_EL    = 0.0
DISTANCES    = {'near': 0.5, 'mid': 1.5, 'far': 3.0}

TARGET_AZ_RANGE     = (-70.0,  70.0)
INTERFERER_AZ_RANGE = (-150.0, 150.0)

# Minimum angular separation between target and any interferer.
# 35° ≥ array angular resolution at 4 kHz (λ/D ≈ 33°).
INTERFERER_AZ_MIN_SEP = 35.0

INTERFERER_AZ_DEFAULT = -60.0
BABBLE_AZ_DEFAULT     = [-45.0, 10.0, 80.0, -120.0]

CONVERSATION_MODES = ['single', 'two_speaker', 'three_speaker']

ACOUSTIC_PROFILES = {
    'target':         (0.12, 0.25),
    'near_interferer':(0.12, 0.25),
    'mid_interferer': (0.18, 0.35),
    'far_interferer': (0.35, 0.70),
    'babble':         (0.45, 0.90),
    'diffuse_noise':  (0.50, 1.00),
}

_DYN_STYLE = {
    'white_noise':      'ou',
    'pink_noise':       'sine',
    'directional_near': 'burst',
    'directional_mid':  'ou',
    'directional_far':  'sine',
    'babble':           'ou',
    'cocktail':         'burst',
}

_BASE = ['white_noise', 'pink_noise',
         'directional_near', 'directional_mid', 'directional_far',
         'babble', 'cocktail']
ALL_SCENARIOS = _BASE + [s + '_dynamic' for s in _BASE]

# ─────────────────────────────────────────────────────────────────────────────
# EasyCom Device_ATFs.h5 field names (verified against actual file layout).
#
# Actual layout:
#   /IR              (768, 1020, 6)  float64  — time-domain RIRs
#                                              axes: (rir_len, n_dirs, n_mics)
#   /RealTF          (385, 1020, 6)  float64  — real part of one-sided FFT
#   /ImagTF          (385, 1020, 6)  float64  — imaginary part
#   /Phi             (1, 1020)       float64  — azimuth  of each direction (°)
#   /Theta           (1, 1020)       float64  — elevation of each direction (°)
#   /SamplingFreq_Hz (1, 1)          float64  — measurement sample rate
#
# We use /IR directly (avoids reconstructing from split real/imag TF).
# ─────────────────────────────────────────────────────────────────────────────
ATF_KEYS = dict(
    ir      = 'IR',               # RIRs  (rir_len, n_dirs, n_mics)
    phi     = 'Phi',              # azimuth  angles (1, n_dirs) in degrees
    theta   = 'Theta',            # elevation angles (1, n_dirs) in degrees
    fs      = 'SamplingFreq_Hz',  # scalar measurement sample rate
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def stable_seed(*items) -> int:
    s = "_".join(map(str, items))
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def resample_to(a, src, dst):
    if src == dst:
        return a
    g = gcd(src, dst)
    return resample_poly(a.astype(np.float64), dst // g, src // g).astype(np.float32)


def lowpass(x, cutoff, fs=FS, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x).astype(np.float32)


def sample_rt60(profile_name: str, rng: np.random.Generator) -> float:
    lo, hi = ACOUSTIC_PROFILES[profile_name]
    return float(rng.uniform(lo, hi))


def az_el_to_unit(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.sin(az) * np.cos(el),
                     np.sin(el),
                     np.cos(az) * np.cos(el)], np.float64)


def room_centre():
    c = np.array(ROOM_DIM, np.float64) / 2.0
    c[1] = 1.6
    return c


def src_pos(az, el, dist, centre):
    return centre + az_el_to_unit(az, el) * dist


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-12


def scale_to_snr(signal, noise, snr_db):
    return noise * (rms(signal) / rms(noise) / 10.0 ** (snr_db / 20.0))


def load_utterances(root, speaker_id=None):
    utts = []
    for f in sorted(Path(root).rglob('*.flac')):
        if speaker_id is not None:
            parts = f.relative_to(root).parts
            if not parts or parts[0] != str(speaker_id):
                continue
        try:
            d, sr = sf.read(str(f), dtype='float32', always_2d=False)
            assert sr == LIBRI_FS
            utts.append(d)
        except Exception as e:
            warnings.warn(f"Skip {f}: {e}")
    return utts


def concat_to_dur(utts, dur_s, sr, rng, sil=(0.5, 2.0), cold_s=3.0):
    target = int(dur_s * sr)
    cold   = int(cold_s * sr)
    ap = [np.zeros(cold, np.float32)]
    vp = [np.zeros(cold, bool)]
    order = rng.permutation(len(utts))
    idx   = 0
    total = cold
    while total < target:
        g = int(np.clip(rng.exponential(0.4) * sr, 0.10 * sr, 0.80 * sr))
        ap.append(np.zeros(g, np.float32))
        vp.append(np.zeros(g, bool))
        u = utts[order[idx % len(order)]]
        idx += 1
        ap.append(u)
        vp.append(np.ones(len(u), bool))
        total += g + len(u)
    audio = np.concatenate(ap)[:target]
    vad   = np.concatenate(vp)[:target]
    r     = rms(audio[vad]) if vad.any() else 1.0
    return (audio * (0.1 / r)).astype(np.float32), vad


def build_turn_taking_audio(speaker_lists, duration_s, sr, rng, cold_s=3.0):
    total_len = int(duration_s * sr)
    cold_len  = int(cold_s * sr)
    audio = np.zeros(total_len, np.float32)
    vad   = np.zeros(total_len, bool)
    speaker_activity = []
    t = cold_len
    while t < total_len:
        spk = int(rng.integers(0, len(speaker_lists)))
        utt = rng.choice(speaker_lists[spk])
        pause   = int(rng.uniform(0.15, 0.8) * sr)
        end     = min(t + len(utt), total_len)
        seg_len = end - t
        audio[t:end] += utt[:seg_len]
        vad[t:end]    = True
        speaker_activity.append((t, end, spk))
        t = end + pause
    r = rms(audio[vad]) if vad.any() else 1.0
    audio = (audio * (0.1 / r)).astype(np.float32)
    return audio, vad, speaker_activity


def make_noise(kind, n, rng):
    if kind == 'white':
        x = rng.standard_normal(n).astype(np.float32)
    else:
        w = rng.standard_normal(n).astype(np.float64)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        psd  = 1.0 / np.sqrt(f)
        psd[0] = 0.0
        x = np.fft.irfft(np.fft.rfft(w) * psd, n=n).astype(np.float32)
    return (x / rms(x)).astype(np.float32)


# ── Noise envelopes ────────────────────────────────────────────────────────────

def env_ornstein_uhlenbeck(n, rng, tau_s=4.0, sigma_db=7.0):
    dt    = 1.0 / FS
    alpha = dt / tau_s
    sigma = sigma_db / 20.0 * np.log(10) * np.sqrt(2 * alpha)
    log_a = np.zeros(n, np.float64)
    noise = rng.standard_normal(n) * sigma
    for i in range(1, n):
        log_a[i] = log_a[i - 1] * (1 - alpha) + noise[i]
    env = 10.0 ** (log_a / 20.0)
    return (env / max(float(env.mean()), 1e-6)).astype(np.float32)


def env_slow_sine(n, rng, period_range=(8.0, 15.0), depth=0.65):
    period = rng.uniform(*period_range) * FS
    phase  = rng.uniform(0, 2 * np.pi)
    t      = np.arange(n, dtype=np.float64)
    env    = 1.0 - depth * 0.5 * (1.0 - np.cos(2 * np.pi * t / period + phase))
    return (env / max(float(env.mean()), 1e-6)).astype(np.float32)


def env_burst(n, rng, rate=0.35, dur_range=(0.4, 2.5), gain_db=12.0):
    env  = np.ones(n, np.float64)
    gain = 10.0 ** (gain_db / 20.0)
    pos  = 0
    while pos < n:
        gap  = int(rng.exponential(1.0 / rate) * FS)
        pos += gap
        if pos >= n:
            break
        dur  = int(rng.uniform(*dur_range) * FS)
        end  = min(pos + dur, n)
        ramp = min(int(0.05 * FS), (end - pos) // 4)
        env[pos:end] = gain
        if ramp > 0 and pos + ramp < n:
            env[pos:pos + ramp] = 1.0 + (gain - 1.0) * np.hanning(2 * ramp)[:ramp]
        if ramp > 0 and end - ramp >= 0:
            env[max(0, end - ramp):end] = (1.0 + (gain - 1.0)
                                           * np.hanning(2 * ramp)[ramp:])
        pos += dur
    return (env / max(float(env.mean()), 1e-6)).astype(np.float32)


def pick_envelope(dynamic_style, n, rng):
    if   dynamic_style == 'ou':    env = env_ornstein_uhlenbeck(n, rng)
    elif dynamic_style == 'sine':  env = env_slow_sine(n, rng)
    elif dynamic_style == 'burst': env = env_burst(n, rng)
    else:                          env = np.ones(n, np.float32)
    if not np.isfinite(env).all():
        raise RuntimeError(f"Non-finite envelope: {dynamic_style}")
    return env.astype(np.float32)


# ── EasyCom ATF loading and convolution ───────────────────────────────────────

# Module-level cache: populated on first call to load_easycom_atfs()
_ATF_CACHE: dict | None = None


def _print_h5_tree(h5file, indent=0):
    """Recursively print an HDF5 file/group structure for diagnostic purposes."""
    import h5py
    for key in h5file.keys():
        item = h5file[key]
        prefix = '  ' * indent + '/'
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}  shape={item.shape}  dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            _print_h5_tree(item, indent + 1)


def load_easycom_atfs(atf_path: str) -> dict | None:
    """
    Load EasyCom Array Transfer Functions from Device_ATFs.h5.

    Actual H5 layout (verified):
      /IR              (768, 1020, 6)  — RIRs, axes: (rir_len, n_dirs, n_mics)
      /Phi             (1,   1020)     — azimuth  of each direction in degrees
      /Theta           (1,   1020)     — elevation of each direction in degrees
      /SamplingFreq_Hz (1,   1)        — measurement sample rate in Hz

    Returns a dict with keys:
      atf_rir    : float32 (n_dirs, n_mics, rir_len) — ready for convolution
      azimuths   : float32 (n_dirs,) — azimuth  of each direction in degrees
      elevations : float32 (n_dirs,) — elevation of each direction in degrees
      meas_fs    : int    — measurement sample rate
    """
    global _ATF_CACHE
    if _ATF_CACHE is not None:
        return _ATF_CACHE

    try:
        import h5py
    except ImportError:
        warnings.warn("h5py not installed — ATF path ignored, falling back to ISM. "
                      "Install with: pip install h5py")
        return None

    try:
        with h5py.File(atf_path, 'r') as f:
            print("\n── EasyCom ATF file structure ──────────────────────────────────")
            _print_h5_tree(f)
            print("────────────────────────────────────────────────────────────────\n")

            # IR: (rir_len, n_dirs, n_mics)
            ir_raw  = f[ATF_KEYS['ir']][:]            # float64
            phi     = f[ATF_KEYS['phi']][:].ravel()   # (n_dirs,) azimuth  in degrees
            theta   = f[ATF_KEYS['theta']][:].ravel() # (n_dirs,) elevation in degrees
            meas_fs = int(round(float(f[ATF_KEYS['fs']][0, 0])))

    except KeyError as e:
        warnings.warn(
            f"ATF key not found: {e}. Check the printed H5 tree above and "
            f"edit ATF_KEYS at the top of the script. Falling back to ISM.")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load ATFs from {atf_path}: {e}. Falling back to ISM.")
        return None

    # ── Transpose to (n_dirs, n_mics, rir_len) ───────────────────────────
    # ir_raw shape: (rir_len=768, n_dirs=1020, n_mics=6)
    atf_rir = np.transpose(ir_raw, (1, 2, 0)).astype(np.float32)
    # atf_rir shape: (1020, 6, 768)

    # ── Resample each RIR to the generation FS if needed ─────────────────
    if meas_fs != FS:
        print(f"  ATF measured at {meas_fs} Hz — resampling to {FS} Hz …")
        g           = gcd(meas_fs, FS)
        up, down    = FS // g, meas_fs // g
        n_dirs, n_m = atf_rir.shape[:2]
        rir_rs_len  = int(round(atf_rir.shape[2] * FS / meas_fs))
        atf_rs      = np.zeros((n_dirs, n_m, rir_rs_len), np.float32)
        for d in range(n_dirs):
            for m in range(n_m):
                atf_rs[d, m] = resample_poly(
                    atf_rir[d, m].astype(np.float64), up, down).astype(np.float32)
        atf_rir = atf_rs
        print(f"  Resampled RIR length: {atf_rir.shape[2]} samples at {FS} Hz")

    # ── Normalise so peak amplitude across all directions/mics is 1.0 ────
    peak = float(np.abs(atf_rir).max())
    if peak > 0:
        atf_rir /= peak

    n_dirs = atf_rir.shape[0]
    print(f"  Loaded EasyCom ATFs: {n_dirs} directions × {N_MICS} mics  "
          f"RIR_len={atf_rir.shape[2]}  meas_fs={meas_fs} Hz")
    print(f"  Azimuth  range: {phi.min():.1f}° … {phi.max():.1f}°")
    print(f"  Elevation range: {theta.min():.1f}° … {theta.max():.1f}°")

    _ATF_CACHE = dict(
        atf_rir    = atf_rir,       # (n_dirs, n_mics, rir_len)
        azimuths   = phi.astype(np.float32),
        elevations = theta.astype(np.float32),
        meas_fs    = meas_fs,
    )
    return _ATF_CACHE


def _nearest_atf_rir(atfs: dict, az_deg: float, el_deg: float) -> np.ndarray:
    """
    Return the (n_mics, rir_len) RIR for the measured direction nearest to
    (az_deg, el_deg), using great-circle distance over the flat (1020,) list.
    """
    azs = atfs['azimuths']    # (n_dirs,)
    els = atfs['elevations']  # (n_dirs,)

    az0 = np.deg2rad(az_deg)
    el0 = np.deg2rad(el_deg)
    az_r = np.deg2rad(azs)
    el_r = np.deg2rad(els)

    cos_d = (np.sin(el0) * np.sin(el_r)
             + np.cos(el0) * np.cos(el_r) * np.cos(az0 - az_r))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    idx   = int(np.argmin(np.arccos(cos_d)))   # nearest direction index

    return atfs['atf_rir'][idx]   # (n_mics, rir_len)


def simulate_room_atf(sources, centre, T: int, atfs: dict) -> np.ndarray:
    """
    Simulate mic-array signals using EasyCom measured ATFs (ATF convolution only).

    Parameters
    ----------
    sources : list of (position_3d, audio_1d) tuples
    centre  : array-centre position (3,)
    T       : output length in samples
    atfs    : dict from load_easycom_atfs()

    Returns
    -------
    mic_sigs : float32 array (N_MICS, T)

    Note: No late-reverberation tail is added.  The EasyCom ATF already
    captures the true device acoustics (head shadow, mic directivity, near-field
    diffraction) for the measured environment.  Adding a synthetic Schroeder
    tail on top creates a model mismatch: the MVDR distortionless constraint
    passes target ⊗ ATF_0 exactly but does NOT recover the Schroeder tail at
    the same amplitude, causing a ~17 dB SI-SDR collapse at high SNR.

    reverberant_reference.wav = target ⊗ ATF_0 only, which exactly matches the
    MVDR output under the distortionless constraint.
    """
    out = np.zeros((N_MICS, T), dtype=np.float32)

    for pos, audio in sources:
        # ── Derive source direction relative to array centre ─────────────
        delta  = (pos - centre).astype(np.float64)
        dist   = float(np.linalg.norm(delta)) + 1e-9
        az_deg = float(np.degrees(np.arctan2(delta[0], delta[2])))
        el_deg = float(np.degrees(np.arcsin(np.clip(delta[1] / dist, -1.0, 1.0))))

        # ── ATF convolution (direct + device acoustics) ───────────────────
        rir_mics = _nearest_atf_rir(atfs, az_deg, el_deg)  # (n_mics, rir_len)
        rir_len  = rir_mics.shape[1]
        sig_len  = len(audio) + rir_len - 1

        # Distance attenuation (1/r)
        atten    = 1.0 / max(dist, 0.05)
        end      = min(sig_len, T)

        for m in range(N_MICS):
            conv = fftconvolve(audio.astype(np.float32), rir_mics[m])
            out[m, :end] += conv[:end] * atten

    return out[:, :T]


# ── Original pyroomacoustics ISM simulation (fallback) ───────────────────────

def simulate_room_direct(sources, centre, T):
    mic_positions = (centre[:, None] + MIC_POSITIONS.T).T
    out = np.zeros((N_MICS, T), dtype=np.float32)
    for sp, audio in sources:
        for m in range(N_MICS):
            dist  = float(np.linalg.norm(sp - mic_positions[m]))
            delay = int(round(dist / C_SOUND * FS))
            amp   = 1.0 / max(dist, 0.01)
            end   = min(T, len(audio) + delay)
            sl    = end - delay
            if sl > 0 and delay < T:
                out[m, delay:end] += amp * audio[:sl].astype(np.float32)
    return out


def simulate_room(sources, centre, rt60, T):
    """
    Simulate mic-array signals using pyroomacoustics ISM.

    Returns
    -------
    sigs        : float32 (N_MICS, T)  — simulated mic signals
    used_rt60   : float               — RT60 actually used in the simulation.
                                        0.0 when the anechoic direct-path
                                        fallback is taken (i.e. pra not
                                        installed, or inverse_sabine fails).
                                        Callers MUST use this value as
                                        effective_rt60 rather than the
                                        originally sampled rt60 — they differ
                                        whenever the fallback fires.

    FIX (v6): Previously simulate_room() returned only the mic signals, so
    callers could not detect whether the anechoic fallback had been taken.
    generate_one() then recorded the originally sampled RT60 in metadata.json
    even when the audio had no reverberation.  This caused compute_vad_stft()
    to add a spurious hangover extension and eval_synthetic.py to apply
    incorrect reference logic for those clips.
    """
    if rt60 <= 0.0:
        return simulate_room_direct(sources, centre, T), 0.0

    try:
        import pyroomacoustics as pra
    except ImportError:
        sys.exit("pip install pyroomacoustics")

    try:
        e_abs, max_ord = pra.inverse_sabine(rt60, ROOM_DIM)
    except ValueError:
        warnings.warn(
            f"inverse_sabine failed for rt60={rt60:.2f}s — anechoic fallback. "
            f"effective_rt60 will be set to 0.0 in metadata.")
        # FIX: return used_rt60=0.0 so callers record the correct value.
        return simulate_room_direct(sources, centre, T), 0.0

    room     = pra.ShoeBox(ROOM_DIM, fs=FS,
                           materials=pra.Material(e_abs), max_order=max_ord)
    mic_locs = (centre[:, None] + MIC_POSITIONS.T).astype(np.float64)
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=FS))
    for pos, audio in sources:
        room.add_source(pos.tolist(), signal=audio.astype(np.float64))
    room.simulate()
    sigs = room.mic_array.signals.astype(np.float32)
    if sigs.shape[1] >= T:
        return sigs[:, :T], rt60
    return np.pad(sigs, ((0, 0), (0, T - sigs.shape[1]))), rt60


def simulate(sources, centre, rt60, T, atfs=None):
    """
    Unified simulation entry point.

    ATF path (atfs is not None):
        Calls simulate_room_atf() — ATF convolution only, no Schroeder tail.
        rt60 is ignored; the EasyCom ATF captures the measured room acoustics.
        Returns (mic_sigs, 0.0) to match the ISM return signature.

    ISM path (atfs is None):
        Calls simulate_room() — pyroomacoustics ISM with the given rt60.
        Returns (mic_sigs, used_rt60) where used_rt60 may be 0.0 if the
        anechoic fallback was taken.

    Returns
    -------
    mic_sigs  : float32 (N_MICS, T)
    used_rt60 : float  — RT60 actually present in the audio (0.0 for ATF path
                         and for any anechoic fallback in the ISM path).
    """
    if atfs is not None:
        return simulate_room_atf(sources, centre, T, atfs), 0.0
    return simulate_room(sources, centre, rt60, T)


# ── Gaze and VAD ───────────────────────────────────────────────────────────────

def compute_gaze_stft(tgt_pos, centre, T, jitter_deg, rng,
                      gaze_model='realistic',
                      saccade_rate=3.0,
                      tremor_std=0.5):
    direction = (tgt_pos - centre).astype(np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    az0 = np.degrees(np.arctan2(direction[0], direction[2]))
    el0 = np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0)))

    n_vid = (T + AUDIO_HOP_VIDEO - 1) // AUDIO_HOP_VIDEO
    dt    = 1.0 / VIDEO_FPS

    if gaze_model == 'oracle' or jitter_deg == 0:
        az = np.full(n_vid, az0)
        el = np.full(n_vid, el0)
    elif gaze_model == 'simple':
        az = az0 + rng.normal(0, jitter_deg,       n_vid)
        el = el0 + rng.normal(0, jitter_deg * 0.5, n_vid)
    else:  # 'realistic'
        tau_s     = 2.0
        sigma     = float(jitter_deg)
        decay     = np.exp(-dt / tau_s)
        noise_std = sigma * np.sqrt(1.0 - decay ** 2)
        az_drift  = np.zeros(n_vid)
        el_drift  = np.zeros(n_vid)
        for i in range(1, n_vid):
            az_drift[i] = decay * az_drift[i-1] + noise_std * rng.standard_normal()
            el_drift[i] = decay * el_drift[i-1] + noise_std * 0.5 * rng.standard_normal()
        saccade_prob = saccade_rate * dt
        az_sacc = az_el = np.zeros(n_vid)
        az_sacc = np.zeros(n_vid)
        el_sacc = np.zeros(n_vid)
        sacc_az = sacc_el = 0.0
        sacc_decay = 0.0
        for i in range(n_vid):
            if rng.random() < saccade_prob:
                amplitude  = float(rng.uniform(5.0, 25.0))
                angle      = float(rng.uniform(0, 2 * np.pi))
                sacc_az    = amplitude * np.cos(angle)
                sacc_el    = amplitude * np.sin(angle) * 0.4
                tau_sacc   = float(rng.uniform(0.10, 0.20))
                sacc_decay = np.exp(-dt / tau_sacc)
            az_sacc[i] = sacc_az
            el_sacc[i] = sacc_el
            sacc_az   *= sacc_decay
            sacc_el   *= sacc_decay
        az_tremor = rng.normal(0, tremor_std,       n_vid)
        el_tremor = rng.normal(0, tremor_std * 0.6, n_vid)
        az = az0 + az_drift + az_sacc + az_tremor
        el = el0 + el_drift + el_sacc + el_tremor

    el    = np.clip(el, -89.0, 89.0)
    gaze  = np.stack([
        np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
        np.sin(np.deg2rad(el)),
        np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
    ], axis=1).astype(np.float32)
    norms = np.linalg.norm(gaze, axis=1, keepdims=True)
    gaze /= np.maximum(norms, 1e-9)

    n_stft  = (T + HOP - 1) // HOP
    centres = np.arange(n_stft) * HOP + HOP // 2
    vid_idx = np.clip(centres // AUDIO_HOP_VIDEO, 0, n_vid - 1)
    return gaze[vid_idx]


def compute_vad_stft(vad_samples: np.ndarray, T: int,
                     rt60: float = 0.0, sr: int = FS) -> np.ndarray:
    """
    Downsample sample-rate VAD to STFT-frame rate, with optional hangover.

    For ATF clips effective_rt60=0.0 is passed (no reverb tail, so no hangover
    needed).  For ISM clips the hangover extends into the reverberant decay.
    """
    vad = np.asarray(vad_samples, dtype=bool)[:T]
    if rt60 > 0.0:
        hangover_samples = int(rt60 * 3 * sr)
        if hangover_samples > 1:
            struct = np.ones(hangover_samples, dtype=bool)
            vad    = binary_dilation(vad, structure=struct).astype(bool)
            vad    = vad[:T]
    n_vid = (T + AUDIO_HOP_VIDEO - 1) // AUDIO_HOP_VIDEO
    vv    = np.array([vad[min(i * AUDIO_HOP_VIDEO, T - 1)]
                      for i in range(n_vid)], dtype=np.float32)
    n_stft  = (T + HOP - 1) // HOP
    centres = np.arange(n_stft) * HOP + HOP // 2
    vid_idx = np.clip(centres // AUDIO_HOP_VIDEO, 0, n_vid - 1)
    return vv[vid_idx] > 0.5


# ── Scenario builder ───────────────────────────────────────────────────────────

def base_scenario(name: str) -> str:
    return name.replace('_dynamic', '')


def is_dynamic(name: str) -> bool:
    return name.endswith('_dynamic')


def _sample_separated_azimuths(rng, n: int, existing=(), min_sep=25.0):
    result = list(existing)
    max_tries = 1000
    while len(result) < len(existing) + n:
        for _ in range(max_tries):
            cand = float(rng.uniform(-180.0, 180.0))
            if all(abs(cand - a) > min_sep for a in result):
                result.append(cand)
                break
        else:
            result.append(float((len(result) * 360.0 / (len(existing) + n)) - 180.0))
    return result[len(existing):]


def build_scenario(scenario_name: str,
                   target_utts,
                   intfr_list,
                   duration_s: float,
                   snr_db: float,
                   rng: np.random.Generator,
                   conversation_mode: str = 'single') -> dict:
    T    = int(duration_s * FS)
    arc  = room_centre()
    dyn  = is_dynamic(scenario_name)
    base = base_scenario(scenario_name)

    if conversation_mode == 'single':
        t16, v16 = concat_to_dur(target_utts, duration_s, LIBRI_FS, rng)
    else:
        n_speakers   = 2 if conversation_mode == 'two_speaker' else 3
        speaker_lists = [target_utts] + [intfr_list[k % len(intfr_list)]
                                          for k in range(n_speakers - 1)]
        t16, v16, _ = build_turn_taking_audio(
            speaker_lists, duration_s, LIBRI_FS, rng)

    tgt_audio = resample_to(t16, LIBRI_FS, FS)[:T]
    tgt_vad   = (resample_to(v16.astype(np.float32), LIBRI_FS, FS)[:T] > 0.5)

    target_az = float(rng.uniform(*TARGET_AZ_RANGE))
    tgt_pos   = src_pos(target_az, TARGET_EL, TARGET_DIST, arc)
    sources   = [(tgt_pos, tgt_audio)]

    add_noise       = None
    noise_envelope  = np.ones(T, np.float32)
    intfr_dist      = DISTANCES['mid']
    interferer_az   = None
    target_rt60     = sample_rt60('target', rng)

    def get_intfr(idx, sil=(0.05, 0.5)):
        utts = intfr_list[idx % len(intfr_list)]
        a16, _ = concat_to_dur(utts, duration_s, LIBRI_FS, rng,
                            sil=sil, cold_s=0.0)
        return resample_to(a16, LIBRI_FS, FS)[:T]

    def get_env():
        style = _DYN_STYLE.get(base, 'ou')
        return pick_envelope(style, T, rng) if dyn else np.ones(T, np.float32)

    if base in ('white_noise', 'pink_noise'):
        kind      = 'white' if base == 'white_noise' else 'pink'
        env       = get_env()
        raw       = make_noise(kind, T, rng)
        add_noise = scale_to_snr(tgt_audio, raw, snr_db) * env
        noise_envelope = env

    elif base in ('directional_near', 'directional_mid', 'directional_far'):
        label      = base.split('_')[1]
        dist       = DISTANCES[label]
        intfr_dist = dist
        env        = get_env()
        intfr_raw  = get_intfr(0)
        if   label == 'far':  intfr_raw = lowpass(intfr_raw, 3500)
        elif label == 'mid':  intfr_raw = lowpass(intfr_raw, 6000)
        intfr_scaled = scale_to_snr(tgt_audio, intfr_raw, snr_db) * env
        [interferer_az] = _sample_separated_azimuths(
            rng, 1, existing=[target_az], min_sep=INTERFERER_AZ_MIN_SEP)
        interferer_az = float(np.clip(interferer_az,
                                      INTERFERER_AZ_RANGE[0],
                                      INTERFERER_AZ_RANGE[1]))
        ipos = src_pos(interferer_az, 0.0, dist, arc)
        sources.append((ipos, intfr_scaled))
        noise_envelope = env

    elif base == 'babble':
        babble_azs = _sample_separated_azimuths(rng, 4, existing=[], min_sep=25.0)
        for i, az in enumerate(babble_azs):
            env  = get_env()
            b48  = get_intfr(i, sil=(0.1, 0.8))
            bs   = scale_to_snr(tgt_audio, b48, snr_db + 6) * env
            bpos = src_pos(az, 0.0, 1.8, arc)
            sources.append((bpos, bs))
        noise_envelope = env
        interferer_az  = babble_azs[0]

    elif base == 'cocktail':
        [intfr_az] = _sample_separated_azimuths(
            rng, 1, existing=[target_az], min_sep=INTERFERER_AZ_MIN_SEP)
        env_i = get_env()
        ia    = get_intfr(0)
        ip    = src_pos(intfr_az, 0.0, DISTANCES['mid'], arc)
        sources.append((ip, scale_to_snr(tgt_audio, ia, snr_db - 3) * env_i))
        babble_azs = _sample_separated_azimuths(
            rng, 3, existing=[target_az, intfr_az], min_sep=25.0)
        for i, az in enumerate(babble_azs):
            env_b = get_env()
            b48   = get_intfr(i + 1, sil=(0.1, 0.8))
            bpos  = src_pos(az, 0.0, 1.8, arc)
            sources.append((bpos, scale_to_snr(tgt_audio, b48, snr_db + 5) * env_b))
        noise_envelope = env_i
        interferer_az  = float(intfr_az)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")

    return dict(
        target_audio      = tgt_audio,
        target_vad        = tgt_vad,
        sources           = sources,
        add_noise         = add_noise,
        noise_envelope    = noise_envelope,
        interferer_dist   = intfr_dist,
        target_azimuth    = float(target_az),
        interferer_azimuth= float(interferer_az) if interferer_az is not None else None,
        target_rt60       = target_rt60,
    )


# ── Main generation function ───────────────────────────────────────────────────

def generate_one(out_dir: Path,
                 scenario: str,
                 snr_db: float,
                 rep: int,
                 target_utts,
                 intfr_list,
                 duration_s: float,
                 rt60: float,
                 jitter_deg: float,
                 conversation_mode: str = 'single',
                 gaze_model: str = 'realistic',
                 saccade_rate: float = 3.0,
                 tremor_std: float = 0.5,
                 atfs: dict | None = None) -> dict:

    seed = stable_seed(scenario, snr_db, rep, conversation_mode)
    rng  = np.random.default_rng(seed)

    sc  = build_scenario(scenario, target_utts, intfr_list,
                         duration_s, snr_db, rng, conversation_mode)
    T   = len(sc['target_audio'])
    arc = room_centre()

    target_az     = sc['target_azimuth']
    interferer_az = sc['interferer_azimuth'] if sc['interferer_azimuth'] is not None else 0

    # 'atf'  → ATF convolution only, no Schroeder tail.
    # 'ism'  → pyroomacoustics ISM fallback.
    sim_mode = 'atf' if atfs is not None else 'ism'

    name = (f"{scenario}"
            f"_taz{int(target_az):+04d}"
            f"_iaz{int(interferer_az):+04d}"
            f"_snr{int(snr_db):+d}"
            f"_rep{rep:02d}")
    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  {name} [{sim_mode}] ...", end='', flush=True)

    base = base_scenario(scenario)

    try:
        all_sigs         = []
        per_source_rt60s = []

        for idx, (pos, audio) in enumerate(sc['sources']):
            if idx == 0:
                profile = 'target'
            elif base.startswith('directional'):
                profile = {'near': 'near_interferer',
                            'mid':  'mid_interferer',
                            'far':  'far_interferer'}[base.split('_')[1]]
            elif base in ('babble', 'cocktail'):
                profile = 'babble'
            else:
                profile = 'diffuse_noise'

            src_rt60 = sample_rt60(profile, rng)
            per_source_rt60s.append(src_rt60)

            # simulate() now returns (mic_sigs, used_rt60).
            # used_rt60 may be 0.0 even when src_rt60 > 0 if inverse_sabine
            # fell back to the anechoic path (FIX 2).
            sig, _ = simulate([(pos, audio)], arc, src_rt60, T, atfs=atfs)
            all_sigs.append(sig)

        mic_sigs = np.sum(all_sigs, axis=0)   # (N_MICS, T)

        # simulate() for the reverberant reference: unpack used_rt60 to get the
        # RT60 actually present in the audio (may differ from the sampled value
        # if inverse_sabine fell back to anechoic — FIX 2).
        reverb_ref_sigs, used_rt60_ref = simulate(
            sc['sources'][:1], arc, per_source_rt60s[0], T, atfs=atfs)
        reverb_ref = reverb_ref_sigs[0]   # channel 0 of reverberant target

        # For ATF clips effective_rt60 is always 0.0 (no reverb tail).
        # For ISM clips use used_rt60_ref — the RT60 actually in the audio —
        # not the originally sampled per_source_rt60s[0] which may not have
        # been honoured if inverse_sabine failed (FIX 2).
        effective_rt60 = used_rt60_ref  # already 0.0 for ATF path

    except Exception as e:
        print(f"\n    [WARN] room sim failed ({e}) — anechoic fallback")
        mic_sigs       = np.tile(sc['target_audio'][None, :], (N_MICS, 1))
        reverb_ref     = sc['target_audio'].copy()
        effective_rt60 = 0.0
        for _, a in sc['sources'][1:]:
            mic_sigs += np.asarray(a, np.float32)[:T][None, :]

    if sc['add_noise'] is None and len(sc['sources']) > 1 and len(all_sigs) > 1:
        tgt_only_sigs   = all_sigs[0]
        intfr_only_sigs = mic_sigs - tgt_only_sigs

        vad_tmp = compute_vad_stft(sc['target_vad'], T, rt60=0.0)
        vad_s   = np.repeat(vad_tmp, HOP)[:T]
        if vad_s.sum() > 512:
            tgt_rms_post   = rms(tgt_only_sigs[0, :T][vad_s.astype(bool)])
            intfr_rms_post = rms(intfr_only_sigs[0, :T])
            actual_snr_db  = 20 * np.log10(tgt_rms_post / (intfr_rms_post + 1e-12))
            desired_scale  = 10 ** ((actual_snr_db - snr_db) / 20.0)
            mic_sigs       = (tgt_only_sigs[:, :T]
                              + intfr_only_sigs[:, :T] * desired_scale)

    if sc['add_noise'] is not None:
        kind       = 'pink' if scenario.startswith('pink') else 'white'
        tgt_rms    = float(np.sqrt(np.mean(sc['add_noise'].astype(np.float64) ** 2)))
        envelope   = sc['noise_envelope'][:T]
        indep_noise = np.zeros((N_MICS, T), np.float32)
        for m in range(N_MICS):
            raw_m    = make_noise(kind, T, rng)
            chan_rms = float(np.sqrt(np.mean(raw_m.astype(np.float64) ** 2))) + 1e-12
            indep_noise[m] = raw_m * (tgt_rms / chan_rms) * envelope
        mic_sigs = mic_sigs[:, :T] + indep_noise

    mic_sigs = mic_sigs[:, :T]
    peak     = np.abs(mic_sigs).max()
    if peak > 0.99:
        scale      = 0.99 / peak
        mic_sigs   *= scale
        reverb_ref *= scale

    tgt_pos    = src_pos(sc['target_azimuth'], TARGET_EL, TARGET_DIST, arc)
    gaze_stft  = compute_gaze_stft(tgt_pos, arc, T, jitter_deg, rng,
                                   gaze_model=gaze_model,
                                   saccade_rate=saccade_rate,
                                   tremor_std=tremor_std)
    # effective_rt60=0.0 for ATF clips → no hangover extension (correct: no tail).
    # For ISM clips, effective_rt60=used_rt60_ref which is already 0.0 if
    # inverse_sabine fell back to anechoic (FIX 2).
    vad_stft = compute_vad_stft(sc['target_vad'], T, rt60=effective_rt60)
    env      = sc['noise_envelope'][:T]

    vad_s = np.repeat(vad_stft, HOP)[:T].astype(bool)
    if vad_s.sum() > 512:
        tgt_rms_log   = rms(reverb_ref[:T][vad_s])
        noise_rms_log = rms(mic_sigs[0] - reverb_ref[:T])
        actual_snr    = 20 * np.log10(tgt_rms_log / (noise_rms_log + 1e-12))
    else:
        actual_snr = float('nan')

    # ── Write output files ──────────────────────────────────────────────────
    sf.write(str(dest / 'array_audio.wav'),          mic_sigs.T,     FS, subtype='FLOAT')

    # reverberant_reference.wav — channel 0 of the reverberant target-only
    # simulation.  This is the correct SI-SDR reference for BOTH the ATF and
    # ISM paths: it is exactly what the MVDR distortionless constraint outputs.
    sf.write(str(dest / 'reverberant_reference.wav'), reverb_ref[:T], FS, subtype='FLOAT')

    # close_mic.wav — semantics differ by simulation mode:
    #
    #   ATF path  (sim_mode == 'atf'):
    #     Dry speech (sc['target_audio']) is stored.  This is required by the
    #     legacy 'atf+reverb' re-convolution path in eval_synthetic.py which
    #     computes close_mic ⊗ ATF_0 to recover the correct reference.  That
    #     operation only makes sense on DRY speech.
    #
    #   ISM path  (sim_mode == 'ism'):
    #     reverb_ref[:T] (channel 0 of reverberant target) is stored.
    #     eval_synthetic.py uses reverberant_reference.wav as the primary
    #     reference for ISM clips and only falls back to close_mic.wav if that
    #     file is missing.  Storing dry speech here was a latent bug: if the
    #     fallback ever fired, eval_synthetic.py would compute against dry
    #     speech and produce a ~0 dB SI-SDR ceiling (FIX 1).
    #
    # metadata['close_mic_is_dry'] records which content is present so
    # eval_synthetic.py can handle both cases unambiguously.
    if sim_mode == 'ism':
        sf.write(str(dest / 'close_mic.wav'), reverb_ref[:T],        FS, subtype='FLOAT')
        close_mic_is_dry = False
    else:
        sf.write(str(dest / 'close_mic.wav'), sc['target_audio'],     FS, subtype='FLOAT')
        close_mic_is_dry = True

    np.save(str(dest / 'gaze.npy'),           gaze_stft.astype(np.float32))
    np.save(str(dest / 'vad.npy'),            vad_stft.astype(bool))
    np.save(str(dest / 'noise_envelope.npy'), env.astype(np.float32))

    meta = {
        'scenario':              scenario,
        'base_scenario':         base_scenario(scenario),
        'is_dynamic':            is_dynamic(scenario),
        'snr_db':                float(snr_db),
        'rep':                   rep,
        'conversation_mode':     conversation_mode,
        'duration_s':            duration_s,
        'room_dim_m':            ROOM_DIM,
        'target_azimuth_deg':    sc['target_azimuth'],
        'target_elevation_deg':  TARGET_EL,
        'target_distance_m':     TARGET_DIST,
        'interferer_azimuth_deg': sc['interferer_azimuth'],
        'interferer_distance_m': sc['interferer_dist'],
        'target_interferer_sep_deg': (
            abs(sc['target_azimuth'] - sc['interferer_azimuth'])
            if sc['interferer_azimuth'] is not None else None),
        'effective_rt60_s':      round(effective_rt60, 3),
        'n_mics':                N_MICS,
        'sample_rate':           FS,
        'n_samples':             T,
        'n_stft_frames':         int(len(vad_stft)),
        'target_speech_frac':    float(sc['target_vad'].mean()),
        'gaze_jitter_deg':       jitter_deg,
        'gaze_model':            gaze_model,
        'gaze_saccade_rate_hz':  saccade_rate,
        'gaze_tremor_std_deg':   tremor_std,
        'actual_snr_db':         round(actual_snr, 2) if np.isfinite(actual_snr) else None,
        'envelope_type':         (_DYN_STYLE.get(base_scenario(scenario), 'static')
                                  if is_dynamic(scenario) else 'static'),
        'noise_env_mean':        float(env.mean()),
        'noise_env_std':         float(env.std()),
        'noise_env_peak_db':     float(20 * np.log10(env.max())),
        'interferer_az_min_sep_deg': INTERFERER_AZ_MIN_SEP,
        # FIX 1: 'simulation_mode' correctly set to 'atf' or 'ism'.
        # eval_synthetic.py branches on this value to select the reference.
        'simulation_mode':       sim_mode,          # 'atf' or 'ism'
        # FIX 1: signals whether close_mic.wav contains dry speech (ATF path)
        # or the reverberant channel-0 reference (ISM path).
        'close_mic_is_dry':      close_mic_is_dry,
        'atf_path':              str(atfs.get('_path', 'N/A')) if atfs else None,
    }
    with open(dest / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    noise_frames = int((~vad_stft).sum())
    warn = f"  [WARN: only {noise_frames} noise frames!]" if noise_frames < 500 else ""
    print(f" done  (speech={sc['target_vad'].mean()*100:.0f}%  "
          f"noise_stft={noise_frames}  snr_act={actual_snr:+.1f}dB  "
          f"env_std={env.std():.3f}  rt60={effective_rt60:.2f}s){warn}")
    return {'name': name, 'path': str(dest), **meta}


# ── CLI ────────────────────────────────────────────────────────────────────────
#"/esat/betelgeuse1/users/kvanhall/Calibration/Array Transfer Functions/Device_ATFs.h5"
def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Generate synthetic multi-channel dataset for beamformer evaluation')
    p.add_argument('--librispeech',
                   default='/esat/betelgeuse1/users/kvanhall/LibriSpeech/test-clean',
                   metavar='DIR')
    p.add_argument('--out',
                   default='/esat/betelgeuse1/users/kvanhall/Demonstration/synthetic_dataset',
                   metavar='DIR')
    p.add_argument('--atf-path',
                   default=None,
                   metavar='H5',
                   help='Path to EasyCom Device_ATFs.h5.  If supplied, ATF-based '
                        'simulation is used (ATF convolution only, no Schroeder tail). '
                        'If omitted, falls back to pyroomacoustics ISM.')
    p.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS,
                   choices=ALL_SCENARIOS, metavar='S')
    p.add_argument('--snrs', nargs='+', type=float,
                   default=[-20, -15, -10, -5, 0, 5, 10, 20], metavar='DB')
    p.add_argument('--scenario-snrs', nargs='*',
                   default=['directional_mid=-5,-4,-3,-2,-1,0,1,2,3,4,5'],
                   metavar='SCENARIO=DB,DB,...',
                   help='Per-scenario SNR grid override, e.g. '
                        '"directional_mid=-5,-4,-3,-2,-1,0,1,2,3,4,5". '
                        'Scenarios not listed use --snrs.  Pass with no '
                        'value to clear all overrides.')
    p.add_argument('--duration', type=float, default=60.0, metavar='S')
    p.add_argument('--rt60', type=float, default=0.15, metavar='S',
                   help='RT60 (seconds) for the ISM fallback path only.  '
                        'Has no effect when --atf-path is supplied (ATF clips '
                        'use no synthetic reverberation; effective_rt60=0).')
    p.add_argument('--n-reps', type=int, default=1)
    p.add_argument('--jitter-deg', type=float, default=2.0)
    p.add_argument('--gaze-model', default='realistic',
                   choices=['oracle', 'simple', 'realistic'])
    p.add_argument('--saccade-rate', type=float, default=3.0)
    p.add_argument('--tremor-std', type=float, default=0.5)
    p.add_argument('--target-speaker', type=int, default=None, metavar='ID')
    p.add_argument('--n-interferer-speakers', type=int, default=5, metavar='N')
    p.add_argument('--conversation-modes', nargs='+', default=['single'],
                   choices=CONVERSATION_MODES)
    args = p.parse_args(argv)

    # ── Parse per-scenario SNR overrides ──────────────────────────────────
    scenario_snrs = {}
    for spec in args.scenario_snrs:
        try:
            scen, vals = spec.split('=', 1)
            if scen not in ALL_SCENARIOS:
                sys.exit(f"--scenario-snrs: unknown scenario '{scen}'")
            scenario_snrs[scen] = [float(v) for v in vals.split(',')]
        except ValueError:
            sys.exit(f"--scenario-snrs: bad spec '{spec}' "
                     f"(expected SCENARIO=DB,DB,...)")
    def snrs_for(scenario):
        return scenario_snrs.get(scenario, args.snrs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load EasyCom ATFs if path provided ────────────────────────────────
    atfs = None
    if args.atf_path:
        print(f"\nLoading EasyCom ATFs from: {args.atf_path}")
        atfs = load_easycom_atfs(args.atf_path)
        if atfs is None:
            print("  [WARN] ATF loading failed — using pyroomacoustics ISM fallback")
        else:
            print(f"  ATF simulation active  "
                  f"(n_dirs={len(atfs['azimuths'])}  no Schroeder tail)")
    else:
        print("\n[INFO] --atf-path not supplied — using pyroomacoustics ISM simulation")
        print("       For more realistic spatial cues, run with:")
        print("       --atf-path /esat/betelgeuse1/users/kvanhall/Calibration/"
              '"Array Transfer Functions"/Device_ATFs.h5\n')

    print(f"\nLoading LibriSpeech from: {args.librispeech}")
    libri     = Path(args.librispeech)
    spk_dirs  = sorted(d for d in libri.iterdir() if d.is_dir() and d.name.isdigit())
    if not spk_dirs:
        sys.exit(f"No speaker dirs under {libri}")
    print(f"  {len(spk_dirs)} speakers available")

    tgt_dir  = libri / str(args.target_speaker) if args.target_speaker else spk_dirs[0]
    tgt_utts = load_utterances(str(libri), int(tgt_dir.name))
    print(f"  Target speaker {tgt_dir.name}: {len(tgt_utts)} utterances")

    intfr_list = []
    for d in spk_dirs[1:args.n_interferer_speakers + 1]:
        u = load_utterances(str(libri), int(d.name))
        if u:
            intfr_list.append(u)
            print(f"  Interferer {d.name}: {len(u)} utts")
    if not intfr_list:
        intfr_list = [tgt_utts] * 5

    n_total = (sum(len(snrs_for(s)) for s in args.scenarios)
               * args.n_reps * len(args.conversation_modes))
    print(f"\nGenerating {n_total} clips  "
          f"({len(args.scenarios)} scenarios, per-scenario SNR grids × "
          f"{args.n_reps} reps × {len(args.conversation_modes)} modes)")
    for scen, grid in scenario_snrs.items():
        if scen in args.scenarios:
            print(f"  SNR override {scen}: {[f'{s:+g}' for s in grid]}")
    print(f"  Interferer min angular separation: {INTERFERER_AZ_MIN_SEP}°")
    print(f"  Simulation mode: {'ATF (anechoic + device acoustics, no tail)' if atfs else 'pyroomacoustics ISM'}\n")

    all_meta = []
    for conv_mode in args.conversation_modes:
        for scenario in args.scenarios:
            for snr in snrs_for(scenario):
                for rep in range(args.n_reps):
                    try:
                        m = generate_one(
                            out_dir, scenario, snr, rep,
                            tgt_utts, intfr_list,
                            args.duration, args.rt60, args.jitter_deg,
                            conv_mode,
                            args.gaze_model, args.saccade_rate, args.tremor_std,
                            atfs=atfs)
                        all_meta.append(m)
                    except Exception as e:
                        import traceback
                        print(f"\n  ERROR {scenario} snr={snr} rep={rep}: {e}")
                        traceback.print_exc()

    manifest = out_dir / 'manifest.json'
    with open(manifest, 'w') as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n{len(all_meta)} clips written to {out_dir}/")
    print(f"Evaluate with:  python eval_synthetic.py --dataset {out_dir} --plot")


if __name__ == '__main__':
    main()