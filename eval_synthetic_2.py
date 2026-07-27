"""
eval_synthetic_2.py — Evaluate pipeline on synthetic dataset + generate summary plots
====================================================================================


"""

from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import sys
import warnings
from math import gcd as _gcd
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_2 import AriaDenoisingPipeline
from doa_2 import ATFSteering
from stft import F_WIN, HOP, FS
from generate_synthetic_dataset import MIC_POSITIONS as SYNTH_MIC_POSITIONS

SYNTH_MIC_POSITIONS_2D = SYNTH_MIC_POSITIONS[:, [0, 2]].astype(np.float32)
AUDIO_HOP_VIDEO = FS // 20

# ── Alpha used for all beamformed modes ───────────────────────────────────────
# α=0.97 → TC ≈ 5 s at 187.5 fps.  Reachable within ~900 noise frames.
# Change to 0.995 only if your dataset has > 20% silence.
BEAMFORMER_ALPHA = 0.97


# ── ATF-based reference computation ──────────────────────────────────────────

def _compute_atf_ref_on_the_fly(close_mic: np.ndarray,
                                 target_az_deg: float,
                                 atf_path: str,
                                 T: int) -> np.ndarray:
    """
    Compute close_mic ⊗ ATF_0(target_az) for OLD datasets ('atf+reverb').
    """
    from scipy.signal import fftconvolve
    import h5py

    with h5py.File(atf_path, 'r') as f:
        ir      = f['IR'][:]
        phi     = f['Phi'][:].ravel()
        meas_fs = int(round(float(f['SamplingFreq_Hz'][0, 0])))

    idx   = int(np.argmin(np.abs(phi - float(target_az_deg))))
    atf_0 = ir[:, idx, 0].astype(np.float64)

    if meas_fs != FS:
        from scipy.signal import resample_poly
        g     = _gcd(meas_fs, FS)
        atf_0 = resample_poly(atf_0, FS // g, meas_fs // g)

    ref = fftconvolve(close_mic.astype(np.float64), atf_0)
    return ref[:T].astype(np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _resample_16k(x):
    try:
        import resampy
        return resampy.resample(x.astype(np.float64), FS, 16_000).astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(FS, 16_000)
        return resample_poly(x, 16_000 // g, FS // g).astype(np.float32)


def _load_clip_mic_positions(clip_path, meta):
    mic_positions_file = clip_path / 'mic_positions.npy'
    if mic_positions_file.exists():
        return np.load(str(mic_positions_file)).astype(np.float32)
    if 'mic_positions' in meta:
        return np.asarray(meta['mic_positions'], dtype=np.float32)
    return SYNTH_MIC_POSITIONS_2D


def si_sdr(est, ref, max_lag=3 * 512):
    """
    Scale-Invariant SDR with FFT cross-correlation lag search at 48 kHz.
    Runs at native 48 kHz — NOT resampled to 16 kHz.
    """
    r = (ref - ref.mean()).astype(np.float64)
    e = (est - est.mean()).astype(np.float64)
    n = min(len(r), len(e))
    r, e = r[:n], e[:n]
    if np.dot(r, r) < 1e-9:
        return float('-inf')

    def _compute(ev, rv):
        nn = min(len(ev), len(rv))
        if nn < 512:
            return float('-inf')
        ev, rv = ev[:nn], rv[:nn]
        a     = np.dot(ev, rv) / (np.dot(rv, rv) + 1e-9)
        proj  = a * rv
        noise = ev - proj
        den   = np.dot(noise, noise)
        if den < 1e-30:
            return float('-inf')
        return float(10 * np.log10((np.dot(proj, proj) + 1e-9) / den))

    if max_lag == 0:
        return _compute(e, r)

    nfft     = int(2 ** np.ceil(np.log2(2 * n)))
    xcorr    = np.fft.irfft(
        np.fft.rfft(e, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
    cands    = np.array([xcorr[lag % nfft]
                         for lag in range(-max_lag, max_lag + 1)])
    best_lag = int(np.argmax(np.abs(cands))) - max_lag

    if best_lag >= 0:
        return _compute(e[best_lag:], r[:n - best_lag])
    else:
        return _compute(e[:n + best_lag], r[-best_lag:])


def pesq_score(e, r):
    try:
        from pesq import pesq
        n = min(len(e), len(r))
        e2, r2 = e[:n], r[:n]
        if max(abs(e2).max(), abs(r2).max()) < 1e-6:
            return float('nan')
        return float(pesq(16_000, r2, e2, 'wb'))
    except Exception as ex:
        warnings.warn(f"PESQ: {ex}")
        return float('nan')


def stoi_score(e, r):
    try:
        from pystoi import stoi
        n = min(len(e), len(r))
        return float(stoi(r[:n], e[:n], 16_000, extended=False))
    except Exception as ex:
        warnings.warn(f"STOI: {ex}")
        return float('nan')


def compute_metrics(enhanced, reference, vad_samples=None, max_lag=3 * 512):
    """SI-SDR at 48 kHz (lag-corrected) + PESQ/STOI at 16 kHz."""
    n    = min(len(enhanced), len(reference))
    e_f  = enhanced[:n].astype(np.float64)
    r_f  = reference[:n].astype(np.float64)

    best_lag = 0
    if max_lag > 0 and n > 2 * max_lag:
        e_z = e_f - e_f.mean()
        r_z = r_f - r_f.mean()
        nfft  = int(2 ** np.ceil(np.log2(2 * n)))
        xcorr = np.fft.irfft(
            np.fft.rfft(e_z, nfft) * np.conj(np.fft.rfft(r_z, nfft)), nfft)
        cands    = np.array([xcorr[lag % nfft]
                             for lag in range(-max_lag, max_lag + 1)])
        best_lag = int(np.argmax(np.abs(cands))) - max_lag

    if best_lag >= 0:
        e_al = e_f[best_lag:]
        r_al = r_f[:n - best_lag]
        v_al = vad_samples[best_lag:n] if vad_samples is not None else None
    else:
        e_al = e_f[:n + best_lag]
        r_al = r_f[-best_lag:]
        v_al = vad_samples[-best_lag:n] if vad_samples is not None else None

    nn   = min(len(e_al), len(r_al))
    e_al, r_al = e_al[:nn], r_al[:nn]
    if v_al is not None:
        v_al = v_al[:nn].astype(bool)

    if v_al is not None and v_al.sum() >= 512:
        e_s, r_s = e_al[v_al], r_al[v_al]
    else:
        e_s, r_s = e_al, r_al

    sdr = si_sdr(e_s.astype(np.float32), r_s.astype(np.float32), max_lag=0)
    e16 = _resample_16k(e_al.astype(np.float32))
    r16 = _resample_16k(r_al.astype(np.float32))

    return {'si_sdr': round(sdr, 3),
            'pesq':   round(pesq_score(e16, r16), 3),
            'stoi':   round(stoi_score(e16, r16), 3)}


# ── Gaze vector → azimuth angle conversion ───────────────────────────────────

def _gaze_vectors_to_azimuths(gaze_stft: np.ndarray) -> np.ndarray:
    """
    Convert (n_frames, 3) unit gaze vectors → (n_frames,) azimuth angles (rad).

    Used for energy_vad mode.  Passing 3-D vectors to the pipeline causes
    _parse_gaze to set is_vec=True, which routes steering through
    gaze_vector_to_theta per frame and works correctly.  However the VAD
    sees the full 3-D vector path and the EnergyVAD gets confused by the
    format.  Converting to scalar azimuths first keeps the VAD logic clean
    and ensures correct noise-frame counting.
    """
    gaze_stft = np.asarray(gaze_stft, dtype=np.float32)
    if gaze_stft.ndim == 1:
        return gaze_stft   # already scalar angles
    # (n_frames, 3) → atan2(x, z) gives azimuth in [-π, +π]
    return np.arctan2(gaze_stft[:, 0], gaze_stft[:, 2]).astype(np.float32)


# ── Pipeline runner ────────────────────────────────────────────────────────────

# ── Ablation mode-name suffixes ─────────────────────────────────────────────
# A mode string may carry '+gazestab', '+saccadehold', and/or '+micsel'
# suffixes to independently toggle the gaze stabilizer / saccade hold /
# adaptive mic selection on top of any base mode (oracle_gaze,
# oracle_target_dir, energy_vad, srp). Examples:
#   'oracle_gaze'                        → all three off (pre-fix behaviour)
#   'oracle_gaze+gazestab'               → gaze stabilizer only
#   'oracle_gaze+micsel'                 → mic selection only
#   'oracle_gaze+gazestab+micsel'        → both
# raw_mic ignores all suffixes (it's a pure passthrough, no pipeline at all).
_ABLATION_SUFFIXES = {
    '+gazestab':     'use_gaze_stabilizer',
    '+saccadehold':  'use_saccade_hold',
    '+micsel':       'use_mic_selection',
    '+projguard':    'use_projection_guard',
}


def parse_ablation_mode(mode_str: str) -> tuple[str, dict]:
    """Split 'oracle_gaze+gazestab+micsel' → ('oracle_gaze', {'use_gaze_stabilizer': True, 'use_mic_selection': True})."""
    parts = mode_str.split('+')
    base  = parts[0]
    flags = {v: False for v in _ABLATION_SUFFIXES.values()}
    for suffix in parts[1:]:
        key = '+' + suffix
        if key not in _ABLATION_SUFFIXES:
            raise ValueError(f"Unknown ablation suffix '+{suffix}' in mode "
                             f"'{mode_str}'. Valid: {list(_ABLATION_SUFFIXES)}")
        flags[_ABLATION_SUFFIXES[key]] = True
    return base, flags


def expand_ablation_modes(base_modes: list[str],
                          factors: list[str] | None = None) -> list[str]:
    """
    Expand each base mode into every combination of the given suffixes
    (default: gazestab × micsel → 4 variants each). Order is stable:
    off/off, gazestab-only, micsel-only, both.
    """
    if factors is None:
        factors = ['+gazestab', '+micsel']
    import itertools
    out = []
    for base in base_modes:
        for combo in itertools.product([False, True], repeat=len(factors)):
            suffix = ''.join(f for f, on in zip(factors, combo) if on)
            out.append(base + suffix)
    return out


def run_mode(mic_audio, gaze_stft, vad_stft, mode, atf_path, use_denoise,
             mic_pos=None, target_azimuth_deg=None, use_gevd_rtf=False,
             return_pipeline=False):
    """
    Run the pipeline in one evaluation mode.

    return_pipeline : if True, return (audio, pipeline_or_None) instead of
        just audio. `pipeline` is the constructed AriaDenoisingPipeline (so
        pipeline.beamformer.diagnostics() is available after process()
        returns), or None for base_mode == 'raw_mic' (no pipeline built).
        Default False preserves the original single-value return exactly,
        so existing callers (evaluate_clip) are unaffected.

    Parameter choices per mode
    --------------------------
    oracle_gaze
        alpha=BEAMFORMER_ALPHA (0.97), rt60_s=0.15, doa_reliable=False
        Gaze 3-D vectors passed directly (saccade noise present but no jump
        detector resets).  Annotated VAD used — rt60_s hangover extension is
        fine because the VAD is not driving noise adaptation here.

    oracle_target_dir
        alpha=BEAMFORMER_ALPHA (0.97), rt60_s=0.15, doa_reliable=True
        Fixed scalar azimuth, no saccades.  Jump detector active but never
        fires (constant angle).  Annotated VAD used.

    energy_vad
        alpha=BEAMFORMER_ALPHA (0.97), rt60_s=0.0, doa_reliable=False
        rt60_s=0.0 is CRITICAL: at RT60=150 ms the hangover would extend to
        84 frames, bridging all silence gaps and producing 99.8% speech
        classification with only 19-22 noise frames.  With rt60_s=0.0 the
        hangover stays at 8 frames so genuine silence is detected.
        Gaze passed as azimuth angles (not 3-D vectors) for clean VAD operation.

    srp
        alpha=BEAMFORMER_ALPHA (0.97), rt60_s=0.15, doa_reliable=False
        SRP angle jumps are noise artefacts not real source movement.
        Annotated VAD used.

    use_gevd_rtf
        Forwarded to AriaDenoisingPipeline for every mode below.  Refines
        the analytic steering vector every frame with a data-driven RTF
        estimate (GEVDRTFEstimator), trust-gated against the analytic/gaze
        direction so it can only add reverberant structure on top of an
        already-agreeing direction, never override it.  No effect when
        atf_path is supplied — AriaDenoisingPipeline ignores use_gevd_rtf on
        the ATFSteering path and prints a warning.
    """
    base_mode, ablation_flags = parse_ablation_mode(mode)

    if base_mode == 'raw_mic':
        return (mic_audio[0], None) if return_pipeline else mic_audio[0]

    atf = ATFSteering(atf_path, n_fft=F_WIN, fs=FS) if atf_path else None

    if base_mode == 'oracle_gaze':
        pipeline = AriaDenoisingPipeline(
            use_gaze=True, atf_steering=atf,
            mic_pos=mic_pos, alpha=BEAMFORMER_ALPHA,
            vad_thr_db=3.0, rt60_s=0.15,
            doa_reliable=False,
            use_gevd_rtf=use_gevd_rtf,
            **ablation_flags)
        vad_bool = np.asarray(vad_stft, dtype=bool)
        out = pipeline.process(mic_audio, gaze=gaze_stft,
                               annotated_vad=vad_bool,
                               skip_denoise=not use_denoise)
        return (out, pipeline) if return_pipeline else out

    elif base_mode == 'oracle_target_dir':
        if target_azimuth_deg is None:
            raise ValueError(
                "oracle_target_dir requires target_azimuth_deg from metadata")
        pipeline = AriaDenoisingPipeline(
            use_gaze=True, atf_steering=atf,
            mic_pos=mic_pos, alpha=BEAMFORMER_ALPHA,
            vad_thr_db=3.0, rt60_s=0.15,
            doa_reliable=True,
            use_gevd_rtf=use_gevd_rtf,
            **ablation_flags)
        vad_bool     = np.asarray(vad_stft, dtype=bool)
        target_angle = np.deg2rad(float(target_azimuth_deg))
        n_gaze       = (gaze_stft.shape[0] if hasattr(gaze_stft, 'shape')
                        else len(gaze_stft))
        target_gaze  = np.full(n_gaze + 64, target_angle, dtype=np.float32)
        out = pipeline.process(mic_audio, gaze=target_gaze,
                               annotated_vad=vad_bool,
                               skip_denoise=not use_denoise)
        return (out, pipeline) if return_pipeline else out

    elif base_mode == 'energy_vad':
        # rt60_s=0.0 — CRITICAL: prevents the 84-frame hangover from
        # bridging all silence gaps and leaving only 19-22 noise frames.
        pipeline = AriaDenoisingPipeline(
            use_gaze=True, atf_steering=atf,
            mic_pos=mic_pos, alpha=BEAMFORMER_ALPHA,
            vad_thr_db=3.0, rt60_s=0.0,
            doa_reliable=False,
            use_gevd_rtf=use_gevd_rtf,
            **ablation_flags)
        # Convert 3-D gaze vectors → scalar azimuth angles so the energy VAD
        # operates on a clean 1-D angle stream.
        gaze_angles = _gaze_vectors_to_azimuths(gaze_stft)
        out = pipeline.process(mic_audio, gaze=gaze_angles,
                               annotated_vad=None,   # use internal energy VAD
                               skip_denoise=not use_denoise)
        return (out, pipeline) if return_pipeline else out

    elif base_mode == 'srp':
        # gaze stabilizer has nothing to smooth here (SRP has no gaze
        # input) — silently ignored if requested. Mic selection still
        # applies (it only needs theta, which SRP provides every frame).
        srp_flags = dict(ablation_flags)
        srp_flags.pop('use_gaze_stabilizer', None)
        srp_flags.pop('use_saccade_hold', None)
        pipeline = AriaDenoisingPipeline(
            use_gaze=False, atf_steering=atf,
            mic_pos=mic_pos, alpha=BEAMFORMER_ALPHA,
            vad_thr_db=3.0, rt60_s=0.15,
            doa_reliable=False,
            use_gevd_rtf=use_gevd_rtf,
            **srp_flags)
        vad_bool = np.asarray(vad_stft, dtype=bool)
        out = pipeline.process(mic_audio, gaze=None,
                               annotated_vad=vad_bool,
                               skip_denoise=not use_denoise)
        return (out, pipeline) if return_pipeline else out

    else:
        raise ValueError(f"Unknown mode: {mode!r} (base={base_mode!r})")


# ── Per-clip evaluation ────────────────────────────────────────────────────────

def evaluate_clip(clip_path, modes, atf_path, use_denoise, use_gevd_rtf=False,
                  collect_diagnostics=False):
    clip_path = Path(clip_path)
    meta_f    = clip_path / 'metadata.json'

    # ── Robust loading ───────────────────────────────────────────────────────
    # Some clips on shared filesystems may be unreadable due to permission
    # issues (e.g. owned by another user, different umask at generation time).
    # Skip those clips with a warning instead of crashing the whole run.
    try:
        if not meta_f.exists():
            return {'error': 'no metadata.json', 'clip': clip_path.name}
        with open(meta_f) as f:
            meta = json.load(f)

        mic_data, sr = sf.read(str(clip_path / 'array_audio.wav'),
                               dtype='float32', always_2d=True)
        assert sr == FS
        mic_audio = mic_data.T          # (N, T)
        gaze_stft = np.load(str(clip_path / 'gaze.npy'))
        vad_stft  = np.load(str(clip_path / 'vad.npy'))
        noise_env = np.load(str(clip_path / 'noise_envelope.npy'))
        mic_pos   = _load_clip_mic_positions(clip_path, meta)
    except PermissionError as e:
        print(f"  [SKIP] Permission denied reading {clip_path.name}: {e}")
        return {'error': f'permission denied: {e}', 'clip': clip_path.name}
    except OSError as e:
        print(f"  [SKIP] OS error reading {clip_path.name}: {e}")
        return {'error': f'os error: {e}', 'clip': clip_path.name}

    target_azimuth_deg = meta.get('target_azimuth_deg')
    T                  = mic_audio.shape[1]
    vad_samples        = np.repeat(vad_stft, HOP)[:T]

    # ── VAD diagnostic ─────────────────────────────────────────────────────
    noise_stft_frames = int((~vad_stft.astype(bool)).sum())
    speech_pct        = float(np.asarray(vad_stft, dtype=bool).mean()) * 100
    if noise_stft_frames < 200:
        print(f"  [VAD WARNING] vad.npy has only {noise_stft_frames} noise "
              f"frames ({100-speech_pct:.1f}% silence). Regenerate dataset.")
    else:
        print(f"  [VAD] speech={speech_pct:.0f}%  noise_frames={noise_stft_frames}")

    # ── Reference signal ───────────────────────────────────────────────────
    # Priority order:
    #   atf          → reverberant_reference.wav  (target ⊗ ATF_0)
    #   ism          → reverberant_reference.wav  (target ⊗ ISM_h_0)
    #   atf+reverb   → compute ATF ref on-the-fly (avoids Schroeder-tail mismatch)
    #   fallback     → close_mic.wav (dry speech — SI-SDR ceiling ≈ 0 dB)
    sim_mode  = meta.get('simulation_mode', 'ism')
    ref_audio = None
    ref_label = 'unknown'

    if sim_mode == 'atf':
        rr_path = clip_path / 'reverberant_reference.wav'
        if rr_path.exists():
            ref_audio, _ = sf.read(str(rr_path), dtype='float32', always_2d=False)
            ref_audio     = ref_audio[:T]
            ref_label     = 'reverberant_ref_atf'
        else:
            warnings.warn(f"reverberant_reference.wav missing in {clip_path}")

    if ref_audio is None and sim_mode == 'ism':
        # reverberant_reference.wav = channel 0 of the ISM target signal
        # (target convolved with the ISM RIR for mic 0).  This is exactly what
        # the MVDR distortionless constraint outputs, so it is the correct
        # reference.  Using close_mic.wav (dry speech) instead gives an
        # SI-SDR ceiling of ≈ 0 dB because it ignores legitimate reverberant
        # energy that the beamformer passes.
        rr_path = clip_path / 'reverberant_reference.wav'
        if rr_path.exists():
            ref_audio, _ = sf.read(str(rr_path), dtype='float32', always_2d=False)
            ref_audio     = ref_audio[:T]
            ref_label     = 'reverberant_ref_ism'
        else:
            warnings.warn(
                f"reverberant_reference.wav missing in {clip_path}. "
                "Falling back to close_mic.wav (dry speech). "
                "SI-SDR ceiling will be ~0 dB. Regenerate dataset.")

    if ref_audio is None and sim_mode == 'atf+reverb':
        cm_path = clip_path / 'close_mic.wav'
        if cm_path.exists() and atf_path and Path(atf_path).exists():
            close_mic, _ = sf.read(str(cm_path), dtype='float32', always_2d=False)
            ref_audio     = _compute_atf_ref_on_the_fly(
                close_mic, float(target_azimuth_deg or 0.0), atf_path, T)
            ref_label     = 'atf_ref_computed_from_closemic'
            print(f"  [REF] Old dataset: computed ATF reference "
                  f"(close_mic ⊗ ATF_0 @ {target_azimuth_deg:+.1f}°)")
        else:
            warnings.warn(
                "Old 'atf+reverb' dataset: close_mic.wav or ATF path "
                "unavailable.  Falling back to reverberant_reference.wav "
                "(SI-SDR will be ~17 dB too low at high SNR). "
                "Regenerate dataset for correct results.")
            rr_path = clip_path / 'reverberant_reference.wav'
            if rr_path.exists():
                ref_audio, _ = sf.read(str(rr_path), dtype='float32',
                                       always_2d=False)
                ref_audio     = ref_audio[:T]
                ref_label     = 'reverberant_ref_schroeder_WARNING'

    if ref_audio is None:
        cm_path = clip_path / 'close_mic.wav'
        if cm_path.exists():
            ref_audio, _ = sf.read(str(cm_path), dtype='float32', always_2d=False)
            ref_audio     = ref_audio[:T]
            ref_label     = 'close_mic'
        else:
            ref_audio = mic_audio[0].copy()
            ref_label = 'mic0_fallback'

    # ── Evaluate each mode ─────────────────────────────────────────────────
    row = {
        'clip':              clip_path.name,
        'scenario':          meta.get('scenario'),
        'base_scenario':     meta.get('base_scenario', meta.get('scenario')),
        'is_dynamic':        meta.get('is_dynamic', False),
        'snr_db':            meta.get('snr_db'),
        'rep':               meta.get('rep'),
        'interferer_dist_m': meta.get('interferer_distance_m'),
        'speech_frac':       round(float(vad_samples.mean()), 3),
        'noise_stft_frames': noise_stft_frames,
        'noise_env_std':     round(float(noise_env.std()), 4),
        'reference':         ref_label,
        'simulation_mode':   sim_mode,
        'use_gevd_rtf':      bool(use_gevd_rtf),
    }

    for mode in modes:
        try:
            want_diag = collect_diagnostics and mode != 'raw_mic'
            result = run_mode(mic_audio, gaze_stft, vad_stft, mode, atf_path,
                              use_denoise, mic_pos=mic_pos,
                              target_azimuth_deg=target_azimuth_deg,
                              use_gevd_rtf=use_gevd_rtf,
                              return_pipeline=want_diag)
            if want_diag:
                enh, pipeline = result
                row[f'{mode}_diag'] = (pipeline.beamformer.diagnostics()
                                        if pipeline is not None else None)
            else:
                enh = result
            m   = compute_metrics(enh, ref_audio, vad_samples, max_lag=512)
            row[f'{mode}_si_sdr'] = m['si_sdr']
            row[f'{mode}_pesq']   = m['pesq']
            row[f'{mode}_stoi']   = m['stoi']
            speech_pct_s = float(vad_samples.mean()) * 100
            print(f"    {mode:<24}  SI-SDR={m['si_sdr']:+6.1f}  "
                  f"PESQ={m['pesq']:.2f}  STOI={m['stoi']:.3f}  "
                  f"(speech={speech_pct_s:.0f}%)")
        except Exception:
            import traceback
            traceback.print_exc()
            row[f'{mode}_si_sdr'] = row[f'{mode}_pesq'] = row[f'{mode}_stoi'] = None
            if want_diag:
                row[f'{mode}_diag'] = None

    return row


# ── Worker for parallel evaluation ────────────────────────────────────────────

def _eval_worker(args_tuple: tuple) -> dict:
    clip_path, modes, atf_path, use_denoise, use_gevd_rtf = args_tuple
    try:
        with open(clip_path / 'metadata.json') as f:
            meta = json.load(f)
        print(f"[{clip_path.name}]  scenario={meta['scenario']}  "
              f"snr={meta['snr_db']} dB  "
              f"dist={meta.get('interferer_distance_m', '?')} m  "
              f"env_std={meta.get('noise_env_std', 0):.3f}",
              flush=True)
    except PermissionError as e:
        print(f"  [SKIP] Permission denied reading metadata for "
              f"{clip_path.name}: {e}", flush=True)
        return {'error': f'permission denied: {e}', 'clip': clip_path.name}
    except OSError as e:
        print(f"  [SKIP] OS error reading metadata for "
              f"{clip_path.name}: {e}", flush=True)
        return {'error': f'os error: {e}', 'clip': clip_path.name}

    return evaluate_clip(clip_path, modes, atf_path, use_denoise, use_gevd_rtf)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(all_results, modes, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    rows  = [r for r in all_results if not r.get('error')]
    scens = sorted({r['scenario'] for r in rows})
    snrs  = sorted({r['snr_db']   for r in rows if r['snr_db'] is not None})

    mode_colors = {
        'raw_mic':           '#888888',
        'oracle_gaze':       '#1a6faf',
        'oracle_target_dir': '#8e44ad',
        'energy_vad':        '#e07b00',
        'srp':               '#2e8b57',
    }
    mode_labels = {
        'raw_mic':           'Raw mic 0',
        'oracle_gaze':       'MVDR + oracle gaze',
        'oracle_target_dir': 'MVDR + oracle target dir',
        'energy_vad':        'MVDR + energy VAD',
        'srp':               'MVDR + SRP DOA',
    }

    def mean_metric(metric_key, scenario=None, snr=None, mode=None):
        vals = []
        for r in rows:
            if scenario is not None and r['scenario'] != scenario: continue
            if snr is not None and r['snr_db'] != snr:             continue
            v = r.get(f'{mode}_{metric_key}' if mode else metric_key)
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    n_scen = len(scens)
    n_cols = min(3, n_scen)
    n_rows = (n_scen + n_cols - 1) // n_cols

    # ── Plot 1: SI-SDR vs SNR per scenario ────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle('SI-SDR vs Input SNR', fontsize=14, fontweight='bold', y=1.01)
    for idx, scen in enumerate(scens):
        ax = axes[idx // n_cols][idx % n_cols]
        for mode in modes:
            y = [mean_metric('si_sdr', scen, snr, mode) for snr in snrs]
            ax.plot(snrs, y, '-o', color=mode_colors.get(mode, 'grey'),
                    label=mode_labels.get(mode, mode), linewidth=2, markersize=5)
        ax.set_title(scen.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Input SNR (dB)'); ax.set_ylabel('SI-SDR (dB)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    for idx in range(n_scen, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    plt.tight_layout()
    p = out_dir / 'plot1_sisdr_vs_snr.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    # ── Plot 2: SI-SDR improvement over raw_mic ────────────────────────────
    beam_modes = [m for m in modes if m != 'raw_mic']
    if beam_modes and snrs:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle('SI-SDR Improvement over Raw Mic (dB)', fontsize=14,
                     fontweight='bold', y=1.01)
        for idx, scen in enumerate(scens):
            ax  = axes[idx // n_cols][idx % n_cols]
            raw = [mean_metric('si_sdr', scen, snr, 'raw_mic') for snr in snrs]
            for mode in beam_modes:
                proc  = [mean_metric('si_sdr', scen, snr, mode) for snr in snrs]
                delta = [p - r if np.isfinite(p) and np.isfinite(r)
                         else float('nan') for p, r in zip(proc, raw)]
                ax.plot(snrs, delta, '-o', color=mode_colors.get(mode, 'grey'),
                        label=mode_labels.get(mode, mode), linewidth=2,
                        markersize=5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_title(scen.replace('_', ' ').title(), fontsize=11)
            ax.set_xlabel('Input SNR (dB)'); ax.set_ylabel('ΔSI-SDR (dB)')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        for idx in range(n_scen, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
        plt.tight_layout()
        p = out_dir / 'plot2_sisdr_improvement.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    # ── Plot 3: STOI heatmap per mode ─────────────────────────────────────
    for mode in modes:
        data = np.full((len(scens), len(snrs)), float('nan'))
        for i, scen in enumerate(scens):
            for j, snr in enumerate(snrs):
                data[i, j] = mean_metric('stoi', scen, snr, mode)
        fig, ax = plt.subplots(figsize=(max(6, len(snrs) * 1.2),
                                        max(4, len(scens) * 0.6)))
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='STOI')
        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([f'{s:+g}' for s in snrs])
        ax.set_yticks(range(len(scens)))
        ax.set_yticklabels([s.replace('_', ' ') for s in scens], fontsize=9)
        ax.set_xlabel('Input SNR (dB)')
        ax.set_title(f'STOI — {mode_labels.get(mode, mode)}',
                     fontsize=12, fontweight='bold')
        for i in range(len(scens)):
            for j in range(len(snrs)):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8,
                            color='black' if 0.3 < v < 0.7 else 'white')
        plt.tight_layout()
        p = out_dir / f'plot3_stoi_heatmap_{mode}.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    # ── Plot 4: Scenario difficulty ranking ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(scens))
    width = 0.8 / max(len(modes), 1)
    for k, mode in enumerate(modes):
        means = [mean_metric('si_sdr', scen, None, mode) for scen in scens]
        ax.bar(x + k * width - (len(modes) - 1) * width / 2, means, width,
               label=mode_labels.get(mode, mode),
               color=mode_colors.get(mode, 'grey'), alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scens], fontsize=9)
    ax.set_ylabel('Mean SI-SDR (dB)')
    ax.set_title('Scenario Difficulty Ranking (all SNRs)', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    p = out_dir / 'plot4_scenario_ranking.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    # ── Plot 5: Distance effect ────────────────────────────────────────────
    dist_scens = [s for s in scens if s.startswith('directional_')]
    if len(dist_scens) >= 2:
        fig, axes = plt.subplots(1, len(modes),
                                 figsize=(5 * len(modes), 4), squeeze=False)
        fig.suptitle('Interferer Distance Effect on SI-SDR', fontsize=13,
                     fontweight='bold')
        dist_colors = {'directional_near': '#d62728',
                       'directional_mid':  '#ff7f0e',
                       'directional_far':  '#2ca02c'}
        for k, mode in enumerate(modes):
            ax = axes[0][k]
            for scen in dist_scens:
                y = [mean_metric('si_sdr', scen, snr, mode) for snr in snrs]
                ax.plot(snrs, y, '-o',
                        color=dist_colors.get(scen, 'grey'),
                        label=scen.replace('directional_', ''),
                        linewidth=2, markersize=5)
            ax.set_title(mode_labels.get(mode, mode), fontsize=10)
            ax.set_xlabel('Input SNR (dB)'); ax.set_ylabel('SI-SDR (dB)')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        plt.tight_layout()
        p = out_dir / 'plot5_distance_effect.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    # ── Plot 6: Static vs Dynamic ──────────────────────────────────────────
    base_scens = sorted({r.get('base_scenario', r['scenario']) for r in rows
                         if not r.get('is_dynamic', False)})
    paired = [s for s in base_scens
              if any(r['scenario'] == s for r in rows)
              and any(r['scenario'] == s + '_dynamic' for r in rows)]
    if paired and snrs:
        mode = 'oracle_gaze' if 'oracle_gaze' in modes else modes[0]
        n_p  = len(paired)
        nc   = min(3, n_p); nr = (n_p + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr),
                                 squeeze=False)
        fig.suptitle(
            f'Static vs Dynamic — {mode_labels.get(mode, mode)} SI-SDR',
            fontsize=12, fontweight='bold', y=1.02)
        for idx, base in enumerate(paired):
            ax = axes[idx // nc][idx % nc]
            ys = [mean_metric('si_sdr', base, snr, mode) for snr in snrs]
            yd = [mean_metric('si_sdr', base + '_dynamic', snr, mode)
                  for snr in snrs]
            ax.plot(snrs, ys, '-o',  color='#1a6faf', linewidth=2,
                    markersize=5, label='Static')
            ax.plot(snrs, yd, '--s', color='#d62728', linewidth=2,
                    markersize=5, label='Dynamic')
            ax.set_title(base.replace('_', ' ').title(), fontsize=10)
            ax.set_xlabel('Mean SNR (dB)'); ax.set_ylabel('SI-SDR (dB)')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        for idx in range(n_p, nr * nc):
            axes[idx // nc][idx % nc].set_visible(False)
        plt.tight_layout()
        p = out_dir / 'plot6_static_vs_dynamic.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        plt.close(fig); saved.append(str(p)); print(f"  Saved: {p}")

    return saved


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(all_results, modes, snrs):
    rows = [r for r in all_results if not r.get('error')]

    def mean_m(metric, scenario=None, snr=None, mode=None):
        vals = []
        for r in rows:
            if scenario and r['scenario'] != scenario:  continue
            if snr is not None and r['snr_db'] != snr:  continue
            v = r.get(f'{mode}_{metric}' if mode else metric)
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    scens = sorted({r['scenario'] for r in rows})
    hdr   = '=' * 90
    sep   = '-' * 91

    print(f"\n{hdr}")
    print(" SUMMARY: Mean SI-SDR (dB) across all reps  "
          "|  rows = scenario  |  cols = SNR")
    print(hdr)

    for mode in modes:
        print(f"\n  Mode: {mode}")
        header = (f"  {'Scenario':<24}"
                  + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
        print(header); print("  " + sep)
        for scen in scens:
            row_vals = [mean_m('si_sdr', scen, snr, mode) for snr in snrs]
            overall  = float(np.nanmean(row_vals))
            val_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                               for v in row_vals)
            print(f"  {scen:<24}{val_strs}  {overall:+7.1f}")

    # oracle_gaze − raw_mic
    if 'oracle_gaze' in modes and 'raw_mic' in modes:
        print(f"\n{hdr}")
        print(" IMPROVEMENT over raw_mic  (oracle_gaze − raw_mic SI-SDR, dB)")
        print(hdr)
        header = (f"  {'Scenario':<24}"
                  + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
        print(header); print("  " + sep)
        for scen in scens:
            raw   = [mean_m('si_sdr', scen, snr, 'raw_mic') for snr in snrs]
            proc  = [mean_m('si_sdr', scen, snr, 'oracle_gaze') for snr in snrs]
            delta = [p - r if np.isfinite(p) and np.isfinite(r)
                     else float('nan') for p, r in zip(proc, raw)]
            mean_d = float(np.nanmean(delta))
            d_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                             for v in delta)
            print(f"  {scen:<24}{d_strs}  {mean_d:+7.1f}")

    # oracle_target_dir − oracle_gaze  (saccade cost)
    if 'oracle_target_dir' in modes and 'oracle_gaze' in modes:
        print(f"\n{hdr}")
        print(" SACCADE COST  (oracle_target_dir − oracle_gaze, dB)")
        print(hdr)
        header = (f"  {'Scenario':<24}"
                  + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
        print(header); print("  " + sep)
        for scen in scens:
            og  = [mean_m('si_sdr', scen, snr, 'oracle_gaze') for snr in snrs]
            otd = [mean_m('si_sdr', scen, snr, 'oracle_target_dir')
                   for snr in snrs]
            delta = [o - g if np.isfinite(o) and np.isfinite(g)
                     else float('nan') for o, g in zip(otd, og)]
            mean_d = float(np.nanmean(delta))
            d_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                             for v in delta)
            print(f"  {scen:<24}{d_strs}  {mean_d:+7.1f}")

    # oracle_gaze − srp  (gaze value)
    if 'oracle_gaze' in modes and 'srp' in modes:
        print(f"\n{hdr}")
        print(" GAZE VALUE  (oracle_gaze − srp SI-SDR, dB)"
              "  — gaze vs audio-only DOA")
        print(hdr)
        header = (f"  {'Scenario':<24}"
                  + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
        print(header); print("  " + sep)
        for scen in scens:
            srp_v = [mean_m('si_sdr', scen, snr, 'srp') for snr in snrs]
            ora_v = [mean_m('si_sdr', scen, snr, 'oracle_gaze') for snr in snrs]
            delta = [o - s if np.isfinite(o) and np.isfinite(s)
                     else float('nan') for o, s in zip(ora_v, srp_v)]
            mean_d = float(np.nanmean(delta))
            d_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                             for v in delta)
            print(f"  {scen:<24}{d_strs}  {mean_d:+7.1f}")

    # oracle_gaze − energy_vad  (VAD value)
    if 'oracle_gaze' in modes and 'energy_vad' in modes:
        print(f"\n{hdr}")
        print(" VAD VALUE  (oracle_gaze − energy_vad SI-SDR, dB)"
              "  — annotated vs energy VAD")
        print(hdr)
        header = (f"  {'Scenario':<24}"
                  + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
        print(header); print("  " + sep)
        for scen in scens:
            evad = [mean_m('si_sdr', scen, snr, 'energy_vad') for snr in snrs]
            ora  = [mean_m('si_sdr', scen, snr, 'oracle_gaze') for snr in snrs]
            delta = [o - e if np.isfinite(o) and np.isfinite(e)
                     else float('nan') for o, e in zip(ora, evad)]
            mean_d = float(np.nanmean(delta))
            d_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                             for v in delta)
            print(f"  {scen:<24}{d_strs}  {mean_d:+7.1f}")


def print_ablation_report(all_results, ablated_base_modes, snrs):
    """
    For each base mode passed to --ablate, print:
      1. The four (or two, for srp) raw mean-SI-SDR tables — already shown
         by print_summary() since the variants are just extra 'modes'.
      2. GAZE STABILIZER EFFECT: mean(+gazestab) − mean(without), averaged
         over both settings of mic-selection (i.e. the stabilizer's effect
         holding mic-selection fixed at each level, then averaged) — an
         unconfounded main effect from the 2×2 factorial design.
      3. MIC SELECTION EFFECT: the symmetric decomposition for micsel.
    """
    rows = [r for r in all_results if not r.get('error')]

    def mean_m(scenario, snr, mode):
        vals = []
        for r in rows:
            if r['scenario'] != scenario or r['snr_db'] != snr:
                continue
            v = r.get(f'{mode}_si_sdr')
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    scens = sorted({r['scenario'] for r in rows})
    hdr, sep = '=' * 90, '-' * 91

    for base in ablated_base_modes:
        has_gazestab = (base != 'srp')

        def table(title, on_suffix, off_suffix):
            print(f"\n{hdr}")
            print(f" {title}  ({base}{on_suffix} − {base}{off_suffix}, dB, "
                  f"averaged over the other factor's on/off)")
            print(hdr)
            header = (f"  {'Scenario':<24}"
                      + "".join(f" {s:>7}" for s in snrs) + "   MEAN")
            print(header); print("  " + sep)
            for scen in scens:
                row_deltas = []
                for snr in snrs:
                    on  = mean_m(scen, snr, base + on_suffix)
                    off = mean_m(scen, snr, base + off_suffix)
                    row_deltas.append(on - off if np.isfinite(on) and np.isfinite(off)
                                      else float('nan'))
                mean_d = float(np.nanmean(row_deltas))
                d_strs = "".join(f" {v:+7.1f}" if np.isfinite(v) else "     n/a"
                                 for v in row_deltas)
                print(f"  {scen:<24}{d_strs}  {mean_d:+7.1f}")

        if has_gazestab:
            # Main effect of gazestab, averaged over micsel off/on:
            #   0.5*[(base+gazestab − base) + (base+gazestab+micsel − base+micsel)]
            # Printed as two rows folded into one table by re-using table()
            # twice and letting the reader eyeball consistency; also print
            # the pooled estimate as its own table using synthetic mode
            # keys built on the fly.
            print(f"\n{hdr}")
            print(f" GAZE STABILIZER EFFECT  (mean SI-SDR delta from turning "
                  f"gazestab ON, dB) — base mode: {base}")
            print(" holding mic-selection OFF:")
            table("  gazestab effect, micsel=OFF", '+gazestab', '')
            print(" holding mic-selection ON:")
            table("  gazestab effect, micsel=ON", '+gazestab+micsel', '+micsel')

        print(f"\n{hdr}")
        print(f" MIC SELECTION EFFECT  (mean SI-SDR delta from turning "
              f"micsel ON, dB) — base mode: {base}")
        if has_gazestab:
            print(" holding gaze-stabilizer OFF:")
            table("  micsel effect, gazestab=OFF", '+micsel', '')
            print(" holding gaze-stabilizer ON:")
            table("  micsel effect, gazestab=ON", '+gazestab+micsel', '+gazestab')
        else:
            table("  micsel effect", '+micsel', '')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', required=True,
                   help='Synthetic dataset root directory')
    p.add_argument('--atf',
                   default=None,
                   metavar='H5',
                   help='Device_ATFs.h5 (required only for ATF-mode datasets '
                        'and old atf+reverb datasets).  Omit for ISM datasets.')
    p.add_argument('--scenarios', nargs='+', default=None)
    p.add_argument('--snrs',      nargs='+', type=float, default=None)
    p.add_argument('--no-denoise', action='store_true')
    p.add_argument('--modes', nargs='+',
                   default=['raw_mic', 'oracle_gaze', 'oracle_target_dir',
                             'energy_vad', 'srp'],
                   help='Pipeline modes to compare')
    p.add_argument('--use-gaze-stabilizer', action='store_true',
                   help='Turn on gaze_processing.GazeStabilizer for every '
                        'mode in --modes that takes a gaze input '
                        '(oracle_gaze, oracle_target_dir, energy_vad). '
                        'Off by default. Ignored for any mode you also '
                        'listed in --ablate (that mode is fully controlled '
                        'by the ablation sweep instead — see --ablate).')
    p.add_argument('--use-saccade-hold', action='store_true',
                   help='Turn on the v6.4 saccade-hold logic (hold the '
                        'previous committed gaze direction for a few '
                        'frames after a jump) for every gaze-input mode in '
                        '--modes. Off by default. Same --ablate exception '
                        'as --use-gaze-stabilizer.')
    p.add_argument('--use-mic-selection', action='store_true',
                   help='Turn on mic_selection.AdaptiveMicSelector for '
                        'every beamformer mode in --modes (oracle_gaze, '
                        'oracle_target_dir, energy_vad, srp). Off by '
                        'default. Same --ablate exception as above.')
    p.add_argument('--ablate', nargs='+', default=None, metavar='MODE',
                   choices=['oracle_gaze', 'oracle_target_dir',
                            'energy_vad', 'srp'],
                   help='For each base mode listed, run all combinations of '
                        'gaze-stabilizer on/off × mic-selection on/off '
                        '(4 pipeline runs per base mode: baseline, '
                        '+gazestab, +micsel, +gazestab+micsel) and print an '
                        'ablation table showing each feature\'s individual '
                        'dB contribution. These variants REPLACE the plain '
                        'base mode in --modes if present, or are appended. '
                        'srp ignores gazestab (no gaze input) — only '
                        'micsel on/off is run for it. A base mode listed '
                        'here is unaffected by --use-gaze-stabilizer / '
                        '--use-saccade-hold / --use-mic-selection: the '
                        'sweep always covers all 4 (or 2) combinations '
                        'regardless of those flags.')
    p.add_argument('--use-gevd-rtf', action='store_true',
                   help='Enable data-driven RTF refinement of the steering '
                        'vector (GEVDRTFEstimator) for every mode, including '
                        'gaze-based modes.  Off by default.  Refines (does '
                        'not override) the gaze/angle-based steering vector '
                        'bin-by-bin where a generalized-eigenvalue RTF '
                        'estimate is available and agrees with it — useful '
                        'because the gaze/angle steering vector alone cannot '
                        'capture room reverberation.  No effect when --atf '
                        'is supplied, since a measured ATF is already a '
                        'better steering source than a data-driven estimate.')
    p.add_argument('--plot',     action='store_true',
                   help='Generate summary plots (requires matplotlib)')
    p.add_argument('--plot-dir', default=None, metavar='DIR')
    p.add_argument('--output-csv', default='eval_synthetic_results.csv')
    p.add_argument('--jobs', type=int, default=1, metavar='N',
                   help='Parallel worker processes (default: 1 = sequential)')
    return p.parse_args(argv)


def main(argv=None):
    args         = parse_args(argv)
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        sys.exit(f"Dataset not found: {dataset_root}")

    manifest = dataset_root / 'manifest.json'
    if manifest.exists():
        with open(manifest) as f:
            clip_names = [m['name'] for m in json.load(f)]
    else:
        clip_names = [d.name for d in sorted(dataset_root.iterdir())
                      if d.is_dir()]

    if args.scenarios:
        clip_names = [n for n in clip_names
                      if any(n.startswith(s) for s in args.scenarios)]
    if args.snrs:
        snr_tags   = [f"snr{int(s):+d}" for s in args.snrs]
        clip_names = [n for n in clip_names
                      if any(t in n for t in snr_tags)]

    clip_paths = [dataset_root / n for n in clip_names
                  if (dataset_root / n).is_dir()]
    all_snrs   = sorted({float(n.split('snr')[1].split('_')[0])
                         for n in clip_names if 'snr' in n})

    modes = list(args.modes)
    if args.ablate:
        for base in args.ablate:
            factors = ['+micsel'] if base == 'srp' else ['+gazestab', '+micsel']
            variants = expand_ablation_modes([base], factors=factors)
            if base in modes:
                idx = modes.index(base)
                modes[idx:idx+1] = variants
            else:
                modes.extend(variants)
        print(f"[ABLATE] Expanded {args.ablate} into: "
              f"{[m for m in modes if any(m.startswith(b) for b in args.ablate)]}")

    # ── Apply plain --use-gaze-stabilizer / --use-saccade-hold /
    # --use-mic-selection flags to every mode that ISN'T part of an
    # --ablate sweep. Ablated base modes are left completely alone here —
    # they already get all 4 (or 2) explicit on/off combinations above,
    # and mixing in a global default would silently break that factorial
    # design (the 'baseline' variant would stop meaning "everything off").
    ablated_bases = set(args.ablate or [])
    global_suffix = ''
    if args.use_gaze_stabilizer:
        global_suffix += '+gazestab'
    if args.use_saccade_hold:
        global_suffix += '+saccadehold'
    mic_suffix = '+micsel' if args.use_mic_selection else ''

    if global_suffix or mic_suffix:
        new_modes = []
        for m in modes:
            base = m.split('+')[0]
            if base in ablated_bases or base == 'raw_mic' or '+' in m:
                new_modes.append(m)   # ablate output or manually-suffixed — leave as-is
                continue
            suffix = ('' if base == 'srp' else global_suffix) + mic_suffix
            new_modes.append(m + suffix)
        modes = new_modes
        print(f"[FLAGS] --use-gaze-stabilizer={args.use_gaze_stabilizer}  "
              f"--use-saccade-hold={args.use_saccade_hold}  "
              f"--use-mic-selection={args.use_mic_selection}  → modes: {modes}")

    if args.use_gevd_rtf and args.atf:
        print("[WARN] --use-gevd-rtf has no effect together with --atf: "
              "AriaDenoisingPipeline ignores use_gevd_rtf whenever a "
              "measured ATF is supplied (it's already a better steering "
              "source than a data-driven estimate).")

    print(f"Evaluating {len(clip_paths)} clips | modes: {modes} | "
          f"jobs: {args.jobs}  alpha={BEAMFORMER_ALPHA}  "
          f"use_gevd_rtf={args.use_gevd_rtf}\n")

    all_results: list[dict] = []

    if args.jobs > 1:
        tasks = [(clip_path, modes, args.atf, not args.no_denoise,
                  args.use_gevd_rtf)
                 for clip_path in clip_paths]
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs) as executor:
            all_results = list(executor.map(_eval_worker, tasks))
    else:
        for clip_path in clip_paths:
            try:
                with open(clip_path / 'metadata.json') as f:
                    meta = json.load(f)
                print(f"[{clip_path.name}]  scenario={meta['scenario']}  "
                      f"snr={meta['snr_db']} dB  "
                      f"dist={meta.get('interferer_distance_m', '?')} m  "
                      f"env_std={meta.get('noise_env_std', 0):.3f}")
            except PermissionError as e:
                print(f"  [SKIP] Permission denied reading metadata for "
                      f"{clip_path.name}: {e}")
                all_results.append(
                    {'error': f'permission denied: {e}', 'clip': clip_path.name})
                continue
            except OSError as e:
                print(f"  [SKIP] OS error reading metadata for "
                      f"{clip_path.name}: {e}")
                all_results.append(
                    {'error': f'os error: {e}', 'clip': clip_path.name})
                continue

            all_results.append(
                evaluate_clip(clip_path, modes, args.atf,
                              not args.no_denoise, args.use_gevd_rtf))

    n_skipped = sum(1 for r in all_results if r.get('error'))
    if n_skipped:
        print(f"\n[SUMMARY] {n_skipped}/{len(all_results)} clip(s) skipped "
              f"due to read errors (see [SKIP] lines above).")

    if all_results:
        # Union of keys across all results — a skipped clip (which only has
        # 'error'/'clip' keys) must not truncate the columns for the rest.
        fieldnames: list[str] = []
        for r in all_results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader(); w.writerows(all_results)
        print(f"\nPer-clip CSV: {args.output_csv}")

    print_summary(all_results, modes, all_snrs)

    if args.ablate:
        print_ablation_report(all_results, args.ablate, all_snrs)

    if args.plot:
        print("\nGenerating plots...")
        plot_dir = (Path(args.plot_dir) if args.plot_dir
                    else dataset_root / 'plots')
        saved    = make_plots(all_results, modes, plot_dir)
        print(f"\n{len(saved)} plots saved to {plot_dir}/")


if __name__ == '__main__':
    main()