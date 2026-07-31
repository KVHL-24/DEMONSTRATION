"""
mic_selection.py — Adaptive microphone subset selection  (v1.1)
=================================================================

Motivation
----------
The eval sweep shows the beamformer's remaining SI-SDR losses cluster at
two extremes:

  • High input SNR (+10/+20 dB): every extra microphone the MVDR weight
    vector has to balance is one more place a small steering-vector /
    calibration error, a stale R_nn⁻¹ entry, or an off-axis reflection can
    leak in and cause the small residual distortion the v7.0 SNR-adaptive
    blend in beamformer_2.py already partially compensates for at the
    *output* stage. Using fewer, more spatially-informative mics when
    there is little real noise to null reduces the number of things that
    can go wrong in the first place, rather than only correcting for it
    after the fact. The IMPROVEMENT-over-raw_mic table in the eval sweep
    is the smoking gun for this: at +20 dB input SNR, EVERY mode and
    EVERY scenario loses 3-6 dB to doing nothing at all — the classic
    "more mics than the noise justifies" signature this module targets.
  • Low input SNR (−20 … −10 dB) / far interferers: this is exactly where
    spatial diversity across the *full* array matters most for nulling.

This module implements a lightweight, per-frame-group re-selection of the
K "most useful" microphones, where K itself adapts to the estimated
input SNR (few mics when quiet, all mics when noisy) — complementing,
not duplicating, the SNR-adaptive *output* blend already in
beamformer_2.py. Re-selection is deliberately infrequent (gated on DOA or
SNR changes exceeding a threshold, not recomputed every frame) so it adds
negligible overhead and doesn't itself become a source of instability.

Integration with MVDRBeamformer (v1.1: now real, not aspirational)
--------------------------------------------------------------------
MVDRBeamformer is constructed once with a fixed mic count N (its R_nn⁻¹
state is an (B, N, N) array). Recomputing that state for an arbitrary,
frequently-changing mic subset would be expensive and would throw away
noise statistics every time the subset changes. Instead, `AdaptiveMicSelector`
only decides which mics *contribute to the final MVDR output*:

  • R_nn⁻¹ keeps being updated from ALL N mics on every noise frame,
    regardless of the current selection — more noise statistics is
    always strictly informative and this is cheap, so there's no reason
    to gate it.
  • The steering vector `a` (and, after solving, the weight vector `w`)
    passed to `compute_weights()` are masked so unselected mics are
    excluded from the frame's *output* combination — see
    `MVDRBeamformer.compute_weights(..., mic_mask=...)` in beamformer_2.py
    (v7.1). Because unselected entries of `a` are exactly zero, the
    distortionless constraint a^H w = 1 is completely unaffected by the
    mask (those terms are zero regardless of what w would otherwise be
    there), so this is a mathematically clean exclusion rather than an
    approximation of a reduced-dimension MVDR solve.

This means: full-array spatial statistics always inform the noise model;
only the final combination step is restricted to the selected subset.

As of beamformer_2.py v7.1, MVDRBeamformer can own an AdaptiveMicSelector
directly and drive it internally — construct the beamformer with
`mic_selector=AdaptiveMicSelector(mic_pos=...)` and pass `theta`/`phi`
into `process_frame()` each call (the same theta/phi already returned by
DOA_Gaze.steering_vector_from_gaze / FreeFieldSteering.steering_vector);
the beamformer will call `update_power()` / `update()` and derive the
mask on its own each frame. Passing an explicit `mic_mask=` to
`process_frame()` overrides this for that call. If you'd rather drive
selection yourself, just call `selector.update(theta)` /
`selector.get_mask()` externally and pass the result in as `mic_mask=`;
the beamformer never requires the internal selector to be configured.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from doa_2 import compute_steering_vector
from stft import F_WIN

# ── Default constants ──────────────────────────────────────────────────────
MIC_SEL_K_MIN      = 2      # never go below a 2-mic array (still spatial)
MIC_SEL_SNR_MID    = 10.0   # dB — SNR at which K sits halfway between K_min/K_max
MIC_SEL_SNR_SCALE  = 6.0    # dB — logistic transition width
MIC_SEL_DOA_THRESH = 10.0   # degrees — DOA change that triggers re-selection
MIC_SEL_SNR_THRESH = 3.0    # dB — SNR change that triggers re-selection


class AdaptiveMicSelector:
    """
    Selects the K most spatially diverse microphones, where K is determined
    dynamically by the estimated per-mic SNR rather than being fixed.

    K schedule (logistic)
    ----------------------
    K = round( K_min + (K_max - K_min) · σ(−(SNR_dB − snr_mid) / snr_scale) )

    At SNR ≪ snr_mid  → K → K_max  (noisy: use all mics, maximum diversity)
    At SNR ≫ snr_mid  → K → K_min  (quiet: use fewest mics, save energy /
                                     minimise distortion sources)

    SNR estimate
    ------------
    Per-mic broadband power is tracked separately during speech-labelled
    (P_speech) and noise-labelled (P_noise) STFT frames via EMA:
        P_speech[m] ← α_snr · P_speech[m] + (1−α_snr) · ‖X[m,:]‖²/B
        P_noise[m]  ← α_snr · P_noise[m]  + (1−α_snr) · ‖X[m,:]‖²/B

    SNR_dB = 10·log10( mean_m(P_speech[m]) / (mean_m(P_noise[m]) + ε) )

    Re-selection triggers
    ---------------------
    Selection is recomputed when either:
      (a) DOA changes by > doa_thresh degrees, or
      (b) SNR_dB changes by > snr_thresh dB since last selection.

    The DOA-diversity ranking (greedy phase-spread maximisation) decides
    *which* K mics to keep given the computed K. A small `stability_bias`
    favours mics that were already active, so the selected set doesn't
    flap between equally-good candidates on every recompute.

    Parameters
    ----------
    mic_pos     : (N, 2) or (N, 3) full array positions.
    n_fft       : STFT FFT size.
    K_min       : minimum number of active mics (default 2).
    K_max       : maximum (default = N, i.e. all mics).
    snr_mid     : SNR in dB at which K = (K_min + K_max) / 2 (default 10).
    snr_scale   : logistic transition width in dB (default 6).
    doa_thresh  : DOA change in degrees that triggers re-selection.
    snr_thresh  : SNR change in dB that triggers re-selection.
    alpha_snr   : EMA coefficient for P_speech / P_noise tracking.
    stability_bias : score bonus (in the same units as the phase-spread
                     score) given to already-active mics, to reduce
                     unnecessary flapping between near-tied candidates.
    """

    def __init__(self,
                 mic_pos:    np.ndarray,
                 n_fft:      int   = F_WIN,
                 K_min:      int   = MIC_SEL_K_MIN,
                 K_max:      Optional[int] = None,
                 snr_mid:    float = MIC_SEL_SNR_MID,
                 snr_scale:  float = MIC_SEL_SNR_SCALE,
                 doa_thresh: float = MIC_SEL_DOA_THRESH,
                 snr_thresh: float = MIC_SEL_SNR_THRESH,
                 alpha_snr:  float = 0.97,
                 stability_bias: float = 0.05):

        self.mic_pos    = np.asarray(mic_pos, dtype=np.float32)
        self.n_fft      = n_fft
        self.N          = self.mic_pos.shape[0]
        self.K_min      = max(1, min(K_min, self.N))
        self.K_max      = self.N if K_max is None else max(self.K_min, min(K_max, self.N))
        self.snr_mid    = float(snr_mid)
        self.snr_scale  = float(snr_scale)
        self.doa_thresh = float(doa_thresh)
        self.snr_thresh = float(snr_thresh)
        self.alpha_snr  = float(alpha_snr)
        self.stability_bias = float(stability_bias)

        # Per-mic power accumulators
        self._P_speech = np.ones(self.N, dtype=np.float64) * 1e-6
        self._P_noise  = np.ones(self.N, dtype=np.float64) * 1e-6

        # State
        self._last_theta:  Optional[float] = None
        self._last_snr_db: Optional[float] = None
        self._selected:    np.ndarray      = np.arange(self.N, dtype=np.int32)
        self._current_K:   int             = self.N
        self._current_snr: float           = 0.0

        # Diagnostics
        self._n_reselections: int = 0
        self._n_updates:      int = 0

    # ── Power tracking (call every STFT frame) ────────────────────────────

    def update_power(self, X: np.ndarray, is_speech: bool) -> None:
        """
        Update per-mic broadband power EMA.

        Parameters
        ----------
        X        : (N, B) complex STFT frame (full mic array, before selection)
        is_speech: True during speech frames, False during noise frames
        """
        power = np.mean(np.abs(X.astype(np.complex128)) ** 2, axis=1)  # (N,)
        a = self.alpha_snr
        if is_speech:
            self._P_speech = a * self._P_speech + (1.0 - a) * power
        else:
            self._P_noise  = a * self._P_noise  + (1.0 - a) * power

    def snr_db(self) -> float:
        """Current broadband SNR estimate in dB (averaged across mics)."""
        p_s = float(np.mean(self._P_speech))
        p_n = float(np.mean(self._P_noise))
        return float(10.0 * np.log10(p_s / (p_n + 1e-12) + 1e-12))

    # ── Selection (call once per frame; internally gates on change) ───────

    def update(self, theta: float, phi: float = 0.0) -> bool:
        """
        Recompute mic selection if DOA or SNR has changed enough.

        Returns True if the active set changed, False otherwise. Cheap to
        call every frame — the expensive greedy re-selection only runs when
        a trigger fires.
        """
        self._n_updates += 1
        snr = self.snr_db()
        self._current_snr = snr

        doa_changed = False
        snr_changed = False

        if self._last_theta is not None:
            delta_deg = abs(np.degrees(theta - self._last_theta))
            if delta_deg > 180.0:
                delta_deg = 360.0 - delta_deg
            doa_changed = delta_deg > self.doa_thresh

        if self._last_snr_db is not None:
            snr_changed = abs(snr - self._last_snr_db) > self.snr_thresh

        if not doa_changed and not snr_changed and self._last_theta is not None:
            return False

        K_new   = self._k_from_snr(snr)
        old_set = set(self._selected.tolist())
        self._selected  = self._greedy_select(theta, phi, K_new)
        self._current_K = K_new
        self._last_theta  = theta
        self._last_snr_db = snr

        new_set = set(self._selected.tolist())
        changed = (new_set != old_set)
        if changed:
            self._n_reselections += 1
        return changed

    @property
    def selected(self) -> np.ndarray:
        """(K,) int32 indices of currently active mics."""
        return self._selected

    @property
    def current_K(self) -> int:
        return self._current_K

    def get_mask(self, dtype=np.float32) -> np.ndarray:
        """
        (N,) mask with 1.0 at selected mic indices, 0.0 elsewhere — the
        form MVDRBeamformer.compute_weights(mic_mask=...) expects.
        """
        mask = np.zeros(self.N, dtype=dtype)
        mask[self._selected] = 1.0
        return mask

    def describe(self) -> str:
        return (f"AdaptiveMicSel: K={self._current_K}/{self.N}  "
                f"SNR={self._current_snr:+.1f}dB  "
                f"mics={self._selected.tolist()}")

    def diagnostics(self) -> dict:
        return {
            'mic_sel_current_K':      self._current_K,
            'mic_sel_current_snr_db': self._current_snr,
            'mic_sel_reselections':   self._n_reselections,
            'mic_sel_updates':        self._n_updates,
            'mic_sel_active_mics':    self._selected.tolist(),
        }

    def print_diagnostics(self, prefix: str = '') -> None:
        d   = self.diagnostics()
        tag = f'[MicSel {prefix}] ' if prefix else '[MicSel] '
        print(f'{tag}K={d["mic_sel_current_K"]}  '
              f'SNR={d["mic_sel_current_snr_db"]:+.1f}dB  '
              f'reselections={d["mic_sel_reselections"]}/{d["mic_sel_updates"]}  '
              f'active={d["mic_sel_active_mics"]}')

    def reset(self) -> None:
        """Reset power accumulators and selection state between recordings."""
        self._P_speech[:]  = 1e-6
        self._P_noise[:]   = 1e-6
        self._last_theta   = None
        self._last_snr_db  = None
        self._selected     = np.arange(self.N, dtype=np.int32)
        self._current_K    = self.N
        self._current_snr  = 0.0
        self._n_reselections = 0
        self._n_updates       = 0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _k_from_snr(self, snr_db: float) -> int:
        """
        Logistic mapping: SNR → K.

        High SNR → K_min (quiet: fewer mics, fewer distortion sources).
        Low  SNR → K_max (noisy: use all mics for spatial rejection).
        """
        x = -(snr_db - self.snr_mid) / (self.snr_scale + 1e-9)
        sigma = 1.0 / (1.0 + np.exp(-float(np.clip(x, -20, 20))))
        K = int(round(self.K_min + (self.K_max - self.K_min) * sigma))
        return int(np.clip(K, self.K_min, self.K_max))

    def _greedy_select(self, theta: float, phi: float, K: int) -> np.ndarray:
        """
        Greedy APERTURE-MINIMISATION: iteratively add the mic that keeps the
        selected subset's physical footprint as COMPACT as possible, i.e.
        minimises the resulting max pairwise inter-mic distance. A small
        stability bias nudges ties toward mics that were already active.

        v1.2 fix — was previously "greedy phase-spread MAXIMISATION"
        --------------------------------------------------------------
        The old criterion picked, at each step, whichever mic maximised
        cross-mic phase std-dev relative to the current selection — i.e. it
        actively sought out the WIDEST-baseline mics. That is backwards for
        what K_min/low-K selections are supposed to achieve: per this
        module's own motivation (see module docstring), a small K is chosen
        specifically at HIGH input SNR to reduce "distortion sources" —
        steering-vector/calibration error and spatial aliasing. But spatial
        aliasing kicks in above c/(2*baseline) (see beamformer_2.py's
        DOA_JUMP_CHECK_FREQ_HZ / spatial-Nyquist notes), so the WIDEST-
        baseline mics are exactly the ones MOST exposed to that error — the
        old criterion was reliably picking the worst possible pair at low K.
        Concretely, on this array (see generate_synthetic_dataset.MIC_POSITIONS),
        at K=2 the old code always selected mics [4, 5] — the single widest
        pair in the whole array (0.150 m baseline, aliasing above ~1.1 kHz)
        — while the closest pair available is only 0.049 m (aliasing-safe to
        ~3.5 kHz). This directly explains the "MIC SELECTION EFFECT" table
        showing micsel reliably hurting SI-SDR (-0.4 to -1.2 dB) across
        every scenario: it was choosing the array's most aliasing-prone
        subset precisely when it mattered most.

        Diversity for interferer-nulling is still available where it
        actually matters: at low estimated SNR, K → K_max (near/at the full
        array) regardless of this criterion, since there's little room left
        to choose a strict subset. This criterion therefore mainly changes
        behaviour in the low-K / high-SNR regime, which is exactly where a
        compact, alias-safe subset was the intended outcome.

        Uses physical mic positions directly (not the look-direction-
        dependent steering phase) — a mic pair's aliasing frequency depends
        on their physical separation, not on which way the array happens to
        be steered right now, so this is both simpler and more robust than
        the phase-based version.
        """
        prev_set = set(self._selected.tolist())

        selected: List[int] = []
        remaining = list(range(self.N))

        # Seed with the most CENTRAL mic (smallest mean distance to every
        # other mic) — the natural starting point for a compact cluster.
        mean_dist = np.linalg.norm(
            self.mic_pos[:, None, :] - self.mic_pos[None, :, :], axis=-1
        ).mean(axis=1)
        seed_scores = -mean_dist  # higher = more central = better seed
        for i in remaining:
            if i in prev_set:
                seed_scores[i] += self.stability_bias
        seed = int(np.argmax(seed_scores))
        selected.append(seed)
        remaining.remove(seed)

        while len(selected) < K and remaining:
            best_score = -np.inf
            best_cand  = remaining[0]
            for cand in remaining:
                cand_set  = selected + [cand]
                pos       = self.mic_pos[cand_set]
                pairwise  = np.linalg.norm(
                    pos[:, None, :] - pos[None, :, :], axis=-1)
                max_baseline = float(pairwise.max())
                score = -max_baseline   # higher = more compact = better
                if cand in prev_set:
                    score += self.stability_bias
                if score > best_score:
                    best_score = score
                    best_cand  = cand
            selected.append(best_cand)
            remaining.remove(best_cand)

        return np.array(sorted(selected), dtype=np.int32)