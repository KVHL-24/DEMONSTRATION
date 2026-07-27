# beamformer_2.py — MVDR Beamformer with Woodbury R_nn⁻¹ update  (v7.2)
# =================================================================
#
# v7.2 additions (this revision) — numerical-divergence fix
# -------------------------------------------------------------
#   Root cause of the "s_max=14704297719.39" style blowups (and the
#   run-to-run non-reproducibility on IDENTICAL data/settings, and the
#   muted/inconsistent gazestab & micsel deltas):
#
#   R_nn⁻¹ (self._Rinv) was stored in single precision (complex64) and
#   downcast back to complex64 after EVERY recursive Woodbury update —
#   hundreds to thousands of sequential in-place updates per clip. Each
#   update is exact in infinite precision, but is also a subtraction of
#   two similar-magnitude terms (Rinv - vvH/s), i.e. exactly the kind of
#   step that is sensitive to rounding. Doing that repeatedly in single
#   precision let the matrix drift both asymmetric and, occasionally,
#   away from positive-definiteness over a long clip. compute_weights()'s
#   trace-normalisation (scale = N / trace(R_nn⁻¹)) only guarded against
#   trace <= 1e-30 — plenty permissive for the trace to have collapsed
#   toward zero from accumulated drift and still pass, producing an
#   unbounded `scale` and the observed ~1e10 weight blowup in that bin.
#
#   This also explains the reproducibility symptom: with no RNG anywhere
#   in this pipeline, the only source of run-to-run difference is
#   floating-point summation/reduction order (e.g. multi-threaded BLAS in
#   np.einsum/np.linalg.inv). Normally that's noise at the 1e-7 level —
#   invisible. But once R_nn⁻¹ has drifted close to singular in a bin,
#   the trace-normalisation is locally chaotic (a near-zero denominator
#   amplifies tiny input differences arbitrarily), so those normally
#   invisible 1e-7-level differences occasionally decided whether a bin
#   diverged on one run and not the other — producing the run-to-run
#   deltas seen even on byte-identical inputs. And because a diverged
#   bin's weights are effectively noise, they add variance that can
#   swamp the (real, but individually small) gains from gazestab/micsel,
#   which is why those ablation deltas looked inconsistent/near-zero
#   almost everywhere except the few scenarios where the effect was
#   large enough to survive it.
#
#   Fix, three parts (see update_noise() / compute_weights() below):
#     1. self._Rinv is now kept in complex128 throughout (init, updates,
#        DOA-jump decay, distortion recovery, reset) — no more per-frame
#        downcast. This alone reduces the per-update rounding error by
#        ~9 orders of magnitude, which stops the drift from accumulating
#        to a divergence within any realistic clip length.
#     2. Every Woodbury update now re-Hermitianises the result
#        (Rinv = (Rinv + Rinv^H)/2). Cheap, and removes the asymmetric-
#        drift half of the problem outright.
#     3. compute_weights() adds an explicit divergence guard: any bin
#        whose trace(R_nn⁻¹) has collapsed far below its expected order
#        of magnitude (N/reg) is treated as diverged, reset to the
#        isotropic prior, and counted in a new diagnostic
#        (trace_guard_resets) — self-healing instead of silently
#        producing an exploding scale.
#   Together these remove the chaotic amplifier rather than papering
#   over its symptoms, which is what actually restores reproducibility
#   (residual run-to-run differences are still theoretically possible
#   from BLAS thread nondeterminism, but no longer have anything left to
#   amplify them into visible SI-SDR deltas).
#
# v7.1 additions
# -------------------------------
#   • Adaptive mic-subset wiring. mic_selection.AdaptiveMicSelector was
#     already written but never actually connected: compute_weights() had
#     no way to exclude mics from the output combination. MVDRBeamformer
#     can now own a selector directly (mic_selector=...) or accept an
#     externally-computed mic_mask per call. R_nn⁻¹ still updates from ALL
#     mics on every noise frame regardless of the mask — only the final
#     weight combination is restricted — so this is a clean exclusion, not
#     an approximation of a reduced-dimension solve (see compute_weights()
#     docstring and mic_selection.py's module docstring for the full
#     argument).
#   • DOA-jump detector aliasing fix. The detector compared steering
#     vectors at bin n_bins//2 (~12 kHz for a 512-pt FFT @ 48 kHz), which
#     is far above the spatial-Nyquist limit of any handheld mic array
#     (half-wavelength at 12 kHz ≈ 1.4 cm; e.g. Aria's aperture is ~14 cm).
#     Above that limit, small true angle changes alias into large apparent
#     phase differences, so the detector could fire on noise rather than
#     real jumps — only visible on oracle_target_dir, the only caller that
#     runs with doa_reliable=True. Now compares at a configurable, safely
#     low frequency bin (default 800 Hz) instead.
#
# v7.0 additions
# -------------------------------
#   • Distortionless-constraint-preserving frequency-axis weight smoothing.
#     The v5.3 smoothing kernel blended weight vectors that were each
#     computed against a DIFFERENT per-bin steering-vector phase, silently
#     breaking the MVDR distortionless constraint a[f]^H w[f] = 1 and
#     adding a broadband target distortion. That distortion is invisible
#     at low input SNR (real noise dominates the error budget) but becomes
#     the dominant source of SI-SDR loss once little real noise remains to
#     remove — exactly the high-SNR regime (e.g. +20 dB) where oracle_gaze
#     was measured to fall below raw_mic across almost every scenario.
#   • SNR-adaptive wet/dry output blending. Every corrective stage (MVDR,
#     cancellation fallback, spectral post-filter) still adds *some*
#     residual distortion, which is only worth paying for when there is
#     real noise to remove. Blends the final output back toward the raw
#     mic-0 signal as a running input-SNR estimate climbs.
#   • Noise-frame purity gate. Rejects candidate noise frames whose
#     Mahalanobis distance against the current R_nn^-1 is an outlier
#     relative to recently-accepted frames, keeping leaked target /
#     off-axis reverberation energy (VAD false negatives) out of R_nn.
#     Aimed at the chronic self-cancellation in directional_* scenes.
#     This is the most speculative of the three changes — ablate via
#     OUTLIER_GATE_ENABLED if it doesn't help on your dataset.
#
# Note on noise-covariance updates during speech
# ------------------------------------------------
# An experiment was run where R_nn was also updated during speech frames,
# by subtracting an estimated noise component from the mixture before
# folding the residual into the Woodbury update. That made things WORSE,
# especially for stationary noise (white/pink). This makes sense in
# hindsight: for stationary noise the noise-only estimate from silence
# frames is already close to the true R_nn (nothing about the noise
# changes between speech and silence), so there was no statistical
# upside to updating during speech — only the downside of injecting
# fresh estimation error (from the subtraction step itself) directly into
# the covariance model on every frame instead of only during confirmed
# silence. R_nn is therefore only ever updated during confirmed
# non-speech frames, as before.


from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stft import B, F_WIN, FS   # FS = 48 000 — always import, never hardcode

if TYPE_CHECKING:
    # Avoid a hard/circular import at module load; MVDRBeamformer only
    # needs this for the mic_selector type hint (v7.1).
    from mic_selection import AdaptiveMicSelector

# ── Module-level constants ─────────────────────────────────────────────────────
ALPHA_DEFAULT  = 0.995
ALPHA_WARMUP   = 0.900   # fast initial convergence (v5.1)
N_WARMUP       = 30      # noise frames before switching to ALPHA_DEFAULT (v5.1)
REG            = 3e-2

# v7.2: numerical-divergence guard (see module docstring above). A bin's
# trace(R_nn⁻¹) is expected to stay within an order of magnitude of its
# initialisation value N/reg. TRACE_GUARD_FRAC is how far below that a
# bin is allowed to fall before it's treated as diverged (accumulated
# rounding drift or a genuinely rank-deficient noise field) and reset to
# the isotropic prior, rather than being allowed to produce an unbounded
# trace-normalisation scale. 1e-4 is deliberately conservative — normal
# operating traces stay within 1-2 orders of magnitude of N/reg even
# under heavy noise, while the observed divergence collapsed trace by
# ~9 orders of magnitude, so there is ample separation.
TRACE_GUARD_FRAC = 1e-4

# v7.7: adaptive trace-CEILING guard — the opposite side of the guard
# above. TRACE_GUARD_FRAC catches trace(R_nn⁻¹) collapsing toward zero; it
# can use a FIXED threshold (a multiple of the fixed init value N/reg)
# because "collapsed near zero" is unambiguous in any scenario. The
# ceiling side can't use a fixed threshold the same way: a single strong,
# correctly-nulled point-source interferer is SUPPOSED to drive trace far
# above N/reg (see the v7.6 eigenvalue-spread hypothesis in
# MVDRBeamformer's diagnostics — a coherent point source concentrates
# inverse-noise-power in the directions orthogonal to it, unlike diffuse
# noise, which spreads it evenly). A fixed ceiling would either be loose
# enough to miss real instability or tight enough to clip legitimate
# directional nulling.
#
# So this guard tracks its own adaptive per-bin baseline instead: a slow
# EMA of trace, updated every frame from bins that are NOT currently over
# their own ceiling (so a genuine spike can't drag its own baseline up
# with it). A bin only trips the guard when it spikes far above ITS OWN
# recent normal level — catching a transient blow-up/instability in that
# bin specifically, not the expected steady-state elevation a real
# directional interferer produces. TRACE_CEIL_MIN_FRAMES gates the guard
# off until the baseline has had enough frames to be trustworthy (every
# bin looks "new" on frame 1).
TRACE_CEIL_MULT       = 50.0   # trace > baseline * this ⇒ treat as runaway
TRACE_CEIL_EMA_RATE   = 0.01   # deliberately slow — must not chase the spike
TRACE_CEIL_MIN_FRAMES = 20     # frames before the ceiling baseline is trusted

# v6.2: lowered from 15 dB and raised blend vs v5.2
CANCEL_THRESH_DB = 10.0  # dB below mean input → suspect target cancellation
CANCEL_BLEND     = 0.65  # fraction of DAS blended in cancelled bins (raised from 0.40)

POST_ALPHA    = 0.92    # PSD smoothing coefficient in SpectralPostFilter (v5.0)
POST_MU       = 1.0     # over-subtraction factor  (1.0 = standard Wiener)
POST_BETA     = 1.0     # Wiener gain exponent  (1 = power, 0.5 = amplitude)
POST_FLOOR    = 0.10    # minimum Wiener gain (prevents full nulling) (v5.0)

SMOOTH_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)   # 3-tap (v5.3)

# v6.0: target-direction projection guard on noise-frame covariance updates
#
# v7.3 fix — disabled by default (root cause of the pink/white-noise
# regression, ~7 dB of lost array gain vs raw_mic/srp)
# -------------------------------------------------------------------------
# The guard subtracts the component of EVERY noise-frame sample along the
# current look direction d before it is folded into R_nn⁻¹ (see
# update_noise()). That is a permanent, unconditional rank-1 projection
# applied on every single noise frame, not just frames that actually
# contain leaked target energy. For a genuinely diffuse/isotropic noise
# field (white_noise, pink_noise — independent per mic, no privileged
# direction) real noise energy DOES exist along d, exactly like every
# other direction, and this guard silently discards that fraction of the
# noise statistics on every update. Over hundreds of noise frames this
# starves R_nn⁻¹ of information specifically along d while it keeps
# learning normally along the other N-1 dimensions, producing an
# artificial anisotropy (R_nn⁻¹ stays near its untouched 1/reg prior along
# d, but shrinks toward the true, larger noise level elsewhere). MVDR then
# "sees" an artificially noise-free look direction and an artificially
# noisy everywhere-else, which is precisely backwards for isotropic noise
# and repeatedly trips the v6.1 distortionless-collapse repair and the
# v5.2/v6.2 cancel-fallback — measured empirically at ~7 dB of lost array
# gain (diffuse noise, clean VAD, no leakage at all) vs the same run with
# the guard off, at every input SNR tested. The loss is specific to
# diffuse noise because directional interferers (babble/cocktail/
# directional_*) don't carry energy along d in the first place, so the
# projection removes ~nothing there (< 0.5 dB either way) — matching the
# observed pattern of pink/white noise losing far more ground than the
# directional scenarios.
#
# The v7.0 noise-frame purity gate (OUTLIER_GATE_ENABLED) already covers
# the failure mode this guard was meant to prevent — VAD false negatives
# leaking target energy into a noise frame — by rejecting the WHOLE
# contaminated frame based on its Mahalanobis distance, rather than
# always removing a fixed subspace from every frame regardless of
# whether that frame is actually contaminated. Measured with synthetic
# VAD-leakage injected at up to 30% of noise frames, purity-gate-only
# (guard off) held array gain essentially flat (6.21 → 6.20 dB) while the
# projection guard stayed pinned at its degraded ~-1.1 dB regardless of
# leakage level — i.e. it wasn't even doing its intended job well.
# Kept as a constructor flag (use_projection_guard=True) for ablation.
PROJECTION_GUARD = False  # v7.3: default OFF — see rationale above

# v6.1: distortionless-constraint recovery
DISTORT_RATIO_MIN       = 0.35   # s/s_das below this → self-null detected
DISTORT_RECOVERY_BLEND  = 0.70   # keep 70% of R_nn⁻¹, blend 30% toward prior
DISTORT_CHECK_FRAC      = 0.20   # fraction of bins collapsed → trigger global recompute

# v7.0: distortionless-constraint-preserving frequency smoothing.
# See module docstring above for rationale.
RENORMALIZE_AFTER_SMOOTH = True

# v7.0: noise-frame purity gate. Rejects candidate noise frames whose
# Mahalanobis distance against the current R_nn^-1 model is an outlier
# relative to recently-accepted noise frames — catches VAD false
# negatives (leaked target tail / off-axis reverberant target energy)
# before they teach R_nn to null part of the target.
OUTLIER_GATE_ENABLED        = True
OUTLIER_MIN_FRAMES          = 20     # frames to establish baseline before gating
OUTLIER_REJECT_Z            = 4.0    # conservative z-score threshold
OUTLIER_BASELINE_ALPHA      = 0.05   # EMA rate for the accepted-frame baseline
OUTLIER_MAX_CONSEC_REJECTS  = 50     # safety valve — force-accept after this many
                                      # consecutive rejections so a genuine
                                      # envelope jump (dynamic noise scenarios)
                                      # can never permanently lock the gate

# v7.0: SNR-adaptive wet/dry output blending. The MVDR + cancellation
# fallback + spectral post-filter chain is tuned to remove real noise; at
# high input SNR there is little real noise to remove, so the small
# residual distortion each stage still introduces becomes the dominant
# error term. Blend the final output back toward the raw mic-0 signal in
# proportion to a running estimate of input SNR.
SNR_BLEND_ENABLED    = True
SNR_BLEND_LOW_DB     = 0.0    # at/below this estimated SNR: fully beamformed
# v7.1.1: eval evidence (babble/cocktail/directional_* at the +20 dB SNR
# column) showed 3-9 dB of residual loss vs raw_mic even with the previous
# HIGH_DB=18/WET_FLOOR=0.25 pair — the corrective chain was still fully (or
# near-fully) wet well past the point where it had real noise left to
# remove. Ramping wetness down starting at a lower estimated SNR, and
# floor-ing it higher, blends more raw signal back in for exactly that
# regime. Re-validate against eval_synthetic_2.py after changing either
# value — too low a HIGH_DB / too high a FLOOR will start giving back the
# gains this pipeline has on pink/white noise at moderate SNR.
SNR_BLEND_HIGH_DB    = 12.0   # at/above this estimated SNR: floor wetness
SNR_BLEND_WET_FLOOR  = 0.45   # minimum beamformer contribution, even at very high SNR
SNR_POWER_ALPHA      = 0.95   # EMA rate for noise/speech frame power tracking
SNR_BLEND_MIN_FRAMES = 10     # frames of each class required before trusting the estimate

# v7.1: DOA-jump detector frequency-bin fix. The detector must compare
# steering-vector phase at a frequency below the array's spatial-Nyquist
# limit (bin_freq < c / (2 * max_aperture)), or angle changes well under
# DOA_JUMP_THRESH alias into large apparent phase differences and the
# detector fires on noise instead of real jumps. n_bins // 2 (~12 kHz for
# a 512-pt FFT at 48 kHz) is far above that limit for any handheld-array
# aperture (e.g. Aria's ~14 cm), which is the likely cause of the detector
# firing on almost every frame — including static scenes — for
# oracle_target_dir (the only mode that runs with doa_reliable=True).
# 800 Hz keeps a safe margin for apertures up to ~21 cm; override via
# doa_jump_freq_hz in the constructor if your array is larger.
DOA_JUMP_CHECK_FREQ_HZ = 800.0


# ── Spectral post-filter ───────────────────────────────────────────────────────
class SpectralPostFilter:
    """
    Single-channel Wiener post-filter applied to the MVDR beamformer output.

    Tracks residual noise PSD P_nn[f] from MVDR output during VAD=0 frames.
    During VAD=1 frames, applies the Wiener-inspired gain:

        G[f] = clip( 1 − mu · P_nn[f] / (P_yy[f] + eps) , floor, 1 )^beta

        y_pf[f] = G[f]^0.5 · y_mvdr[f]          (magnitude only, phase kept)

    For POST_BETA = 1.0 (default) this is the standard Wiener filter.
    For POST_BETA = 0.5 the gain is applied to the amplitude spectrum, which
    reduces musical noise at the cost of slightly less noise suppression.

    Parameters
    ----------
    n_bins    : number of STFT bins (B = n_fft//2 + 1)
    alpha     : EMA for noise PSD tracking (0 < alpha < 1, larger = slower)
    mu        : over-subtraction factor  (1.0 standard, > 1.0 more aggressive)
    beta      : exponent applied to the Wiener gain before multiplying y
    floor     : minimum gain value in [0, 1]  (prevents total nulling)
    """

    def __init__(self,
                 n_bins: int,
                 alpha: float = POST_ALPHA,
                 mu:    float = POST_MU,
                 beta:  float = POST_BETA,
                 floor: float = POST_FLOOR):
        self.n_bins = n_bins
        self.alpha  = float(alpha)
        self.mu     = float(mu)
        self.beta   = float(beta)
        self.floor  = float(floor)
        self._P_nn  = np.ones(n_bins, dtype=np.float64) * 1e-6

        # Diagnostic
        self._gain_history: list[float] = []

    def update_noise(self, y: np.ndarray) -> None:
        """Update noise PSD estimate from an MVDR output during a noise frame."""
        P = np.abs(y.astype(np.complex128)) ** 2          # (B,)
        self._P_nn = self.alpha * self._P_nn + (1.0 - self.alpha) * P

    def apply(self, y: np.ndarray) -> np.ndarray:
        """
        Apply Wiener gain to MVDR output y during a speech frame.

        Parameters
        ----------
        y : (B,) complex MVDR beamformer output

        Returns
        -------
        y_pf : (B,) complex, post-filtered output
        """
        P_yy = np.abs(y.astype(np.complex128)) ** 2        # (B,)
        # Wiener gain: G = (1 - mu * P_nn / P_yy) ^ beta
        G = np.clip(
            1.0 - self.mu * self._P_nn / (P_yy + 1e-12),
            self.floor,
            1.0
        ) ** self.beta                                      # (B,) in [floor^beta, 1]

        self._gain_history.append(float(G.mean()))
        return (np.sqrt(G) * y.astype(np.complex128)).astype(np.complex64)

    def reset(self) -> None:
        """Reset noise PSD estimate (call at start of each clip)."""
        self._P_nn[:] = 1e-6
        self._gain_history.clear()

    def diagnostics(self) -> dict:
        g = self._gain_history
        arr = np.asarray(g) if g else np.array([float('nan')])
        return {
            'post_gain_mean': float(np.nanmean(arr)),
            'post_gain_std':  float(np.nanstd(arr)),
            'post_gain_min':  float(np.nanmin(arr)),
        }


# ── MVDR Beamformer ────────────────────────────────────────────────────────────
class MVDRBeamformer:
    """
    Per-frequency-bin MVDR beamformer with Woodbury R_nn⁻¹ update.

    Internally stores R_nn⁻¹[f] for all B bins.  During noise frames the
    inverse is updated via the Woodbury rank-1 identity (O(N²) per bin).
    During speech frames the MVDR weight vector is computed and applied,
    optionally followed by the SpectralPostFilter (v5.0).

    v6 adds a target-direction projection guard on noise-frame updates (v6.0),
    a distortionless-constraint recovery mechanism (v6.1), and a more
    aggressive DAS fallback backstop (v6.2).

    v7.0 adds a fix for a real distortionless-constraint violation introduced
    by v5.3's frequency smoothing (RENORMALIZE_AFTER_SMOOTH), a noise-frame
    purity gate that rejects R_nn updates from frames whose statistics look
    anomalous relative to recently-accepted noise (OUTLIER_GATE_ENABLED),
    and an SNR-adaptive wet/dry output blend that backs off all of the
    above machinery once the estimated input SNR is high enough that the
    machinery's own residual distortion outweighs its benefit
    (SNR_BLEND_ENABLED). See module docstring for the full rationale.

    v7.1 adds real integration with mic_selection.AdaptiveMicSelector via
    an explicit `mic_mask` argument on compute_weights()/process_frame()
    (own a selector directly with `mic_selector=`, or drive selection
    externally and pass `mic_mask=` per call), and fixes a spatial-aliasing
    bug in the DOA-jump detector (see DOA_JUMP_CHECK_FREQ_HZ above).

    Parameters
    ----------
    N          : number of active microphones
    n_fft      : FFT size (default 512 → 257 bins)
    alpha      : EMA coefficient for R_nn (0 < α < 1).  Larger → slower.
                 α=0.97 (TC ≈ 5 s) when noise frames are scarce.
                 α=0.995 (TC ≈ 27 s) when silence is abundant.
    reg        : diagonal regularisation added to R_nn⁻¹ initialisation.
    use_postfilter : enable SpectralPostFilter stage (v5.0, default True).
    use_projection_guard : enable v6.0 steering-vector projection guard
                 on noise-frame covariance updates (default True).
    use_outlier_gate : enable v7.0 noise-frame purity gate (default True).
    outlier_reject_z : z-score threshold for the purity gate (default 4.0).
    use_snr_adaptive_blend : enable v7.0 SNR-adaptive wet/dry output blend
                 (default True).
    snr_blend_low_db, snr_blend_high_db, snr_blend_wet_floor :
                 thresholds/floor for the SNR-adaptive blend — see module
                 constants for defaults and rationale.
    mic_selector : optional mic_selection.AdaptiveMicSelector instance
                 (v7.1). If given, process_frame() drives it automatically
                 each speech frame (update_power() every frame, update()
                 + get_mask() on speech frames) whenever an explicit
                 `mic_mask` isn't passed in for that call. R_nn⁻¹ always
                 keeps updating from the full mic array regardless of the
                 mask; only the final weight combination is restricted.
    doa_jump_freq_hz : frequency (Hz) at which the DOA-jump detector
                 compares steering-vector phase (v7.1). Must stay below
                 the array's spatial-Nyquist limit c / (2 * max_aperture)
                 or the detector aliases — see module docstring.
    """

    def __init__(self,
                 N:               int,
                 n_fft:           int   = F_WIN,
                 alpha:           float = ALPHA_DEFAULT,
                 reg:             float = REG,
                 use_postfilter:  bool  = True,
                 use_projection_guard: bool = PROJECTION_GUARD,
                 use_outlier_gate: bool = OUTLIER_GATE_ENABLED,
                 outlier_reject_z: float = OUTLIER_REJECT_Z,
                 use_snr_adaptive_blend: bool = SNR_BLEND_ENABLED,
                 snr_blend_low_db:   float = SNR_BLEND_LOW_DB,
                 snr_blend_high_db:  float = SNR_BLEND_HIGH_DB,
                 snr_blend_wet_floor: float = SNR_BLEND_WET_FLOOR,
                 mic_selector: 'AdaptiveMicSelector | None' = None,
                 doa_jump_freq_hz: float = DOA_JUMP_CHECK_FREQ_HZ):
        self.N             = N
        self.n_bins        = n_fft // 2 + 1
        self.alpha         = alpha
        self.reg           = reg
        self.use_postfilter       = use_postfilter
        self.use_projection_guard = use_projection_guard

        # R_nn⁻¹[f] — shape (B, N, N) complex128 (v7.2: was complex64 — see
        # module docstring for why single-precision recursive storage was
        # the root cause of the numerical-divergence/reproducibility bug).
        self._Rinv = np.stack(
            [(1.0 / reg) * np.eye(N, dtype=np.complex128)
             for _ in range(self.n_bins)]
        )

        # MVDR weights — shape (B, N) complex64
        self._weights       = np.zeros((self.n_bins, N), dtype=np.complex64)
        self._weights_valid = False

        # ── Frequency-dependent alpha (v4) ────────────────────────────────
        # Low frequencies have more noise corr. structure → faster adaptation.
        freqs     = np.fft.rfftfreq(n_fft, 1.0 / FS)
        F_HI      = 1000.0
        alpha_low = max(0.97, alpha - 0.015)
        t_f       = np.clip(freqs / F_HI, 0.0, 1.0)
        self._alpha_f     = (alpha_low + t_f * (alpha - alpha_low)
                             ).astype(np.float64)
        self._scale_f     = self._alpha_f / (1.0 - self._alpha_f)
        self._inv_alpha_f = 1.0 / self._alpha_f

        # ── v7.1: aliasing-safe bin for the DOA-jump detector ──────────────
        # Replaces the old n_bins // 2 midpoint, which sits at ~12 kHz for a
        # 512-pt FFT @ 48 kHz — well past the spatial-Nyquist limit of any
        # handheld array, causing false jump detections (see module docstring).
        self.doa_jump_freq_hz = float(doa_jump_freq_hz)
        self._doa_check_bin   = int(np.argmin(np.abs(freqs - doa_jump_freq_hz)))

        # Pre-built identity stack
        self._eye_stack = np.eye(N, dtype=np.complex128)[None]   # (1, N, N)

        # Isotropic prior for DOA-jump reset and v6.1 recovery
        sigma2 = 1e-3
        self._iso_prior = (
            (1.0 / sigma2) * np.eye(N, dtype=np.complex128)[None]
        )                                                          # (1, N, N)

        # ── Warmup tracking (v5.1) ────────────────────────────────────────
        self._noise_frame_count = 0

        # ── Post-filter (v5.0) ────────────────────────────────────────────
        self._postfilter = SpectralPostFilter(self.n_bins) if use_postfilter else None

        # ── v6.0: last known steering vector for projection guard ─────────
        # Set by process_frame() on every call so update_noise() can access it.
        self._last_d: np.ndarray | None = None   # (N, B) complex64

        # ── v7.0: noise-frame purity gate state ────────────────────────────
        self.use_outlier_gate  = use_outlier_gate
        self.outlier_reject_z  = outlier_reject_z
        self._mahal_mean             = 0.0
        self._mahal_var              = 1.0
        self._mahal_n                = 0
        self._consec_outlier_rejects = 0
        self._outlier_rejects        = 0
        self._outlier_forced_accepts = 0

        # ── v7.0: SNR-adaptive blend state ─────────────────────────────────
        self.use_snr_adaptive_blend = use_snr_adaptive_blend
        self.snr_blend_low_db    = snr_blend_low_db
        self.snr_blend_high_db   = snr_blend_high_db
        self.snr_blend_wet_floor = snr_blend_wet_floor
        self._noise_power_ema  = 1e-6
        self._speech_power_ema = 1e-6
        self._n_noise_pwr      = 0
        self._n_speech_pwr     = 0
        self._wet_history: list[float] = []

        # ── v7.1: adaptive mic selection (optional) ─────────────────────────
        self.mic_selector = mic_selector

        # ── Diagnostics ───────────────────────────────────────────────────
        self._nan_resets      = 0
        self._frames_noise    = 0
        self._frames_speech   = 0
        self._cancel_events   = 0     # bins blended back to DAS (v5.2)
        self._distort_resets  = 0     # partial R_nn resets from v6.1
        self._trace_guard_resets = 0  # v7.2: numerical-divergence resets
        # v7.7: adaptive trace-CEILING guard state — see module docstring.
        self._trace_ceil_baseline: np.ndarray | None = None  # (n_bins,) EMA
        self._trace_ceil_nframes  = 0   # frames contributing to baseline
        self._trace_ceil_resets   = 0   # v7.7: runaway-trace resets
        self._sigma2_init     = 1e-3
        # Per-frame output/input power ratio (for std, v5.4)
        self._out_in_ratio_dB: list[float] = []
        # v7.6: per-frame trace-normalization scale (N/trace(Rinv)) and raw
        # trace, min/mean across bins each speech frame. `scale` uniformly
        # rescales Rinv before adding reg*I, so a large spread in Rinv's
        # eigenvalues (expected to be much larger for a single strong
        # point-source interferer than for diffuse/multi-talker noise,
        # since the former drives most directions' inverse-noise-power
        # very high while leaving the interferer's own direction low)
        # could make reg*I's effective contribution swing frame-to-frame
        # in a way that diffuse/babble noise doesn't — this is the next
        # candidate explanation for directional_* running ~10x higher
        # cancel/distort counts than babble/cocktail after PROJECTION_GUARD
        # was ruled out (it made every scenario worse, not better).
        self._scale_history: list[float] = []   # per-frame mean(scale)
        self._scale_min_history: list[float] = []  # per-frame min(scale)
        self._trace_history: list[float] = []   # per-frame mean(trace)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        ratios = np.asarray(self._out_in_ratio_dB) if self._out_in_ratio_dB \
                 else np.array([float('nan')])
        d = {
            'n_bins':            self.n_bins,
            'nan_resets':        self._nan_resets,
            'frames_noise':      self._frames_noise,
            'frames_speech':     self._frames_speech,
            'cancel_events':     self._cancel_events,
            'distort_resets':    self._distort_resets,
            'trace_guard_resets': self._trace_guard_resets,
            'trace_ceil_resets': self._trace_ceil_resets,  # v7.7
            'rinv_nan_vals':     int(np.sum(~np.isfinite(self._Rinv))),
            'out_in_ratio_mean': float(np.nanmean(ratios)),
            'out_in_ratio_std':  float(np.nanstd(ratios)),
            'out_in_ratio_min':  float(np.nanmin(ratios)),
        }
        if self._postfilter is not None:
            d.update(self._postfilter.diagnostics())

        # ── v7.0 ─────────────────────────────────────────────────────────
        d['outlier_rejects']        = self._outlier_rejects
        d['outlier_forced_accepts'] = self._outlier_forced_accepts
        d['mahal_baseline_mean']    = self._mahal_mean
        d['mahal_baseline_std']     = float(np.sqrt(max(self._mahal_var, 0.0)))
        if self._wet_history:
            wa = np.asarray(self._wet_history)
            d['wet_mean'] = float(wa.mean())
            d['wet_min']  = float(wa.min())
        else:
            d['wet_mean'] = float('nan')
            d['wet_min']  = float('nan')
        d['est_snr_db_last'] = getattr(self, '_last_est_snr_db', float('nan'))

        # ── v7.6 ─────────────────────────────────────────────────────────
        if self._scale_history:
            sa = np.asarray(self._scale_history)
            sm = np.asarray(self._scale_min_history)
            ta = np.asarray(self._trace_history)
            d['scale_mean']      = float(sa.mean())
            d['scale_min']       = float(sm.min())
            d['scale_spread']    = float(sa.max() / max(sa.min(), 1e-30))
            d['trace_mean']      = float(ta.mean())
        else:
            d['scale_mean'] = d['scale_min'] = d['scale_spread'] = d['trace_mean'] = float('nan')

        # ── v7.1 ─────────────────────────────────────────────────────────
        d['doa_check_bin']    = self._doa_check_bin
        d['doa_check_freq_hz'] = self.doa_jump_freq_hz
        if self.mic_selector is not None:
            d.update(self.mic_selector.diagnostics())
        return d

    def print_diagnostics(self, prefix: str = '') -> None:
        """Pretty-print all diagnostic values including mean ± std."""
        d = self.diagnostics()
        tag = f'[BF {prefix}] ' if prefix else '[BF] '
        print(f'{tag}frames  noise={d["frames_noise"]}  speech={d["frames_speech"]}')
        print(f'{tag}nan_resets={d["nan_resets"]}  '
              f'cancel_events={d["cancel_events"]}  '
              f'distort_resets={d["distort_resets"]}  '
              f'trace_guard_resets={d["trace_guard_resets"]}  '
              f'trace_ceil_resets={d["trace_ceil_resets"]}  '
              f'rinv_nans={d["rinv_nan_vals"]}')
        print(f'{tag}out/in ratio (dB): '
              f'mean={d["out_in_ratio_mean"]:+.2f}  '
              f'std={d["out_in_ratio_std"]:.2f}  '
              f'min={d["out_in_ratio_min"]:+.2f}')
        if 'post_gain_mean' in d:
            print(f'{tag}post-filter gain: '
                  f'mean={d["post_gain_mean"]:.3f}  '
                  f'std={d["post_gain_std"]:.3f}  '
                  f'min={d["post_gain_min"]:.3f}')
        print(f'{tag}outlier_gate: rejects={d["outlier_rejects"]}  '
              f'forced_accepts={d["outlier_forced_accepts"]}  '
              f'baseline={d["mahal_baseline_mean"]:.3f}±{d["mahal_baseline_std"]:.3f}')
        print(f'{tag}snr_blend: wet_mean={d["wet_mean"]:.3f}  '
              f'wet_min={d["wet_min"]:.3f}  '
              f'est_input_snr_last={d["est_snr_db_last"]:+.1f} dB')
        print(f'{tag}doa_jump_check: bin={d["doa_check_bin"]} '
              f'(~{d["doa_check_freq_hz"]:.0f} Hz)')
        if self.mic_selector is not None:
            print(f'{tag}mic_sel: K={d.get("mic_sel_current_K")}  '
                  f'SNR={d.get("mic_sel_current_snr_db", float("nan")):+.1f}dB  '
                  f'reselections={d.get("mic_sel_reselections")}')

    def reset_diagnostics(self) -> None:
        self._nan_resets = self._frames_noise = self._frames_speech = 0
        self._cancel_events = 0
        self._distort_resets = 0
        self._trace_guard_resets = 0
        self._trace_ceil_resets = 0
        self._out_in_ratio_dB.clear()
        self._outlier_rejects        = 0
        self._outlier_forced_accepts = 0
        self._wet_history.clear()
        self._scale_history.clear()
        self._scale_min_history.clear()
        self._trace_history.clear()

    # ── Diffuse-field / isotropic initialisation ──────────────────────────────

    def init_diffuse(self, H_rel: np.ndarray) -> None:
        """
        Initialise R_nn⁻¹ from a measured ATF table (spherically isotropic
        diffuse-field approximation).

        Parameters
        ----------
        H_rel : (B, D, N) complex — mic-0-normalised ATF table.
        """
        B_atf, D, N = H_rel.shape
        assert N == self.N
        for f in range(self.n_bins):
            H_f = H_rel[min(f, B_atf - 1)].astype(np.complex128)
            R_f = (H_f.T @ H_f.conj()) / D + self.reg * np.eye(N, dtype=np.complex128)
            self._Rinv[f] = np.linalg.inv(R_f)   # v7.2: no complex64 downcast
        self._weights_valid = False
        print(f'[Beamformer] Diffuse-field R_nn⁻¹ init: {D} dirs × {self.n_bins} bins '
              f'(N={self.N})')

    def init_isotropic(self) -> None:
        """Spatially flat noise prior → Delay-and-Sum weights from frame 0."""
        Rinv_init = (1.0 / self._sigma2_init) * np.eye(self.N, dtype=np.complex128)
        for f in range(self.n_bins):
            self._Rinv[f] = Rinv_init.copy()
        self._weights_valid = False
        print(f'[Beamformer] Isotropic R_nn⁻¹ init (sigma²={self._sigma2_init:.0e})')

    # ── Stage 3A+3B : Woodbury rank-1 update (vectorised) ─────────────────────

    def update_noise(self, X: np.ndarray) -> None:
        """
        Update R_nn⁻¹ from a noise-only STFT frame — vectorised over all B bins.

        Woodbury per bin:
            v[b]       = R⁻¹[b] u[b]
            mahal[b]   = Re(u†v)                       (Mahalanobis term)
            s[b]       = mahal[b] + α/(1−α)
            R⁻¹_new[b] = (1/α)(R⁻¹[b] − vv† / s[b])

        v5.1: For the first N_WARMUP noise frames, alpha = ALPHA_WARMUP (0.90)
              to allow rapid initial convergence from the identity prior.

        v6.0: Projection guard — project out the current target steering
              direction from each STFT bin before the Woodbury update.
              This prevents target energy that leaked into nominally
              noise-only frames from teaching R_nn⁻¹ to null the target
              direction.

              For each bin b:
                d̂[b]   = d[:, b] / ‖d[:, b]‖
                u_proj  = u − (d̂ d̂ᴴ) u = (I − d̂d̂ᴴ) u

              The projection is applied whenever self._last_d is available
              (i.e. after process_frame() has run at least once).  Bins
              with a near-zero steering vector norm are passed through
              unmodified via the norm guard.

        v7.0: Noise-frame purity gate. The projection guard above only
              removes leakage along the exact direct-path steering
              direction; it cannot remove target energy that arrives via
              room reflections from other directions, nor a VAD false
              negative that lets a chunk of genuine (un-attenuated) target
              through. Before folding the (post-projection) residual into
              R_nn, its frame-level Mahalanobis score against the CURRENT
              R_nn⁻¹ is compared to a running baseline built from
              recently-accepted noise frames. A frame that doesn't look
              like "normal noise" relative to that baseline is rejected
              outright — R_nn is left untouched and the frame does not
              count toward the warmup schedule. The baseline is a fast
              EMA so genuinely time-varying (dynamic) noise is still
              tracked; a safety valve force-accepts after
              OUTLIER_MAX_CONSEC_REJECTS consecutive rejections so a real
              noise-envelope jump can never permanently lock the gate.

        Note (v7.1): this method deliberately still consumes the FULL
        N-mic frame regardless of any mic_mask in effect for the output
        stage — the noise-covariance model always benefits from more
        statistics, so it is never restricted to a mic subset. Only
        compute_weights()/apply() are subset-aware. See mic_selection.py.

        Energy gate skips near-silent frames to preserve null shape during
        dynamic noise envelope troughs.
        """
        ENERGY_FLOOR = 1e-6
        if float(np.mean(np.abs(X) ** 2)) < ENERGY_FLOOR:
            return

        # ── v5.1 warmup alpha (based on count BEFORE any update this call) ──
        if self._noise_frame_count < N_WARMUP:
            warmup_ratio  = self._noise_frame_count / max(N_WARMUP - 1, 1)
            eff_alpha_f   = ALPHA_WARMUP + warmup_ratio * (self._alpha_f - ALPHA_WARMUP)
            eff_scale_f   = eff_alpha_f / (1.0 - eff_alpha_f)
        else:
            eff_alpha_f   = self._alpha_f
            eff_scale_f   = self._scale_f

        u = X.T.astype(np.complex128)            # (B, N)

        # ── v6.0 : projection guard ───────────────────────────────────────
        if self.use_projection_guard and self._last_d is not None:
            d_bn   = self._last_d.T.astype(np.complex128)              # (B, N)
            d_norm = np.linalg.norm(d_bn, axis=1, keepdims=True)        # (B, 1)
            d_safe = np.where(d_norm > 1e-9, d_bn / (d_norm + 1e-12), 0.0)  # (B, N)
            proj_coeff = np.einsum('bn,bn->b', d_safe.conj(), u)        # (B,)
            u = u - proj_coeff[:, None] * d_safe                        # (B, N)

        Rinv = self._Rinv     # (B, N, N) complex128 — pre-update model (v7.2: no cast needed)

        v       = np.einsum('bij,bj->bi', Rinv, u)             # (B, N)
        mahal_b = np.real(np.einsum('bi,bi->b', u.conj(), v))  # (B,)  pure Mahalanobis term
        s       = mahal_b + eff_scale_f                         # (B,)

        # ── v7.0 : noise-frame purity gate ────────────────────────────────
        if self.use_outlier_gate:
            frame_score = float(np.median(mahal_b))

            if self._mahal_n < OUTLIER_MIN_FRAMES:
                # Still building the baseline — accept unconditionally.
                delta = frame_score - self._mahal_mean
                self._mahal_mean += OUTLIER_BASELINE_ALPHA * delta
                self._mahal_var = ((1.0 - OUTLIER_BASELINE_ALPHA)
                                    * (self._mahal_var
                                       + OUTLIER_BASELINE_ALPHA * delta * delta))
                self._mahal_n += 1
                self._consec_outlier_rejects = 0
            else:
                std = float(np.sqrt(max(self._mahal_var, 1e-12)))
                z   = (frame_score - self._mahal_mean) / max(std, 1e-6)
                is_outlier = z > self.outlier_reject_z

                if is_outlier and self._consec_outlier_rejects < OUTLIER_MAX_CONSEC_REJECTS:
                    self._outlier_rejects += 1
                    self._consec_outlier_rejects += 1
                    return   # reject — R_nn untouched, not a warmup frame

                if is_outlier:
                    # Safety valve: too many consecutive rejections likely
                    # means a genuine noise-envelope jump, not contamination.
                    self._outlier_forced_accepts += 1

                self._consec_outlier_rejects = 0
                delta = frame_score - self._mahal_mean
                self._mahal_mean += OUTLIER_BASELINE_ALPHA * delta
                self._mahal_var = ((1.0 - OUTLIER_BASELINE_ALPHA)
                                    * (self._mahal_var
                                       + OUTLIER_BASELINE_ALPHA * delta * delta))
                self._mahal_n += 1

        self._noise_frame_count += 1   # only accepted frames advance warmup

        valid = s > 1e-30

        vvH     = np.einsum('bi,bj->bij', v, v.conj())
        s_safe  = np.where(valid, s, 1.0)[:, None, None]
        Rinv_new = (Rinv - vvH / s_safe) / eff_alpha_f[:, None, None]

        # ── v7.2 : re-Hermitianise after every update ─────────────────────
        # Exact in infinite precision, but repeated subtraction of
        # similar-magnitude terms accumulates rounding error over many
        # sequential updates per clip; forcing exact Hermitian symmetry
        # every step is cheap and removes the asymmetric-drift half of the
        # divergence problem (see module docstring). Positive-definiteness
        # isn't guaranteed by this alone, which is why compute_weights()
        # also carries an explicit divergence guard.
        Rinv_new = 0.5 * (Rinv_new + np.conj(np.transpose(Rinv_new, (0, 2, 1))))

        finite_ok    = np.isfinite(Rinv_new).all(axis=(1, 2))
        self._nan_resets += int((valid & ~finite_ok).sum())

        mask       = (valid & finite_ok)[:, None, None]
        # v7.2: kept in complex128 (was complex64) — see module docstring.
        self._Rinv = np.where(mask, Rinv_new, Rinv)

        # ── v7.8 : persistent trace renormalization ─────────────────────────
        # Root cause (confirmed against the real update_noise() recursion in
        # diag_trace_growth.py, not just observed in aggregate diagnostics):
        # each Woodbury step only shrinks the eigenvalue along the CURRENT
        # update direction u; every other direction is simply divided by
        # eff_alpha_f (~0.995) with nothing to check it. Diffuse noise
        # excites all directions across frames, so this stays bounded.
        # A single coherent point-source interferer excites essentially the
        # same direction every noise frame, so every orthogonal direction
        # grows by ~1/alpha, UNOPPOSED, frame after frame — genuine unbounded
        # growth over a clip's noise-frame count, not a steady-state
        # elevation. Measured: 78x trace growth and eigenvalue spread
        # reaching ~6e11 (numerically near-singular) over 900 point-source
        # noise frames with this step disabled, vs. ~1x/~1.8e3 with it
        # enabled — diffuse-noise behavior is unaffected either way.
        #
        # compute_weights() already computes and applies a trace-normalizing
        # `scale` factor (see TRACE_GUARD_FRAC / TRACE_CEIL_MULT above), but
        # only transiently, at read time — it doesn't feed back into
        # self._Rinv, so the underlying eigen-structure keeps compounding
        # more extreme every noise frame regardless of what compute_weights
        # does with it. That's also why the v7.7 ceiling guard reads
        # ceil/c=0 on exactly the clips with the worst trace blowup: its own
        # baseline is an EMA of this same self._Rinv, so a monotonic drift
        # just drags the baseline up with it — self-referential and
        # structurally blind to sustained drift, as opposed to a transient
        # spike. Renormalizing self._Rinv itself, every accepted frame, so
        # the recursion never drifts far from a fixed reference in the
        # first place, fixes this at the source instead of trying to react
        # to it downstream.
        #
        # This does NOT remove the null: rescaling by a single per-bin
        # scalar preserves the relative eigenvalue structure (the ratio
        # between the suppressed and unsuppressed directions), it just
        # keeps the absolute magnitude from compounding without bound.
        trace_now = np.real(np.einsum('bii->b', self._Rinv))
        trace_ref = float(self.N) / self.reg
        persist_scale = np.where(trace_now > 1e-30, trace_ref / trace_now, 1.0)
        self._Rinv = self._Rinv * persist_scale[:, None, None]

        self._weights_valid = False

    # ── Stage 3C : MVDR weight computation (vectorised) ───────────────────────

    def compute_weights(self, d: np.ndarray, mic_mask: np.ndarray | None = None) -> None:
        """
        Compute MVDR weights — vectorised over all B bins.

        w[b] = R̃⁻¹[b] a[b] / (aᴴ R̃⁻¹[b] a[b])
        where a = d* (array response), R̃ = trace-normalised R_nn⁻¹ + reg·I.

        v5.3: After solving for w, apply a mild 3-tap frequency-axis smoothing
        to suppress bin-level outliers that cause musical-noise artefacts.

        v7.0: Smoothing across frequency mixes weight vectors that were each
        tailored to a DIFFERENT per-bin steering-vector phase a[f], so the
        smoothed w no longer satisfies a[f]^H w[f] = 1 for any individual
        bin. This silently introduces a broadband target distortion. After
        smoothing, each bin's weight is renormalised by 1 / (a[f]^H w[f])
        so the distortionless constraint holds exactly again, while the
        smoothed (less erratic, less musical-noise-prone) spatial response
        is otherwise preserved. See RENORMALIZE_AFTER_SMOOTH.

        v6.1: After computing s = aᴴ R̃⁻¹ a (the distortionless denominator),
        check for bins where s < DISTORT_RATIO_MIN · s_das.  These bins have
        formed a null toward the target (self-cancellation).  For those bins,
        partially reset R_nn⁻¹ toward the isotropic prior so the null can
        dissolve.  If the collapse is widespread (more than
        DISTORT_CHECK_FRAC of bins), Rinv_n / v / s are recomputed from the
        repaired R_nn⁻¹ before the weights are finalised.

        v7.1: mic_mask — optional (N,) array of 1.0 (selected) / 0.0
        (excluded) per microphone, from mic_selection.AdaptiveMicSelector
        or supplied directly. `a` is zeroed at excluded mic entries BEFORE
        any of the R_inv/v/s computation above, and the final weight
        vector is zeroed at those same entries again after smoothing (the
        R_inv matrix mixes mics, so v/w at an excluded entry are not
        guaranteed zero just because a is). Because a[excluded] = 0, the
        distortionless constraint a^H w = 1 is completely unaffected by
        either masking step — those terms contribute exactly zero
        regardless of what w would otherwise be there. R_nn⁻¹ itself is
        untouched by the mask; it keeps being estimated from the full
        array (see update_noise()).
        """
        # ── First-call diagnostic (once per clip) ──────────────────────────
        if not hasattr(self, '_cw_called'):
            self._cw_called = True
            f_mid = self.n_bins // 2
            df0   = d[:, f_mid].astype(np.complex128)
            R0    = self._Rinv[f_mid].astype(np.complex128)
            tr0   = np.real(np.trace(R0))
            Rn0   = (R0 / (tr0 / self.N) if tr0 > 1e-30 else R0) \
                    + self.reg * np.eye(self.N, dtype=np.complex128)
            a0    = df0.conj()
            v0    = Rn0 @ a0
            s0    = np.real(a0.conj() @ v0)
            w0    = v0 / s0 if s0 > 1e-12 else v0
            eigs  = np.sort(np.real(np.linalg.eigvalsh(Rn0)))
            print(f'[BF compute_weights] f={f_mid}  s={s0:.4f}  '
                  f'min_eig={eigs[0]:.4f}  '
                  f'|w|={np.abs(w0).round(3).tolist()}', flush=True)

        self._s_diag_count = getattr(self, '_s_diag_count', 0) + 1
        do_s_diag = self._s_diag_count <= 3

        # ── Vectorised weight computation ──────────────────────────────────
        a    = d.T.conj().astype(np.complex128)          # (B, N)

        # ── v7.1 : apply mic mask to the array-response vector FIRST ───────
        if mic_mask is not None:
            m = mic_mask.astype(np.complex128)            # (N,)
            a = a * m[None, :]

        Rinv = self._Rinv          # (B, N, N) complex128 (v7.2: no cast needed)

        trace = np.real(np.einsum('bii->b', Rinv))

        # ── v7.2 : numerical-divergence guard ───────────────────────────
        # trace(R_nn⁻¹) is expected to stay within an order of magnitude
        # of its initialisation value N/reg. A bin whose trace has
        # collapsed far below that (accumulated recursive rounding error,
        # or a genuinely rank-deficient noise field) makes
        # scale = N/trace blow up and corrupts that bin's weights — this
        # is the root cause of the observed s_max~1.5e10 blowups and the
        # run-to-run non-reproducibility they caused (see module
        # docstring). Treat it as a divergence event: reset that bin to
        # the isotropic prior instead of letting scale explode.
        trace_floor = TRACE_GUARD_FRAC * (float(self.N) / self.reg)
        diverged    = trace < trace_floor
        if diverged.any():
            self._trace_guard_resets += int(diverged.sum())
            prior64 = (1.0 / self.reg) * np.eye(self.N, dtype=np.complex128)
            self._Rinv[diverged] = prior64
            Rinv  = self._Rinv
            trace = np.real(np.einsum('bii->b', Rinv))

        # ── v7.7 : adaptive trace-CEILING guard ─────────────────────────
        # See TRACE_CEIL_MULT rationale in the module docstring. Gated off
        # until the per-bin baseline has accumulated enough frames to be
        # trustworthy — every bin looks "new" on the very first call.
        if self._trace_ceil_baseline is None:
            self._trace_ceil_baseline = trace.copy()
            self._trace_ceil_nframes  = 1
        else:
            baseline_trusted = self._trace_ceil_nframes >= TRACE_CEIL_MIN_FRAMES
            if baseline_trusted:
                ceiling = self._trace_ceil_baseline * TRACE_CEIL_MULT
                runaway = trace > ceiling
                if runaway.any():
                    self._trace_ceil_resets += int(runaway.sum())
                    prior64 = (1.0 / self.reg) * np.eye(self.N, dtype=np.complex128)
                    self._Rinv[runaway] = prior64
                    Rinv  = self._Rinv
                    trace = np.real(np.einsum('bii->b', Rinv))
                    # Recompute so the just-reset bins don't get folded
                    # into their own baseline as "stable" below.
                    ceiling = self._trace_ceil_baseline * TRACE_CEIL_MULT
                stable = trace <= ceiling
            else:
                stable = np.ones_like(trace, dtype=bool)

            # Update the baseline only from bins NOT currently flagged as
            # runaway, so a genuine spike can't drag its own ceiling up
            # with it and mask itself on the next frame.
            self._trace_ceil_baseline = np.where(
                stable,
                (1 - TRACE_CEIL_EMA_RATE) * self._trace_ceil_baseline
                + TRACE_CEIL_EMA_RATE * trace,
                self._trace_ceil_baseline)
            self._trace_ceil_nframes += 1

        scale  = np.where(trace > 1e-30, float(self.N) / trace, 1.0)
        Rinv_n = Rinv * scale[:, None, None] + self.reg * self._eye_stack

        self._scale_history.append(float(scale.mean()))
        self._scale_min_history.append(float(scale.min()))
        self._trace_history.append(float(trace.mean()))

        v = np.einsum('bij,bj->bi', Rinv_n, a)          # (B, N)
        s = np.real(np.einsum('bi,bi->b', a.conj(), v))  # (B,)

        if do_s_diag:
            print(f'[BF weights call#{self._s_diag_count}] '
                  f's_min={float(s.min()):.4f}  s_max={float(s.max()):.2f}  '
                  f'bins_s<1e-6={int((s < 1e-6).sum())}', flush=True)

        # ── v6.1 : distortionless-constraint monitor ───────────────────────
        # s_das = ‖a‖² — the denominator if R̃⁻¹ were identity (DAS case).
        # When s collapses far below s_das, R_nn⁻¹ has formed a null toward
        # the target direction itself — repair before computing weights.
        s_das_ref     = np.real(np.einsum('bn,bn->b', a.conj(), a))  # (B,)
        collapse_mask = (s < DISTORT_RATIO_MIN * np.maximum(s_das_ref, 1e-12))
        n_collapsed   = int(collapse_mask.sum())

        if n_collapsed > 0:
            self._distort_resets += n_collapsed
            prior = (1.0 / self.reg) * np.eye(self.N, dtype=np.complex128)
            idx = np.where(collapse_mask)[0]
            self._Rinv[idx] = (
                DISTORT_RECOVERY_BLEND * self._Rinv[idx]
                + (1.0 - DISTORT_RECOVERY_BLEND) * prior[None, :, :]
            )

            if n_collapsed > DISTORT_CHECK_FRAC * self.n_bins:
                Rinv   = self._Rinv          # v7.2: already complex128
                trace  = np.real(np.einsum('bii->b', Rinv))
                scale  = np.where(trace > 1e-30, float(self.N) / trace, 1.0)
                Rinv_n = Rinv * scale[:, None, None] + self.reg * self._eye_stack
                v = np.einsum('bij,bj->bi', Rinv_n, a)
                s = np.real(np.einsum('bi,bi->b', a.conj(), v))

        # DAS fallback for degenerate bins
        degenerate = s < 1e-12
        s_final    = np.where(degenerate, s_das_ref, s)
        v_final    = np.where(degenerate[:, None], a, v)

        weights = (v_final / np.maximum(s_final, 1e-30)[:, None]).astype(np.complex64)

        # ── v7.1 : re-zero excluded mics before smoothing ───────────────────
        # a[excluded] is already 0, but Rinv_n mixes mics, so v/weights at
        # excluded entries are not automatically zero — enforce it here so
        # apply() truly excludes those mics from the output combination.
        # Doing this before the frequency-axis smoothing below (which only
        # convolves within a mic's own row) keeps excluded columns at
        # exactly zero across all bins afterward too.
        if mic_mask is not None:
            weights = weights * mic_mask.astype(np.complex64)[None, :]

        # ── v5.3 : mild frequency-axis smoothing ──────────────────────────
        k = SMOOTH_KERNEL.astype(np.float32)
        for n in range(self.N):
            wr = np.pad(weights[:, n].real, 1, mode='edge')
            wi = np.pad(weights[:, n].imag, 1, mode='edge')
            weights[:, n] = (np.convolve(wr, k, mode='valid') +
                             1j * np.convolve(wi, k, mode='valid'))

        # ── v7.0 : restore the distortionless constraint after smoothing ───
        # Each bin's pre-smoothing weight satisfied a[f]^H w[f] = 1 exactly,
        # but a[f] itself rotates with frequency (different steering phase
        # per bin for the same physical direction), so blending
        # w[f-1], w[f], w[f+1] — each tailored to a different a — does NOT
        # preserve a[f]^H w_smoothed[f] = 1. The result is a smooth,
        # broadband multiplicative distortion of the target spectrum: small
        # in absolute terms, but it becomes the dominant error term once the
        # beamformer has little real noise left to remove (high input SNR).
        # Renormalising restores exact target transparency while keeping the
        # smoothed (less erratic) spatial response in other directions.
        if RENORMALIZE_AFTER_SMOOTH:
            w128  = weights.astype(np.complex128)
            denom = np.einsum('bn,bn->b', a.conj(), w128)
            safe  = np.abs(denom) > 1e-8
            scale_renorm = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 1.0)
            weights = (w128 * scale_renorm[:, None]).astype(np.complex64)

        self._weights       = weights
        self._weights_valid = True

    # ── Stage 3D : Apply weights ──────────────────────────────────────────────

    def apply(self, X: np.ndarray) -> np.ndarray:
        """y[f] = wᴴ[f] x[f]  →  shape (B,) complex."""
        if not self._weights_valid:
            raise RuntimeError(
                'Weights not computed.  Call compute_weights() before apply().')
        return np.einsum('fn,fn->f',
                         self._weights.conj(),
                         X.T.astype(np.complex64))

    def _phase_ref_mic0(self, X: np.ndarray, d: np.ndarray,
                       mic_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Single-mic reference, phase-rotated into the SAME a/d convention as
        y (so it's safe to linearly blend with y without comb-filtering),
        but — unlike _steered_das() — with NO cross-mic combination at all.

        y_ref[f] = (d[m0,f] / |d[m0,f]|) * X[m0,f]

        where m0 is mic 0 (or the first mic still selected, if mic_mask
        excludes it). Dividing by |d[m0,f]| makes this a pure phase
        rotation regardless of whether `d`'s magnitude convention is
        exactly 1 everywhere (e.g. under GEVD-estimated RTFs it may not
        be) — X[m0]'s own magnitude/SNR passes through completely
        unchanged, only its phase is rotated to match `a`.

        WHY NOT y_das HERE: y_das is phase-coherent too, but it is a
        genuine N-mic combination, so its quality depends on `d` actually
        matching the true inter-mic phase relationship. Any steering-
        vector error (gaze/DOA angle error, reverberant/near-field
        mismatch from the free-field delay model, GEVD estimation noise,
        ...) causes real partial destructive summation in y_das itself —
        it is not risk-free just because it shares y's phase convention.
        Using it as the SNR-blend's "safe, low-risk" dry reference baked
        that risk directly into the reference most heavily weighted at
        HIGH estimated SNR (wet floors to snr_blend_wet_floor there) —
        i.e. exactly where any added distortion dominates SI-SDR (little
        real background noise remains to dilute it against). That is
        what caused the further, uniform (all-scenario, including
        white/pink) regression at the +10/+20 dB SNR columns after y_das
        was first introduced as the blend target: it swapped one
        phase-mismatch problem (raw X[0], wrong frame) for a different,
        worse one (y_das, right frame but with real N-mic combination
        risk) in the one place that most needs a reference with NO risk
        at all. A single channel, phase-rotated but never combined with
        any other channel, cannot suffer destructive-summation loss no
        matter how wrong `d` is — its magnitude is mathematically
        identical to raw X[m0] regardless.

        The cancellation fallback (CANCEL_BLEND, above) deliberately
        keeps using y_das instead of this: it engages rarely, on already-
        failed bins, where the real array gain from a genuine multi-mic
        combination is worth the (small, narrow-band) risk. The SNR
        blend engages broadly across most speech frames at high SNR,
        where that trade-off flips — safety matters far more than the
        marginal extra suppression a combination could add, since little
        suppression is even needed there.
        """
        m0 = 0
        if mic_mask is not None:
            active = np.nonzero(mic_mask)[0]
            if active.size > 0:
                m0 = int(active[0])
        phase = d[m0] / (np.abs(d[m0]) + 1e-12)
        return (phase * X[m0]).astype(np.complex64)

    def _steered_das(self, X: np.ndarray, d: np.ndarray,
                     mic_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Delay-and-sum combination in the SAME steering convention as
        compute_weights()/apply() (a = d.conj(), constraint a^H w = 1),
        so it is phase-coherent with the MVDR output and safe to blend
        with it.

        y_das[f] = (1/N_eff) * sum_m d[m,f] * X[m,f]

        This is the uniform-weight vector w_das = a / N_eff (since
        a^H(a/N_eff) = ||a||²/N_eff = N_eff/N_eff = 1, |a_m|=1), giving
        y_das = w_das^H x = sum_m conj(a_m/N_eff) x_m
              = sum_m (d_m/N_eff) x_m   (conj(a_m) = d_m).

        BUG THIS FIXES: the previous fallback used a plain, un-delayed
        `X.mean(axis=0)` (and its masked variant) as "DAS". That is NOT a
        delay-and-sum beamformer — it never applies the per-mic steering
        phase, so mic signals arriving with different physical delays
        partially cancel/comb-filter when summed for any source direction
        where the mics aren't all equidistant (i.e. almost every real
        direction, given a real array geometry). That naive average was
        used (a) as the CANCEL_BLEND fallback — engaged most often on
        exactly the self-nulling-prone, spatially-coherent-interferer
        scenarios (babble/cocktail/directional_*) this bug most affects —
        and (b) implicitly wrong even as a *reference point*: raw mic-0
        (see the SNR-adaptive blend below) carries its own uncompensated
        physical delay relative to the array-manifold phase reference
        `d`/`a` used everywhere else, so blending straight into it
        introduces the same kind of comb-filtering artifact, worse for
        off-broadside directions and worse as more of it gets blended in
        (i.e. at high estimated SNR — matching the observed pattern of
        oracle_gaze/oracle_target_dir losing MORE ground to raw_mic at
        high SNR across almost every scenario, not just a few).
        """
        if mic_mask is not None:
            m     = mic_mask.astype(np.complex128)
            n_eff = max(float(np.real(m).sum()), 1.0)
            return (((d * X) * m[:, None]).sum(axis=0)
                    / n_eff).astype(np.complex64)
        n_eff = X.shape[0]
        return ((d * X).sum(axis=0) / n_eff).astype(np.complex64)

    # ── Convenience: process one frame end-to-end ─────────────────────────────

    def process_frame(self,
                      X: np.ndarray,
                      d: np.ndarray,
                      is_noise: bool,
                      doa_reliable: bool = True,
                      theta: 'float | None' = None,
                      phi: float = 0.0,
                      mic_mask: 'np.ndarray | None' = None) -> np.ndarray:
        """
        Process one STFT frame end-to-end.

        DOA-change detector (v7.1: aliasing fix; otherwise unchanged from v4.2)
        ------------------------------------------------------------------------
        When doa_reliable=True, fires if the steering direction changes by
        more than DOA_JUMP_THRESH degrees between consecutive frames.
        On a jump, R_nn⁻¹ is decayed 70% toward the isotropic prior so the
        null can relocate within ~2 s.

        The comparison is made at self._doa_check_bin (default ~800 Hz, see
        DOA_JUMP_CHECK_FREQ_HZ) rather than the FFT midpoint. The midpoint
        (~12 kHz at 512-pt/48 kHz) sits well past the spatial-Nyquist limit
        of any handheld mic array, so small true angle changes could alias
        into large apparent phase differences there and fire the detector
        on noise — visible only on doa_reliable=True callers (i.e.
        oracle_target_dir), since that's the only mode that has the
        detector active at all.

        v5.2 / v6.2 Output monitor
        ---------------------------
        After apply(), the per-bin output/input power ratio is checked.
        Bins where MVDR attenuates the signal by > CANCEL_THRESH_DB (10 dB,
        lowered from 15 dB) relative to the mean input power are blended
        back toward DAS at ratio CANCEL_BLEND (0.65, raised from 0.40).
        This catches residual target-cancellation that v6.0 and v6.1 did
        not prevent. (v7.1: the DAS fallback itself now also respects the
        active mic mask, if any, for consistency with compute_weights().)

        v6.0 Projection guard
        ----------------------
        self._last_d is updated on every call (before update_noise() runs)
        so the guard in update_noise() always has access to the most recent
        steering vector, including on the very first frame of a clip.

        v7.0 SNR-tracking + adaptive blend
        ------------------------------------
        Every call updates a running EMA of frame power, split by branch
        (is_noise → noise-floor EMA, else → speech-segment EMA). Once both
        EMAs have seen enough frames, the speech branch uses their ratio as
        a causal proxy for input SNR and blends the final output back
        toward the raw mic-0 signal as that estimate climbs (see
        SNR_BLEND_* constants and the module docstring).

        v7.1 Adaptive mic selection
        ------------------------------
        Every call feeds frame power to self.mic_selector (if one is
        attached) via update_power(). On speech frames, if `mic_mask` isn't
        given explicitly, the selector's own update()/get_mask() are used
        to derive one from `theta`/`phi` (pass the same values already
        returned by DOA_Gaze.steering_vector_from_gaze() /
        FreeFieldSteering.steering_vector()). If no selector is attached
        and no mic_mask is given, behaviour is identical to v7.0 (all mics
        used). R_nn⁻¹ itself is never restricted by the mask — see
        update_noise().

        Parameters
        ----------
        X            : (N, B) STFT frame
        d            : (N, B) steering vector
        is_noise     : True → update R_nn⁻¹;  False → compute weights + apply
        doa_reliable : True  → DOA-jump detector active (oracle_target_dir)
                       False → detector suppressed (oracle_gaze, energy_vad, srp)
        theta, phi   : current steering azimuth/elevation (radians), used
                       only to drive an attached self.mic_selector when
                       mic_mask isn't supplied explicitly. Optional.
        mic_mask     : optional (N,) 1.0/0.0 mask overriding self.mic_selector
                       for this call.
        """
        DOA_JUMP_THRESH = 25.0   # degrees (v4.2)
        DECAY_ON_JUMP   = 0.30   # blend coefficient toward isotropic prior

        # ── v6.0: always refresh last known steering vector first ────────
        self._last_d = d

        # ── v7.0: track noise/speech frame power for SNR-adaptive blending ──
        if self.use_snr_adaptive_blend:
            frame_power = float(np.mean(np.abs(X) ** 2))
            if is_noise:
                self._noise_power_ema = ((1.0 - SNR_POWER_ALPHA) * frame_power
                                          + SNR_POWER_ALPHA * self._noise_power_ema)
                self._n_noise_pwr += 1
            else:
                self._speech_power_ema = ((1.0 - SNR_POWER_ALPHA) * frame_power
                                           + SNR_POWER_ALPHA * self._speech_power_ema)
                self._n_speech_pwr += 1

        # ── v7.1: feed adaptive mic selector, every frame (cheap) ───────────
        if self.mic_selector is not None:
            self.mic_selector.update_power(X, is_speech=not is_noise)

        # ── DOA-jump detector (v7.1: aliasing-safe bin) ─────────────────────
        if doa_reliable and hasattr(self, '_prev_d'):
            f_check = self._doa_check_bin
            d_now   = d[:, f_check]
            d_prev  = self._prev_d
            cos_sim = np.abs(np.dot(d_now.conj(), d_prev)) / (
                np.linalg.norm(d_now) * np.linalg.norm(d_prev) + 1e-12)
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, 0.0, 1.0))))
            if angle_deg > DOA_JUMP_THRESH:
                # v7.2: no complex64 downcast — self._Rinv/_iso_prior are
                # both complex128, see module docstring.
                self._Rinv = (
                    DECAY_ON_JUMP * self._Rinv
                    + (1.0 - DECAY_ON_JUMP) * self._iso_prior
                )
                self._weights_valid = False

        self._prev_d = d[:, self._doa_check_bin].copy()

        if is_noise:
            self._frames_noise += 1
            self.update_noise(X)

            if self._weights_valid:
                y = self.apply(X)
                if self._postfilter is not None:
                    self._postfilter.update_noise(y)
                return y
            return np.zeros(self.n_bins, dtype=np.complex64)

        else:
            self._frames_speech += 1

            # ── v7.1: resolve the mic mask for this speech frame ────────────
            eff_mask = mic_mask
            if eff_mask is None and self.mic_selector is not None and theta is not None:
                self.mic_selector.update(theta, phi)
                eff_mask = self.mic_selector.get_mask()

            self.compute_weights(d, mic_mask=eff_mask)
            y = self.apply(X)

            # ── v5.2 / v6.2 : output-power monitor + DAS fallback ─────────
            # v7.4: this now measures the drop relative to y_das (the
            # phase-coherent DAS reference, computed just below), NOT raw
            # mixture input power. Comparing against raw input power cannot
            # distinguish "noise correctly suppressed, target intact" from
            # "target itself got nulled" — output power dropping well below
            # input power is the ENTIRE POINT of a working beamformer,
            # especially in any bin/frame where noise dominates the
            # mixture (i.e. most bins at most SNRs tested). Confirmed via
            # diag_summary.py: the old input-referenced check fired on
            # nearly every bin of nearly every speech frame in every
            # scenario sampled — including the healthy white/pink-noise
            # controls (cancel_events in the hundreds of thousands to
            # >1.5M per clip, against a per-clip ceiling of roughly
            # n_speech_frames * n_bins) — meaning the "rare backstop" had
            # become the primary signal path, permanently diluting real
            # MVDR interferer-nulling with a much less selective DAS blend.
            # That hit babble/cocktail/directional_* hardest (DAS can't
            # reject a directional interferer the way adaptive nulling
            # can) while partially masking itself on diffuse noise (DAS
            # alone still gives reasonable array gain there) — exactly the
            # pattern observed (directional_far worst, white/pink still
            # positive but everything narrower than it should be).
            #
            # y_das already approximately preserves the target (it's a
            # phase-steered average, not adaptively nulled), so comparing
            # MVDR's output against IT is a meaningful anomaly signal:
            # MVDR should not normally do markedly worse than plain DAS at
            # retaining target-direction energy. Comparing against the raw
            # mixture never was.
            out_power = np.abs(y) ** 2                            # (B,)

            # informational only now — total suppression achieved,
            # unrelated to the cancel-fallback decision below.
            mean_input_power = np.mean(np.abs(X) ** 2, axis=0)     # (B,)
            input_ratio_dB = 10.0 * np.log10(
                out_power / (mean_input_power + 1e-12) + 1e-12)   # (B,)
            self._out_in_ratio_dB.append(float(input_ratio_dB.mean()))

            y_das = self._steered_das(X, d, eff_mask)

            # ── v7.2, partially reverted: the cancel-fallback below now uses
            # a properly phase-steered DAS instead of an un-delay-compensated
            # average — that part is a clean win, verified numerically
            # (old plain average had >100% relative reconstruction error
            # even at broadside; steered version is ~exact). See
            # _steered_das() docstring.
            #
            # v7.2 ALSO changed the SNR-adaptive blend below to target this
            # same y_das instead of raw X[0] — that part is REVERTED here.
            # Reason: y_das depends on `d`, and `d` is not always a clean
            # physical quantity — when GEVDRTFEstimator supplies it (--use
            # -gevd-rtf), `d` is a noisy, adaptively-estimated RTF (solved
            # periodically from limited noise/speech-frame statistics), not
            # a known-exact geometric model. The whole point of the
            # SNR-adaptive blend is to fall back, at high SNR, to something
            # LOW-RISK when little correction is needed — raw X[0] has zero
            # dependency on any estimated quantity, while a `d`-dependent
            # combination inherits GEVD's estimation noise on every frame,
            # which is the opposite of "low risk". Measured on real data,
            # blending toward y_das here made high-SNR results markedly
            # WORSE (not better) than the original raw-X[0] blend, most
            # visible exactly where GEVD is in play — consistent with this
            # explanation. The cancel-fallback keeps y_das because it only
            # engages in narrow, already-degraded bins where a `d`-dependent
            # combination (even a noisily-estimated one) reliably beats an
            # un-delay-compensated average — see beamformer_2.py's earlier
            # angular-error sweep showing steered-with-error stays >= naive
            # even at large direction errors.

            # Bins where MVDR falls well short of the DAS reference →
            # suspect target self-cancellation (see v7.4 rationale above).
            das_power = np.abs(y_das) ** 2                          # (B,)
            cancel_ratio_dB = 10.0 * np.log10(
                out_power / (das_power + 1e-12) + 1e-12)             # (B,)
            cancel_mask = cancel_ratio_dB < -CANCEL_THRESH_DB         # (B,) bool
            if cancel_mask.any():
                y = np.where(cancel_mask,
                             CANCEL_BLEND     * y_das
                             + (1.0 - CANCEL_BLEND) * y,
                             y)
                self._cancel_events += int(cancel_mask.sum())

            # ── v5.0 : spectral post-filter ───────────────────────────────
            if self._postfilter is not None:
                y = self._postfilter.apply(y)

            # ── v7.0 : SNR-adaptive wet/dry blend ──────────────────────────
            # The corrective machinery above (MVDR, cancellation fallback,
            # spectral post-filter) is tuned to remove real noise; when very
            # little noise is actually present (high input SNR) every one
            # of those stages still introduces SOME residual distortion,
            # and that residual distortion is no longer small relative to
            # the (now tiny) noise it is removing. Blend the final output
            # back toward a "dry" reference in proportion to how little
            # noise reduction is actually needed. At low estimated SNR this
            # is a no-op (wet=1, identical to v6 behaviour); at high
            # estimated SNR it floors out at snr_blend_wet_floor so some
            # spatial suppression is always retained.
            #
            # v7.5: the dry reference is y_ref (_phase_ref_mic0) — a SINGLE
            # channel, phase-rotated into the same a/d convention as y, NOT
            # y_das and NOT raw X[0].
            #
            # v7.3 moved this from raw X[0] to y_das to fix comb-filtering
            # (X[0] is in the wrong phase frame relative to y). That part
            # of the diagnosis was correct — but y_das, while phase-
            # coherent, is a genuine N-mic combination, so its quality
            # still depends on `d` matching the true inter-mic phase
            # relationship. Any steering-vector error (gaze/DOA angle
            # error, near-field/reverb mismatch in the free-field delay
            # model, ...) causes real partial destructive summation
            # inside y_das itself — sharing y's phase convention doesn't
            # make it risk-free. Using it as the "safe" fallback baked
            # that risk into the reference most heavily weighted at HIGH
            # estimated SNR (wet floors to snr_blend_wet_floor there),
            # i.e. exactly where any added distortion dominates SI-SDR
            # (little real noise left to dilute it against) — and it
            # showed up as a FURTHER uniform regression at the +10/+20 dB
            # columns, across every scenario including white/pink, right
            # after y_das was introduced here.
            #
            # y_ref has no such risk: it's one channel, phase-rotated but
            # never combined with any other channel, so it is
            # mathematically identical in magnitude/SNR to raw X[m0]
            # regardless of how wrong `d` is — there is nothing for it to
            # destructively sum against. It still shares y's phase
            # convention, so blending with y is still comb-filter-safe,
            # same fix as v7.3 intended, just without inheriting `d`'s
            # estimation risk. See _phase_ref_mic0() docstring.
            #
            # The cancellation fallback above deliberately still uses
            # y_das, not y_ref: it engages rarely, on already-failed
            # bins, where the real array gain from an actual multi-mic
            # combination is worth the small, narrow-band risk. The SNR
            # blend engages broadly across most high-SNR speech frames,
            # where that trade-off flips the other way.
            if (self.use_snr_adaptive_blend
                    and self._n_noise_pwr  >= SNR_BLEND_MIN_FRAMES
                    and self._n_speech_pwr >= SNR_BLEND_MIN_FRAMES):
                est_snr_db = 10.0 * np.log10(
                    self._speech_power_ema / (self._noise_power_ema + 1e-12) + 1e-12)
                span = max(self.snr_blend_high_db - self.snr_blend_low_db, 1e-6)
                wet  = 1.0 - (est_snr_db - self.snr_blend_low_db) / span
                wet  = float(np.clip(wet, self.snr_blend_wet_floor, 1.0))
                y_ref = self._phase_ref_mic0(X, d, eff_mask)
                y    = (wet * y.astype(np.complex64)
                        + (1.0 - wet) * y_ref)
                self._wet_history.append(wet)
                self._last_est_snr_db = est_snr_db

            return y

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Re-initialise R_nn⁻¹ to scaled identity and clear all per-clip state.

        Call init_diffuse() or init_isotropic() afterwards.
        The warmup counter is also reset so the fast-convergence phase
        applies again at the start of the next clip (v5.1).
        """
        self._Rinv = np.stack(
            [(1.0 / self.reg) * np.eye(self.N, dtype=np.complex128)
             for _ in range(self.n_bins)]
        )
        self._weights_valid     = False
        self._noise_frame_count = 0   # restart warmup (v5.1)
        self._last_d            = None  # clear projection-guard state (v6.0)

        # ── v7.0: clear noise-frame purity gate baseline (fresh per clip) ──
        self._mahal_mean             = 0.0
        self._mahal_var              = 1.0
        self._mahal_n                = 0
        self._consec_outlier_rejects = 0

        # ── v7.7: clear adaptive trace-ceiling baseline (fresh per clip) ───
        self._trace_ceil_baseline = None
        self._trace_ceil_nframes  = 0

        # ── v7.0: clear SNR-adaptive blend state (fresh per clip) ──────────
        self._noise_power_ema  = 1e-6
        self._speech_power_ema = 1e-6
        self._n_noise_pwr      = 0
        self._n_speech_pwr     = 0

        # Clear per-clip state to prevent leakage (v4.3)
        for attr in ('_prev_d', '_cw_called', '_s_diag_count', '_last_est_snr_db'):
            if hasattr(self, attr):
                delattr(self, attr)

        if self._postfilter is not None:
            self._postfilter.reset()

        # ── v7.1: clear adaptive mic selector state (fresh per clip) ───────
        if self.mic_selector is not None:
            self.mic_selector.reset()

        self.reset_diagnostics()