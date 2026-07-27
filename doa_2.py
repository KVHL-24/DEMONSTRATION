"""
doa_2.py — Direction-of-Arrival estimation  (v7.2)
===================================================

v7.2 additions (this revision) — steering-vector phase-reference fix
----------------------------------------------------------------------
  Root cause of "beamformer loses to raw_mic across the board (worst at
  high SNR)", "GEVD-RTF trust stuck at 3-10%", and "gazestab/micsel
  ablation deltas round to ~0.0 everywhere": _delay_samples() computed
  each mic's delay relative to the array's geometric coordinate origin,
  not relative to any actual microphone. For real layouts (e.g.
  aria_mic_positions()) no mic sits at that origin, so the analytic
  steering vector d(θ) had d[0] ≠ 1 for almost every direction — a
  direction- and frequency-dependent phase offset between the MVDR
  beamformer's output and the physical mic-0 signal. beamformer_2.py's
  SNR-adaptive blend and cancellation fallback both mix the beamformed
  output directly against raw X[0] in the complex STFT domain (see
  MVDRBeamformer._steered_das()'s docstring, which already diagnosed this
  exact mismatch but only fixed the DAS-vs-naive-average half of it); with
  a frequency-dependent phase offset that mixing is destructive
  interference, not a clean SNR-weighted average — worse the more of it
  gets blended in, i.e. at high estimated SNR, matching the observed
  pattern exactly. The same referencing bug was independently present in
  ATFSteering (normalising by the reference channel's magnitude only,
  discarding its phase) and caused GEVDRTFEstimator.refine_steering()'s
  cosine-similarity trust check to compare against a spuriously-phased
  analytic vector, suppressing its measured trust fraction.
  Fix: _delay_samples() now subtracts the reference mic's own delay so
  d[REFERENCE_MIC_IDX] == 1 for every θ, φ, f (see its docstring); and
  ATFSteering now divides by the full complex reference channel instead
  of just its magnitude. Both changes are pure re-originations of an
  existing phase convention — pairwise mic delays / relative phases
  (what DOA estimation, the DOA-jump detector, and AdaptiveMicSelector's
  ranking actually depend on) are provably unaffected — so this only
  removes the reference mismatch, it does not change any DOA/selection
  behaviour.

v7.1 additions
-------------------------------
  • DOA_Gaze now wires in gaze_processing.GazeStabilizer by default
    (use_gaze_stabilizer=True). The raw 3-D gaze vector is passed through
    the stabilizer's rolling, confidence/recency-weighted window BEFORE
    the existing saccade-hold logic below ever sees it, so continuous
    frame-to-frame gaze jitter is smoothed out while genuine saccades
    still pass through and reach the hold logic unchanged (see
    gaze_processing.py's module docstring for the full rationale). This
    directly targets the "oracle_gaze loses to oracle_target_dir / SRP
    outside the _dynamic subset" pattern the eval sweep's SACCADE COST /
    GAZE VALUE tables show — the signature of continuous steering noise
    rather than discrete saccades.
"""

from __future__ import annotations
import numpy as np
from stft import B, F_WIN
from gaze_processing import GazeStabilizer

# ── Geometry / DSP constants ───────────────────────────────────────────────────
C      = 343.0
FS     = 48_000
K      = 181
THETAS = np.deg2rad(np.linspace(-90, 90, K))   # azimuth search grid

# ── v6.0  Speech-band mask parameters ────────────────────────────────────────
F_LOW_HZ  =   100.0   # Hz — remove sub-bass and DC-offset leakage
F_HIGH_HZ = 7_500.0   # Hz — Nyquist is 24 kHz; remove high-freq noise

# ── v6.2  Confidence / adaptive alpha ─────────────────────────────────────────
CONFIDENCE_PEAK_RATIO_CLIP = 20.0   # clip peak/mean ratio to [0, this]
ALPHA_HIGH  = 0.985    # EMA alpha when confidence is high
ALPHA_LOW   = 0.940    # EMA alpha when confidence is low

# ── v6.4  Saccade hold ────────────────────────────────────────────────────────
SACCADE_THRESH_DEG    = 5.0    # degrees; smaller = more sensitive
SACCADE_HOLD_FRAMES   = 8      # number of frames to hold old direction

# ── v7.0  GEVD RTF estimation ─────────────────────────────────────────────────
GEVD_COV_ALPHA      = 0.97    # EMA coefficient for R_yy / R_nn accumulation
GEVD_MIN_NOISE_FR   = 15      # minimum noise frames before first GEVD solve
GEVD_MIN_SPEECH_FR  = 15      # minimum speech frames before first GEVD solve
GEVD_RECOMPUTE_EVERY = 20     # recompute GEVD every N speech frames
GEVD_REG_FRAC       = 1e-2    # regularisation as a fraction of mean trace
GEVD_TRUST_COS      = 0.75    # min |cos similarity| vs analytic d to trust GEVD bin
                               # (calibrated to keep false-trust rate low even at
                               #  N=4 mics, where random vectors exceed cos=0.5 about
                               #  a third of the time; genuine RTF/analytic matches sit
                               #  at ~0.97-1.0, so 0.75 cleanly separates signal from
                               #  chance overlap without rejecting true matches)
GEVD_BLEND_HOLD     = 0.85    # EMA smoothing of the committed RTF across recomputes


# ── Helpers ────────────────────────────────────────────────────────────────────

def _freq_bins(n_fft: int = F_WIN) -> np.ndarray:
    """Frequency in Hz for each STFT bin, shape (B,)."""
    return np.fft.rfftfreq(n_fft, d=1.0 / FS).astype(np.float32)


# Which mic column the analytic steering vector (and therefore the MVDR
# distortionless constraint a^H w = 1) is phase-referenced to. This MUST
# match the channel used everywhere else as "the" reference channel:
# raw_mic mode returns mic_audio[REFERENCE_MIC_IDX], AriaDenoisingPipeline.
# process() passes through mic_audio[REFERENCE_MIC_IDX] until weights are
# valid, and eval_synthetic_2.py's reference audio is documented as
# "channel 0 of the ISM target signal" — i.e. mic 0.
REFERENCE_MIC_IDX = 0


def _delay_samples(mic_pos: np.ndarray, theta: float,
                   phi: float = 0.0) -> np.ndarray:
    """
    Fractional sample delay τ_m(θ, φ) for each mic, referenced to
    REFERENCE_MIC_IDX (mic 0).

    Parameters
    ----------
    mic_pos : (N, 2) [X, Z] in metres  OR  (N, 3) [X, Y, Z] in metres
    theta   : azimuth  in radians (0 = forward +Z, +π/2 = right +X)
    phi     : elevation in radians (0 = horizontal, +π/2 = up +Y)

    Returns
    -------
    tau : (N,) delay in samples relative to mic REFERENCE_MIC_IDX
          (positive = mic closer to source than the reference mic).
          tau[REFERENCE_MIC_IDX] is always exactly 0.

    Bug fix (root cause of the "beamformer loses to raw_mic / gazestab &
    micsel ablations wash out to ~0.0" symptoms)
    --------------------------------------------------------------------
    Previously this returned delays relative to the array's geometric
    coordinate origin (wherever mic_pos happens to place (0, 0)), which
    for every real mic layout (e.g. aria_mic_positions() — no mic actually
    sits at (0, 0)) is NOT the same point as any physical microphone. The
    resulting steering vector d(θ) therefore had d[REFERENCE_MIC_IDX] ≠ 1
    for almost every θ: a direction- and frequency-dependent phase offset
    of several samples (up to ~10 samples of delay for
    aria_mic_positions(), i.e. multiple radians of phase at mid/high
    frequencies) between the MVDR beamformer's implicit output phase and
    the actual mic-0 signal.

    That silent mismatch broke several things that all assume the
    steering vector is mic-0-referenced:
      • beamformer_2.py's SNR-adaptive wet/dry blend and cancellation
        fallback both mix the MVDR output directly against raw X[0] in
        the complex STFT domain. With a frequency-dependent phase offset
        between them, that mixing is destructive interference (comb
        filtering), not a clean SNR-weighted average — the dominant cause
        of oracle_gaze / oracle_target_dir / energy_vad / srp losing
        several dB to raw_mic specifically at high input SNR (where wet/
        dry blending is most active).
      • eval_synthetic_2.py's reference signal is documented as "channel 0
        of the ISM target signal" (mic 0). A phase-misreferenced
        beamformer output is penalised by SI-SDR beyond what a single
        broadband lag search (compute_metrics' max_lag) can correct,
        because the error is frequency-dependent, not a pure delay.
      • GEVDRTFEstimator.refine_steering() compares its own h_hat (which
        *is* correctly ref_mic-normalised — see GEVDRTFEstimator) against
        this analytic d via cosine similarity. A spurious relative phase
        offset between the two directly suppresses the measured cos_sim,
        consistent with the very low GEVD "trusted" bin fractions (3-10%)
        seen in eval runs: GEVD wasn't actually disagreeing with the
        physical direction, it was disagreeing with an arbitrarily-phased
        analytic reference.
      • Because this mismatch affects every beamformed mode identically
        (oracle_gaze, oracle_target_dir, energy_vad, srp all build `d`
        the same way), it also swamps the real but smaller ablation
        effects (gaze-stabilizer, mic-selection) in the aggregate SI-SDR
        tables — most of the variance being measured was this shared
        referencing bug, not the ablated feature.

    Referencing tau to a fixed physical mic (mic 0, matching every other
    "channel 0" convention already used in this codebase) fixes all of
    the above at once, and is a pure re-origination of the delay model:
    pairwise delays between mics — what DOA_GCCSRP's tau_table, the
    DOA-jump detector's cosine similarity, and AdaptiveMicSelector's
    phase-spread ranking all actually depend on — are completely
    unaffected, since subtracting the same per-θ constant from every
    mic's tau leaves every pairwise difference tau_m − tau_n unchanged.
    """
    if mic_pos.shape[1] == 2:
        unit = np.array([np.sin(theta), np.cos(theta)], dtype=np.float64)
    else:
        unit = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(phi),
            np.cos(theta) * np.cos(phi),
        ], dtype=np.float64)
    proj = mic_pos.astype(np.float64) @ unit
    tau  = proj / C * FS
    tau  = tau - tau[REFERENCE_MIC_IDX]
    return tau.astype(np.float32)


def gaze_vector_to_theta(gaze_vec: np.ndarray,
                          mic_plane_normal: np.ndarray | None = None) -> float:
    """Azimuth θ (radians) from a 3-D gaze unit vector."""
    g = np.asarray(gaze_vec, dtype=np.float64).ravel()
    n = np.linalg.norm(g)
    if n < 1e-9:
        return 0.0
    g /= n

    if mic_plane_normal is None:
        mic_plane_normal = np.array([0.0, 1.0, 0.0])
    nv = np.asarray(mic_plane_normal, dtype=np.float64)
    nv /= np.linalg.norm(nv)

    g_proj = g - np.dot(g, nv) * nv
    pn     = np.linalg.norm(g_proj)
    if pn < 1e-9:
        return 0.0
    g_proj /= pn

    forward = np.array([0.0, 0.0, 1.0])
    forward -= np.dot(forward, nv) * nv
    fn = np.linalg.norm(forward)
    forward = forward / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(nv, forward)
    return float(np.arctan2(np.dot(g_proj, right), np.dot(g_proj, forward)))


def gaze_vector_to_angles(gaze_vec: np.ndarray,
                           mic_plane_normal: np.ndarray | None = None
                           ) -> tuple[float, float]:
    """Extract azimuth θ AND elevation φ from a 3-D gaze unit vector."""
    g = np.asarray(gaze_vec, dtype=np.float64).ravel()
    n = np.linalg.norm(g)
    if n < 1e-9:
        return 0.0, 0.0
    g /= n

    if mic_plane_normal is None:
        mic_plane_normal = np.array([0.0, 1.0, 0.0])
    nv = np.asarray(mic_plane_normal, dtype=np.float64)
    nv /= np.linalg.norm(nv)

    sin_phi = np.clip(np.dot(g, nv), -1.0, 1.0)
    phi     = float(np.arcsin(sin_phi))

    g_proj = g - sin_phi * nv
    pn     = np.linalg.norm(g_proj)
    if pn < 1e-9:
        return 0.0, phi
    g_proj /= pn

    forward = np.array([0.0, 0.0, 1.0])
    forward -= np.dot(forward, nv) * nv
    fn = np.linalg.norm(forward)
    forward = forward / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(nv, forward)
    theta = float(np.arctan2(np.dot(g_proj, right), np.dot(g_proj, forward)))
    return theta, phi


def compute_steering_vector(mic_pos: np.ndarray,
                             theta:   float,
                             n_fft:   int   = F_WIN,
                             phi:     float = 0.0) -> np.ndarray:
    """
    Analytic far-field steering vector d_m[f] = exp(−j 2π f/fs τ_m(θ, φ)).

    Returns
    -------
    d : (N, B) complex64,  |d_m[f]| = 1 for all m, f.
        d[REFERENCE_MIC_IDX, f] == 1+0j for all f, θ, φ (see
        _delay_samples()'s bug-fix note) — the steering vector is
        phase-referenced to mic REFERENCE_MIC_IDX (mic 0), matching
        raw_mic / the SNR-adaptive blend / the eval reference signal.
    """
    freqs = _freq_bins(n_fft)
    tau   = _delay_samples(mic_pos, theta, phi)
    phase = -1j * 2.0 * np.pi * (freqs[None, :] / FS) * tau[:, None]
    return np.exp(phase).astype(np.complex64)


# ── GCC-PHAT ───────────────────────────────────────────────────────────────────

def gcc_phat(Xm: np.ndarray, Xn: np.ndarray,
             band_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Phase-Transform cross-correlation for one mic pair.

    Parameters
    ----------
    Xm, Xn    : (B, n_frames) STFT arrays
    band_mask : (B,) boolean mask — if provided, zeros out cross-spectrum
                bins outside the speech band (v6.0).

    Returns
    -------
    psi : (F_WIN, n_frames) real GCC-PHAT correlation
    """
    G   = Xm * np.conj(Xn)
    if band_mask is not None:
        G = G * band_mask[:, None]
    mag = np.abs(G) + 1e-12
    Psi = G / mag
    return np.fft.irfft(Psi, n=F_WIN, axis=0).astype(np.float32)


# ── SRP-PHAT DOA  ──────────────────────────────────────────────────────────────

class DOA_GCCSRP:
    """
    DOA estimation via GCC-PHAT + Steered Response Power search.
    Searches K=181 azimuth candidates (−90° … +90°, 1° steps).

    v6 Additions
    ────────────
    • Speech-band mask on GCC cross-spectrum (v6.0).
    • Parabolic sub-bin interpolation after argmax (v6.1).
    • Confidence-weighted accumulation (v6.2).
    • Adaptive EMA alpha from confidence (v6.3).
    • Per-frame (theta, confidence) history for diagnostics (v6.6).

    Parameters
    ----------
    mic_pos    : (N, 2) or (N, 3) mic positions in metres.
    n_fft      : FFT window size (default 512).
    srp_alpha  : base EMA coefficient.  Actual alpha is adapted per frame
                 by confidence (v6.3).  0.0 = single-frame mode.
    f_low_hz   : lower speech-band edge for GCC mask (v6.0).
    f_high_hz  : upper speech-band edge for GCC mask (v6.0).
    """

    def __init__(self,
                 mic_pos:   np.ndarray,
                 n_fft:     int   = F_WIN,
                 srp_alpha: float = 0.97,
                 f_low_hz:  float = F_LOW_HZ,
                 f_high_hz: float = F_HIGH_HZ):
        self.mic_pos   = mic_pos[:, :2]
        self.n_fft     = n_fft
        self.N         = mic_pos.shape[0]
        self.srp_alpha = float(np.clip(srp_alpha, 0.0, 1.0))

        pairs = [(m, n) for m in range(self.N) for n in range(m + 1, self.N)]
        self.pairs = pairs
        self.P     = len(pairs)

        # ── Precompute angle_units → tau → lag tables ─────────────────────
        angle_units = np.stack([np.sin(THETAS), np.cos(THETAS)], axis=1)
        proj_all    = self.mic_pos.astype(np.float64) @ angle_units.T
        tau_all     = (proj_all / C * FS).astype(np.float32)

        pair_m = np.array([m for m, n in pairs], dtype=np.int32)
        pair_n = np.array([n for m, n in pairs], dtype=np.int32)
        self._pair_m   = pair_m
        self._pair_n   = pair_n

        self.tau_table = tau_all[pair_m] - tau_all[pair_n]
        self._lag_int  = np.mod(
            np.round(self.tau_table).astype(np.int32), n_fft)
        self._pair_idx = np.arange(self.P, dtype=np.int32)[:, None]

        # ── v6.0 : speech-band mask ───────────────────────────────────────
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)           # (B,)
        self._band_mask = ((freqs >= f_low_hz) &
                           (freqs <= f_high_hz)).astype(np.float32)  # (B,)

        # Temporal accumulation state
        self._srp_accum  = np.ones(K, dtype=np.float64) / K
        self._srp_frames = 0

        # ── v6.6 : diagnostics history ────────────────────────────────────
        self._theta_history:      list[float] = []
        self._confidence_history: list[float] = []

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset accumulator and history.  Call between independent clips."""
        self._srp_accum[:]  = 1.0 / K
        self._srp_frames    = 0
        self._theta_history.clear()
        self._confidence_history.clear()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """
        Return per-clip mean ± std of DOA angle and confidence.
        Call after the clip is processed and before reset().
        """
        th  = np.asarray(self._theta_history)      if self._theta_history      else np.array([float('nan')])
        cf  = np.asarray(self._confidence_history) if self._confidence_history else np.array([float('nan')])
        return {
            'srp_theta_mean_deg':  float(np.nanmean(np.degrees(th))),
            'srp_theta_std_deg':   float(np.nanstd(np.degrees(th))),
            'srp_confidence_mean': float(np.nanmean(cf)),
            'srp_confidence_std':  float(np.nanstd(cf)),
            'srp_frames':          self._srp_frames,
        }

    def print_diagnostics(self, prefix: str = '') -> None:
        d   = self.diagnostics()
        tag = f'[SRP {prefix}] ' if prefix else '[SRP] '
        print(f'{tag}frames={d["srp_frames"]}  '
              f'theta={d["srp_theta_mean_deg"]:+.1f}°±{d["srp_theta_std_deg"]:.1f}°  '
              f'confidence={d["srp_confidence_mean"]:.3f}±{d["srp_confidence_std"]:.3f}')

    # ── Estimate ──────────────────────────────────────────────────────────────

    def estimate(self, X: np.ndarray) -> float:
        """
        Estimate azimuth DOA from one STFT speech frame — fully vectorised.

        Returns
        -------
        theta_est : DOA in radians (0 = forward, ±π/2 = right/left)
        """
        # ── Step 1: batch GCC-PHAT (v6.0 band mask applied) ───────────────
        Xm  = X[self._pair_m]                          # (P, B) complex
        Xn  = X[self._pair_n]                          # (P, B) complex
        G   = Xm * np.conj(Xn)                        # (P, B)
        G   = G * self._band_mask[None, :]             # v6.0 zero out-of-band
        Psi = np.fft.irfft(G / (np.abs(G) + 1e-12),
                            n=self.n_fft, axis=-1)     # (P, n_fft)

        # ── Step 2: batch SRP via advanced indexing ────────────────────────
        srp = Psi[self._pair_idx, self._lag_int].sum(axis=0)   # (K,)

        # ── Step 3: L1 normalise ──────────────────────────────────────────
        srp_sum = float(np.abs(srp).sum())
        if srp_sum > 1e-12:
            srp = srp / srp_sum

        # ── Step 4: confidence score (v6.2) ──────────────────────────────
        peak_to_mean = float(srp.max()) / (float(np.abs(srp).mean()) + 1e-12)
        confidence   = np.clip(
            (peak_to_mean - 1.0) / (CONFIDENCE_PEAK_RATIO_CLIP - 1.0),
            0.0, 1.0
        )

        # ── Step 5: adaptive alpha (v6.3) ────────────────────────────────
        if self.srp_alpha > 0.0:
            eff_alpha = ALPHA_LOW + confidence * (ALPHA_HIGH - ALPHA_LOW)
        else:
            eff_alpha = 0.0

        # ── Step 6: confidence-weighted EMA accumulation (v6.2) ──────────
        if eff_alpha > 0.0:
            w_new = (1.0 - eff_alpha) * (0.1 + 0.9 * confidence)
            self._srp_accum = (
                (1.0 - w_new) * self._srp_accum + w_new * srp
            )
            s = self._srp_accum.sum()
            if s > 1e-30:
                self._srp_accum /= s
            self._srp_frames += 1

            srp_to_use = self._srp_accum if self._srp_frames >= 5 else srp
        else:
            srp_to_use = srp

        # ── Step 7: argmax + parabolic sub-bin interpolation (v6.1) ──────
        k_peak = int(np.argmax(srp_to_use))
        theta_est = self._parabolic_interpolate(srp_to_use, k_peak)

        # ── Step 8: store history (v6.6) ──────────────────────────────────
        self._theta_history.append(theta_est)
        self._confidence_history.append(confidence)

        return theta_est

    @staticmethod
    def _parabolic_interpolate(srp: np.ndarray, k: int) -> float:
        """
        Fit a parabola through srp[k-1], srp[k], srp[k+1] and return the
        sub-integer peak location mapped to an angle via THETAS.

        If k is at the boundary or the parabola is flat, returns THETAS[k].
        """
        if k == 0 or k == len(srp) - 1:
            return float(THETAS[k])
        y0, y1, y2 = float(srp[k - 1]), float(srp[k]), float(srp[k + 1])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) < 1e-30:
            return float(THETAS[k])
        delta = 0.5 * (y0 - y2) / denom   # fractional bin offset
        delta = np.clip(delta, -1.0, 1.0)
        k_frac = k + delta
        k_lo   = int(np.floor(k_frac))
        k_hi   = min(k_lo + 1, len(THETAS) - 1)
        k_lo   = max(k_lo, 0)
        t      = k_frac - k_lo
        return float((1.0 - t) * THETAS[k_lo] + t * THETAS[k_hi])


# ── Gaze-based DOA ─────────────────────────────────────────────────────────────

class DOA_Gaze:
    """
    DOA from gaze: θ (and optionally φ) provided externally.

    v6.4 : Saccade hold
    -------------------
    If the gaze direction changes by more than SACCADE_THRESH_DEG between
    consecutive calls to steering_vector_from_gaze(), the previous direction
    is held for SACCADE_HOLD_FRAMES frames before the new direction is
    committed.  This prevents the beamformer from momentarily misdirecting
    its null during the mid-saccade interval (~50–200 ms) when the gaze is
    between two fixation points.

    Set hold_frames=0 to disable (matches v5 behaviour).

    v6.5 : reset() method clears saccade-hold state between clips (now also
    clears the gaze stabilizer's window, if one is attached — see v7.1).

    v7.1 : Gaze stabilization (upstream jitter smoothing)
    -------------------------------------------------------
    Before the saccade-hold logic above ever inspects the incoming gaze
    direction, the raw 3-D gaze vector is first passed through a
    gaze_processing.GazeStabilizer (enabled by default). The stabilizer
    smooths continuous frame-to-frame jitter (eye-tracker noise, vergence
    glitches, blinks) that never crosses the saccade threshold but would
    otherwise perturb the steering vector every frame; genuine saccades
    still pass through it and reach the (unchanged) hold logic below. Set
    use_gaze_stabilizer=False to fully recover pre-v7.1 behaviour, or pass
    a pre-configured `gaze_stabilizer` instance to share/tune its state
    externally (e.g. across multiple DOA_Gaze instances, or with
    non-default constructor args).

    Parameters
    ----------
    mic_pos             : (N, 2) or (N, 3) mic positions in metres.
    n_fft               : STFT FFT size.
    mic_plane_normal    : (3,) normal of the mic array plane (default y-axis).
    hold_frames         : number of frames to hold direction after a saccade.
    saccade_thresh      : azimuth change (degrees) that triggers a hold.
    use_gaze_stabilizer : enable the v7.1 upstream jitter smoother (default
                          True). Ignored if `gaze_stabilizer` is given.
    gaze_stabilizer     : optional pre-built GazeStabilizer instance to use
                          instead of constructing a default one.
    """

    def __init__(self,
                 mic_pos:          np.ndarray,
                 n_fft:            int                   = F_WIN,
                 mic_plane_normal: np.ndarray | None     = None,
                 hold_frames:      int                   = SACCADE_HOLD_FRAMES,
                 saccade_thresh:   float                 = SACCADE_THRESH_DEG,
                 use_gaze_stabilizer: bool                = True,
                 gaze_stabilizer:  GazeStabilizer | None = None):
        self.mic_pos          = mic_pos
        self.n_fft            = n_fft
        self.mic_plane_normal = mic_plane_normal
        self.hold_frames      = int(hold_frames)
        self.saccade_thresh   = float(saccade_thresh)

        # ── v7.1: upstream jitter smoother, applied before saccade detection ──
        if gaze_stabilizer is not None:
            self._gaze_stabilizer = gaze_stabilizer
        elif use_gaze_stabilizer:
            self._gaze_stabilizer = GazeStabilizer()
        else:
            self._gaze_stabilizer = None

        # Saccade-hold state (v6.4)
        self._prev_theta:  float | None = None
        self._hold_count:  int          = 0          # frames remaining in hold
        self._held_d:      np.ndarray | None = None  # steering vector being held

        # Diagnostics
        self._saccade_events: int = 0
        self._total_calls:    int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def steering_vector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """Return (N, B) complex steering vector for azimuth θ, elevation φ."""
        return compute_steering_vector(self.mic_pos, theta, self.n_fft, phi)

    def steering_vector_from_gaze(self,
                                   gaze_vec:      np.ndarray,
                                   use_elevation: bool = False,
                                   confidence:    float = 1.0,
                                   ) -> tuple[np.ndarray, float, float]:
        """
        Compute steering vector from a 3-D gaze unit vector.

        v7.1: The raw gaze vector is first run through the attached
        GazeStabilizer (if any), which returns a recency/confidence-
        weighted rolling average that down-weights (rather than drops)
        single-frame outliers. Everything below then operates on that
        smoothed vector exactly as it did pre-v7.1.

        v6.4: If the implied azimuth differs from the previous call by more
        than saccade_thresh degrees, the previous steering vector is returned
        for hold_frames frames before the new direction is committed.

        Parameters
        ----------
        gaze_vec      : (3,) array-like raw gaze unit vector for this frame.
        use_elevation : also extract elevation φ (default False → φ=0).
        confidence    : optional external per-sample confidence in [0, 1],
                        forwarded to the GazeStabilizer's outlier gate
                        (default 1.0 — trust it, subject to the gate).

        Returns
        -------
        d     : (N, B) complex steering vector
        theta : committed azimuth in radians
        phi   : committed elevation in radians (0 if use_elevation=False)
        """
        self._total_calls += 1

        # ── v7.1: smooth the raw gaze vector before anything else sees it ──
        if self._gaze_stabilizer is not None:
            gaze_vec = self._gaze_stabilizer.update(
                gaze_vec, is_vec=True, confidence=confidence)

        # Decode new gaze direction
        if use_elevation:
            theta_new, phi_new = gaze_vector_to_angles(
                gaze_vec, self.mic_plane_normal)
        else:
            theta_new = gaze_vector_to_theta(gaze_vec, self.mic_plane_normal)
            phi_new   = 0.0

        # ── Saccade detection (v6.4) ──────────────────────────────────────
        if (self._prev_theta is not None) and (self.hold_frames > 0):
            delta_deg = abs(np.degrees(theta_new - self._prev_theta))
            # Wrap to [-180, 180]
            if delta_deg > 180.0:
                delta_deg = 360.0 - delta_deg

            if delta_deg > self.saccade_thresh:
                # Saccade detected — start / extend hold
                self._saccade_events += 1
                self._hold_count = self.hold_frames
                # _held_d is the direction we were using before the saccade

        # Return held direction if still in hold period
        if self._hold_count > 0 and self._held_d is not None:
            self._hold_count -= 1
            theta_commit = self._prev_theta
            phi_commit   = 0.0   # we don't track held elevation separately
            return self._held_d, theta_commit, phi_commit

        # Commit new direction
        d = compute_steering_vector(
            self.mic_pos, theta_new, self.n_fft, phi_new)

        self._prev_theta = theta_new
        self._held_d     = d          # save for potential future hold

        return d, theta_new, phi_new

    def reset(self) -> None:
        """
        Reset saccade-hold state and diagnostics. Call between clips.
        Also clears the gaze stabilizer's rolling window (v7.1), if one is
        attached, so its history doesn't leak across clips.
        """
        self._prev_theta    = None
        self._hold_count    = 0
        self._held_d        = None
        self._saccade_events= 0
        self._total_calls   = 0
        if self._gaze_stabilizer is not None:
            self._gaze_stabilizer.reset()

    def diagnostics(self) -> dict:
        d = {
            'saccade_events': self._saccade_events,
            'total_calls':    self._total_calls,
            'saccade_rate':   (self._saccade_events / max(self._total_calls, 1)),
        }
        # v7.1: merge in the gaze stabilizer's own diagnostics, if attached.
        if self._gaze_stabilizer is not None:
            d.update(self._gaze_stabilizer.diagnostics())
        return d

    def print_diagnostics(self, prefix: str = '') -> None:
        d   = self.diagnostics()
        tag = f'[Gaze {prefix}] ' if prefix else '[Gaze] '
        print(f'{tag}saccade_events={d["saccade_events"]}  '
              f'total_calls={d["total_calls"]}  '
              f'saccade_rate={d["saccade_rate"]:.3f}')
        if self._gaze_stabilizer is not None:
            self._gaze_stabilizer.print_diagnostics(prefix)


# ── Free-field analytic steering ──────────────────────────────────────────────

class FreeFieldSteering:
    """
    Free-field (anechoic) steering vector computed analytically from mic geometry.

    Drop-in replacement for ATFSteering when no measured ATF is available.
    Uses the far-field plane-wave model:  d_m[f] = exp(−j 2π f / fs · τ_m(θ, φ)).

    Interface mirrors ATFSteering so the pipeline can use either class
    transparently via self.steerer.

    Note: FreeFieldSteering cannot initialise a diffuse R_nn prior (no ATF
    table); the pipeline should call beamformer.init_isotropic() instead.

    Parameters
    ----------
    mic_pos : (N, 2) or (N, 3) mic positions in metres.
    n_fft   : STFT FFT size (default 512 → B = 257 bins).
    fs      : audio sample rate (default 48 000 Hz).
    """

    def __init__(self, mic_pos: np.ndarray, n_fft: int = F_WIN, fs: int = FS):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float32)
        self.n_fft   = n_fft
        self.fs      = fs
        self.N       = self.mic_pos.shape[0]
        self.B       = n_fft // 2 + 1

    def steering_vector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """(N, B) complex64 analytic steering vector."""
        return compute_steering_vector(self.mic_pos, theta, self.n_fft, phi)

    def steering_vector_from_gaze(self,
                                   gaze_vec:         np.ndarray,
                                   use_elevation:    bool                  = False,
                                   mic_plane_normal: np.ndarray | None     = None,
                                   ) -> tuple[np.ndarray, float, float]:
        """Mirror ATFSteering interface for drop-in use in the pipeline."""
        if use_elevation:
            theta, phi = gaze_vector_to_angles(gaze_vec, mic_plane_normal)
        else:
            theta = gaze_vector_to_theta(gaze_vec, mic_plane_normal)
            phi   = 0.0
        d = compute_steering_vector(self.mic_pos, theta, self.n_fft, phi)
        return d, theta, phi


# ── ATF-based steering ─────────────────────────────────────────────────────────

class ATFSteering:
    """
    Steering vector lookup from a measured Array Transfer Function (ATF) HDF5.

    The ATF file (e.g. EasyCom Device_ATFs.h5) contains:
      RealTF / ImagTF : (F_atf, D, N)  complex freq-domain responses
      Theta           : (1, D) ELEVATION  in degrees  (ISO/physics convention)
      Phi             : (1, D) AZIMUTH    in degrees  (ISO/physics convention)
      SamplingFreq_Hz : scalar

    Bug fix (v5): EasyCom uses the ISO convention where Theta=elevation,
    Phi=azimuth — the *opposite* of the common audio convention.
    Code now reads f['Phi'] → azimuth, f['Theta'] → elevation.

    Bug fix (v5): steering_vector() returns np.conj(H_rel.T) so that
    d* = H_rel is the true array response vector satisfying the MVDR
    distortionless constraint d^H w = 1.

    Parameters
    ----------
    atf_path : path to Device_ATFs.h5
    n_fft    : STFT FFT size (default 512 → B=257 bins)
    fs       : audio sample rate (default 48 000 Hz)
    """

    def __init__(self, atf_path: str, n_fft: int = F_WIN, fs: int = FS):
        import h5py
        self.n_fft = n_fft
        self.fs    = fs
        self.B     = n_fft // 2 + 1

        with h5py.File(atf_path, 'r') as f:
            real_tf   = f['RealTF'][:]          # (F_atf, D, N)
            imag_tf   = f['ImagTF'][:]          # (F_atf, D, N)
            # BUG FIX (v5): f['Phi'] is azimuth, f['Theta'] is elevation
            theta_deg = f['Phi'][:].ravel()     # (D,) azimuth   ← was f['Theta']
            phi_deg   = f['Theta'][:].ravel()   # (D,) elevation ← was f['Phi']
            fs_atf    = float(f['SamplingFreq_Hz'][()])

        atf_cplx = real_tf + 1j * imag_tf      # (F_atf, D, N)
        F_atf, D, N = atf_cplx.shape
        self.N = N

        # Interpolate ATF onto STFT frequency grid
        freqs_atf  = np.fft.rfftfreq(2 * (F_atf - 1), d=1.0 / fs_atf)
        freqs_stft = np.fft.rfftfreq(n_fft,            d=1.0 / fs)
        atf_interp = np.zeros((self.B, D, N), dtype=np.complex128)
        for d_idx in range(D):
            for n_idx in range(N):
                atf_interp[:, d_idx, n_idx] = np.interp(
                    freqs_stft, freqs_atf,
                    atf_cplx[:, d_idx, n_idx].real
                ) + 1j * np.interp(
                    freqs_stft, freqs_atf,
                    atf_cplx[:, d_idx, n_idx].imag
                )

        # Normalise to mic-0 (relative ATF).
        #
        # Bug fix: this previously divided by np.abs(ref) — the reference
        # channel's MAGNITUDE only — which normalises amplitude but leaves
        # mic 0's own phase untouched (atf_rel[..., 0] = ref/|ref|, a
        # unit-magnitude complex number with ARBITRARY phase, not the 1.0
        # the "relative to mic 0" comment and the class's own
        # distortionless-constraint claim require). That silently broke
        # the same assumption as the FreeFieldSteering path (see
        # _delay_samples()'s bug-fix note): d[0] was not actually 1, so
        # the MVDR output's phase reference didn't match the physical
        # mic-0 channel that raw_mic, the SNR-adaptive blend, and the
        # eval reference signal all use. A full complex division makes
        # atf_rel[..., 0] exactly 1 (both magnitude and phase), matching
        # the FreeFieldSteering fix and the GEVDRTFEstimator convention
        # (which already normalises by the full complex ref_mic entry).
        ref      = atf_interp[:, :, 0:1]                       # (B, D, 1)
        ref_safe = np.where(np.abs(ref) < 1e-12, 1e-12 + 0j, ref)
        self._atf_rel = atf_interp / ref_safe                  # (B, D, N)
        self._H_rel   = self._atf_rel   # alias for beamformer.init_diffuse()

        # Direction table in radians
        self._theta_rad = np.deg2rad(theta_deg)   # (D,)
        self._phi_rad   = np.deg2rad(phi_deg)     # (D,)

        print(f'[ATFSteering] Loaded {D} directions × {N} mics × {self.B} bins '
              f'from {atf_path}')

    @property
    def H_rel(self) -> np.ndarray:
        """(B, D, N) relative ATF — for beamformer.init_diffuse()."""
        return self._atf_rel

    def _nearest_idx(self, theta: float, phi: float) -> int:
        """Great-circle nearest neighbour lookup."""
        cos_sim = (np.cos(phi) * np.cos(self._phi_rad)
                   * np.cos(theta - self._theta_rad)
                   + np.sin(phi) * np.sin(self._phi_rad))
        return int(np.argmax(cos_sim))

    def steering_vector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """
        (N, B) complex steering vector for (azimuth θ, elevation φ).

        Returns np.conj(H_rel.T) so that the MVDR distortionless constraint
        d^H w = 1 has a real-positive denominator (v5 bug fix).
        """
        idx = self._nearest_idx(theta, phi)
        H   = self._atf_rel[:, idx, :].astype(np.complex128)   # (B, N)
        return np.conj(H.T).astype(np.complex64)               # (N, B)

    def steering_vector_from_gaze(self,
                                   gaze_vec:         np.ndarray,
                                   use_elevation:    bool                  = False,
                                   mic_plane_normal: np.ndarray | None     = None,
                                   ) -> tuple[np.ndarray, float, float]:
        """Compute steering vector from a 3-D gaze unit vector."""
        if use_elevation:
            theta, phi = gaze_vector_to_angles(gaze_vec, mic_plane_normal)
        else:
            theta = gaze_vector_to_theta(gaze_vec, mic_plane_normal)
            phi   = 0.0
        d = self.steering_vector(theta, phi)
        return d, theta, phi


# ── EasyCom approximate mic positions ─────────────────────────────────────────

def aria_mic_positions() -> np.ndarray:
    """Approximate 2-D (X, Z) positions of Aria's 7 mics (metres)."""
    return np.array([
        [-0.070,  0.005],
        [-0.060, -0.005],
        [-0.020,  0.010],
        [ 0.000,  0.015],
        [ 0.020,  0.010],
        [ 0.060, -0.005],
        [ 0.070,  0.005],
    ], dtype=np.float32)


# ── v7.0 : Data-driven RTF estimation via GEVD ─────────────────────────────────

class GEVDRTFEstimator:
    """
    Generalized-eigenvalue-decomposition (GEVD) relative transfer function
    (RTF) estimator, for use when no measured ATF table is available.

    Motivation
    ----------
    FreeFieldSteering assumes a pure anechoic plane wave: a fixed delay-only
    relationship between mics with no reflections.  Real rooms have early
    reflections and a non-trivial RTF that a measured ATF table would
    normally capture.  When no ATF is available, this class estimates the
    RTF directly from the recording itself using the standard
    covariance-whitening / generalized-eigenvector method (Markovich-Golan,
    Cohen & Gannot 2009; widely used in RTF-MVDR literature):

        R_yy[f] = covariance of mic signals during speech(+noise) frames
        R_nn[f] = covariance of mic signals during noise-only frames

        Solve the generalized eigenproblem   R_yy[f] v = λ R_nn[f] v.
        Let v_max be the eigenvector for the largest eigenvalue λ_max.

        Under the single-dominant-source model R_yy = R_nn + σ_s² h hᴴ,
        algebra shows that  R_nn v_max  is proportional to h (NOT v_max
        itself).  So the RTF estimate is:

            h_hat[f] = (R_nn[f] @ v_max) / (R_nn[f] @ v_max)[ref_mic]

        This is the "covariance whitening + principal eigenvector" method:
        solving the generalized pencil directly is mathematically equivalent
        to whitening by R_nn^{-1/2} and taking the principal eigenvector of
        the whitened speech covariance, but avoids computing a matrix square
        root (better conditioned, and scipy.linalg.eigh supports generalized
        Hermitian-definite pencils natively).

    Trust gating (critical for multi-talker scenes)
    -------------------------------------------------
    A pure GEVD estimate has no notion of which source is "the target" — in
    babble or cocktail-party scenes it will lock onto whichever source is
    most dominant in the speech-labelled frames, which need not be the
    gaze/SRP-indicated target.  To avoid silently steering toward the wrong
    speaker, GEVDRTFEstimator never replaces the angle-based steering vector
    outright.  Instead, refine_steering() blends bin-by-bin against the
    analytic vector d_analytic already computed from gaze/SRP:

        cos_sim[f] = |<h_hat[f], d_analytic[f]>| / (‖h_hat[f]‖ ‖d_analytic[f]‖)

        if cos_sim[f] > GEVD_TRUST_COS:  use h_hat[f]       (refine with RTF)
        else:                             use d_analytic[f]  (fall back)

    This means GEVD can only ADD reverberant/reflection structure on top of
    a direction the angle estimator already agrees with; it can never
    override the angle estimate to point somewhere completely different.
    When GEVD is unreliable (cold start, ill-conditioned covariances,
    disagreement with the angle estimate) the result is identical to not
    using GEVD at all.

    Update cadence
    --------------
    R_yy / R_nn are accumulated every frame via EMA (separate from the
    beamformer's own R_nn⁻¹ Woodbury state — this is an independent pair of
    accumulators used only for RTF estimation).  The generalized eigenproblem
    is only re-solved every GEVD_RECOMPUTE_EVERY speech frames (and not
    before GEVD_MIN_SPEECH_FR / GEVD_MIN_NOISE_FR frames have been seen),
    since eigh is more expensive than a Woodbury update and the true RTF
    should not change rapidly frame-to-frame.  The committed RTF is smoothed
    across recomputes with GEVD_BLEND_HOLD to avoid abrupt jumps when the
    eigenproblem is re-solved.

    Parameters
    ----------
    N            : number of microphones.
    n_bins       : number of STFT bins (B = n_fft // 2 + 1).
    ref_mic      : reference microphone index for RTF normalisation (default 0).
    cov_alpha    : EMA coefficient for R_yy / R_nn accumulation.
    min_noise_frames  : minimum noise frames before first GEVD solve.
    min_speech_frames : minimum speech frames before first GEVD solve.
    recompute_every   : recompute GEVD every N speech frames after warmup.
    reg_frac     : diagonal regularisation as a fraction of mean trace,
                   applied to both R_yy and R_nn before solving (numerical
                   stability; also guards against a near-singular R_nn).
    trust_cos    : minimum |cosine similarity| against the analytic steering
                   vector for a bin's GEVD estimate to be trusted.
    blend_hold   : EMA coefficient smoothing the committed RTF estimate
                   across successive recomputes (higher = smoother/slower).
    """

    def __init__(self,
                 N:                  int,
                 n_bins:             int,
                 ref_mic:            int   = 0,
                 cov_alpha:          float = GEVD_COV_ALPHA,
                 min_noise_frames:   int   = GEVD_MIN_NOISE_FR,
                 min_speech_frames:  int   = GEVD_MIN_SPEECH_FR,
                 recompute_every:    int   = GEVD_RECOMPUTE_EVERY,
                 reg_frac:           float = GEVD_REG_FRAC,
                 trust_cos:          float = GEVD_TRUST_COS,
                 blend_hold:         float = GEVD_BLEND_HOLD):
        self.N       = N
        self.n_bins  = n_bins
        self.ref_mic = ref_mic
        self.cov_alpha   = float(cov_alpha)
        self.min_noise_frames  = int(min_noise_frames)
        self.min_speech_frames = int(min_speech_frames)
        self.recompute_every   = int(recompute_every)
        self.reg_frac    = float(reg_frac)
        self.trust_cos   = float(trust_cos)
        self.blend_hold  = float(blend_hold)

        # EMA covariance accumulators — independent of the beamformer's R_nn⁻¹
        eye = np.eye(N, dtype=np.complex128)
        self._Ryy = np.stack([eye.copy() * 1e-6 for _ in range(n_bins)])  # (B,N,N)
        self._Rnn = np.stack([eye.copy() * 1e-6 for _ in range(n_bins)])  # (B,N,N)

        # Committed RTF estimate, (N, B) complex64, mic-0-normalised.
        # Initialised to None: until the first successful solve, refine_steering
        # simply passes the analytic vector through unchanged.
        self._h_hat: np.ndarray | None = None   # (N, B) complex64

        # Frame counters
        self._n_speech = 0
        self._n_noise  = 0
        self._frames_since_solve = 0
        self._n_solves = 0

        # Diagnostics
        self._trusted_bin_history: list[int] = []   # bins trusted per solve
        self._last_trust_mask: np.ndarray | None = None

    # ── Update covariance accumulators ─────────────────────────────────────────

    def update(self, X: np.ndarray, is_speech: bool) -> None:
        """
        Accumulate one STFT frame into the appropriate covariance estimate.

        Parameters
        ----------
        X         : (N, B) complex STFT frame.
        is_speech : True → accumulate into R_yy (speech+noise covariance).
                    False → accumulate into R_nn (noise-only covariance).
        """
        ENERGY_FLOOR = 1e-8
        power = float(np.mean(np.abs(X) ** 2))
        if power < ENERGY_FLOOR:
            return   # skip near-silent frames (numerically uninformative)

        u   = X.T.astype(np.complex128)                      # (B, N)
        uuH = np.einsum('bi,bj->bij', u, u.conj())            # (B, N, N)

        a = self.cov_alpha
        if is_speech:
            self._Ryy = a * self._Ryy + (1.0 - a) * uuH
            self._n_speech += 1
            self._frames_since_solve += 1
        else:
            self._Rnn = a * self._Rnn + (1.0 - a) * uuH
            self._n_noise += 1

    # ── GEVD solve ────────────────────────────────────────────────────────────

    def _ready(self) -> bool:
        return (self._n_speech >= self.min_speech_frames and
                self._n_noise  >= self.min_noise_frames)

    def maybe_recompute(self) -> bool:
        """
        Re-solve the GEVD if enough new speech frames have accumulated since
        the last solve (and the minimum frame-count warmup is satisfied).

        Returns
        -------
        bool : True if a recompute was performed this call.
        """
        if not self._ready():
            return False
        if (self._n_solves > 0 and
                self._frames_since_solve < self.recompute_every):
            return False
        self._solve()
        self._frames_since_solve = 0
        return True

    def _solve(self) -> None:
        """
        Solve the generalized eigenproblem R_yy v = λ R_nn v per bin and
        extract the RTF estimate h_hat[f] = (R_nn v_max) / (R_nn v_max)[ref].
        """
        from scipy.linalg import eigh

        N = self.N
        h_new = np.zeros((self.n_bins, N), dtype=np.complex128)
        solved_ok = np.zeros(self.n_bins, dtype=bool)

        for f in range(self.n_bins):
            Ryy_f = self._Ryy[f]
            Rnn_f = self._Rnn[f]

            tr_nn = float(np.real(np.trace(Rnn_f)))
            tr_yy = float(np.real(np.trace(Ryy_f)))
            if tr_nn < 1e-30 or tr_yy < 1e-30:
                continue

            reg_nn = self.reg_frac * (tr_nn / N)
            reg_yy = self.reg_frac * (tr_yy / N)
            Rnn_reg = Rnn_f + reg_nn * np.eye(N, dtype=np.complex128)
            Ryy_reg = Ryy_f + reg_yy * np.eye(N, dtype=np.complex128)

            # Hermitise defensively (EMA accumulation can leave tiny
            # asymmetric numerical residue)
            Rnn_reg = 0.5 * (Rnn_reg + Rnn_reg.conj().T)
            Ryy_reg = 0.5 * (Ryy_reg + Ryy_reg.conj().T)

            try:
                # Generalized Hermitian-definite eigenproblem: Ryy v = λ Rnn v
                # scipy returns eigenvalues in ASCENDING order.
                eigvals, eigvecs = eigh(Ryy_reg, Rnn_reg)
            except np.linalg.LinAlgError:
                continue

            v_max = eigvecs[:, -1]                       # largest eigenvalue
            h_unnorm = Rnn_reg @ v_max                    # ∝ h  (see docstring)

            ref_val = h_unnorm[self.ref_mic]
            if np.abs(ref_val) < 1e-12:
                continue

            h_new[f] = h_unnorm / ref_val
            solved_ok[f] = True

        if not solved_ok.any():
            return

        h_new_T = h_new.T.astype(np.complex64)   # (N, B)

        if self._h_hat is None:
            self._h_hat = h_new_T
            if not solved_ok.all():
                self._h_hat[:, ~solved_ok] = np.nan
        else:
            blend = self.blend_hold
            ok_idx = np.where(solved_ok)[0]
            self._h_hat[:, ok_idx] = (
                blend * self._h_hat[:, ok_idx]
                + (1.0 - blend) * h_new_T[:, ok_idx]
            )

        self._n_solves += 1
        self._trusted_bin_history.append(int(solved_ok.sum()))

    # ── Trust-gated refinement of an analytic steering vector ──────────────────

    def refine_steering(self, d_analytic: np.ndarray) -> np.ndarray:
        """
        Bin-wise trust-gated blend of the GEVD RTF estimate against an
        analytic/angle-based steering vector.

        Parameters
        ----------
        d_analytic : (N, B) complex steering vector from FreeFieldSteering
                     or ATFSteering, evaluated at the current gaze/SRP angle.

        Returns
        -------
        d_refined : (N, B) complex64.  Bins where the GEVD estimate is
                    available and agrees with d_analytic (cosine similarity
                    above trust_cos) use the GEVD estimate; all other bins
                    pass d_analytic through unchanged.
        """
        if self._h_hat is None:
            self._last_trust_mask = np.zeros(self.n_bins, dtype=bool)
            return d_analytic

        h = self._h_hat                                    # (N, B) complex64
        valid = np.isfinite(h).all(axis=0)                 # (B,) bins with a solve

        # Fix: don't conjugate h — h (from _solve(), ref-mic-normalised) and
        # d_analytic (from FreeFieldSteering/ATFSteering, also ref-mic-
        # normalised) are both in the same conjugate/manifold convention, so
        # the correct inner product for cosine similarity is <h, d_analytic>,
        # not <h.conj(), d_analytic>. The conjugated version tests the wrong
        # relationship and was suppressing genuine agreement down to
        # cos_sim ~0.355 even for a mathematically perfect RTF estimate.
        num = np.abs(np.einsum('nb,nb->b',
                                h.astype(np.complex128),
                                d_analytic.astype(np.complex128)))
        h_norm = np.linalg.norm(h, axis=0)
        d_norm = np.linalg.norm(d_analytic, axis=0)
        denom  = h_norm * d_norm + 1e-12
        cos_sim = np.where(valid, num / denom, 0.0)         # (B,)

        trust_mask = valid & (cos_sim > self.trust_cos)      # (B,)
        self._last_trust_mask = trust_mask

        d_refined = np.where(trust_mask[None, :],
                              h.astype(np.complex64),
                              d_analytic.astype(np.complex64))
        return d_refined

    # ── Reset / diagnostics ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset covariance accumulators, committed RTF, and counters."""
        eye = np.eye(self.N, dtype=np.complex128)
        self._Ryy = np.stack([eye.copy() * 1e-6 for _ in range(self.n_bins)])
        self._Rnn = np.stack([eye.copy() * 1e-6 for _ in range(self.n_bins)])
        self._h_hat = None
        self._n_speech = 0
        self._n_noise  = 0
        self._frames_since_solve = 0
        self._n_solves = 0
        self._trusted_bin_history.clear()
        self._last_trust_mask = None

    def diagnostics(self) -> dict:
        trusted_now = (int(self._last_trust_mask.sum())
                       if self._last_trust_mask is not None else 0)
        hist = self._trusted_bin_history
        return {
            'gevd_n_speech':        self._n_speech,
            'gevd_n_noise':         self._n_noise,
            'gevd_n_solves':        self._n_solves,
            'gevd_solved_bins_last': (hist[-1] if hist else 0),
            'gevd_trusted_bins_last': trusted_now,
            'gevd_trusted_frac_last': trusted_now / max(self.n_bins, 1),
        }

    def print_diagnostics(self, prefix: str = '') -> None:
        d   = self.diagnostics()
        tag = f'[GEVD {prefix}] ' if prefix else '[GEVD] '
        print(f'{tag}speech_frames={d["gevd_n_speech"]}  '
              f'noise_frames={d["gevd_n_noise"]}  solves={d["gevd_n_solves"]}')
        print(f'{tag}last solve: {d["gevd_solved_bins_last"]}/{self.n_bins} bins solved, '
              f'{d["gevd_trusted_bins_last"]}/{self.n_bins} bins trusted '
              f'({100*d["gevd_trusted_frac_last"]:.0f}%)')