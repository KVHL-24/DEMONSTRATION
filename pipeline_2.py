"""
pipeline_2.py — Full Audio Denoising Pipeline
============================================

"""

from __future__ import annotations
import time
import numpy as np

from stft       import stft_multichannel, istft, make_window, F_WIN, HOP, FS
from doa_2        import (DOA_GCCSRP, DOA_Gaze, ATFSteering, FreeFieldSteering,
                        GEVDRTFEstimator,
                        aria_mic_positions,
                        gaze_vector_to_theta, gaze_vector_to_angles,
                        compute_steering_vector,
                        SACCADE_HOLD_FRAMES, SACCADE_THRESH_DEG)
from beamformer_2 import MVDRBeamformer
from vad        import EnergyVAD
from denoiser   import DeepFilterNetDenoiser
from gaze_processing import GazeStabilizer
from mic_selection   import AdaptiveMicSelector

# ── NOTE on use_gaze_stabilizer / use_saccade_hold / use_mic_selection ──────
# Historically these three mechanisms were documented as "wired in by
# default" (see gaze_processing.py / mic_selection.py module docstrings and
# doa_2.DOA_Gaze's own docstring), but AriaDenoisingPipeline never actually
# routed through them:
#   • _get_steering() computed theta/phi via the *module-level*
#     gaze_vector_to_theta/gaze_vector_to_angles helpers directly, instead
#     of calling self.doa.steering_vector_from_gaze() (the method that owns
#     both the GazeStabilizer and the v6.4 saccade-hold logic). self.doa was
#     constructed but, for use_gaze=True, never called anywhere in process().
#   • MVDRBeamformer was always constructed with mic_selector=None, and
#     process_frame() was always called without theta/phi, so
#     AdaptiveMicSelector could never activate even if one had been built.
# Both were therefore complete no-ops in every eval run to date. The
# implementation below wires them in for real, each behind its own
# constructor flag, so their individual contribution can be measured
# (see eval_synthetic_2.py --ablate).
#
# v7.2 fix: the saccade-hold logic wired in below had its own bug —
# _sacc_prev_theta (the reference used to detect the NEXT jump) was only
# updated while NOT holding, so during a hold it stayed frozen at the
# pre-saccade direction. Once gaze had genuinely settled on a new
# sustained direction farther than saccade_thresh from that frozen
# reference, the trigger condition stayed true every subsequent frame,
# continuously re-arming the hold — it could never release, permanently
# freezing the beamformer on the ORIGINAL pre-saccade direction after the
# first real saccade in a clip. See the inline comment in _get_steering()
# below. This only affects use_saccade_hold=True (not the default
# gazestab/micsel ablation sweep), but is a real correctness bug whenever
# that flag is used with genuinely moving sources (`_dynamic` scenarios).


# ── Gaze input normalisation ──────────────────────────────────────────────────

def _parse_gaze(gaze, n_frames: int):
    """
    Normalise any supported gaze input.

    Returns
    -------
    ga     : (n_frames,) float32  azimuth angles in radians, OR
             (n_frames, 3) float32  3-D unit gaze vectors
    is_vec : True if ga is a vector array, False if it holds angles

    Supported input shapes
    ----------------------
    scalar (0-D or size-1)     → fixed azimuth angle, broadcast to (n_frames,)
    (3,) array AND n_frames==1 → fixed 3-D unit gaze vector, tiled to (1, 3)
    (n_frames,) array          → per-frame azimuth angles.
                                 BUG FIX: a (3,) array with n_frames > 1 is
                                 now correctly treated as azimuth angles, not a
                                 3-D vector.  The old code silently fed a
                                 nonsense unit vector for all clips whose gaze
                                 array happened to have exactly 3 frames.
    (n_frames, 3) array        → per-frame 3-D unit gaze vectors
    (1, 3) array               → fixed 3-D unit gaze vector, tiled to (n_frames, 3)
    """
    if gaze is None:
        raise ValueError("use_gaze=True but no gaze input provided.")

    g = np.asarray(gaze, dtype=np.float32)

    if g.ndim == 0:
        return np.full(n_frames, float(g), dtype=np.float32), False

    if g.ndim == 1:
        # Only treat a length-3 1-D array as a unit vector for single-frame
        # clips.  For all other cases it is per-frame azimuth angles.
        if g.size == 3 and n_frames == 1:
            norm = np.linalg.norm(g)
            g    = g / norm if norm > 1e-9 else g
            return np.tile(g[None, :], (1, 1)).astype(np.float32), True
        if len(g) < n_frames:
            g = np.pad(g, (0, n_frames - len(g)), mode='edge')
        return g[:n_frames], False

    if g.ndim == 2 and g.shape[1] == 3:
        if len(g) < n_frames:
            pad = np.tile(g[-1:], (n_frames - len(g), 1))
            g   = np.concatenate([g, pad], axis=0)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return (g[:n_frames] / norms[:n_frames]).astype(np.float32), True

    raise ValueError(
        f"Unsupported gaze shape {g.shape}. Expected: scalar, (n_frames,), "
        f"(n_frames,3), or a single (3,) vector (only when n_frames==1).")


def _gaze_to_theta_phi(ga_entry, is_vec: bool,
                       mic_plane_normal,
                       use_elevation: bool) -> tuple[float, float]:
    if not is_vec:
        return float(ga_entry), 0.0
    if use_elevation:
        return gaze_vector_to_angles(ga_entry, mic_plane_normal)
    return gaze_vector_to_theta(ga_entry, mic_plane_normal), 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AriaDenoisingPipeline:
    """
    End-to-end spatial audio denoising pipeline.

    Parameters
    ----------
    use_gaze        : True  → DOA from gaze input (angle or 3-D vector)
                      False → GCC-PHAT + SRP estimated on speech frames.
    use_elevation   : Only active when use_gaze=True and gaze vectors are
                      provided.  Extracts elevation φ from the 3-D gaze vector.
    atf_steering    : Optional ATFSteering instance for measured steering
                      vectors.  When None, FreeFieldSteering is used (analytic
                      plane-wave model), optionally refined by GEVD-RTF
                      estimation if use_gevd_rtf=True.
    use_gevd_rtf    : (v7, NEW) When True and atf_steering is None, refine
                      the analytic FreeFieldSteering vector every frame with
                      a data-driven RTF estimate from GEVDRTFEstimator.  The
                      refinement is trust-gated against the analytic/gaze
                      direction (see module docstring) so it cannot lock
                      onto the wrong speaker in multi-talker scenes.  Has no
                      effect when atf_steering is supplied (a measured ATF
                      is already better than any data-driven estimate).
    mic_pos         : (N, 2) or (N, 3) mic positions in metres.
    mic_plane_normal: (3,) normal of the mic plane for gaze projection.
    n_fft           : FFT window length (default 512)
    hop             : hop size in samples (default 256)
    alpha           : R_nn⁻¹ EMA forgetting factor.
                      α=0.97  → TC ≈ 5 s  — recommended when noise frames are
                                scarce (~8 % silence in a 60 s clip ≈ 4.8 s).
                      α=0.995 → TC ≈ 27 s — only suitable when silence is
                                abundant (> 20 % of the clip).
    vad_thr_db      : VAD broadband speech threshold in dB above noise floor.
    rt60_s          : Room RT60.  Extends VAD hangover to 3×RT60.
                      Pass 0.0 for energy_vad mode to prevent the extended
                      hangover from bridging all silence gaps.
    doa_reliable    : Controls the beamformer DOA-change detector.
                      None  → True if use_gaze, False for SRP (default)
                      True  → detector active   (oracle_target_dir, real-time)
                      False → detector disabled  (oracle_gaze, energy_vad, srp)
    gevd_kwargs     : optional dict of keyword overrides forwarded to
                      GEVDRTFEstimator (e.g. {'trust_cos': 0.8}).  Ignored
                      when use_gevd_rtf=False.
    verbose         : print per-frame DOA status
    """

    def __init__(self,
                 use_gaze:          bool              = False,
                 use_elevation:     bool              = False,
                 atf_steering:      ATFSteering | None = None,
                 use_gevd_rtf:      bool              = False,
                 mic_pos:           np.ndarray | None = None,
                 mic_plane_normal:  np.ndarray | None = None,
                 n_fft:             int   = F_WIN,
                 hop:               int   = HOP,
                 alpha:             float = 0.97,
                 vad_thr_db:        float = 3.0,
                 rt60_s:            float = 0.15,
                 doa_reliable:      bool | None = None,
                 gevd_kwargs:       dict | None = None,
                 use_gaze_stabilizer: bool        = False,
                 gaze_stabilizer_kwargs: dict | None = None,
                 use_saccade_hold:  bool          = False,
                 hold_frames:       int           = SACCADE_HOLD_FRAMES,
                 saccade_thresh:    float         = SACCADE_THRESH_DEG,
                 use_mic_selection: bool          = False,
                 mic_selector_kwargs: dict | None = None,
                 use_projection_guard: bool       = False,
                 weight_stride:     int           = 1,
                 use_bypass_gate:   bool          = False,
                 gate_on_db:        float         = 15.0,
                 gate_off_db:       float         = 10.0,
                 observer:          'object|None' = None,
                 verbose:           bool  = False):
        """
        use_gaze_stabilizer : (bug fix) actually apply gaze_processing.
                    GazeStabilizer to the raw per-frame gaze sample before
                    it is turned into theta/phi. Previously constructed
                    inside a DOA_Gaze instance that process() never called,
                    so it had zero effect regardless of this flag's
                    predecessor. Default False so existing callers see
                    unchanged behaviour until they opt in.
        gaze_stabilizer_kwargs : optional kwargs forwarded to GazeStabilizer.
        use_saccade_hold : (bug fix) actually apply the v6.4 saccade-hold
                    logic (hold the previous committed direction for
                    hold_frames frames after a jump > saccade_thresh
                    degrees) — same no-op history as use_gaze_stabilizer.
        use_mic_selection : (bug fix) actually construct a
                    mic_selection.AdaptiveMicSelector and attach it to the
                    MVDRBeamformer, AND pass theta/phi into
                    beamformer.process_frame() every frame so it can
                    activate. Previously the selector was never constructed
                    at all and process_frame() was never given theta/phi,
                    so mic_selector could never fire even if attached.
        mic_selector_kwargs : optional kwargs forwarded to
                    AdaptiveMicSelector (e.g. {'K_min': 3}).
        weight_stride : (v8.0, runtime_demo) recompute MVDR weights only
                    every k-th speech frame, reusing them in between.
                    1 (default) is bit-identical to previous behaviour.
                    See MVDRBeamformer.weight_stride.
        use_bypass_gate : (v8.0, runtime_demo) when the beamformer's causal
                    SNR estimate exceeds gate_on_db, bypass stages 2–3
                    entirely and pass mic 0 through unprocessed (same
                    passthrough the warmup period already uses), until the
                    estimate falls back below gate_off_db (hysteresis).
                    The SNR EMAs keep updating during a bypass stretch —
                    via MVDRBeamformer.update_snr_stats() — so the gate
                    can re-close; R_nn⁻¹ does NOT update while bypassed,
                    and stored weights are invalidated on re-close.
                    Default False: no behaviour change.
        gate_on_db / gate_off_db : bypass-gate hysteresis thresholds (dB
                    estimated input SNR). on > off, or the gate chatters.
        observer : (v8.0, runtime_demo) optional callable
                    observer(frame_idx: int, data: dict) invoked once per
                    STFT frame AFTER the frame is processed, from both
                    process() and process_frame(). data holds references
                    (NOT copies — copy before storing) to per-frame
                    internals: speech flag, theta/phi, steering d, MVDR
                    weights, mic mask, gate state, and stage timings in
                    µs. None (default): zero overhead, no timing calls.
        use_projection_guard : forwarded to MVDRBeamformer (v6.0 steering-
                    vector projection guard on noise-frame updates,
                    default OFF — see beamformer_2.py's PROJECTION_GUARD
                    module comment for the diffuse-noise-vs-directional-
                    interferer trade-off). Previously only editable by
                    hand in beamformer_2.py; exposed here so it can be
                    A/B tested via eval_synthetic_2.py's mode+suffix
                    mechanism, e.g. mode='oracle_gaze+projguard'.
        """

        self.use_gaze         = use_gaze
        self.use_elevation    = use_elevation and use_gaze
        self.atf_steering     = atf_steering
        self.n_fft            = n_fft
        self.hop              = hop
        self.verbose          = verbose
        self.window           = make_window(n_fft)
        self.mic_plane_normal = mic_plane_normal

        # ── Gaze stabilizer (bug fix — see module-level note above) ────────
        self.use_gaze_stabilizer = bool(use_gaze_stabilizer)
        if self.use_gaze_stabilizer:
            kwargs = dict(gaze_stabilizer_kwargs) if gaze_stabilizer_kwargs else {}
            self._gaze_stabilizer = GazeStabilizer(fs=FS, hop=hop, **kwargs)
        else:
            self._gaze_stabilizer = None

        # ── Saccade hold (bug fix — see module-level note above) ───────────
        self.use_saccade_hold = bool(use_saccade_hold)
        self._hold_frames     = int(hold_frames)
        self._saccade_thresh  = float(saccade_thresh)
        self._sacc_prev_theta: float | None = None
        self._sacc_hold_cnt:   int   = 0
        self._sacc_held_theta: float = 0.0
        self._sacc_held_phi:   float = 0.0
        self._saccade_events:  int   = 0

        self.use_mic_selection = bool(use_mic_selection)
        self._mic_selector_kwargs = (dict(mic_selector_kwargs)
                                     if mic_selector_kwargs else {})

        # ── Resolve doa_reliable ──────────────────────────────────────────
        # For oracle_gaze / energy_vad: always pass doa_reliable=False.
        # Saccades (5–25°, ~3 Hz) fire the detector ~90×/clip with the old
        # 15° threshold, each time decaying R_nn 70% toward the isotropic
        # prior.  With α=0.995 (27 s TC) R_nn never recovers → stays at DAS.
        if doa_reliable is None:
            self._doa_reliable = bool(use_gaze)
        else:
            self._doa_reliable = bool(doa_reliable)

        # ── N mics, mic positions, and steering ────────────────────────────
        # ATFSteering path  (measured ATF supplied):
        #   self.steerer = atf_steering
        #   beamformer initialised with diffuse R_nn prior from H_rel.
        #
        # FreeFieldSteering path  (no ATF — ISM or anechoic simulation):
        #   self.steerer = FreeFieldSteering(mic_pos, n_fft=n_fft)
        #   beamformer initialised with isotropic prior.
        #   v7: optionally refined per-frame by GEVDRTFEstimator.
        if atf_steering is not None:
            self.N       = atf_steering.N
            self.mic_pos = mic_pos
            self.steerer = atf_steering                              # ATF-based
            if use_gevd_rtf:
                print('[Pipeline] WARNING: use_gevd_rtf=True ignored — '
                      'a measured ATF was supplied and is already a better '
                      'steering source than a data-driven RTF estimate.')
            self._use_gevd_rtf = False
        else:
            self.mic_pos = mic_pos if mic_pos is not None else aria_mic_positions()
            self.N       = self.mic_pos.shape[0]
            self.steerer = FreeFieldSteering(self.mic_pos, n_fft=n_fft)  # analytic
            self._use_gevd_rtf = bool(use_gevd_rtf)

        # ── v7: GEVD RTF estimator (only constructed in FreeFieldSteering path) ──
        n_bins = n_fft // 2 + 1
        if self._use_gevd_rtf:
            kwargs = dict(gevd_kwargs) if gevd_kwargs else {}
            self.gevd = GEVDRTFEstimator(N=self.N, n_bins=n_bins, **kwargs)
            print('[Pipeline] GEVD-RTF refinement enabled '
                  f'(N={self.N} mics, {n_bins} bins).')
        else:
            self.gevd = None

        # ── Stage 2: DOA estimator ────────────────────────────────────────
        if use_gaze:
            if atf_steering is None:
                if self.mic_pos is None:
                    raise ValueError(
                        "use_gaze=True requires either atf_steering or mic_pos.")
                if (use_elevation and self.mic_pos.ndim == 2
                        and self.mic_pos.shape[1] == 2):
                    raise ValueError(
                        "use_elevation=True requires 3-D mic positions (N,3).")
                self.doa = DOA_Gaze(self.mic_pos, n_fft=n_fft,
                                    mic_plane_normal=mic_plane_normal)
            else:
                self.doa = None  # ATF handles steering directly
        else:
            srp_pos = self.mic_pos if self.mic_pos is not None else aria_mic_positions()
            self.doa = DOA_GCCSRP(srp_pos, n_fft=n_fft, srp_alpha=0.97)

        # ── Stage 3: MVDR beamformer ──────────────────────────────────────
        if self.use_mic_selection:
            mic_pos_for_sel = (self.mic_pos if self.mic_pos is not None
                               else aria_mic_positions())
            self.mic_selector = AdaptiveMicSelector(
                mic_pos=mic_pos_for_sel, n_fft=n_fft,
                **self._mic_selector_kwargs)
            print(f'[Pipeline] AdaptiveMicSelector enabled '
                  f'(N={self.mic_selector.N}, K_min={self.mic_selector.K_min}, '
                  f'K_max={self.mic_selector.K_max}).')
        else:
            self.mic_selector = None

        self.beamformer = MVDRBeamformer(self.N, n_fft=n_fft, alpha=alpha,
                                         mic_selector=self.mic_selector,
                                         use_projection_guard=use_projection_guard,
                                         weight_stride=weight_stride)

        # ── v8.0 (runtime_demo): SNR bypass gate + observer ────────────────
        self.use_bypass_gate = bool(use_bypass_gate)
        self.gate_on_db      = float(gate_on_db)
        self.gate_off_db     = float(gate_off_db)
        self._gate_open        = False
        self._gate_frames      = 0     # frames bypassed (diagnostics)
        self._gate_transitions = 0
        self.observer = observer
        self._stream_frame_idx = 0     # observer frame index (streaming path)

        if atf_steering is not None:
            self.beamformer.init_diffuse(atf_steering.H_rel)
        else:
            self.beamformer.init_isotropic()

        # ── VAD ───────────────────────────────────────────────────────────
        # use_spectral=True enables the mid-band spectral ratio condition (B),
        # which is the only reliable way to detect speech at negative broadband
        # SNR (white/pink noise at −20 to −10 dB).
        # rt60_s is forwarded so the hangover can be controlled per mode:
        #   oracle_gaze / oracle_target_dir / srp: rt60_s=0.15 (extend hangover
        #     — they use annotated_vad so the energy VAD is not critical)
        #   energy_vad: rt60_s=0.0 (no extension — energy VAD must see silence
        #     frames to update its noise floor estimate)
        self.vad = EnergyVAD(threshold_db=vad_thr_db,
                             rt60_s=rt60_s,
                             fs=FS,
                             hop=hop,
                             use_spectral=True,
                             spectral_thr_db=1.5,
                             n_fft=n_fft)

        # Stage 4: DeepFilterNet (lazy-loaded on first enhance() call)
        self.denoiser = DeepFilterNetDenoiser()

        self._last_theta = 0.0
        self._last_phi   = 0.0

    # ── Internal: compute d for one frame ────────────────────────────────────

    def _get_steering(self, Xk: np.ndarray,
                      ga_entry, is_vec: bool,
                      speech: bool) -> tuple[np.ndarray, float, float]:
        if self.use_gaze:
            # ── Bug fix: actually run the raw gaze sample through the
            # stabilizer before decoding it to theta/phi (previously this
            # lived inside a DOA_Gaze instance that was never called). ──
            ga_use = ga_entry
            if self._gaze_stabilizer is not None:
                ga_use = self._gaze_stabilizer.update(ga_entry, is_vec=is_vec,
                                                       confidence=1.0)

            theta_new, phi_new = _gaze_to_theta_phi(
                ga_use, is_vec, self.mic_plane_normal, self.use_elevation)

            # ── Bug fix: actually apply the v6.4 saccade-hold logic
            # (previously also dead code — same DOA_Gaze instance). ──
            if self.use_saccade_hold:
                if self._sacc_prev_theta is not None and self._hold_frames > 0:
                    delta_deg = abs(np.degrees(theta_new - self._sacc_prev_theta))
                    if delta_deg > 180.0:
                        delta_deg = 360.0 - delta_deg
                    if delta_deg > self._saccade_thresh:
                        self._saccade_events += 1
                        self._sacc_hold_cnt = self._hold_frames

                # ── Bug fix (v7.2): _sacc_prev_theta must track the raw
                # incoming theta EVERY frame — used to detect the NEXT
                # frame-to-frame jump — independent of whether we are
                # currently holding. It previously only updated in the
                # "not holding" branch below, so during a hold it stayed
                # frozen at the pre-saccade direction. Once gaze had
                # genuinely settled on a new sustained direction more
                # than saccade_thresh away from that frozen reference,
                # the trigger check above stayed true on every following
                # frame (comparing the new steady gaze against the stale
                # pre-saccade reference), continuously resetting
                # _sacc_hold_cnt back up to hold_frames — the hold could
                # never release and the beamformer stayed pointed at the
                # ORIGINAL pre-saccade direction indefinitely. Invisible
                # on static-source scenes (no real settle-elsewhere to
                # trigger it); devastating on the `_dynamic` (moving-
                # source) scenarios, exactly where gaze/saccade handling
                # matters most.
                self._sacc_prev_theta = theta_new

                if self._sacc_hold_cnt > 0:
                    self._sacc_hold_cnt -= 1
                    theta, phi = self._sacc_held_theta, self._sacc_held_phi
                else:
                    theta, phi = theta_new, phi_new
                    self._sacc_held_theta = theta_new
                    self._sacc_held_phi   = phi_new
            else:
                theta, phi = theta_new, phi_new

            self._last_theta = theta
            self._last_phi   = phi
        else:
            if speech:
                theta = self.doa.estimate(Xk)
                self._last_theta = theta
            else:
                theta = self._last_theta
            phi = self._last_phi

        # Both ATFSteering and FreeFieldSteering expose the same interface;
        # no conditional needed here.
        d = self.steerer.steering_vector(theta, phi)

        # ── v7: GEVD-RTF trust-gated refinement ───────────────────────────
        # Feed the covariance accumulators every frame (regardless of
        # speech/noise — update() routes internally), then periodically
        # re-solve and blend.  Refinement only ever substitutes bins where
        # the data-driven estimate agrees with this analytic vector, so a
        # disagreeing/uninitialised GEVD estimate is exactly equivalent to
        # not using it at all.
        if self.gevd is not None:
            self.gevd.update(Xk, is_speech=speech)
            self.gevd.maybe_recompute()
            d = self.gevd.refine_steering(d)

        return d, theta, phi

    # ── v8.0 (runtime_demo): SNR bypass gate ─────────────────────────────────

    def _gate_check(self, X_frame: np.ndarray, speech: bool) -> bool:
        """Decide whether THIS frame bypasses stages 2–3 entirely.

        Called before any steering/beamforming work. Hysteresis: opens above
        gate_on_db, closes below gate_off_db. While open, the beamformer is
        never invoked, so its v7.0 SNR EMAs are fed from here instead —
        otherwise the estimate would freeze at its gate-opening value and
        the gate could never close. R_nn⁻¹ is deliberately NOT updated while
        bypassed (that is the compute being saved); on re-close the stored
        weights are invalidated so the stride logic cannot reuse them.

        Requires the beamformer's use_snr_adaptive_blend (default on): with
        it disabled the EMAs never update, estimated_snr_db() stays None,
        and the gate simply never opens.
        """
        if not self.use_bypass_gate:
            return False
        est = self.beamformer.estimated_snr_db()
        if est is not None:
            if self._gate_open:
                if est < self.gate_off_db:
                    self._gate_open = False
                    self._gate_transitions += 1
                    self.beamformer.invalidate_weights()
            elif est > self.gate_on_db:
                self._gate_open = True
                self._gate_transitions += 1
        if self._gate_open:
            self._gate_frames += 1
            self.beamformer.update_snr_stats(X_frame, is_noise=not speech)
        return self._gate_open

    # ── Full-recording batch processing ──────────────────────────────────────

    def process(self,
                audio: np.ndarray,
                gaze:  np.ndarray | float | None = None,
                annotated_vad: np.ndarray | None = None,
                skip_denoise: bool = False) -> np.ndarray:
        """
        Process a full multi-channel recording end-to-end.

        Parameters
        ----------
        audio          : (N, T) or (T,) float32 at 48 kHz
        gaze           : gaze input (only used when use_gaze=True)
        annotated_vad  : optional (n_stft_frames,) bool array; overrides
                         the energy-based VAD when provided.
        skip_denoise   : skip DeepFilterNet, return beamformed iSTFT directly.

        Returns
        -------
        enhanced : (T,) float32
        """
        if audio.ndim == 1:
            audio = np.tile(audio[None, :], (self.N, 1))

        T = audio.shape[1]

        print("[Pipeline] Stage 1: STFT...")
        X        = stft_multichannel(audio, window=self.window,
                                     n_fft=self.n_fft, hop=self.hop)
        n_frames = X.shape[2]

        if self.use_gaze:
            ga, is_vec = _parse_gaze(gaze, n_frames)
            mode_str   = "3-D vectors" if is_vec else "azimuth angles"
            elev_str   = " + elevation" if (is_vec and self.use_elevation) else ""
            print(f"[Pipeline] Gaze input: {mode_str}{elev_str}")
        else:
            ga, is_vec = None, False

        steerer_type = type(self.steerer).__name__
        gevd_str     = '+GEVD-RTF' if self.gevd is not None else ''
        doa_mode     = ('gaze (' + ('vectors' if is_vec else 'angles') + ')'
                        if self.use_gaze
                        else 'GCC-PHAT+SRP (speech frames, α=0.97)')
        reliable_str = 'on' if self._doa_reliable else 'OFF (saccade-safe)'
        print(f"[Pipeline] Stages 2-3: {doa_mode}, steering={steerer_type}{gevd_str}, "
              f"DOA-jump-detector={reliable_str}")

        doa_reliable = self._doa_reliable

        n_bins = self.n_fft // 2 + 1
        Y = np.zeros((n_bins, n_frames), dtype=np.complex64)
        Y[:, :] = X[0, :, :]   # mic-0 passthrough until weights are valid

        doa_angles_deg = []
        obs = self.observer
        for k in range(n_frames):
            Xk     = X[:, :, k]
            speech = (bool(annotated_vad[k]) if annotated_vad is not None
                      else self.vad.is_speech(Xk))

            ga_k          = ga[k] if ga is not None else None

            # ── v8.0: SNR bypass gate ──────────────────────────────────────
            # Y[:, k] already holds the mic-0 passthrough from the init
            # above, so a bypassed frame needs no write at all.
            if self._gate_check(Xk, speech):
                doa_angles_deg.append(np.rad2deg(self._last_theta))
                if obs is not None:
                    obs(k, {'speech': speech, 'gated': True,
                            'theta': self._last_theta, 'phi': self._last_phi,
                            'd': None, 'w': None, 'mask': None,
                            'weights_valid': False,
                            't_doa_us': 0.0, 't_bf_us': 0.0})
                continue

            t0 = time.perf_counter() if obs is not None else 0.0
            d, theta, phi = self._get_steering(Xk, ga_k, is_vec, speech)
            t1 = time.perf_counter() if obs is not None else 0.0
            doa_angles_deg.append(np.rad2deg(theta))

            if self.verbose and k % 100 == 0:
                phi_str = (f"  elev={np.rad2deg(phi):+5.1f}°"
                           if self.use_elevation else "")
                gevd_diag = ''
                if self.gevd is not None:
                    gd = self.gevd.diagnostics()
                    gevd_diag = (f"  gevd_trust={gd['gevd_trusted_frac_last']*100:.0f}%")
                print(f"  frame {k:5d}/{n_frames}  "
                      f"speech={'yes' if speech else ' no '}  "
                      f"DOA={np.rad2deg(theta):+6.1f}°{phi_str}{gevd_diag}")

            t2 = time.perf_counter() if obs is not None else 0.0
            bf_out = self.beamformer.process_frame(
                Xk, d, is_noise=not speech, doa_reliable=doa_reliable,
                theta=theta, phi=phi)

            if self.beamformer._weights_valid or not (not speech):
                Y[:, k] = bf_out

            if obs is not None:
                bf = self.beamformer
                obs(k, {'speech': speech, 'gated': False,
                        'theta': theta, 'phi': phi,
                        'd': d, 'w': bf._weights, 'mask': bf._last_mask,
                        'weights_valid': bf._weights_valid,
                        't_doa_us': (t1 - t0) * 1e6,
                        't_bf_us': (time.perf_counter() - t2) * 1e6})

        angles          = np.array(doa_angles_deg)
        n_speech_frames = self.beamformer._frames_speech
        n_noise_frames  = self.beamformer._frames_noise
        print(f"[Pipeline] DOA: mean={angles.mean():+.1f}°  std={angles.std():.1f}°  "
              f"min={angles.min():+.1f}°  max={angles.max():+.1f}°")
        print(f"[Pipeline] VAD: {n_speech_frames}/{n_frames} speech "
              f"({100*n_speech_frames/max(n_frames,1):.1f}%)  "
              f"{n_noise_frames} noise frames")
        if self.gevd is not None:
            self.gevd.print_diagnostics()

        if self._gaze_stabilizer is not None:
            self._gaze_stabilizer.print_diagnostics()

        if self.use_saccade_hold:
            print(f'[Pipeline] SaccadeHold: events={self._saccade_events}  '
                  f'hold_frames={self._hold_frames}  '
                  f'thresh={self._saccade_thresh:.1f}deg')

        if self.mic_selector is not None:
            self.mic_selector.print_diagnostics()

        # ── beamformer diagnostics ──────────────────────────────────────────
        # These were being computed every clip (cancel_events, distort_resets,
        # trace_guard_resets, outlier_rejects, wet_mean/min, ...) but never
        # surfaced anywhere — unlike gevd/gaze_stabilizer/mic_selector above.
        # Without this, there's no way to see whether the persistent
        # babble/cocktail/directional_* underperformance vs raw_mic is coming
        # from repeated self-nulling + v6.1 repair cycling (distort_resets),
        # frequent v6.2 DAS-fallback engagement (cancel_events), the v7.0
        # purity gate rejecting genuine non-stationary interferer frames as
        # if they were target leakage (outlier_rejects vs frames_noise), or
        # something else entirely.
        bf_diag = self.beamformer.diagnostics()
        print(
            f"[Pipeline] Beamformer: cancel_events={bf_diag['cancel_events']}  "
            f"distort_resets={bf_diag['distort_resets']}  "
            f"trace_guard_resets={bf_diag['trace_guard_resets']}  "
            f"outlier_rejects={bf_diag['outlier_rejects']}/{bf_diag['frames_noise']} noise frames  "
            f"outlier_forced_accepts={bf_diag['outlier_forced_accepts']}  "
            f"wet_mean={bf_diag['wet_mean']:.2f}  wet_min={bf_diag['wet_min']:.2f}  "
            f"out/in_ratio_mean={bf_diag['out_in_ratio_mean']:.1f}dB"
        )

        print("[Pipeline] Stage 4: DeepFilterNet...")
        if skip_denoise:
            enhanced = istft(Y, window=self.window, n_fft=self.n_fft,
                             hop=self.hop, length=T)
            print("[Pipeline] Done (DeepFilterNet skipped).")
            return enhanced

        enhanced = self.denoiser.enhance_from_stft(
            Y,
            istft_fn=lambda S, length=None: istft(
                S, window=self.window, n_fft=self.n_fft, hop=self.hop,
                length=length),
            length=T,
        )

        if not np.all(np.isfinite(enhanced)):
            n_bad = int((~np.isfinite(enhanced)).sum())
            print(f"[Pipeline] WARNING: DeepFilterNet produced {n_bad} NaN/inf "
                  f"— falling back to beamformed output.")
            enhanced = istft(Y, window=self.window, n_fft=self.n_fft,
                             hop=self.hop, length=T)
        print("[Pipeline] Done.")
        return enhanced

    # ── Frame-by-frame streaming ──────────────────────────────────────────────

    def process_frame(self,
                      X_frame: np.ndarray,
                      gaze:    np.ndarray | float | None = None,
                      speech_override: bool | None = None) -> np.ndarray:
        """
        Process one STFT frame (streaming / real-time use).

        Parameters
        ----------
        X_frame         : (N, B) complex STFT frame
        gaze            : scalar angle (rad) or (3,) unit vector
        speech_override : overrides the energy VAD for this frame

        Returns
        -------
        Y_frame : (B,) complex beamformed spectrum
        """
        speech = (speech_override if speech_override is not None
                  else self.vad.is_speech(X_frame))

        obs = self.observer
        k_idx = self._stream_frame_idx
        self._stream_frame_idx = k_idx + 1

        # ── v8.0: SNR bypass gate (mirrors the process() loop) ────────────
        if self._gate_check(X_frame, speech):
            if obs is not None:
                obs(k_idx, {'speech': speech, 'gated': True,
                            'theta': self._last_theta, 'phi': self._last_phi,
                            'd': None, 'w': None, 'mask': None,
                            'weights_valid': False,
                            't_doa_us': 0.0, 't_bf_us': 0.0})
            return X_frame[0].astype(np.complex64, copy=True)

        if self.use_gaze:
            if gaze is None:
                raise ValueError("use_gaze=True: gaze argument required.")
            g = np.asarray(gaze, dtype=np.float32)
            if g.ndim == 0 or g.size == 1:
                ga_k, is_vec = float(g.ravel()[0]), False
            elif g.ndim == 1 and g.size == 3:
                ga_k, is_vec = g, True
            else:
                raise ValueError(
                    f"Per-frame gaze must be scalar or (3,); got {g.shape}.")
        else:
            ga_k, is_vec = None, False

        t0 = time.perf_counter() if obs is not None else 0.0
        d, theta, phi = self._get_steering(X_frame, ga_k, is_vec, speech)
        t1 = time.perf_counter() if obs is not None else 0.0
        y = self.beamformer.process_frame(
            X_frame, d, is_noise=not speech,
            doa_reliable=self._doa_reliable, theta=theta, phi=phi)
        if obs is not None:
            bf = self.beamformer
            obs(k_idx, {'speech': speech, 'gated': False,
                        'theta': theta, 'phi': phi,
                        'd': d, 'w': bf._weights, 'mask': bf._last_mask,
                        'weights_valid': bf._weights_valid,
                        't_doa_us': (t1 - t0) * 1e6,
                        't_bf_us': (time.perf_counter() - t1) * 1e6})
        return y

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset all per-clip state: beamformer, DOA estimator, VAD, gaze
        saccade-hold state, and the GEVD RTF estimator (if enabled).
        Call between independent clips when reusing the same pipeline
        instance.
        """
        self.beamformer.reset()
        if self.atf_steering is not None:
            self.beamformer.init_diffuse(self.atf_steering.H_rel)
        else:
            self.beamformer.init_isotropic()

        if self.doa is not None:
            self.doa.reset()

        self.vad.reset()

        if self.gevd is not None:
            self.gevd.reset()

        if self._gaze_stabilizer is not None:
            self._gaze_stabilizer.reset()

        self._sacc_prev_theta = None
        self._sacc_hold_cnt   = 0
        self._sacc_held_theta = 0.0
        self._sacc_held_phi   = 0.0
        self._saccade_events  = 0

        self._last_theta = 0.0
        self._last_phi   = 0.0

        # -- v8.0 (runtime_demo): gate + observer per-clip state ------------
        self._gate_open        = False
        self._gate_frames      = 0
        self._gate_transitions = 0
        self._stream_frame_idx = 0
