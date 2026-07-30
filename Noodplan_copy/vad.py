"""
vad.py — Simple energy-based Voice Activity Detector
=====================================================
Used to decide whether a frame is noise-only (update R_nn) or
speech-present (apply beamformer weights).

Three conditions:
  (A) Broadband energy threshold — detects loud speech frames.
  (B) Mid-band spectral ratio    — detects speech-shaped frames even when
                                   broadband SNR is negative (white/pink noise).
  (C) Direction-aware coherence gate — an *asymmetric downgrade* applied on
                                   top of (A)/(B). See "Direction-aware gate"
                                   below.

(A) and (B) are combined with OR, exactly as before. (C) never participates
in that OR — it can only veto a positive (A)/(B) verdict, never produce one.

Direction-aware gate (condition C)
-----------------------------------
(A) and (B) only look at *how much* energy is present and *how it's shaped*
spectrally — neither knows *where* the energy came from. That is fine when
the only thing competing with the target is stationary noise, but it breaks
down when the interferer is also speech: a directional talker 55° away is
loud and speech-shaped, so both (A) and (B) fire on it, and those frames get
treated as "speech" — the beamformer holds its weights instead of updating
R_nn, so the interferer's covariance is never learned and MVDR can't null it.

Condition (C) asks a third question of any frame where (A) or (B) already
fired: does the energy's *spatial pattern* match the target's steering
vector? It computes the magnitude of the complex cosine similarity between
the observed N-mic frame and the candidate target steering vector, averaged
over the same mid-band (500–4000 Hz) used by (B) — the band where a 6-mic
array has enough spatial resolution to tell a 55°-separated interferer from
the target. A frame whose energy lines up with the target direction scores
close to 1; a frame dominated by a differently-directed interferer scores
much lower.

This is applied *only* as a downgrade: if (A)/(B) said "speech" but the
coherence score is below the current adaptive threshold, the frame is
reclassified as noise (and so becomes eligible for the R_nn update /
noise-floor adaptation that a confirmed-silence frame would normally get).
It can never do the reverse — it cannot turn an (A)/(B) "not speech"
verdict into "speech" — and it never touches the hangover extension (a
frame held "speech" purely by hangover is left alone). This asymmetry is
deliberate and load-bearing: a false positive here (an interferer frame
wrongly kept as "speech") just means one missed opportunity to sharpen
R_nn — recoverable next frame. A false negative (real target speech
wrongly downgraded to "noise") teaches MVDR to null the target itself,
which is far more damaging and much harder to recover from. So the gate
is built to only ever err on the side of under-claiming noise frames,
never over-claiming them.

Why the threshold is adaptive, not a fixed constant
------------------------------------------------------
An earlier version of this gate used a fixed absolute coherence cutoff
(e.g. "downgrade below 0.45"), tuned against an anechoic synthetic test
where genuine on-axis target frames score close to 1.0. On a real
reverberant clip (RT60 ~150 ms) that assumption breaks: even frames
measured against the *true* target steering vector average only ~0.44
coherence, because early reflections decorrelate the array response
relative to the analytic free-field (pure plane-wave) model the steering
vector assumes. A fixed 0.45 cutoff in that room fired on roughly half of
*all* frames regardless of which talker was dominant — collapsing the
VAD's speech rate from ~98% to ~29% on a clip where the target actually
speaks ~90% of the time. A fixed threshold that has to be retuned per
room/array/distance is not a workable gate.

Instead the gate tracks its own ceiling: `_coh_ceiling` is a max-decay
estimate of "the best coherence this room/array has recently produced"
(jumps up immediately to a new high score, decays slowly otherwise via
`direction_ceil_decay`), and the effective threshold is
`direction_rel_thr * _coh_ceiling` (floored at `direction_abs_floor` so a
ceiling that has decayed to near zero, e.g. after a long interferer-only
stretch, can't make the gate permanently inert). This makes the gate
self-calibrate to whatever coherence ceiling the room/array/distance
actually allows, rather than assuming the anechoic ideal.

The candidate steering vector is supplied by the caller (pipeline_2.py
passes the last-known target direction — from gaze or the last committed
SRP estimate — computed *before* this frame's speech/noise decision, so
condition C never uses information derived from the very decision it is
gating). When no steering vector is supplied (`steering_vector=None`, e.g.
single-channel use, or callers that haven't been updated), condition (C) is
a no-op and behaviour is identical to the (A) OR (B) gate described below.

Why condition (B) is needed
----------------------------
At −20 dB broadband SNR, speech raises total frame power by only 0.04 dB —
far below any practical threshold_db.  Condition (A) never fires and the
beamformer defaults permanently to mic-0 passthrough.

Speech concentrates energy in the 500–4000 Hz band.  White noise has a flat
spectrum.  The ratio of mid-band power to full-band power (spectral_ratio) is
consistently higher during speech than during white noise, even at −20 dB
broadband SNR.  Condition (B) detects this spectral lift and correctly labels
speech frames, allowing the beamformer to compute MVDR weights.

Caveats
-------
• The spectral ratio is less discriminative in babble (many speech-like
  interferers raise the ratio even during silence), adding a small number of
  false positives.  Their impact is limited because false-positive speech
  frames cause a weight recompute, not an R_nn update toward the wrong
  direction.
• Mid-band bin range is computed from n_fft and fs; pass both correctly.
  Default n_fft=512 at fs=48000 → 93.75 Hz/bin, 500 Hz ≈ bin 5,
  4000 Hz ≈ bin 43.

RT60 hangover
-------------
If rt60_s > 0, the hangover is extended to 3×RT60 so reverb tails of the
target source are not labelled as noise frames (which would teach R_nn the
target direction and cause MVDR self-nulling).

IMPORTANT: pass rt60_s=0.0 for the energy_vad evaluation mode.  That mode
uses the internal VAD to drive noise-floor adaptation.  A long hangover
(84 frames at RT60=150 ms) bridges every silence gap, leaving almost no
confirmed noise frames for the noise-floor estimate and causing 99.8%
speech classification.
"""

import numpy as np


class EnergyVAD:
    """
    Adaptive energy-based VAD with optional sub-band spectral feature.

    The broadband EMA maintains a running noise-floor estimate updated only
    during confirmed silence.  A frame is declared speech if either of two
    conditions is met:

      (A) Broadband energy exceeds the noise floor by threshold_db dB.
      (B) [optional] The mid-band (500–4000 Hz) energy ratio exceeds its
          noise-mode baseline by spectral_thr_db dB.

    Condition (B) is enabled by default (use_spectral=True) because it is
    the only way to detect speech reliably when broadband SNR is negative
    (white noise at −20 to −10 dB).

    Parameters
    ----------
    threshold_db    : broadband detection threshold above noise floor (dB).
                      Default 10.0 — the pipeline sets this to 3.0 via
                      vad_thr_db.
    hangover        : minimum frames to remain in speech mode after energy
                      drops below threshold (avoids clipping word endings).
    adapt_rate      : EMA coefficient for noise floor updates.  Only applied
                      during confirmed silence so speech frames do not corrupt
                      the noise estimate.
    rt60_s          : room RT60 in seconds.  When > 0, the hangover is
                      extended to 3×RT60 seconds.  Pass 0.0 for energy_vad
                      evaluation mode so silence gaps are not bridged.
    fs              : sample rate in Hz.
    hop             : STFT hop size in samples.
    use_spectral    : enable sub-band spectral ratio condition (B).
    spectral_thr_db : condition (B) detection threshold above the spectral
                      ratio noise baseline (dB).
    n_fft           : FFT window size.  Used to compute mid-band bin indices.
                      Must match the STFT configuration used in the pipeline.
    stuck_speech_max_frames : safety valve (see "Stuck-in-speech" below).
                      Set to 0 to disable.
    use_direction   : enable condition (C), the direction-aware coherence
                      downgrade described in the module docstring. Default
                      True. Has no effect on any call where
                      `is_speech(X, steering_vector=None)` is used (no
                      candidate direction supplied) — those calls behave
                      exactly as the (A) OR (B) gate did before condition
                      (C) existed.
    direction_thr   : *relative* multiplier in [0, 1) against the tracked
                      coherence ceiling (see _coh_ceiling / direction_ceil_decay
                      / direction_abs_floor below) -- the effective cutoff is
                      direction_thr * _coh_ceiling, floored at
                      direction_abs_floor. Default 0.85, empirically retuned:
                      on a representative reverberant clip (55° separation,
                      0 dB SNR, RT60~150 ms) the smoothed target-speaking vs.
                      non-target coherence distributions overlap heavily, and
                      sweeping this value shows a flat "2.8% of true
                      interferer-only frames caught, 0 real target frames
                      wrongly downgraded" plateau from ~0.80-0.88, with false
                      positives appearing sharply above ~0.90. 0.85 sits in
                      the middle of that safe plateau. Lower = more
                      conservative (harder to downgrade, fewer interferer
                      frames reclaimed but lower risk of ever touching real
                      target frames); higher = more aggressive, but past the
                      plateau the false-positive rate rises much faster than
                      the catch rate -- this is not a smooth trade-off, so do
                      not tune it past the plateau without re-checking on
                      real data.
    direction_smooth : EMA coefficient (0 disables) that smooths the
                      per-frame coherence score before thresholding.
                      Default 0.05 (~20-frame / ~100 ms time constant).
                      This needs to be much slower than it might first
                      look: when the interferer is also speech and near-
                      continuously active (the case this condition exists
                      for), most "speech" frames are actually target+
                      interferer *mixtures*, not cleanly one or the other,
                      so a fast/no-smoothing per-frame coherence estimate
                      swings with whichever talker happens to be louder
                      in that instant and can spuriously dip below
                      direction_thr on a genuinely target-dominant frame.
                      Empirically (synthetic 2-talker A/B test, 55 deg
                      separation, comparable levels, ~90%+ mutual overlap,
                      8 seeds), smoothing rates of 0.15-0.3 fixed most
                      seeds but still produced an occasional net loss vs.
                      both raw-mic and SRP baselines from over-eager
                      single-frame downgrades; 0.05 removed that failure
                      mode across all 8 seeds tested while leaving the
                      gate free to downgrade sustained interferer-only
                      stretches (which is what actually matters for R_nn).
                      Smoothing is applied to the score only, never used
                      to carry a decision across frames.
    direction_warmup_frames : number of frames to leave condition (C)
                      inactive at the very start of a clip / after reset(),
                      before `self._last_theta`-derived candidate steering
                      vectors have had a chance to settle on a real
                      direction. Default 5.
    direction_ceil_decay : per-frame decay applied to `_coh_ceiling` when the
                      current coherence score does not set a new high.
                      Default 0.999 (slow — the ceiling should track "the
                      best this room/array has recently done", not chase
                      every dip).
    direction_abs_floor : minimum effective threshold (`direction_thr *
                      _coh_ceiling` is floored at this value), so a ceiling
                      that has decayed to near zero after a long
                      interferer-only stretch can't make the gate
                      permanently inert. Default 0.05.

    Stuck-in-speech safety valve
    -----------------------------
    Both conditions only adapt their baselines during *confirmed* silence.
    If a chronic run of false positives occurs — e.g. reverberant coloration
    keeps nudging the mid-band ratio (condition B) above threshold even
    though no target speech is present — silence is never confirmed, the
    baselines set near clip-start never move again, and the VAD can get
    permanently stuck (observed as near-100% "speech" classification and
    collapsed SI-SDR on stationary-noise clips under reverb). After
    `stuck_speech_max_frames` consecutive frames with no confirmed silence,
    a very slow EMA nudge (rate 0.01) pulls both baselines toward the
    current frame so the state can't freeze indefinitely. This mirrors the
    escape-hatch pattern beamformer_2.py already uses for its noise-frame
    purity gate (OUTLIER_MAX_CONSEC_REJECTS) — a rare, capped correction,
    not a change to normal operation.
    """

    _MID_LO_HZ = 500.0
    _MID_HI_HZ = 4000.0

    def __init__(self,
                 threshold_db:    float = 10.0,
                 hangover:        int   = 8,
                 adapt_rate:      float = 0.05,
                 rt60_s:          float = 0.0,
                 fs:              int   = 16000,
                 hop:             int   = 256,
                 use_spectral:    bool  = True,
                 spectral_thr_db: float = 1.5,
                 n_fft:           int   = 512,
                 stuck_speech_max_frames: int = 150,
                 use_direction:   bool  = True,
                 direction_thr:   float = 0.85,
                 direction_smooth: float = 0.05,
                 direction_warmup_frames: int = 5,
                 direction_ceil_decay: float = 0.999,
                 direction_abs_floor: float = 0.05):
        self.thr_db       = threshold_db
        self.adapt_rate   = adapt_rate
        self.use_spectral = use_spectral
        self.spectral_thr = spectral_thr_db
        self.stuck_max     = int(stuck_speech_max_frames)
        self._stuck_adapt_rate = 0.01   # deliberately much slower than the
                                        # normal confirmed-silence adapt_rate
        self._consec_no_silence = 0
        self._n_stuck_nudges    = 0     # diagnostics

        # ── Direction-aware gate (condition C) ──────────────────────────────
        self.use_direction  = bool(use_direction)
        # NOTE: direction_thr is a *relative* multiplier against the tracked
        # coherence ceiling (see _coh_ceiling below), not an absolute cutoff.
        # This is what makes the gate self-calibrating (module docstring,
        # "Why the threshold is adaptive, not a fixed constant").
        self.direction_thr  = float(direction_thr)
        self.direction_smooth = float(direction_smooth)
        self.direction_warmup = int(direction_warmup_frames)
        self.direction_ceil_decay = float(direction_ceil_decay)
        self.direction_abs_floor  = float(direction_abs_floor)
        self._coh_ema        = None   # smoothed coherence score, reset per clip
        self._coh_ceiling     = None  # max-decay estimate of the best
                                      # coherence recently seen; reset per clip
        self._n_frames_seen  = 0      # for warm-up gating
        self._n_dir_downgrades = 0    # diagnostics: (A)/(B) speech -> noise

        # ── RT60-aware hangover ─────────────────────────────────────────────
        if rt60_s > 0.0:
            reverb_frames      = round(3.0 * rt60_s * fs / hop)
            effective_hangover = max(hangover, reverb_frames)
            if effective_hangover > hangover:
                print(
                    f"[EnergyVAD] RT60={rt60_s*1e3:.0f} ms → reverb hangover "
                    f"extended from {hangover} to {effective_hangover} frames "
                    f"({3*rt60_s*1e3:.0f} ms @ {fs/hop:.1f} fps)"
                )
        else:
            effective_hangover = hangover
        self.hangover = effective_hangover

        # ── Mid-band bin range for condition (B) ────────────────────────────
        n_bins       = n_fft // 2 + 1
        bin_hz       = fs / n_fft
        self._mid_lo = max(1, int(self._MID_LO_HZ / bin_hz))
        self._mid_hi = min(n_bins - 1, int(self._MID_HI_HZ / bin_hz))
        if self._mid_lo >= self._mid_hi:
            self._mid_lo = 1
            self._mid_hi = n_bins // 2

        # ── Running state ───────────────────────────────────────────────────
        self._noise_power  = None
        self._noise_ratio  = None
        self._hangover_cnt = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _spectral_ratio(self, X: np.ndarray) -> float:
        """
        Ratio of mid-band (500–4000 Hz) power to full-band power, averaged
        over all microphone channels.  Returns a value in (0, 1].
        """
        if X.ndim == 2:
            ps = np.mean(np.abs(X) ** 2, axis=0)
        else:
            ps = np.abs(X) ** 2
        full = float(np.mean(ps)) + 1e-12
        mid  = float(np.mean(ps[self._mid_lo:self._mid_hi])) + 1e-12
        return mid / full

    def _direction_coherence(self, X: np.ndarray, d: np.ndarray) -> float:
        """
        Magnitude of the complex cosine similarity between the observed
        N-mic frame and the candidate steering vector, averaged over the
        mid-band (500-4000 Hz) bins used by condition (B).

        X : (N, B) complex STFT frame
        d : (N, B) complex steering vector, |d[n, f]| == 1 for all n, f

        Returns a value in [0, 1]. For a single frequency bin f, cosine
        similarity is |d_f^H x_f| / (||d_f|| ||x_f||) — by Cauchy-Schwarz
        this is 1 iff x_f is a scalar multiple of d_f (energy arriving
        exactly along the candidate direction) and falls off as the
        spatial pattern of the energy diverges from that direction (e.g.
        an interferer arriving from a different angle).
        """
        Xm = X[:, self._mid_lo:self._mid_hi]   # (N, K)
        dm = d[:, self._mid_lo:self._mid_hi]   # (N, K)

        num  = np.abs(np.sum(np.conj(dm) * Xm, axis=0))          # (K,)
        d_nrm = np.sqrt(np.sum(np.abs(dm) ** 2, axis=0)) + 1e-12  # (K,)
        x_nrm = np.sqrt(np.sum(np.abs(Xm) ** 2, axis=0)) + 1e-12  # (K,)

        cos_sim = num / (d_nrm * x_nrm)
        return float(np.clip(np.mean(cos_sim), 0.0, 1.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def is_speech(self, X: np.ndarray,
                  steering_vector: np.ndarray | None = None) -> bool:
        """
        Decide whether the current STFT frame contains speech.

        Applies a two-condition OR gate, then an optional asymmetric
        direction-aware downgrade:
          (A) Broadband energy > noise_floor + threshold_db
          (B) Mid-band spectral ratio > ratio_baseline + spectral_thr_db
              (only evaluated when use_spectral=True)
          (C) If (A) or (B) fired and a steering_vector was supplied: downgrade
              back to "not speech" if the frame's spatial coherence with the
              candidate direction is below direction_thr (only evaluated when
              use_direction=True). Never fires on its own, never upgrades a
              (A)/(B) "not speech" verdict, never overrides hangover.

        Hangover is applied after (A)/(B)/(C) are resolved.

        Parameters
        ----------
        X : (N, B) or (B,) complex STFT frame
        steering_vector : optional (N, B) complex candidate steering vector
              for the target direction (e.g. last-known gaze/SRP direction),
              |steering_vector[n, f]| == 1. Required (together with X being
              multi-channel) for condition (C) to be evaluated; ignored
              otherwise.

        Returns
        -------
        bool : True if speech detected
        """
        self._n_frames_seen += 1
        power = float(np.mean(np.abs(X) ** 2))
        power = max(power, 1e-12)

        # Initialise from very first frame
        if self._noise_power is None:
            self._noise_power = power
            if self.use_spectral:
                self._noise_ratio = self._spectral_ratio(X)
            return False

        # ── Condition A: broadband energy ──────────────────────────────────
        snr_db   = 10.0 * np.log10(power / self._noise_power)
        speech_A = snr_db > self.thr_db

        # ── Condition B: mid-band spectral ratio ───────────────────────────
        speech_B = False
        if self.use_spectral and self._noise_ratio is not None:
            ratio    = self._spectral_ratio(X)
            ratio_db = 10.0 * np.log10(ratio / max(self._noise_ratio, 1e-12))
            speech_B = ratio_db > self.spectral_thr

        speech = speech_A or speech_B

        # ── Condition C: direction-aware downgrade ─────────────────────────
        # Only ever turns a positive (A)/(B) verdict OFF. Never fires when
        # speech is already False, never turns False into True. See module
        # docstring for the asymmetry rationale.
        if (speech and self.use_direction and steering_vector is not None
                and X.ndim == 2 and self._n_frames_seen > self.direction_warmup):
            coh = self._direction_coherence(X, steering_vector)
            if self.direction_smooth > 0.0:
                self._coh_ema = (coh if self._coh_ema is None else
                                 (1 - self.direction_smooth) * self._coh_ema
                                 + self.direction_smooth * coh)
                coh_eval = self._coh_ema
            else:
                coh_eval = coh

            # ── Adaptive ceiling ────────────────────────────────────────────
            # Track "the best coherence this room/array has recently
            # produced": jump up immediately on a new high, else decay
            # slowly. The gate threshold is direction_thr * ceiling, floored
            # at direction_abs_floor. This is what lets the gate work in a
            # reverberant room where even genuine on-axis target frames only
            # average ~0.44 coherence (a fixed 0.45 cutoff would downgrade
            # ~half of all real target frames — see module docstring).
            if self._coh_ceiling is None or coh_eval > self._coh_ceiling:
                self._coh_ceiling = coh_eval
            else:
                self._coh_ceiling *= self.direction_ceil_decay

            eff_thr = max(self.direction_thr * self._coh_ceiling,
                         self.direction_abs_floor)

            if coh_eval < eff_thr:
                speech = False
                self._n_dir_downgrades += 1

        # ── Hangover logic ─────────────────────────────────────────────────
        if speech:
            self._hangover_cnt = self.hangover
        else:
            if self._hangover_cnt > 0:
                self._hangover_cnt -= 1
                speech = True
            else:
                # Confirmed silence — update noise floor estimates
                self._noise_power = ((1 - self.adapt_rate) * self._noise_power
                                     + self.adapt_rate * power)
                if self.use_spectral:
                    ratio       = self._spectral_ratio(X)
                    ratio_adapt = self.adapt_rate * 0.5
                    self._noise_ratio = ((1 - ratio_adapt) * self._noise_ratio
                                         + ratio_adapt * ratio)

        # ── Stuck-in-speech safety valve ─────────────────────────────────────
        # Track consecutive frames with no *confirmed* silence (i.e. every
        # frame reaching this point with speech=True, whether from condition
        # A/B or from hangover). If confirmed silence never arrives for a
        # long stretch, the baselines above are frozen at whatever they were
        # near clip-start. See class/module docstring for why this matters
        # for reverberant stationary noise.
        if speech:
            self._consec_no_silence += 1
        else:
            self._consec_no_silence = 0

        if self.stuck_max > 0 and self._consec_no_silence >= self.stuck_max:
            self._noise_power = ((1 - self._stuck_adapt_rate) * self._noise_power
                                 + self._stuck_adapt_rate * power)
            if self.use_spectral:
                ratio = self._spectral_ratio(X)
                self._noise_ratio = ((1 - self._stuck_adapt_rate) * self._noise_ratio
                                     + self._stuck_adapt_rate * ratio)
            self._consec_no_silence = 0   # restart the countdown
            self._n_stuck_nudges   += 1

        return speech

    def reset(self) -> None:
        """Reset all running state (call between recordings)."""
        self._noise_power  = None
        self._noise_ratio  = None
        self._hangover_cnt = 0
        self._consec_no_silence = 0
        self._n_stuck_nudges    = 0
        self._coh_ema           = None
        self._coh_ceiling       = None
        self._n_frames_seen     = 0
        self._n_dir_downgrades  = 0

    def diagnostics(self) -> dict:
        """Frame counts useful for spotting a stuck-VAD run post-hoc."""
        return {
            'stuck_nudges':        self._n_stuck_nudges,
            'consec_no_silence':   self._consec_no_silence,
            'direction_downgrades': self._n_dir_downgrades,
            'coh_ceiling_last':    self._coh_ceiling,
        }