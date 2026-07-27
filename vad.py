"""
vad.py — Simple energy-based Voice Activity Detector
=====================================================
Used to decide whether a frame is noise-only (update R_nn) or
speech-present (apply beamformer weights).

Two conditions (OR gate):
  (A) Broadband energy threshold — detects loud speech frames.
  (B) Mid-band spectral ratio    — detects speech-shaped frames even when
                                   broadband SNR is negative (white/pink noise).

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
                 stuck_speech_max_frames: int = 150):
        self.thr_db       = threshold_db
        self.adapt_rate   = adapt_rate
        self.use_spectral = use_spectral
        self.spectral_thr = spectral_thr_db
        self.stuck_max     = int(stuck_speech_max_frames)
        self._stuck_adapt_rate = 0.01   # deliberately much slower than the
                                        # normal confirmed-silence adapt_rate
        self._consec_no_silence = 0
        self._n_stuck_nudges    = 0     # diagnostics

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

    # ── Public API ────────────────────────────────────────────────────────────

    def is_speech(self, X: np.ndarray) -> bool:
        """
        Decide whether the current STFT frame contains speech.

        Applies a two-condition OR gate:
          (A) Broadband energy > noise_floor + threshold_db
          (B) Mid-band spectral ratio > ratio_baseline + spectral_thr_db
              (only evaluated when use_spectral=True)

        Hangover is applied after either condition fires.

        Parameters
        ----------
        X : (N, B) or (B,) complex STFT frame

        Returns
        -------
        bool : True if speech detected
        """
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

    def diagnostics(self) -> dict:
        """Frame counts useful for spotting a stuck-VAD run post-hoc."""
        return {
            'stuck_nudges':        self._n_stuck_nudges,
            'consec_no_silence':   self._consec_no_silence,
        }