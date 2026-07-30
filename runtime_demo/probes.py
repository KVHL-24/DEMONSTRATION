"""Visualization probes — turn observer internals into drawable data.

Everything here is DISPLAY-ONLY compute: it runs outside the pipeline's
timed sections, so the stage-time / duty numbers shown in the UI never
include it (otherwise those numbers would lie).
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doa_2 import FreeFieldSteering                     # noqa: E402
from configs import mic_positions                        # noqa: E402

FS     = 48000
N_FFT  = 512
N_AZ   = 72                       # 5° azimuth grid
# Speech-critical subband for the displayed pattern, bounded by the
# array's physics: min mic spacing 4.9 cm → spatial Nyquist ≈ 3.5 kHz
# (grating lobes above), aperture 15 cm → angular resolution collapses
# below ≈ 1.1 kHz. 800–3000 Hz stays inside both limits; averaging a
# wider band was measured to wash the null out entirely.
BAND_HZ   = (800.0, 3000.0)
N_BAND_BINS = 12


def _band_bins() -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    lo = int(np.searchsorted(freqs, BAND_HZ[0]))
    hi = int(np.searchsorted(freqs, BAND_HZ[1]))
    return np.unique(np.linspace(lo, hi - 1, N_BAND_BINS).astype(int))


class BeamPatternProbe:
    """|wᴴ d(θ)|² over a 5° azimuth grid, averaged over a speech subband.

    The scan steering vectors are precomputed once per mic-count from the
    same FreeFieldSteering model the pipeline itself uses, so the pattern
    is exactly the array response the beamformer believes it has.
    """

    def __init__(self, n_mics: int) -> None:
        self.n_mics = n_mics
        self.bins = _band_bins()                          # (Bs,)
        self.az_deg = np.linspace(-180.0, 180.0, N_AZ, endpoint=False)
        steerer = FreeFieldSteering(mic_positions(n_mics), n_fft=N_FFT)
        # D[a, n, b] — scan steering vector per azimuth, band bins only
        D = np.stack([
            steerer.steering_vector(np.deg2rad(az), 0.0)[:, self.bins]
            for az in self.az_deg
        ])                                                # (A, N, Bs)
        self._D = D.astype(np.complex64)

    def pattern_db(self, w: np.ndarray,
                   floor_db: float = -30.0) -> list[float] | None:
        """w: (B, N) complex64 MVDR weights → 72 dB values in [floor, 0]."""
        if w is None:
            return None
        w_sel = w[self.bins, :]                           # (Bs, N)
        # Convention check (matches beamformer_2.compute_weights): the
        # distortionless constraint there is aᴴw = 1 with a = d*, i.e.
        # dᵀw = 1 — the array response to the steered direction is
        # Σ_n w_n·d_n, NOT wᴴd. Using wᴴd here (first attempt) scrambles
        # the phases and washes the pattern into mush.
        resp = np.einsum('bn,anb->ab', w_sel, self._D)
        p = np.mean(np.abs(resp) ** 2, axis=1)            # (A,)
        peak = float(p.max())
        if peak <= 0.0 or not np.isfinite(peak):
            return None
        db = 10.0 * np.log10(np.maximum(p / peak, 10 ** (floor_db / 10)))
        return np.round(db, 1).tolist()


def spec_column(X_frame: np.ndarray, n_out: int = 64,
                floor_db: float = -70.0) -> list[float]:
    """One spectrogram column: mic-0 magnitude in dB, log-ish bin pooling
    down to n_out values (display only)."""
    mag = np.abs(X_frame[0]) ** 2                         # (B,)
    b = mag.shape[0]
    edges = np.unique(np.geomspace(1, b - 1, n_out + 1).astype(int))
    pooled = np.maximum.reduceat(mag, edges[:-1])
    db = 10.0 * np.log10(pooled + 10 ** (floor_db / 10))
    return np.round(db, 1).tolist()
