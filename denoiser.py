"""
denoiser.py — DeepFilterNet wrapper (Stage 4)
==============================================
Wraps the open-source DeepFilterNet model to denoise the single-channel
beamformer output.

Install with:
    pip install deepfilternet

DeepFilterNet expects:
  - mono float32 audio at 48 kHz
  - input as a torch.Tensor of shape (1, T) or (T,)

The beamformer output is in the STFT domain; we first reconstruct the
time-domain signal with iSTFT, then pass it through DeepFilterNet,
and optionally return the enhanced STFT as well.

Reference:
    Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement
    Framework for Full-Band Audio based on Deep Filtering",
    ICASSP 2022.  https://github.com/Rikorose/DeepFilterNet
"""

from __future__ import annotations
import numpy as np


class DeepFilterNetDenoiser:
    """
    Lazy-loading wrapper around DeepFilterNet.

    The model is downloaded and loaded on the first call to `enhance()`.
    Subsequent calls reuse the loaded model.

    Parameters
    ----------
    post_filter : use DeepFilterNet's post-filter (DF) module in addition
                  to the ERB-domain filter (recommended, default True)
    """

    def __init__(self, post_filter: bool = True):
        self.post_filter = post_filter
        self._model      = None
        self._df_state   = None

    def _load(self) -> None:
        """Download / load model on first use."""
        try:
            from df.enhance import enhance, init_df
        except ImportError:
            raise ImportError(
                "DeepFilterNet is not installed.\n"
                "Install it with:  pip install deepfilternet"
            )

        model, df_state, _ = init_df()
        self._model    = model
        self._df_state = df_state
        print("[DeepFilterNet] model loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(self, audio: np.ndarray, sr: int = 48_000) -> np.ndarray:
        """
        Denoise a mono time-domain signal.

        Parameters
        ----------
        audio : (T,) float32 mono signal at 48 kHz
        sr    : sample rate (must be 48000 for DeepFilterNet)

        Returns
        -------
        enhanced : (T,) float32 denoised signal
        """
        if sr != 48_000:
            raise ValueError(
                f"DeepFilterNet requires 48 kHz input; got {sr} Hz."
            )

        if self._model is None:
            self._load()

        try:
            import torch
            from df.enhance import enhance
        except ImportError:
            raise ImportError("Install deepfilternet: pip install deepfilternet")

        # Ensure float32 and add batch dimension → (1, T)
        x_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            enhanced_t = enhance(
                self._model,
                self._df_state,
                x_t,
                pad=True,
            )

        return enhanced_t.squeeze(0).numpy()

    def enhance_from_stft(self,
                          Y: np.ndarray,
                          istft_fn,
                          length: int | None = None,
                          sr: int = 48_000) -> tuple[np.ndarray, np.ndarray]:
        """
        Denoise starting from a beamformed STFT spectrum.

        Performs iSTFT → DeepFilterNet → returns both time-domain and
        STFT of the enhanced signal.

        Parameters
        ----------
        Y        : (B, n_frames) complex beamformed spectrum
        istft_fn : callable matching signature of stft.istft()
        length   : original signal length (for trimming)
        sr       : sample rate

        Returns
        -------
        enhanced_audio : (T,) float32 enhanced signal
        """
        # iSTFT: STFT domain → time domain
        y_time = istft_fn(Y, length=length)

        # DeepFilterNet denoising
        enhanced = self.enhance(y_time, sr=sr)
        return enhanced
