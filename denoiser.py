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
        except ImportError as e:
            # deepfilternet 0.5.6 uses torchaudio APIs that were removed in
            # torchaudio 2.1 (`torchaudio.backend.common.AudioMetaData`,
            # `torchaudio.info`). With a modern torchaudio installed, the
            # import fails here — but the message names torchaudio, not
            # deepfilternet, so distinguish the two cases rather than always
            # claiming DeepFilterNet is missing.
            msg = str(e)
            if "torchaudio" in msg:
                raise ImportError(
                    f"DeepFilterNet is installed but incompatible with the "
                    f"installed torchaudio ({msg}).\n"
                    f"\n"
                    f"deepfilternet 0.5.6 needs torchaudio <= 2.0.2, which "
                    f"pins torch to 2.0.1 (max GPU arch sm_86). On newer GPUs "
                    f"(H100 = sm_90) that torch cannot run at all, so this "
                    f"project keeps modern torch and leaves Stage 4 "
                    f"unavailable.\n"
                    f"\n"
                    f"Options:\n"
                    f"  • Run with --no-denoise to skip Stage 4 (Stages 1-3, "
                    f"the beamformer, are unaffected).\n"
                    f"  • For Stage 4, build a separate CPU/sm_86 environment "
                    f"with torch==2.0.1 and torchaudio==2.0.2."
                ) from e
            raise ImportError(
                "DeepFilterNet is not installed.\n"
                "Install it with:  pip install deepfilternet"
            ) from e

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
                          sr: int = 48_000) -> np.ndarray:
        """
        Denoise starting from a beamformed STFT spectrum.

        Performs iSTFT → DeepFilterNet and returns the enhanced time-domain
        signal (this is what pipeline_2.AriaDenoisingPipeline.process()
        expects).

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
