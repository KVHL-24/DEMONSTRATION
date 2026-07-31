"""
stft.py — Short-Time Fourier Transform utilities  (v2 — torch-accelerated)
===========================================================================
Matches slide 1 exactly:
  Window F = 512 samples, 50% overlap (hop = 256),
  R = 48000 / 256 = 187.5 frames/s, B = 257 bins.

v2 changes
----------
The public API (`stft`, `istft`, `stft_multichannel`, `make_window`) is
UNCHANGED — same signatures, same shapes, same dtypes (complex64 / float32
numpy arrays) — so every other module (beamformer_2.py, doa_2.py,
pipeline_2.py, eval_synthetic_2.py) keeps working without modification.

What changed under the hood is *how* the transform is computed:

  • The old implementation looped over frames one at a time in Python
    (`for k in range(n_frames): ...np.fft.rfft(...)`). That loop is pure
    Python overhead — the FFT itself is cheap, but doing 10,000+ of them
    one-by-one in an interpreted loop is not.
  • The framing step (STFT) and the overlap-add step (ISTFT) are both
    "embarrassingly parallel" across frames and — for stft_multichannel —
    across microphone channels too: every frame's FFT is independent of
    every other frame. That makes them a perfect fit for batched GPU
    execution, unlike the beamformer's frame-by-frame Woodbury update
    (beamformer_2.py) or the per-frame DOA/VAD logic in doa_2.py /
    pipeline_2.py, which are inherently *sequential* (each frame's state
    depends on the previous frame's R_nn⁻¹ / EMA / hold-counter) and would
    gain little from batching — attempting to vectorise those would mean
    either giving up the causal, online-style adaptation entirely or
    reimplementing it as an explicit recurrence in a GPU kernel, which is
    a much bigger and riskier change for uncertain benefit. So *only* the
    STFT / ISTFT transforms are moved to torch here.
  • Framing is done with `Tensor.unfold` (creates a strided view over all
    frames at once, no copy) and the forward transform is a single batched
    `torch.fft.rfft` call over *all* frames (and all mics, for
    stft_multichannel) simultaneously. The inverse overlap-add is done
    with `torch.nn.functional.fold`, which performs the scatter-add over
    all frames in one vectorised call instead of a Python loop of
    in-place slice additions.
  • Everything runs on whatever device `get_device()` resolves to: CUDA if
    available, then Apple-Silicon MPS, else CPU. Even on CPU, removing the
    Python per-frame loop is a meaningful speedup because torch's batched
    FFT still avoids per-call Python/dispatch overhead; on a CUDA GPU the
    speedup for a full 60 s / 7-mic clip is typically an order of
    magnitude or more.
  • torch is a soft dependency: if it isn't installed, both functions fall
    back automatically to the original pure-NumPy per-frame loop
    (`_stft_numpy` / `_istft_numpy` below) so nothing breaks in an
    environment without torch.

Usage note
----------
If you want to force a specific device (e.g. keep everything on CPU even
though a GPU is present, or target an Apple-Silicon MPS device explicitly),
call `set_device("cpu")` / `set_device("mps")` once at start-up, before any
STFT calls. `get_device()` shows what is currently active.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    _HAS_TORCH = False


# ── Default parameters (match slides) ────────────────────────────────────────
FS      = 48_000
F_WIN   = 512          # window length
HOP     = F_WIN // 2   # 256  (50% overlap)
B       = F_WIN // 2 + 1  # 257 frequency bins


# ── Device management ──────────────────────────────────────────────────────

_DEVICE = None   # resolved lazily on first use, cached thereafter


def _auto_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> "torch.device":
    """
    Return the torch device STFT/ISTFT run on (CUDA > MPS > CPU by default).
    Resolved once and cached; call set_device() to override.
    """
    global _DEVICE
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed — get_device() unavailable.")
    if _DEVICE is None:
        _DEVICE = _auto_device()
        print(f"[stft] torch device: {_DEVICE}")
    return _DEVICE


def set_device(device) -> None:
    """
    Force a specific torch device for all subsequent STFT/ISTFT calls, e.g.
    set_device("cpu"), set_device("cuda:1"), set_device("mps").
    """
    global _DEVICE
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed — set_device() unavailable.")
    _DEVICE = torch.device(device)
    print(f"[stft] torch device forced to: {_DEVICE}")


# ── Public helpers ───────────────────────────────────────────────────────────

def make_window(n_fft: int = F_WIN) -> np.ndarray:
    """Return a normalised Hann window."""
    return np.hanning(n_fft).astype(np.float32)


# ── Torch core (vectorised over frames AND channels) ────────────────────────

def _to_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """(T,) → (1,T) with a flag saying whether we should squeeze back."""
    if x.ndim == 1:
        return x[None, :], True
    return x, False


def _stft_torch(x2d: np.ndarray, window: np.ndarray,
                n_fft: int, hop: int) -> np.ndarray:
    """
    x2d : (C, T) real  →  returns (C, n_bins, n_frames) complex64 numpy.
    Batched over channels AND frames in one FFT call.
    """
    dev = get_device()
    xt  = torch.as_tensor(x2d, dtype=torch.float32, device=dev)      # (C, T)
    win = torch.as_tensor(window, dtype=torch.float32, device=dev)   # (n_fft,)

    T        = xt.shape[-1]
    n_frames = 1 + (T - n_fft) // hop
    if n_frames < 1:
        raise ValueError(f"Signal too short for n_fft={n_fft}: T={T}")

    # (C, n_frames_all, n_fft) strided view, no copy; keep only n_frames
    frames = xt.unfold(-1, n_fft, hop)[:, :n_frames, :]              # (C,F,n_fft)
    frames = frames * win                                            # broadcast
    spec   = torch.fft.rfft(frames, n=n_fft, dim=-1)                 # (C,F,B)
    spec   = spec.transpose(-1, -2).contiguous()                     # (C,B,F)

    return spec.to(torch.complex64).cpu().numpy()


def _istft_torch(X: np.ndarray, window: np.ndarray,
                 n_fft: int, hop: int, length: int | None) -> np.ndarray:
    """
    X : (n_bins, n_frames) complex  →  (T,) float32 numpy, via vectorised
    overlap-add (torch.nn.functional.fold) instead of a per-frame Python loop.
    """
    import torch.nn.functional as Fnn

    dev = get_device()
    Xt  = torch.as_tensor(X, dtype=torch.complex64, device=dev)      # (B,F)
    win = torch.as_tensor(window, dtype=torch.float32, device=dev)   # (n_fft,)

    n_bins, n_frames = Xt.shape
    frames = torch.fft.irfft(Xt.transpose(0, 1), n=n_fft, dim=-1)     # (F,n_fft)
    frames = frames * win                                            # synth window
    out_len = n_fft + hop * (n_frames - 1)

    # torch.nn.functional.fold expects (N, C*kh*kw, L); use a "1-D image"
    # trick: kernel_size=(1, n_fft), stride=(1, hop), output_size=(1, out_len).
    cols = frames.transpose(0, 1).unsqueeze(0)                        # (1,n_fft,F)
    y    = Fnn.fold(cols, output_size=(1, out_len),
                    kernel_size=(1, n_fft), stride=(1, hop))           # (1,1,1,out_len)
    y    = y.reshape(out_len)

    win2      = (win ** 2).unsqueeze(0).unsqueeze(0)                  # (1,1,n_fft)
    ones_cols = win2.expand(n_frames, -1, -1).reshape(n_frames, n_fft) \
                    .transpose(0, 1).unsqueeze(0)                     # (1,n_fft,F)
    norm = Fnn.fold(ones_cols, output_size=(1, out_len),
                    kernel_size=(1, n_fft), stride=(1, hop))
    norm = norm.reshape(out_len)
    norm = torch.where(norm < 1e-8, torch.ones_like(norm), norm)
    y    = y / norm

    y = y.to(torch.float32).cpu().numpy()
    if length is not None:
        # Match the original NumPy implementation exactly: trim only, never
        # pad, so callers see byte-for-byte identical shapes to v1.
        y = y[:length]
    return y.astype(np.float32)


# ── Pure-NumPy fallback (identical to the original v1 implementation) ───────

def _stft_numpy(x2d: np.ndarray, window: np.ndarray,
                n_fft: int, hop: int) -> np.ndarray:
    C        = x2d.shape[0]
    n_frames = 1 + (x2d.shape[1] - n_fft) // hop
    out      = np.zeros((C, n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for c in range(C):
        for k in range(n_frames):
            frame        = x2d[c, k*hop : k*hop + n_fft] * window
            out[c, :, k] = np.fft.rfft(frame, n=n_fft)
    return out


def _istft_numpy(X: np.ndarray, window: np.ndarray,
                 n_fft: int, hop: int, length: int | None) -> np.ndarray:
    n_bins, n_frames = X.shape
    out_len = n_fft + hop * (n_frames - 1)
    y       = np.zeros(out_len, dtype=np.float32)
    norm    = np.zeros(out_len, dtype=np.float32)
    for k in range(n_frames):
        frame     = np.fft.irfft(X[:, k], n=n_fft).real
        start     = k * hop
        y[start:start+n_fft]    += frame * window
        norm[start:start+n_fft] += window ** 2
    norm = np.where(norm < 1e-8, 1.0, norm)
    y   /= norm
    if length is not None:
        y = y[:length]
    return y.astype(np.float32)


# ── Public API (unchanged signatures) ────────────────────────────────────────

def stft(x: np.ndarray,
         window: np.ndarray | None = None,
         n_fft: int = F_WIN,
         hop: int = HOP) -> np.ndarray:
    """
    Compute STFT of a single-channel signal.

    Parameters
    ----------
    x       : (T,) time-domain signal
    window  : (n_fft,) analysis window; defaults to Hann
    n_fft   : FFT size
    hop     : hop size in samples

    Returns
    -------
    X : (n_fft//2+1, n_frames) complex64 spectrum
    """
    if window is None:
        window = make_window(n_fft)
    x2d, _ = _to_2d(np.asarray(x, dtype=np.float32))
    if _HAS_TORCH:
        out = _stft_torch(x2d, window, n_fft, hop)
    else:
        out = _stft_numpy(x2d, window, n_fft, hop)
    return out[0]


def istft(X: np.ndarray,
          window: np.ndarray | None = None,
          n_fft: int = F_WIN,
          hop: int = HOP,
          length: int | None = None) -> np.ndarray:
    """
    Inverse STFT (overlap-add reconstruction).

    Parameters
    ----------
    X      : (n_fft//2+1, n_frames) complex spectrum
    window : synthesis window; defaults to Hann
    n_fft  : FFT size
    hop    : hop size
    length : expected output length (trims or zero-pads)

    Returns
    -------
    x : (T,) reconstructed time-domain signal
    """
    if window is None:
        window = make_window(n_fft)
    X = np.asarray(X, dtype=np.complex64)
    if _HAS_TORCH:
        return _istft_torch(X, window, n_fft, hop, length)
    return _istft_numpy(X, window, n_fft, hop, length)


def stft_multichannel(x: np.ndarray,
                      window: np.ndarray | None = None,
                      n_fft: int = F_WIN,
                      hop: int = HOP) -> np.ndarray:
    """
    STFT for all N microphone channels — computed in a single batched call
    (all channels × all frames at once) rather than one Python-level call
    per channel. See module docstring for rationale.

    Parameters
    ----------
    x : (N, T) multi-channel time-domain signal

    Returns
    -------
    X : (N, B, n_frames) complex64 STFT array
    """
    if window is None:
        window = make_window(n_fft)
    x = np.asarray(x, dtype=np.float32)
    if _HAS_TORCH:
        return _stft_torch(x, window, n_fft, hop)
    return _stft_numpy(x, window, n_fft, hop)