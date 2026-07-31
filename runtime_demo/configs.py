"""Configuration space of the runtime demo.

A `PipeConfig` is one point in the knob space the demo lets partners
compare. `build_pipeline()` turns it into a ready AriaDenoisingPipeline
(plus the FrameTap observer the engine reads per-frame internals from).

The knob set:
    steering       'gaze' (oracle gaze vectors) | 'srp' (GCC-PHAT+SRP)
                   | 'manual' (operator-set azimuth, fed per frame by the
                     engine through the same scalar-gaze input path)
    n_mics         2..6  — leading subset of the array
    gazestab       GazeStabilizer on the raw gaze samples (gaze only)
    micsel         AdaptiveMicSelector (output-mask mic selection)

chao-v2 note: the weight_stride / SNR-bypass-gate runtime knobs were
removed when the pipeline was re-aligned to main's algorithm code — the
demo now always runs main's default configuration. Re-introduce them
together with their pipeline_2/beamformer_2 support if the runtime
optimizations are ported forward again.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, asdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_2 import AriaDenoisingPipeline            # noqa: E402
from eval_synthetic_2 import SYNTH_MIC_POSITIONS_2D     # noqa: E402

# Match the eval harness so demo numbers are comparable with eval numbers.
BEAMFORMER_ALPHA = 0.97
VAD_THR_DB       = 3.0
RT60_S           = 0.15


@dataclass
class PipeConfig:
    steering:      str  = "gaze"     # 'gaze' | 'srp'
    n_mics:        int  = 6
    gazestab:      bool = False
    micsel:        bool = False

    def validate(self) -> "PipeConfig":
        if self.steering not in ("gaze", "srp", "manual"):
            raise ValueError(
                f"steering must be gaze|srp|manual, got {self.steering!r}")
        if not 2 <= int(self.n_mics) <= SYNTH_MIC_POSITIONS_2D.shape[0]:
            raise ValueError(f"n_mics out of range: {self.n_mics}")
        return self

    def short(self) -> str:
        bits = [self.steering, f"{self.n_mics}mic"]
        if self.gazestab:
            bits.append("stab")
        if self.micsel:
            bits.append("msel")
        return "+".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipeConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed}).validate()


class FrameTap:
    """Observer that keeps the most recent frame's internals for the engine."""

    def __init__(self) -> None:
        self.last: dict | None = None
        self.frame_idx: int = -1

    def __call__(self, frame_idx: int, data: dict) -> None:
        self.frame_idx = frame_idx
        self.last = data


def mic_positions(n_mics: int):
    return SYNTH_MIC_POSITIONS_2D[:n_mics]


def build_pipeline(cfg: PipeConfig) -> tuple[AriaDenoisingPipeline, FrameTap]:
    cfg.validate()
    tap = FrameTap()
    # 'manual' rides the gaze input path: the engine feeds the operator's
    # azimuth as a per-frame scalar gaze. The stabilizer is gaze-only —
    # smoothing a hand-set constant would just add lag.
    use_gaze = cfg.steering in ("gaze", "manual")
    pipe = AriaDenoisingPipeline(
        use_gaze=use_gaze,
        mic_pos=mic_positions(cfg.n_mics),
        alpha=BEAMFORMER_ALPHA,
        vad_thr_db=VAD_THR_DB,
        rt60_s=RT60_S,
        doa_reliable=False,
        use_gaze_stabilizer=cfg.gazestab and cfg.steering == "gaze",
        use_mic_selection=cfg.micsel,
        observer=tap,
    )
    return pipe, tap
