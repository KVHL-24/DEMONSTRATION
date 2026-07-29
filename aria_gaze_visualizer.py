"""
aria_gaze_visualizer.py — Oracle gaze visualisation on Aria RGB frames
=======================================================================

Projects oracle eye-gaze estimates onto calibrated, upright-rotated RGB
frames from a Project Aria VRS recording and encodes them straight into an
annotated MP4.

Coordinate system
-----------------
All sensors share the CPF (Central Pupil Frame) via the device-factory
calibration stored inside the VRS:

  • Eye gaze is natively in CPF (yaw / pitch angles, CPF origin).
  • ``project_gaze_manual`` (see the v1.6 note below) applies the CPF→RGB
    extrinsic + the RGB camera's own intrinsic (fisheye / linear) to
    yield the pixel-space gaze point on the (upright-rotated) RGB image.
  • Microphone positions are also expressed in the device frame, which
    is coincident with CPF; their calibration is printed on startup.

Gaze depth (v1.3)
------------------
Reprojecting a CPF-frame gaze ray into the RGB camera requires an assumed
*depth* along that ray, because CPF and the RGB camera are physically
offset (a real parallax baseline) — the same gaze direction reprojects to
different RGB pixels depending on how far away the gazed-at point is
assumed to be. The new eye-gaze model (vergence-based) estimates this
per-frame — it's the ``depth_m`` column in the CSV and the ``.depth``
attribute on each parsed ``EyeGaze`` sample. That per-sample depth is used
for every frame instead of a single flat assumption. ``--depth-m`` is only
a *fallback*, used solely for samples that have no usable per-frame depth.

Note from live testing: vergence-derived depth is genuinely noisy frame
to frame (a single recording can show ``depth_m`` swing from ~0.3 m to
the ~4 m ceiling between nearby samples) even while yaw/pitch stay
comparatively stable. That's expected sensor behaviour, not a bug — the
gaze marker can legitimately barely move even as ``depth_m`` swings
widely, if the gaze direction happens to be close to the CPF↔RGB baseline
axis for that stretch of the recording.

Nearest-gaze lookup + out-of-frame indicator (v1.4)
----------------------------------------------------
Two related bugs could make the on-screen marker look frozen at a single
pixel for the whole video even though the underlying gaze angles/depth
are genuinely varying frame to frame:

  1. The old "gaze is outside the frame" arrow clamped u/v *independently*
     to the frame border. If the reprojected point is consistently far
     outside the visible frame in roughly the same general direction,
     independent-axis clamping collapses to the *same* boundary pixel
     every frame. This is fixed with a proper ray/frame-border
     intersection (``_edge_intersection``), so the indicator continuously
     tracks the true gaze bearing instead of saturating to a corner.
  2. The previous version called the SDK's nearest-timestamp gaze lookup
     as an opaque black box. That's replaced with an explicit,
     inspectable nearest-neighbour search (binary search over the gaze
     sample timestamps) plus a maximum time-gap guard: if the closest
     available gaze sample is farther than ``--max-gaze-gap-ms`` from the
     frame's own timestamp, it's treated as MISSING rather than silently
     reused as if it were current.

Gaze status on each frame
--------------------------
For every RGB frame, the gaze is in exactly one of three states:

  • OK            — a gaze sample was found close enough in time AND it
                     reprojects to a valid pixel coordinate.
  • NO_PROJECTION — a gaze sample was found (yaw/pitch angles are known)
                     but reprojection to pixel space failed.
  • MISSING       — no gaze sample was found within --max-gaze-gap-ms of
                     this frame's timestamp.

Output (v1.2)
-------------
No per-frame PNGs are written — only a single annotated MP4 per recording,
named after the input VRS file's stem (e.g. recording123.vrs →
recording123.mp4), so multiple recordings can safely share one
--output-dir. Annotated frames are streamed directly into a
cv2.VideoWriter as soon as they're ready, in frame order.

Parallelism
-----------
Per-frame work is farmed out to a process pool via ``--workers`` (default:
all CPU cores). Each worker process opens its *own* independent VRS
data_provider and re-reads the gaze CSV in an initializer, since Aria's
provider/calibration objects are pybind11-wrapped C++ objects and are not
picklable.

Manual reprojection, bypassing get_gaze_vector_reprojection (v1.6)
---------------------------------------------------------------------
Root-caused a persistent "gaze always NO_PROJECTION" failure to the eye
gaze CSV format itself: ``general_eye_gaze.csv`` (the generalized model's
output) has NO spatial-gaze-point columns at all -- only yaw/pitch/depth
and per-eye positions. Despite that, the SDK's parsed ``EyeGaze`` samples
report ``spatial_gaze_point_valid = True`` for this file and populate
``spatial_gaze_point_in_cpf`` with leftover/uninitialized memory (tiny
denormal floats, effectively a near-zero point) instead of a real 3D
point or a proper invalid flag.

``get_gaze_vector_reprojection`` trusts that flag internally: whenever
``spatial_gaze_point_valid`` is True it uses that (bogus) point and
ignores the ``depth_m`` argument entirely -- so no combination of
``--depth-m`` or ``make_upright`` could ever fix it for this CSV type,
because the function was silently projecting a near-zero-magnitude point
sitting a few centimetres from the CPF origin, which reliably lands
behind or right next to the camera.

The fix: this script no longer calls ``get_gaze_vector_reprojection`` at
all. It instead builds the gaze point itself from yaw/pitch/depth via
``mps.get_eyegaze_point_at_depth`` (which does not depend on the broken
spatial-point field), and reprojects it manually through the CPF→camera
extrinsic and the upright camera's own ``project()`` -- see
``project_gaze_manual`` below. That extrinsic is computed once per worker
(it's constant for the whole recording), not recomputed every frame.

Path/robustness hardening (v1.5)
----------------------------------
Diagnosing a real "gaze always out of frame" report surfaced a separate,
unrelated failure mode: ``create_vrs_data_provider`` and
``mps.read_eyegaze`` both fail *silently or confusingly* on a bad path —
the former returns ``None`` (leading to a confusing
``AttributeError: 'NoneType' object has no attribute ...`` several lines
later), and the latter can print a native-library warning and return zero
samples rather than raising. On shared/cluster filesystems it's also
possible for ``os.path.isfile()`` to say a path exists while the C++
reader still can't open it (hidden characters from copy-paste, broken
symlinks, automount quirks). All of that is now checked explicitly, in
the main process AND inside each worker's initializer (so you get one
clear error instead of N confusing tracebacks from N worker processes),
with actionable messages instead of a bare stack trace.

Usage
-----
    python aria_gaze_visualizer.py \\
        --vrs        /path/to/recording.vrs \\
        --gaze-csv   /path/to/mps/eye_gaze/general_eye_gaze.csv \\
        --output-dir /path/to/output/ \\
        [--depth-m          1.0] \\
        [--every-n          5  ] \\
        [--no-video             ] \\
        [--video-fps        10.0] \\
        [--no-mic-calib         ] \\
        [--online-calib  /path/to/mps/slam/online_calibration.jsonl] \\
        [--max-gaze-gap-ms  150 ] \\
        [--debug                ] \\
        [--workers          0  ]   # 0 = use all available CPU cores

The output MP4 is written to <output-dir>/<vrs-stem>.mp4.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Aria SDK ──────────────────────────────────────────────────────────────────
try:
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core.sophus import SE3
    from projectaria_tools.utils.calibration_utils import (
        rotate_upright_image_and_calibration,
    )
    _ARIA_AVAILABLE = True
except ImportError:
    _ARIA_AVAILABLE = False
    warnings.warn(
        "projectaria_tools not installed — "
        "install with: pip install projectaria-tools"
    )

# ── Stream IDs ────────────────────────────────────────────────────────────────
RGB_STREAM_ID  = StreamId("214-1")
ET_STREAM_ID   = StreamId("211-1")
SLAM_L_STREAM  = StreamId("1201-1")
SLAM_R_STREAM  = StreamId("1201-2")

# ── Annotation style ──────────────────────────────────────────────────────────
GAZE_COLOUR_BGR  = (0, 220, 0)        # green circle / dot — gaze OK
CROSS_COLOUR_BGR = (255, 255, 255)    # white cross-hair
OUTER_RING_BGR   = (0, 0, 0)          # black shadow ring
CIRCLE_RADIUS    = 18
CROSS_LEN        = 30
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.52
FONT_THICK       = 1
OSD_COLOUR       = (0, 220, 0)        # default OSD colour — green (gaze OK)
OSD_COLOUR_WARN  = (0, 200, 255)      # amber — gaze angle known, no pixel fix
OSD_COLOUR_BAD   = (0, 0, 255)        # red — no gaze sample at all
OSD_SHADOW       = (0, 0, 0)

# Nominal eye-gaze sampling rate varies by model/session, so a match
# within 150 ms of the frame's own timestamp is a reasonable default for
# "current"; beyond that there's a real dropout and we should say so
# (MISSING) rather than silently reusing a stale match as if it were live.
DEFAULT_MAX_GAZE_GAP_MS = 150.0


class GazeStatus(Enum):
    """Tri-state outcome of trying to place a gaze marker on a given frame."""
    OK            = "ok"             # sample found + reprojected to a pixel
    NO_PROJECTION = "no_projection"  # sample found, yaw/pitch known, pixel fix failed
    MISSING       = "missing"        # no gaze sample near this timestamp


# ══════════════════════════════════════════════════════════════════════════════
# Path validation helpers (v1.5)
# ══════════════════════════════════════════════════════════════════════════════

def _describe_path(label: str, path: str) -> str:
    """One diagnostic line per path: repr() (shows hidden chars) + isfile()."""
    return f"{label}: {path!r}  isfile={os.path.isfile(path)}"


def validate_input_paths(vrs_path: str, gaze_csv_path: str) -> None:
    """
    Fail loudly and specifically for the two most common real-world causes
    of confusing downstream errors: a wrong/mangled path (often from
    terminal line-wrap + copy-paste truncation) and a path that exists per
    Python's stat() but that the SDK's own C++ reader still can't open
    (hidden whitespace/unicode, broken symlink, NFS/automount mismatch).

    Raises FileNotFoundError with an actionable message rather than
    letting the caller hit an opaque 'NoneType has no attribute ...'
    several calls later.
    """
    problems = []
    if not os.path.isfile(vrs_path):
        problems.append(_describe_path("--vrs", vrs_path))
    if not os.path.isfile(gaze_csv_path):
        problems.append(_describe_path("--gaze-csv", gaze_csv_path))

    if problems:
        raise FileNotFoundError(
            "One or more input paths do not resolve to a real file:\n  "
            + "\n  ".join(problems)
            + "\nCheck for shell line-wrapping / copy-paste truncation, and "
              "confirm with `ls -la '<path>'` or `find <dir> -iname '<name>'` "
              "rather than retyping a long path by hand."
        )


def open_vrs_provider(vrs_path: str):
    """
    create_vrs_data_provider() returns None on failure instead of raising,
    which turns into a confusing AttributeError several lines later if
    unchecked. Convert that into a clear, immediate RuntimeError instead.
    """
    prov = data_provider.create_vrs_data_provider(vrs_path)
    if prov is None:
        raise RuntimeError(
            f"create_vrs_data_provider returned None for:\n  {vrs_path}\n"
            "The path exists but the VRS SDK could not open it — it may be "
            "corrupt, an incomplete/interrupted transfer, or not a valid "
            "Aria VRS recording."
        )
    return prov


def load_gaze_data(gaze_csv_path: str):
    """
    mps.read_eyegaze() can print a native-library error and still return
    an empty list rather than raising (e.g. when os.path.isfile() says the
    path exists but the C++ reader can't actually open it — seen in
    practice on NFS/automounted cluster paths). Surface that clearly.
    """
    gaze_data = mps.read_eyegaze(gaze_csv_path)
    if len(gaze_data) == 0:
        raise RuntimeError(
            f"mps.read_eyegaze parsed zero samples from:\n  {gaze_csv_path}\n"
            "If os.path.isfile() reported this path as existing, this mismatch "
            "usually means a hidden character in the path, a broken symlink, or "
            "an NFS/automount issue — not a CSV formatting problem. Try, in the "
            f"same shell: `cat -A '{gaze_csv_path}' | head -3`. If that ALSO "
            "fails with 'No such file or directory', re-derive the path with "
            "`find`/tab-completion rather than retyping or copy-pasting it."
        )
    return gaze_data


# ══════════════════════════════════════════════════════════════════════════════
# Calibration helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_mic_calibration(device_calib) -> None:
    """Print microphone array calibration summary."""
    try:
        mic_calibs = device_calib.get_aria_microphone_calib()
        print(f"\n[MicCalib] {len(mic_calibs)} microphones in device calibration:")
        for i, mc in enumerate(mic_calibs):
            label = mc.get_label() if hasattr(mc, "get_label") else f"mic{i}"
            print(f"  [{i:02d}] label={label}")
    except Exception as exc:
        warnings.warn(f"Could not read mic calibration: {exc}")


def print_et_calibration(device_calib) -> None:
    """Print eye-tracking camera calibration summary."""
    try:
        et_calibs = device_calib.get_aria_et_camera_calib()
        print(f"\n[ETCalib] Eye-tracking cameras ({len(et_calibs)}):")
        for i, ec in enumerate(et_calibs):
            label = ec.get_label() if hasattr(ec, "get_label") else f"et{i}"
            print(f"  [{i}] label={label}")
    except Exception as exc:
        warnings.warn(f"Could not read ET calibration: {exc}")


def print_rgb_calibration(device_calib, rgb_label: str) -> None:
    """Print RGB camera calibration and CPF→RGB extrinsic."""
    try:
        cam_calib = device_calib.get_camera_calib(rgb_label)
        t_cpf_rgb = device_calib.get_transform_cpf_sensor(rgb_label)
        print(f"\n[RGBCalib] label={rgb_label}")
        print(f"  T_cpf_rgb  = {t_cpf_rgb}")
        print(f"  cam_calib  = {cam_calib}")
    except Exception as exc:
        warnings.warn(f"Could not read RGB calibration: {exc}")


def load_online_calibration(jsonl_path: str) -> list:
    """Load per-frame online calibration from SLAM MPS JSONL output."""
    if not os.path.isfile(jsonl_path):
        warnings.warn(
            f"--online-calib path does not exist, skipping: {jsonl_path!r}"
        )
        return []
    try:
        calibs = mps.read_online_calibration(jsonl_path)
        print(f"\n[OnlineCalib] {len(calibs)} entries from {jsonl_path}")
        if calibs:
            entry = calibs[0]
            for imu_c in entry.imu_calibs:
                print(f"  IMU  : {imu_c.get_label()}")
            for cam_c in entry.camera_calibs:
                print(f"  Camera: {cam_c.get_label()}")
        return calibs
    except Exception as exc:
        warnings.warn(f"Could not load online calibration: {exc}")
        return []


# 90-degree camera-frame rotation applied when the RGB image is rotated
# upright (portrait sensor -> landscape-looking display convention).
_CAMERA_CW90 = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])


def build_upright_camera_from_cpf_transform(device_calib, rgb_label: str):
    """
    Build the constant CPF->upright-RGB-camera extrinsic once per recording.

    This replaces relying on get_gaze_vector_reprojection's internal
    extrinsic handling (see the v1.6 changelog note at the top of this
    file for why that function can't be trusted for general_eye_gaze.csv
    inputs). The transform only depends on device calibration + which
    camera label we're using, not on any per-frame gaze sample, so it's
    computed once and reused for every frame.
    """
    transform_device_cpf    = device_calib.get_transform_device_cpf()
    transform_device_camera = device_calib.get_transform_device_sensor(rgb_label, True)
    transform_camera_cw90   = SE3.from_matrix(_CAMERA_CW90)
    transform_device_camera_upright = transform_device_camera @ transform_camera_cw90
    return transform_device_camera_upright.inverse() @ transform_device_cpf


def project_gaze_manual(
    yaw: float,
    pitch: float,
    depth_m: float,
    camera_from_cpf,
    upright_calib,
) -> Optional[np.ndarray]:
    """
    Reproject a yaw/pitch/depth gaze sample into upright-RGB pixel space,
    entirely independent of eye_gaze.spatial_gaze_point_valid /
    spatial_gaze_point_in_cpf.

    Returns None if the point is behind the camera or falls outside the
    calibrated sensor's valid FOV mask (a genuine "not visible in this
    frame" case) -- but never silently substitutes a bogus point, which is
    the failure mode this replaces.
    """
    gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(yaw, pitch, depth_m)
    gaze_center_in_camera = np.asarray(camera_from_cpf @ gaze_center_in_cpf).reshape(-1)
    if gaze_center_in_camera.size != 3 or not np.all(np.isfinite(gaze_center_in_camera)):
        return None
    if gaze_center_in_camera[2] <= 0:
        return None  # behind the camera -- not a valid pixel, don't hand it to project()
    return upright_calib.project(gaze_center_in_camera)


def resolve_sample_depth_m(eye_gaze, fallback_depth_m: float) -> float:
    """
    Resolve the depth (metres) to use for reprojecting one gaze sample.

    Prefers the sample's own vergence-derived ``.depth`` (populated by the
    new eye-gaze model — this is the same value as the CSV's ``depth_m``
    column). Falls back to ``fallback_depth_m`` only when the sample has
    no usable per-frame depth — missing attribute, non-finite, or
    non-positive.
    """
    sample_depth_m = getattr(eye_gaze, "depth", None)
    if sample_depth_m is None or not np.isfinite(sample_depth_m) or sample_depth_m <= 0:
        return fallback_depth_m
    return float(sample_depth_m)


def _edge_intersection(
    cx0: float, cy0: float,
    u: float, v: float,
    width: int, height: int,
    margin: int = 10,
) -> Tuple[int, int]:
    """
    Find where the ray from the frame centre ``(cx0, cy0)`` through the
    (possibly far outside the frame) gaze pixel ``(u, v)`` crosses the
    image border, inset by ``margin`` pixels. Replaces independently
    clamping u and v, which collapses any far-outside point in roughly the
    same direction to the exact same boundary pixel.
    """
    dx = u - cx0
    dy = v - cy0
    if dx == 0 and dy == 0:
        return int(round(cx0)), int(round(cy0))

    x_min, x_max = margin, width - margin
    y_min, y_max = margin, height - margin

    t_candidates: List[float] = []
    if dx > 0:
        t_candidates.append((x_max - cx0) / dx)
    elif dx < 0:
        t_candidates.append((x_min - cx0) / dx)
    if dy > 0:
        t_candidates.append((y_max - cy0) / dy)
    elif dy < 0:
        t_candidates.append((y_min - cy0) / dy)

    positive_ts = [t for t in t_candidates if t > 0]
    t = min(positive_ts) if positive_ts else 0.0

    ex = cx0 + t * dx
    ey = cy0 + t * dy
    ex = float(np.clip(ex, x_min, x_max))
    ey = float(np.clip(ey, y_min, y_max))
    return int(round(ex)), int(round(ey))


# ══════════════════════════════════════════════════════════════════════════════
# Annotation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_osd_line(img_bgr: np.ndarray,
                   text: str,
                   row: int,
                   x_pad: int = 8,
                   colour: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a single OSD text line with a dark shadow for readability."""
    y = 22 + row * 22
    cv2.putText(img_bgr, text, (x_pad + 1, y + 1),
                FONT, FONT_SCALE, OSD_SHADOW, FONT_THICK + 1, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x_pad, y),
                FONT, FONT_SCALE, colour or OSD_COLOUR, FONT_THICK, cv2.LINE_AA)


def annotate_frame(
    rgb_image:    np.ndarray,              # (H, W, 3) uint8, upright
    gaze_uv:      Optional[Tuple[float, float]],
    gaze_status:  GazeStatus,
    frame_idx:    int,
    timestamp_ns: int,
    depth_m:      float,
    yaw_deg:      Optional[float] = None,
    pitch_deg:    Optional[float] = None,
) -> np.ndarray:
    """
    Draw the gaze projection and OSD metadata onto an RGB frame.
    Returns a new (H, W, 3) uint8 array in RGB colour order.
    """
    canvas = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    H, W   = canvas.shape[:2]

    if gaze_status is GazeStatus.OK and gaze_uv is not None:
        u, v = int(round(gaze_uv[0])), int(round(gaze_uv[1]))
        in_frame = (0 <= u < W) and (0 <= v < H)

        if in_frame:
            cv2.circle(canvas, (u, v), CIRCLE_RADIUS + 4, OUTER_RING_BGR, 2)
            cv2.circle(canvas, (u, v), CIRCLE_RADIUS, GAZE_COLOUR_BGR, 2)
            cv2.circle(canvas, (u, v), 5, GAZE_COLOUR_BGR, -1)
            cv2.line(canvas, (u - CROSS_LEN, v), (u + CROSS_LEN, v),
                     CROSS_COLOUR_BGR, 1, cv2.LINE_AA)
            cv2.line(canvas, (u, v - CROSS_LEN), (u, v + CROSS_LEN),
                     CROSS_COLOUR_BGR, 1, cv2.LINE_AA)
        else:
            cx, cy = _edge_intersection(W / 2.0, H / 2.0, u, v, W, H, margin=10)
            dx = u - W // 2
            dy = v - H // 2
            n  = max(np.hypot(dx, dy), 1e-6)
            ax = int(cx - 50 * dx / n)
            ay = int(cy - 50 * dy / n)
            cv2.arrowedLine(canvas, (ax, ay), (cx, cy),
                            GAZE_COLOUR_BGR, 2, tipLength=0.35)
            cv2.putText(canvas, "gaze outside frame",
                        (cx + 6, cy - 6), FONT, 0.42,
                        OSD_SHADOW, 2, cv2.LINE_AA)
            cv2.putText(canvas, "gaze outside frame",
                        (cx + 5, cy - 7), FONT, 0.42,
                        GAZE_COLOUR_BGR, 1, cv2.LINE_AA)

    elif gaze_status is GazeStatus.NO_PROJECTION:
        badge = "gaze: angle only, no pixel fix"
        cv2.putText(canvas, badge, (10, H - 16),
                    FONT, 0.45, OSD_SHADOW, 2, cv2.LINE_AA)
        cv2.putText(canvas, badge, (9, H - 17),
                    FONT, 0.45, OSD_COLOUR_WARN, 1, cv2.LINE_AA)

    ts_s = timestamp_ns * 1e-9

    if gaze_status is GazeStatus.OK and gaze_uv is not None:
        gaze_str    = f"gaze ({gaze_uv[0]:.1f}, {gaze_uv[1]:.1f}) px"
        gaze_colour = OSD_COLOUR
    elif gaze_status is GazeStatus.NO_PROJECTION:
        gaze_str    = "gaze: no pixel projection (angle only)"
        gaze_colour = OSD_COLOUR_WARN
    else:
        gaze_str    = "gaze: NO DATA"
        gaze_colour = OSD_COLOUR_BAD

    lines: List[Tuple[str, Optional[Tuple[int, int, int]]]] = [
        (f"Frame {frame_idx:05d}   t = {ts_s:.3f} s", None),
        (f"{gaze_str}   depth = {depth_m:.2f} m", gaze_colour),
    ]
    if yaw_deg is not None and pitch_deg is not None:
        lines.append(
            (f"yaw = {yaw_deg:+.1f} deg   pitch = {pitch_deg:+.1f} deg", None)
        )

    for row, (line, colour) in enumerate(lines):
        _draw_osd_line(canvas, line, row, colour=colour)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame worker (runs in a subprocess)
# ══════════════════════════════════════════════════════════════════════════════

_W: Dict[str, object] = {}


def _init_worker(
    vrs_path: str,
    gaze_csv_path: str,
    depth_m: float,
    max_gaze_gap_ns: int,
    debug: bool,
) -> None:
    """
    Pool initializer — runs once per worker process.

    Paths are re-validated here (not just in the main process) because
    each worker independently opens its own provider/CSV — if the main
    process's check somehow passed but a worker's environment differs
    (rare, but possible with per-process working directories or NFS mount
    timing), you want one clear error per worker, not a silent empty
    gaze array that then produces spurious MISSING for every single frame
    that worker touches.
    """
    validate_input_paths(vrs_path, gaze_csv_path)
    prov             = open_vrs_provider(vrs_path)
    device_calib     = prov.get_device_calibration()
    rgb_label        = prov.get_label_from_stream_id(RGB_STREAM_ID)
    rgb_camera_calib = device_calib.get_camera_calib(rgb_label)
    gaze_data        = load_gaze_data(gaze_csv_path)

    # Constant for the whole recording -- computed once here rather than
    # per frame. See project_gaze_manual / the v1.6 changelog note.
    camera_from_cpf = build_upright_camera_from_cpf_transform(device_calib, rgb_label)

    gaze_ts_ns = np.array(
        [int(g.tracking_timestamp.total_seconds() * 1e9) for g in gaze_data],
        dtype=np.int64,
    )
    if gaze_ts_ns.size and not np.all(np.diff(gaze_ts_ns) >= 0):
        warnings.warn(
            "Eye-gaze samples are not sorted by timestamp — "
            "nearest-neighbour matching may be unreliable."
        )

    _W["prov"]             = prov
    _W["device_calib"]     = device_calib
    _W["rgb_label"]        = rgb_label
    _W["rgb_camera_calib"] = rgb_camera_calib
    _W["camera_from_cpf"]  = camera_from_cpf
    _W["gaze_data"]        = gaze_data
    _W["gaze_ts_ns"]       = gaze_ts_ns
    _W["fallback_depth_m"] = depth_m
    _W["max_gaze_gap_ns"]  = max_gaze_gap_ns
    _W["debug"]            = debug


def _find_nearest_gaze(query_ns: int):
    """Explicit nearest-timestamp lookup over the precomputed gaze timestamp index."""
    gaze_data  = _W["gaze_data"]
    gaze_ts_ns = _W["gaze_ts_ns"]

    n = gaze_ts_ns.shape[0]
    if n == 0:
        return None, None

    pos = int(np.searchsorted(gaze_ts_ns, query_ns))
    candidate_idxs = []
    if pos > 0:
        candidate_idxs.append(pos - 1)
    if pos < n:
        candidate_idxs.append(pos)

    best_idx = min(candidate_idxs, key=lambda i: abs(int(gaze_ts_ns[i]) - query_ns))
    gap_ns   = abs(int(gaze_ts_ns[best_idx]) - query_ns)
    return gaze_data[best_idx], gap_ns


def _process_one_frame(idx: int) -> Tuple[int, str, np.ndarray]:
    """Process a single RGB frame end-to-end and return its annotated image."""
    prov             = _W["prov"]
    fallback_depth_m = _W["fallback_depth_m"]
    max_gaze_gap_ns  = _W["max_gaze_gap_ns"]
    debug            = _W["debug"]

    img_data = prov.get_image_data_by_index(RGB_STREAM_ID, idx)
    raw_np   = img_data[0].to_numpy_array()
    ts_ns    = int(img_data[1].capture_timestamp_ns)

    upright_img, upright_calib = rotate_upright_image_and_calibration(
        raw_np, _W["rgb_camera_calib"]
    )

    eye_gaze, gap_ns = _find_nearest_gaze(ts_ns)

    gaze_uv:      Optional[Tuple[float, float]] = None
    yaw_deg:      Optional[float] = None
    pitch_deg:    Optional[float] = None
    gaze_status:  GazeStatus      = GazeStatus.MISSING
    used_depth_m: float           = fallback_depth_m

    have_fresh_sample = (
        eye_gaze is not None
        and gap_ns is not None
        and gap_ns <= max_gaze_gap_ns
    )

    if have_fresh_sample:
        yaw_deg   = float(np.degrees(eye_gaze.yaw))
        pitch_deg = float(np.degrees(eye_gaze.pitch))
        used_depth_m = resolve_sample_depth_m(eye_gaze, fallback_depth_m)

        # NOTE: deliberately NOT calling get_gaze_vector_reprojection here.
        # For general_eye_gaze.csv inputs it internally trusts
        # eye_gaze.spatial_gaze_point_valid, which this CSV format reports
        # as True while actually holding uninitialized/garbage data (see
        # the v1.6 changelog note at the top of this file) -- so it silently
        # projects a bogus near-zero-magnitude point instead of the real
        # yaw/pitch/depth gaze ray, regardless of the depth_m passed in.
        # project_gaze_manual bypasses that field entirely.
        proj = project_gaze_manual(
            eye_gaze.yaw,
            eye_gaze.pitch,
            used_depth_m,
            _W["camera_from_cpf"],
            upright_calib,
        )
        if proj is not None:
            gaze_uv     = (float(proj[0]), float(proj[1]))
            gaze_status = GazeStatus.OK
        else:
            gaze_status = GazeStatus.NO_PROJECTION
    else:
        gaze_status = GazeStatus.MISSING

    if debug:
        gap_ms_str = f"{gap_ns / 1e6:.1f}ms" if gap_ns is not None else "n/a"
        print(
            f"[dbg] frame={idx:05d} ts_ns={ts_ns} gaze_gap={gap_ms_str} "
            f"yaw={yaw_deg} pitch={pitch_deg} depth={used_depth_m:.3f} "
            f"status={gaze_status.value} uv={gaze_uv}",
            flush=True,
        )

    annotated_rgb = annotate_frame(
        rgb_image    = upright_img,
        gaze_uv      = gaze_uv,
        gaze_status  = gaze_status,
        frame_idx    = idx,
        timestamp_ns = ts_ns,
        depth_m      = used_depth_m,
        yaw_deg      = yaw_deg,
        pitch_deg    = pitch_deg,
    )

    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    return idx, gaze_status.value, annotated_bgr


# ══════════════════════════════════════════════════════════════════════════════
# Main visualisation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def visualise_gaze_on_rgb(
    vrs_path:          str,
    gaze_csv_path:     str,
    output_dir:        str,
    depth_m:           float        = 1.0,
    every_n:           int          = 1,
    create_video:      bool         = True,
    video_fps:         float        = 10.0,
    show_mic_calib:    bool         = True,
    online_calib_path: Optional[str] = None,
    max_gaze_gap_ms:   float        = DEFAULT_MAX_GAZE_GAP_MS,
    debug:             bool         = False,
    workers:           int          = 0,
) -> None:
    """
    Main entry point: iterate over RGB frames, reproject eye gaze, annotate,
    and stream the result into <output_dir>/<vrs-stem>.mp4.
    """
    if not _ARIA_AVAILABLE:
        raise RuntimeError("projectaria_tools is required — "
                           "pip install projectaria-tools")

    # Fail fast, in the main process, with a clear message -- before
    # spinning up a whole process pool that would otherwise hit the same
    # problem N times over.
    validate_input_paths(vrs_path, gaze_csv_path)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mp4_path = out_root / f"{Path(vrs_path).stem}.mp4"

    print(f"\n[VRS] Opening  {vrs_path}")
    prov = open_vrs_provider(vrs_path)

    device_calib = prov.get_device_calibration()
    print(f"[Calib] Device subtype : {device_calib.get_device_subtype()}")

    rgb_label = prov.get_label_from_stream_id(RGB_STREAM_ID)
    print_rgb_calibration(device_calib, rgb_label)

    if show_mic_calib:
        print_mic_calibration(device_calib)
        print_et_calibration(device_calib)

    if online_calib_path:
        load_online_calibration(online_calib_path)

    # Confirm the gaze CSV parses in the main process too, so a bad file
    # is reported once, clearly, before any workers are spawned.
    load_gaze_data(gaze_csv_path)

    n_rgb  = prov.get_num_data(RGB_STREAM_ID)
    frames = list(range(0, n_rgb, every_n))
    if not frames:
        raise RuntimeError(
            f"No RGB frames found on stream {RGB_STREAM_ID} (n_rgb={n_rgb}). "
            "Confirm this VRS actually contains an RGB stream and that "
            "--every-n isn't larger than the total frame count."
        )

    n_workers = workers if workers and workers > 0 else (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(frames)))

    max_gaze_gap_ns = int(max_gaze_gap_ms * 1e6)

    print(f"\n[Loop] {n_rgb} RGB frames total  →  processing {len(frames)} "
          f"(every_n={every_n})  fallback depth={depth_m:.2f} m "
          f"(per-sample vergence depth used when available)  "
          f"max_gaze_gap={max_gaze_gap_ms:.0f}ms  workers={n_workers}")

    n_gaze_ok      = 0
    n_gaze_no_proj = 0
    n_gaze_missing = 0

    writer:       Optional[cv2.VideoWriter] = None
    pending:      Dict[int, np.ndarray]     = {}
    next_write_i  = 0
    t0 = time.monotonic()
    done = 0

    def _flush_ready() -> None:
        nonlocal next_write_i, writer
        while next_write_i < len(frames) and frames[next_write_i] in pending:
            frame_bgr = pending.pop(frames[next_write_i])
            if create_video:
                if writer is None:
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(mp4_path), fourcc, video_fps, (w, h))
                writer.write(frame_bgr)
            next_write_i += 1

    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _init_worker,
        initargs    = (vrs_path, gaze_csv_path, depth_m, max_gaze_gap_ns, debug),
    ) as pool:
        futures = {pool.submit(_process_one_frame, idx): idx for idx in frames}

        for fut in as_completed(futures):
            idx, status, frame_bgr = fut.result()
            pending[idx] = frame_bgr

            if status == GazeStatus.OK.value:
                n_gaze_ok += 1
            elif status == GazeStatus.NO_PROJECTION.value:
                n_gaze_no_proj += 1
            else:
                n_gaze_missing += 1

            _flush_ready()

            done += 1
            if done % 50 == 0 or done == len(frames):
                elapsed = time.monotonic() - t0
                rate    = done / max(elapsed, 1e-9)
                eta_s   = (len(frames) - done) / max(rate, 1e-9)
                print(f"  [{done:5d}/{len(frames)}]  "
                      f"{rate:5.1f} frames/s  ETA {eta_s:6.0f}s")

    if writer is not None:
        writer.release()

    elapsed_total = time.monotonic() - t0
    print(f"\n[Done] Processed {len(frames)} frames in {elapsed_total:.1f}s "
          f"({len(frames)/max(elapsed_total,1e-9):.1f} fps)")
    print(f"       Gaze projected (OK)        : {n_gaze_ok}")
    print(f"       Angle only, no pixel fix   : {n_gaze_no_proj}")
    print(f"       No gaze sample (missing)   : {n_gaze_missing}")

    if create_video and writer is not None:
        size_mb = mp4_path.stat().st_size / 1e6
        print(f"[Video] Saved {mp4_path}  ({size_mb:.1f} MB)")
    elif not create_video:
        print("[Video] --no-video set — nothing written to disk.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Project Aria — oracle gaze projection onto RGB frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vrs",        required=True,
                   help="Path to the Aria VRS recording")
    p.add_argument("--gaze-csv",   required=True,
                   help="Path to MPS general_eye_gaze.csv")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the output MP4 (named after the "
                        "VRS file's stem)")
    p.add_argument("--depth-m",    type=float, default=1.0,
                   metavar="M",
                   help="Fallback gaze-ray depth (m), used only for "
                        "samples that don't carry their own per-frame "
                        "vergence depth.")
    p.add_argument("--every-n",    type=int, default=1,
                   metavar="N",
                   help="Sample every N-th RGB frame (1 = all frames)")
    p.add_argument("--no-video",   action="store_true",
                   help="Process frames for stats only — write no MP4")
    p.add_argument("--video-fps",  type=float, default=10.0,
                   metavar="FPS",
                   help="Frame rate for the output MP4")
    p.add_argument("--no-mic-calib", action="store_true",
                   help="Skip printing microphone calibration on startup")
    p.add_argument("--online-calib", default=None,
                   metavar="PATH",
                   help="Path to SLAM online_calibration.jsonl (optional)")
    p.add_argument("--max-gaze-gap-ms", type=float, default=DEFAULT_MAX_GAZE_GAP_MS,
                   metavar="MS",
                   help="Maximum time gap between a frame and its nearest "
                        "gaze sample for that sample to count as current.")
    p.add_argument("--debug", action="store_true",
                   help="Print per-frame yaw/pitch/depth/pixel/gap diagnostics.")
    p.add_argument("--workers", type=int, default=0,
                   metavar="N",
                   help="Number of worker processes. 0 = all CPU cores.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    visualise_gaze_on_rgb(
        vrs_path          = args.vrs,
        gaze_csv_path     = args.gaze_csv,
        output_dir        = args.output_dir,
        depth_m           = args.depth_m,
        every_n           = args.every_n,
        create_video      = not args.no_video,
        video_fps         = args.video_fps,
        show_mic_calib    = not args.no_mic_calib,
        online_calib_path = args.online_calib,
        max_gaze_gap_ms   = args.max_gaze_gap_ms,
        debug             = args.debug,
        workers           = args.workers,
    )


if __name__ == "__main__":
    main()