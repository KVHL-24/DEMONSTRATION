"""
Requirements:
    pip install PySide6 sounddevice opencv-python numpy

The stream object is expected to implement:
    get_duration()
    fetch_camera(t) -> (frame, next_t)
    fetch_audio(t)  -> (samples, next_t)
    get_param_metadata()
    get_param(name)
    set_param(name, value)
"""

from __future__ import annotations

import math
import queue
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

class ClickableVideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_image_size = None  # Stores (width, height) of raw frame
        self.on_image_click = None   # Callback function
        self._is_dragging = False

    def set_raw_image_size(self, width: int, height: int):
        self.raw_image_size = (width, height)

    def _process_mouse_pos(self, pos):
        """Helper to convert screen position to raw image coordinates."""
        if not self.pixmap() or not self.raw_image_size:
            return

        raw_w, raw_h = self.raw_image_size
        pm_size = self.pixmap().size()
        lbl_size = self.size()

        # Calculate black bar margins caused by KeepAspectRatio centering
        offset_x = (lbl_size.width() - pm_size.width()) / 2.0
        offset_y = (lbl_size.height() - pm_size.height()) / 2.0

        click_x = pos.x()
        click_y = pos.y()

        # Check if click/drag position lands inside the actual rendered image area
        if (offset_x <= click_x <= offset_x + pm_size.width()) and \
           (offset_y <= click_y <= offset_y + pm_size.height()):

            # Map from label coordinates to pixmap coordinates
            pix_x = click_x - offset_x
            pix_y = click_y - offset_y

            # Scale to raw image pixel coordinates
            scale_x = raw_w / pm_size.width()
            scale_y = raw_h / pm_size.height()

            img_x = int(pix_x * scale_x)
            img_y = int(pix_y * scale_y)

            # Clamp coordinates safety check
            img_x = max(0, min(raw_w - 1, img_x))
            img_y = max(0, min(raw_h - 1, img_y))

            if self.on_image_click:
                self.on_image_click(img_x, img_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._process_mouse_pos(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Triggered while dragging if the left button is held
        if self._is_dragging and (event.buttons() & Qt.LeftButton):
            self._process_mouse_pos(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
        super().mouseReleaseEvent(event)


class GUI(QWidget):
    def __init__(self, stream):
        super().__init__()

        self.stream = stream
        self.duration = stream.get_duration()

        self.current_time = 0.0
        self.playing = False
        self.playback_lock = threading.Lock()
        self.playback_started_at = None
        self.playback_origin = 0.0

        self.audio_stream = sd.OutputStream(
            samplerate=48000,
            channels=2,
            dtype="int32",
            blocksize=2048,
        )
        self.audio_stream.start()

        self._build_ui()
        self._apply_style()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(16)  # ~60 FPS GUI refresh

        self.last_update = time.perf_counter()

        self.next_audio_t = 0.0
        self.audio_queue = queue.Queue(maxsize=8)
        self.audio_stop = threading.Event()

        self.audio_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True,
        )
        self.audio_thread.start()

        self.audio_producer_thread = threading.Thread(
            target=self._audio_producer,
            daemon=True,
        )
        self.audio_producer_thread.start()

    def _audio_worker(self):
        while not self.audio_stop.is_set():
            audio = self.audio_queue.get()
            if audio is None:
                break
            self.audio_stream.write(audio)

    def _clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _playback_position(self):
        with self.playback_lock:
            if not self.playing or self.playback_started_at is None:
                return self.current_time

            elapsed = time.perf_counter() - self.playback_started_at
            return min(self.duration, self.playback_origin + elapsed)

    def _start_playback(self, start_time=None):
        if start_time is None:
            start_time = self.current_time

        with self.playback_lock:
            self.current_time = start_time
            self.playback_origin = start_time
            self.playback_started_at = time.perf_counter()
            self.playing = True
            self.next_audio_t = start_time

        self._clear_audio_queue()

    def _pause_playback(self):
        position = self._playback_position()

        with self.playback_lock:
            self.current_time = position
            self.playback_origin = position
            self.playback_started_at = None
            self.playing = False
            self.next_audio_t = position

        self._clear_audio_queue()

    def _audio_producer(self):
        lead_time = self.stream.audio_dt * 4

        while not self.audio_stop.is_set():
            if not self.playing:
                time.sleep(0.01)
                continue

            playback_position = self._playback_position()

            if playback_position >= self.duration:
                time.sleep(0.01)
                continue

            with self.playback_lock:
                next_audio_t = self.next_audio_t

            if next_audio_t - playback_position >= lead_time:
                # we have enough audio buffered already ahead of time
                time.sleep(0.002)
                continue
            
            if playback_position < next_audio_t:
                # trying to fetch audio we already have buffered in the previous iteration
                time.sleep(0.002)
                continue

            audio_t = max(playback_position, next_audio_t)
            audio, next_audio_t = self.stream.fetch_audio(audio_t)
            with self.playback_lock:
                if self.audio_stop.is_set() or not self.playing:
                    continue
                self.next_audio_t = next_audio_t

            while not self.audio_stop.is_set() and self.playing:
                try:
                    # convert to 1 or 2-channel
                    num_ch = audio.shape[1] if audio.ndim > 1 else 1
                    if num_ch > 2:
                        audio_left_i = math.ceil(num_ch/2)
                        audio_right_i = math.floor(num_ch/2)
                        audio_left = audio[:, :audio_left_i].mean(axis=1)
                        audio_right = audio[:, audio_right_i:].mean(axis=1)
                        audio = np.stack((audio_left, audio_right), axis=1).astype(np.int32)
                    elif num_ch == 1:
                        audio = np.stack((audio[:, 0], audio[:, 0]), axis=1).astype(np.int32)

                    self.audio_queue.put(np.ascontiguousarray(audio), timeout=0.05)
                    break
                except queue.Full:
                    time.sleep(0.002)

    ###########################################################################
    # UI
    ###########################################################################

    def _build_ui(self):
        self.setWindowTitle("Data Player")
        self.resize(1000, 750)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        # Main viewport & controls area split
        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)

        # Left Card: Video display
        video_card = QFrame()
        video_card.setObjectName("card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(16, 16, 16, 16)

        self.video_label = ClickableVideoLabel()
        self.video_label.on_image_click = self._handle_image_click
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(512, 512)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        video_layout.addStretch()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_layout.addStretch()

        # Right Card: Parameters side panel
        params_frame = QFrame()
        params_frame.setObjectName("card")
        params_layout = QVBoxLayout(params_frame)

        title = QLabel("Parameters")
        title.setObjectName("sectionTitle")
        params_layout.addWidget(title)

        self.param_sliders = {}

        for name, (minimum, maximum, step) in self.stream.get_param_bounds().items():
            print(f"Adding slider for {name}: {minimum} - {maximum} (step {step})")
            row = QVBoxLayout()
            header = QHBoxLayout()

            label = QLabel(name.replace("_", " ").title())
            value_label = QLabel(f"{self.stream.get_param(name):.2f}")

            header.addWidget(label)
            header.addStretch()
            header.addWidget(value_label)

            slider = QSlider(Qt.Horizontal)
            
            num_steps = int((maximum - minimum) / step)
            
            slider.setMinimum(0)
            slider.setMaximum(num_steps)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            
            
            value = self.stream.get_param(name)
            slider.setValue(
                int((value - minimum) / (maximum - minimum) * num_steps)
            )

            def make_callback(
                param=name, lo=minimum, hi=maximum, label=value_label, num_steps=num_steps,
            ):
                def callback(v):
                    val = lo + (hi - lo) * v / num_steps
                    self.stream.set_param(param, val)
                    label.setText(f"{val:.2f}")

                return callback

            slider.valueChanged.connect(make_callback())

            row.addLayout(header)
            row.addWidget(slider)
            params_layout.addLayout(row)

            self.param_sliders[name] = slider

        params_layout.addStretch()

        main_layout.addWidget(video_card, 3)
        main_layout.addWidget(params_frame, 1)

        root.addLayout(main_layout)

        #######################################################################
        # Bottom Bar: Timeline + Play/Pause Controls
        #######################################################################

        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        # Fixed width icon-only play/pause button
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("playButton")
        self.play_button.setFixedWidth(42)
        self.play_button.clicked.connect(self.toggle_play)

        # Horizontal progress timeline
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._timeline_pressed)
        self.timeline.sliderReleased.connect(self._timeline_released)

        bottom_bar.addWidget(self.play_button)
        bottom_bar.addWidget(self.timeline)

        root.addLayout(bottom_bar)

    ###########################################################################
    # Style
    ###########################################################################

    def _apply_style(self):
        self.setStyleSheet("""
        * {
            font-family: "Segoe UI";
            font-size: 13px;
            color: #edf2f7;
        }

        QWidget {
            background: #0f1117;
        }

        QFrame#card {
            background: rgba(30, 34, 45, 220);
            border: 1px solid #2c3446;
            border-radius: 18px;
            padding: 14px;
        }

        QLabel#sectionTitle {
            font-size: 18px;
            font-weight: 700;
            color: white;
            padding-bottom: 8px;
        }

        QLabel {
            background: transparent;
        }

        QPushButton#playButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #5d5fef,
                stop:1 #45caff
            );
            border: none;
            border-radius: 10px;
            padding: 8px 0px;
            font-size: 14px;
            font-weight: 600;
        }

        QPushButton#playButton:hover {
            background: #69d6ff;
        }

        QPushButton#playButton:pressed {
            background: #3aa7d4;
        }

        /* --- SLIDER FIXES --- */

        QSlider::horizontal {
            min-height: 24px;  /* Ensures enough vertical clearance for handle */
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #242b39;
            border-radius: 3px;
        }

        QSlider::sub-page:horizontal {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #5d5fef,
                stop:1 #45caff
            );
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            width: 18px;
            height: 18px;
            margin: -6px 0;  /* Vertically centers 18px handle on 6px groove */
            border-radius: 9px;
            background: white;
            border: 3px solid #45caff;
        }

        QSlider::handle:hover {
            background: #dff8ff;
        }

        QSlider::add-page:horizontal {
            background: #1d2330;
            border-radius: 3px;
        }
        """)

    ###########################################################################
    # Playback
    ###########################################################################

    def toggle_play(self):
        if self.playing:
            self._pause_playback()
            self.play_button.setText("▶")
        else:
            self._start_playback(self.current_time)
            self.play_button.setText("❚❚")

    ###########################################################################
    # Timeline
    ###########################################################################

    def _timeline_pressed(self):
        self.was_playing = self.playing
        self._pause_playback()

    def _timeline_released(self):
        x = self.timeline.value() / 1000.0
        self.current_time = x * self.duration
        self.playback_origin = self.current_time
        self.next_audio_t = self.current_time

        if self.was_playing:
            self._start_playback(self.current_time)
            self.play_button.setText("❚❚")
        else:
            self._clear_audio_queue()
            self._draw_current_frame()

    ###########################################################################
    # Main update
    ###########################################################################

    def _update(self):
        if not self.playing:
            self._draw_current_frame()
            return

        self.current_time = self._playback_position()

        if self.current_time >= self.duration:
            self.current_time = self.duration
            self._pause_playback()
            self.play_button.setText("▶")

        self.timeline.blockSignals(True)
        self.timeline.setValue(int(1000 * self.current_time / self.duration))
        self.timeline.blockSignals(False)

        self._draw_current_frame()

    ###########################################################################
    # Video
    ###########################################################################

    def _draw_current_frame(self):
        frame, _ = self.stream.fetch_camera(self.current_time)
        
        _, gaze_proj = self.stream.fetch_gaze(self.current_time, return_projection=True)
        if gaze_proj is not None:
            frame = cv2.circle(
                frame,
                (int(gaze_proj[0]), int(gaze_proj[1])),
                10,
                (255, 0, 255),
                2,
            )
            # crosshair
            frame = cv2.line(
                frame,
                (int(gaze_proj[0]) - 10, int(gaze_proj[1])),
                (int(gaze_proj[0]) + 10, int(gaze_proj[1])),
                (255, 0, 255),
                2,
            )
            frame = cv2.line(
                frame,
                (int(gaze_proj[0]), int(gaze_proj[1]) - 10),
                (int(gaze_proj[0]), int(gaze_proj[1]) + 10),
                (255, 0, 255),
                2,
            )

        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # assume RGB
        qimg = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888,
        )

        pix = QPixmap.fromImage(qimg)

        pix = pix.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(pix)


    def _handle_image_click(self, x: int, y: int):
        """Called when the video image is clicked."""
        if hasattr(self.stream, "set_gaze_xy"):
            self.stream.set_gaze_xy(x, y)
            # Re-render immediately to reflect the new gaze point visually
            self._draw_current_frame()


    def _draw_crosshair(self, frame, x, y, color, thickness):
        frame = cv2.circle(
            frame,
            (int(x), int(y)),
            10,
            color,
            thickness,
        )
        frame = cv2.line(
            frame,
            (int(x) - 10, int(y)),
            (int(x) + 10, int(y)),
            color,
            thickness,
        )
        frame = cv2.line(
            frame,
            (int(x), int(y) - 10),
            (int(x), int(y) + 10),
            color,
            thickness,
        )
        return frame


    def _draw_current_frame(self):
        frame, _ = self.stream.fetch_camera(self.current_time)
        
        # Pass raw image dimensions to video label for coordinate scaling
        self.video_label.set_raw_image_size(frame.shape[1], frame.shape[0])
        
        _, gaze_proj_orig = self.stream.fetch_gaze(self.current_time, return_projection=True, ignore_override=True)
        _, gaze_proj = self.stream.fetch_gaze(self.current_time, return_projection=True)
        if gaze_proj_orig is not None:
            frame = self._draw_crosshair(
                frame,
                gaze_proj_orig[0],
                gaze_proj_orig[1],
                color=(128, 128, 128),
                thickness=1,
            )
        if gaze_proj is not None:
            frame = self._draw_crosshair(
                frame,
                gaze_proj[0],
                gaze_proj[1],
                color=(255, 0, 255),
                thickness=2,
            )


        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # assume RGB
        qimg = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888,
        )

        pix = QPixmap.fromImage(qimg)

        pix = pix.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(pix)

    ###########################################################################
    def closeEvent(self, event):
        self.audio_stop.set()
        self._clear_audio_queue()

        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass

        if hasattr(self, "audio_producer_thread"):
            self.audio_producer_thread.join(timeout=1.0)

        if hasattr(self, "audio_thread"):
            self.audio_thread.join(timeout=1.0)

        self.audio_stream.stop()
        self.audio_stream.close()

        event.accept()