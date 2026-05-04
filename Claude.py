"""
gesture_app.py — PyQt6 desktop UI for Hand Gesture Mouse Controller
Run: python gesture_app.py
"""

import sys
import time
import cv2
import numpy as np
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QCheckBox, QGroupBox, QGridLayout, QComboBox,
    QStatusBar, QFrame, QSizePolicy
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from gesture_mouse_control import GestureMouseController
import config


# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────
class VideoThread(QThread):
    frame_ready    = pyqtSignal(np.ndarray)   # processed BGR frame
    gesture_update = pyqtSignal(str, float)   # gesture name, confidence
    mouse_update   = pyqtSignal(int, int)     # screen x, y
    fps_update     = pyqtSignal(float)

    def __init__(self, controller: GestureMouseController, camera_index: int = 0):
        super().__init__()
        self.controller   = controller
        self.camera_index = camera_index
        self._run         = True
        self._paused      = False
        self._show_skeleton   = True
        self._show_hud        = True
        self._mouse_control   = True

        # FPS tracking
        self._fps_times: deque = deque(maxlen=30)

    # ── public control methods (called from main thread) ──────────────────

    def stop(self):
        self._run = False
        self.wait()

    def set_paused(self, paused: bool):
        self._paused = paused

    def set_show_skeleton(self, v: bool):
        self._show_skeleton = v

    def set_show_hud(self, v: bool):
        self._show_hud = v

    def set_mouse_control(self, v: bool):
        self._mouse_control = v

    def set_camera(self, index: int):
        self.camera_index = index

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        cap = self._open_camera(self.camera_index)

        while self._run:
            t0 = time.perf_counter()

            # Camera switch requested?
            if getattr(self, '_camera_changed', False):
                cap.release()
                cap = self._open_camera(self.camera_index)
                self._camera_changed = False

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            if not self._paused:
                hand_landmarks_list = self.controller.hand_tracker.detect(frame)

                if hand_landmarks_list:
                    for landmarks in hand_landmarks_list:
                        gesture  = self.controller.get_hand_gesture(landmarks)
                        conf     = self._estimate_confidence(landmarks)

                        if self._mouse_control:
                            self.controller.control_mouse(
                                landmarks,
                                frame.shape[1], frame.shape[0],
                                gesture
                            )

                        if self._show_skeleton:
                            self.controller.hand_tracker.draw_landmarks(frame, landmarks)

                        if self._show_hud:
                            self._draw_hud(frame, gesture, conf)

                        self.gesture_update.emit(gesture, conf)
                        mx, my = self.controller.prev_x, self.controller.prev_y
                        self.mouse_update.emit(mx, my)
                else:
                    if self._show_hud:
                        self._draw_no_hand(frame)
                    self.gesture_update.emit("—", 0.0)
            else:
                self._draw_paused(frame)

            # FPS
            now = time.perf_counter()
            self._fps_times.append(now)
            if len(self._fps_times) >= 2:
                fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
                self.fps_update.emit(fps)

            self.frame_ready.emit(frame)
            elapsed = time.perf_counter() - t0
            sleep   = max(0.0, (1 / config.CAMERA_FPS) - elapsed)
            time.sleep(sleep)

        cap.release()

    # ── helpers ───────────────────────────────────────────────────────────

    def _open_camera(self, index: int):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        return cap

    def _estimate_confidence(self, landmarks) -> float:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        size = ((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) ** 0.5
        return min(1.0, max(0.0, size / 0.35))

    def _draw_hud(self, frame, gesture: str, conf: float):
        h, w = frame.shape[:2]
        # semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        color = (80, 220, 120) if gesture not in ("FIST", "—") else (80, 80, 220)
        cv2.putText(frame, f"GESTURE: {gesture}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        bar_w = int(conf * 180)
        cv2.rectangle(frame, (w - 200, 10), (w - 20, 28), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - 200, 10), (w - 200 + bar_w, 28), (60, 200, 100), -1)
        cv2.putText(frame, f"{conf:.0%}", (w - 205, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                    cv2.LINE_AA)

    def _draw_no_hand(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "No hand detected", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)

    def _draw_paused(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (200, 200, 200), 3)


# ─────────────────────────────────────────────
#  SMALL REUSABLE WIDGETS
# ─────────────────────────────────────────────

def _make_label(text: str, bold=False, size=10, color="#ccc") -> QLabel:
    lbl = QLabel(text)
    font = QFont()
    font.setPointSize(size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    return line


# ─────────────────────────────────────────────
#  STAT CARD (gesture / fps / position)
# ─────────────────────────────────────────────

class StatCard(QWidget):
    def __init__(self, label: str, initial: str = "—", accent="#4ade80"):
        super().__init__()
        self._accent = accent
        self.setStyleSheet(f"""
            StatCard {{
                background: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.title_lbl = _make_label(label.upper(), bold=False, size=8, color="#666")
        self.value_lbl = _make_label(initial, bold=True, size=14, color=accent)
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)

    def set_value(self, text: str):
        self.value_lbl.setText(text)


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):

    DARK_BG   = "#0f0f0f"
    PANEL_BG  = "#161616"
    BORDER    = "#252525"
    ACCENT    = "#4ade80"
    TEXT_PRI  = "#e0e0e0"
    TEXT_SEC  = "#666"
    BTN_STYLE = """
        QPushButton {
            background: #1e1e1e;
            color: #ccc;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 6px 14px;
            font-size: 11px;
        }
        QPushButton:hover  { background: #2a2a2a; color: #fff; }
        QPushButton:pressed{ background: #111; }
        QPushButton:disabled{ color: #444; border-color: #222; }
    """
    ACCENT_BTN = """
        QPushButton {
            background: #166534;
            color: #4ade80;
            border: 1px solid #15803d;
            border-radius: 5px;
            padding: 6px 14px;
            font-size: 11px;
            font-weight: bold;
        }
        QPushButton:hover  { background: #14532d; }
        QPushButton:pressed{ background: #052e16; }
    """
    DANGER_BTN = """
        QPushButton {
            background: #3b0a0a;
            color: #f87171;
            border: 1px solid #7f1d1d;
            border-radius: 5px;
            padding: 6px 14px;
            font-size: 11px;
        }
        QPushButton:hover { background: #450a0a; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Mouse Controller")
        self.setFixedSize(1280, 740)
        self.setStyleSheet(f"background: {self.DARK_BG}; color: {self.TEXT_PRI};")

        self._controller = GestureMouseController()
        self._thread     = VideoThread(self._controller, camera_index=config.DEFAULT_CAMERA_INDEX)
        self._paused     = False

        self._build_ui()
        self._connect_thread()
        self._thread.start()
        self.statusBar().setStyleSheet("color: #555; font-size: 10px;")
        self.statusBar().showMessage("Ready — hand tracking active")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # LEFT: video feed
        root_layout.addWidget(self._build_video_panel(), stretch=3)
        # RIGHT: controls
        root_layout.addWidget(self._build_control_panel(), stretch=1)

    # ── video panel ───────────────────────────────────────────────────────

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {self.PANEL_BG}; border: 1px solid {self.BORDER}; border-radius: 8px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background: #000; border-radius: 8px 8px 0 0;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        # stat bar below video
        stat_bar = QWidget()
        stat_bar.setStyleSheet(f"background: #111; border-top: 1px solid {self.BORDER}; border-radius: 0 0 8px 8px;")
        stat_layout = QHBoxLayout(stat_bar)
        stat_layout.setContentsMargins(12, 8, 12, 8)
        stat_layout.setSpacing(12)

        self.card_gesture = StatCard("Gesture",  "—",         "#4ade80")
        self.card_conf    = StatCard("Confidence","—",         "#60a5fa")
        self.card_pos     = StatCard("Cursor",    "—, —",      "#c084fc")
        self.card_fps     = StatCard("FPS",       "—",         "#fb923c")

        for card in (self.card_gesture, self.card_conf, self.card_pos, self.card_fps):
            stat_layout.addWidget(card)

        layout.addWidget(stat_bar)
        return panel

    # ── control panel ─────────────────────────────────────────────────────

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {self.PANEL_BG}; border: 1px solid {self.BORDER}; border-radius: 8px;")
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(_make_label("GESTURE CONTROLLER", bold=True, size=11, color=self.ACCENT))
        layout.addWidget(_separator())

        # ── Camera ──────────────────────────────────────────────
        layout.addWidget(_make_label("Camera", bold=True, size=9, color="#999"))
        cam_row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["0 — Built-in", "1 — DroidCam / External", "2 — USB External"])
        self.cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: #1e1e1e; color: {self.TEXT_PRI};
                border: 1px solid {self.BORDER}; border-radius: 4px;
                padding: 4px 8px; font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: #1e1e1e; color: {self.TEXT_PRI};
                selection-background-color: #2a2a2a;
            }}
        """)
        self.cam_combo.currentIndexChanged.connect(self._on_camera_change)
        cam_row.addWidget(self.cam_combo)
        layout.addLayout(cam_row)

        layout.addWidget(_separator())

        # ── Sensitivity sliders ─────────────────────────────────
       # ── Sensitivity sliders ─────────────────────────────────
        layout.addWidget(_make_label("Sensitivity", bold=True, size=9, color="#999"))

        # Smoothing Slider
        self.slider_smooth, self.lbl_smooth, row_smooth = self._make_slider(
            "Smoothing", 1, 99, int(config.SMOOTHING_FACTOR * 100), "%",
            tooltip="Higher = smoother but more lag"
        )
        self.slider_smooth.valueChanged.connect(
            lambda v: self._update_config("SMOOTHING_FACTOR", v / 100, self.lbl_smooth, f"{v/100:.2f}")
        )
        layout.addWidget(row_smooth) # Add it here!

        # Click Threshold Slider
        self.slider_thresh, self.lbl_thresh, row_thresh = self._make_slider(
            "Click threshold", 2, 20, int(config.CLICK_THRESHOLD * 100), "",
            tooltip="Distance for pinch detection (lower = harder to click)"
        )
        self.slider_thresh.valueChanged.connect(
            lambda v: self._update_config("CLICK_THRESHOLD", v / 100, self.lbl_thresh, f"{v/100:.2f}")
        )
        layout.addWidget(row_thresh) # Add it here!

        # Palm Gain Slider
        self.slider_gain, self.lbl_gain, row_gain = self._make_slider(
            "Palm gain", 10, 30, int(config.PALM_CONTROL_GAIN * 10), "",
            tooltip="How much hand motion maps to screen"
        )
        self.slider_gain.valueChanged.connect(
            lambda v: self._update_config("PALM_CONTROL_GAIN", v / 10, self.lbl_gain, f"{v/10:.1f}×")
        )
        layout.addWidget(row_gain) # Add it here!
        # ── Feature toggles ─────────────────────────────────────
        layout.addWidget(_make_label("Features", bold=True, size=9, color="#999"))

        self.chk_left   = self._make_checkbox("Left click (pinch index)",  config.ENABLE_LEFT_CLICK,
                                               lambda v: setattr(config, "ENABLE_LEFT_CLICK",  bool(v)))
        self.chk_right  = self._make_checkbox("Right click (pinch middle)", config.ENABLE_RIGHT_CLICK,
                                               lambda v: setattr(config, "ENABLE_RIGHT_CLICK", bool(v)))
        self.chk_mouse  = self._make_checkbox("Mouse control active",       True,
                                               lambda v: self._thread.set_mouse_control(bool(v)))
        self.chk_skel   = self._make_checkbox("Show hand skeleton",         True,
                                               lambda v: self._thread.set_show_skeleton(bool(v)))
        self.chk_hud    = self._make_checkbox("Show HUD overlay",           True,
                                               lambda v: self._thread.set_show_hud(bool(v)))

        for chk in (self.chk_left, self.chk_right, self.chk_mouse, self.chk_skel, self.chk_hud):
            layout.addWidget(chk)

        layout.addWidget(_separator())

        # ── Gesture reference ───────────────────────────────────
        layout.addWidget(_make_label("Gesture map", bold=True, size=9, color="#999"))
        gestures = [
            ("🖐", "PALM",        "Move cursor"),
            ("☝",  "POINT",       "Precision mode"),
            ("🤌", "PINCH idx",   "Left click"),
            ("🤏", "PINCH mid",   "Right click"),
            ("✌",  "PEACE",       "Right mode"),
            ("✊", "FIST",        "Standby"),
        ]
        grid = QGridLayout()
        grid.setSpacing(3)
        for row, (icon, name, action) in enumerate(gestures):
            grid.addWidget(_make_label(f"{icon} {name}", size=9, color="#ccc"), row, 0)
            grid.addWidget(_make_label(action, size=9, color="#555"),           row, 1)
        layout.addLayout(grid)

        layout.addStretch()
        layout.addWidget(_separator())

        # ── Action buttons ──────────────────────────────────────
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setStyleSheet(self.BTN_STYLE)
        self.btn_pause.clicked.connect(self._toggle_pause)

        self.btn_reset = QPushButton("↺  Reset smoothing")
        self.btn_reset.setStyleSheet(self.BTN_STYLE)
        self.btn_reset.clicked.connect(self._reset_smoothing)

        self.btn_quit = QPushButton("✕  Quit")
        self.btn_quit.setStyleSheet(self.DANGER_BTN)
        self.btn_quit.clicked.connect(self.close)

        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_reset)
        layout.addWidget(self.btn_quit)

        return panel

    # ── helper builders ───────────────────────────────────────────────────

    def _make_slider(self, label_text: str, lo: int, hi: int, val: int,
                     unit: str, tooltip: str = ""):
        """Returns (QSlider, value_label). Adds both rows to self's layout via caller."""
        row_widget = QWidget()
        row_widget.setStyleSheet("background: transparent;")
        if tooltip:
            row_widget.setToolTip(tooltip)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        lbl_title = _make_label(label_text, size=9, color="#888")
        lbl_title.setFixedWidth(100)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(val)
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px; background: #2a2a2a; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {self.ACCENT}; width: 12px; height: 12px;
                border-radius: 6px; margin: -4px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: #166534; border-radius: 2px;
            }}
        """)
        lbl_val = _make_label(f"{val/100:.2f}{unit}", size=9, color=self.ACCENT)
        lbl_val.setFixedWidth(38)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row_layout.addWidget(lbl_title)
        row_layout.addWidget(slider)
        row_layout.addWidget(lbl_val)

        # add to parent layout via caller — we return the widget
        self.centralWidget().layout().itemAt(1).widget().layout().addWidget(row_widget)
        return slider, lbl_val

    def _make_checkbox(self, text: str, checked: bool, callback) -> QCheckBox:
        chk = QCheckBox(text)
        chk.setChecked(checked)
        chk.setStyleSheet(f"""
            QCheckBox {{ color: #aaa; font-size: 11px; spacing: 6px; background: transparent; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid #444; border-radius: 3px; background: #1a1a1a;
            }}
            QCheckBox::indicator:checked {{
                background: {self.ACCENT}; border-color: {self.ACCENT};
            }}
        """)
        chk.stateChanged.connect(callback)
        return chk

    # ── thread connections ─────────────────────────────────────────────────

    def _connect_thread(self):
        self._thread.frame_ready.connect(self._update_frame)
        self._thread.gesture_update.connect(self._update_gesture)
        self._thread.mouse_update.connect(self._update_mouse)
        self._thread.fps_update.connect(self._update_fps)

    # ── slot handlers ─────────────────────────────────────────────────────

    def _update_frame(self, frame: np.ndarray):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix   = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _update_gesture(self, gesture: str, conf: float):
        self.card_gesture.set_value(gesture)
        self.card_conf.set_value(f"{conf:.0%}" if conf > 0 else "—")

    def _update_mouse(self, x: int, y: int):
        self.card_pos.set_value(f"{x}, {y}")

    def _update_fps(self, fps: float):
        self.card_fps.set_value(f"{fps:.0f}")

    def _toggle_pause(self):
        self._paused = not self._paused
        self._thread.set_paused(self._paused)
        self.btn_pause.setText("▶  Resume" if self._paused else "⏸  Pause")
        self.statusBar().showMessage("Paused" if self._paused else "Resumed — hand tracking active")

    def _reset_smoothing(self):
        self._controller.prev_x = 0
        self._controller.prev_y = 0
        self.statusBar().showMessage("Smoothing buffer cleared")

    def _on_camera_change(self, index: int):
        self._thread.camera_index = index
        self._thread._camera_changed = True
        self.statusBar().showMessage(f"Switching to camera {index}…")

    def _update_config(self, attr: str, value, label_widget: QLabel, display: str):
        setattr(config, attr, value)
        label_widget.setText(display)

    # ── cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._thread.stop()
        self._controller.hand_tracker.close()
        event.accept()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette so native widgets inherit the theme
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(15,  15,  15))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base,            QColor(20,  20,  20))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(25,  25,  25))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(40,  40,  40))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(200, 200, 200))
    pal.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button,          QColor(30,  30,  30))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(74,  222, 128))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0,   0,   0))
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())