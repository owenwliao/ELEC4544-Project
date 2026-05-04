import math
import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from gesture_utils import HandAnalyzer, GestureRecognizer

# Import your existing advanced logic
from gesture_mouse_advanced import AdvancedGestureMouseController
from config import CAMERA_WIDTH, CAMERA_HEIGHT

# --- HIGH-TECH UI STYLESHEET ---
STYLESHEET = """
QMainWindow {
    background-color: #050505;
}
QLabel {
    color: #00d4ff;
    font-family: 'Consolas', 'Courier New', monospace;
}
#Title {
    font-size: 22px;
    font-weight: bold;
    color: #ffffff;
    border-bottom: 2px solid #00d4ff;
    margin-bottom: 15px;
}
#StatBox {
    background-color: #10141b;
    border: 1px solid #1e252e;
    border-radius: 5px;
    padding: 10px;
}
#ValueText {
    font-size: 18px;
    font-weight: bold;
    color: #00ff88;
}
QFrame#Sidebar {
    background-color: #0a0e14;
    border-left: 2px solid #1e252e;
    min-width: 320px;
}
QPushButton {
    background-color: #161b22;
    color: #00d4ff;
    border: 1px solid #00d4ff;
    padding: 12px;
    font-weight: bold;
    border-radius: 4px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #00d4ff;
    color: #000000;
}
QPushButton#Active {
    background-color: #004d3d;
    border-color: #00ff88;
    color: #00ff88;
}
QPushButton#Quit {
    border: 1px solid #ff4b2b;
    color: #ff4b2b;
}
QPushButton#Quit:hover {
    background-color: #ff4b2b;
    color: white;
}
QTextEdit {
    background-color: #000000;
    color: #00ff88;
    border: 1px solid #1e252e;
    font-size: 11px;
}
"""

class AdvancedWorkerThread(QThread):
    """Handles Camera Capture, MediaPipe, and Mouse Logic"""
    frame_signal = pyqtSignal(np.ndarray, dict)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._active = True
        self.control_enabled = False # Master safety switch

    def run(self):
        cap = cv2.VideoCapture(self.controller.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        while self._active:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for intuitive use
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process with MediaPipe
            hand_landmarks_list = self.controller.hand_tracker.detect(frame)
            
            telemetry = {
                "gesture": "IDLE",
                "active_function": "NAVIGATING", # Add this line
                "confidence": 0.0,
                "mouse_pos": (self.controller.prev_x, self.controller.prev_y),
                "fps": self.controller.fps
            }

            if hand_landmarks_list:
                for landmarks in hand_landmarks_list:
                    # Detect Gesture
                    gesture_info = self.controller.detect_advanced_gesture(landmarks)
                    gesture_name = gesture_info['name']
                    telemetry["gesture"] = gesture_name
                    
                    # --- Update Active Function based on gesture ---
                    if gesture_name == "PINKY_UP":
                        telemetry["active_function"] = "VOLUME CONTROL"
                    elif gesture_name == "POINT":
                        telemetry["active_function"] = "LEFT CLICK"
                    elif gesture_name == "PEACE":
                        telemetry["active_function"] = "RIGHT CLICK"
                    elif gesture_name == "THUMBS_UP":
                        telemetry["active_function"] = "DOUBLE LEFT CLICK"
                    elif gesture_name == "PALM":
                        telemetry["active_function"] = "NAVIGATINGs"
                    elif gesture_name == "OK":
                        telemetry["active_function"] = "MIDDLE CLICK"
                    elif gesture_name == "FIST":
                        telemetry["active_function"] = "STANDBY"
                    elif gesture_name == "THREE":
                        telemetry["active_function"] = "SCROLLING"
                    elif gesture_name == "LOVE":
                        telemetry["active_function"] = "VOLUME CONTROL"

            if hand_landmarks_list:
                for landmarks in hand_landmarks_list:
                    # Draw visual feedback
                    self.controller.hand_tracker.draw_landmarks(frame, landmarks)
                    
                    # Detect Gesture
                    gesture_info = self.controller.detect_advanced_gesture(landmarks)
                    telemetry["gesture"] = gesture_info['name']
                    telemetry["confidence"] = gesture_info['confidence']

                    #
                    # 1. Get the Index Finger Tip coordinates (Landmark 8)
                    if gesture_name == "POINT":
                        index_tip = landmarks[8]
                        
                        # 2. Convert normalized coordinates to pixel coordinates
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        
                        # 3. Draw a high-tech reticle
                        # Outer circle (Cyan)
                        cv2.circle(frame, (ix, iy), 15, (255, 255, 0), 2) 
                        # Inner solid dot (Green)
                        cv2.circle(frame, (ix, iy), 5, (0, 255, 0), -1) 
                        
                        # Optional: Add a label that follows the finger
                        cv2.putText(frame, "CLICK READY", (ix + 20, iy - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    # 2. If the detected gesture is "POINT", check for pinch and provide click feedback
                    if gesture_name == "POINT":
                        # Using the pinch logic from your controller
                        is_pinched = GestureRecognizer.detect_pinch(landmarks, 8, threshold=0.04)
                        
                        if is_pinched:
                            # Change to Red if clicking
                            cv2.circle(frame, (ix, iy), 20, (0, 0, 255), 3) 
                            cv2.putText(frame, "CLICK!", (ix - 20, iy - 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 3. If the detected gesture is "PEACE", draw dual tech-circles on the Index and Middle finger tips
                    if gesture_name == "PEACE":
                        # 1. Get coordinates for Index (8) and Middle (12) tips
                        idx_tip = landmarks[8]
                        mid_tip = landmarks[12]
                        
                        # 2. Convert to pixel coordinates
                        ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
                        mx, my = int(mid_tip.x * w), int(mid_tip.y * h)
                        
                        # 3. Draw dual tech-circles
                        # Index tip (Cyan)
                        cv2.circle(frame, (ix, iy), 12, (255, 255, 0), 2)
                        # Middle tip (Cyan)
                        cv2.circle(frame, (mx, my), 12, (255, 255, 0), 2)
                        
                        # 4. Draw a connecting line to visualize the "V"
                        cv2.line(frame, (ix, iy), (mx, my), (255, 255, 0), 1)
                        
                        cv2.putText(frame, "R-CLICK READY", (ix + 20, iy - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                    if gesture_name == "PEACE":
                        # Use Landmark 12 for right-click pinch detection
                        is_pinched_right = GestureRecognizer.detect_pinch(landmarks, 12, threshold=0.04)
                        
                        if is_pinched_right:
                            # Flash Magenta/Red for Right Click
                            cv2.circle(frame, (mx, my), 18, (255, 0, 255), 3) 
                            cv2.putText(frame, "RIGHT CLICK!", (mx - 30, my - 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    #  If the active function is volume control, draw the dial and needle
                    if telemetry["active_function"] == "VOLUME CONTROL":
                    # 1. Get pixel coordinates for the palm center
                        palm_x_norm, palm_y_norm = HandAnalyzer.get_palm_center(landmarks)
                        cx, cy = int(palm_x_norm * w), int(palm_y_norm * h)
                        
                        # 2. Draw the outer dial ring (Cyan)
                        cv2.circle(frame, (cx, cy), 60, (255, 212, 0), 2) # BGR: Cyan-ish
                        
                        # 3. Draw the rotating needle (Green)
                        # Use the angle we saved in the controller
                        angle_rad = math.radians(self.controller.current_angle)
                        nx = int(cx + 55 * math.cos(angle_rad))
                        ny = int(cy + 55 * math.sin(angle_rad))
                        
                        cv2.line(frame, (cx, cy), (nx, ny), (136, 255, 0), 4) # BGR: Bright Green
                        
                        # 4. Add a small 'VOL' label next to the dial
                        cv2.putText(frame, "VOL", (cx - 20, cy - 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Execute Mouse Control ONLY if the UI button is toggled ON
                    if self.control_enabled:
                        self.controller.control_mouse_advanced(landmarks, w, h, gesture_info)
                        telemetry["mouse_pos"] = (self.controller.prev_x, self.controller.prev_y)

            # Update FPS logic inside the controller
            self.controller.update_fps()
            telemetry["fps"] = self.controller.fps

            self.frame_signal.emit(frame, telemetry)
        
        cap.release()

    def stop(self):
        self._active = False
        self.wait()

class AdvancedGUIMouse(QMainWindow):
    def __init__(self):
            super().__init__()
            self.setWindowTitle("ADVANCED GESTURE CONTROL SYSTEM")
            self.setFixedSize(1600, 850)
            self.setStyleSheet(STYLESHEET)
            
            # Initialize Logic Controller
            self.controller = AdvancedGestureMouseController()
            
            # UI Layouts
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.layout = QHBoxLayout(self.central_widget)

            # 1. LEFT PANEL: Video Display
            self.video_area = QVBoxLayout()
            self.video_label = QLabel()
            self.video_label.setFixedSize(1200, 720)
            self.video_label.setStyleSheet("background-color: black; border: 2px solid #1e252e;")
            self.video_area.addWidget(self.video_label)
            
            # Control Buttons
            self.btn_row = QHBoxLayout()
            self.btn_toggle = QPushButton("ENGAGE MOUSE CONTROL")
            self.btn_toggle.setCheckable(True)
            self.btn_toggle.clicked.connect(self.toggle_control)
            
            self.btn_quit = QPushButton("TERMINATE SYSTEM")
            self.btn_quit.setObjectName("Quit")
            self.btn_quit.clicked.connect(self.close)
            
            self.btn_row.addWidget(self.btn_toggle)
            self.btn_row.addStretch()
            self.btn_row.addWidget(self.btn_quit)
            self.video_area.addLayout(self.btn_row)
            
            self.layout.addLayout(self.video_area)

            # 2. RIGHT PANEL: Telemetry & Log
            self.sidebar = QFrame()
            self.sidebar.setObjectName("Sidebar")
            self.side_layout = QVBoxLayout(self.sidebar)
            
            self.side_layout.addWidget(QLabel("SYSTEM TELEMETRY", objectName="Title"))
            
            # --- Gesture Display ---
            self.gesture_box = QWidget(objectName="StatBox")
            self.g_box_layout = QVBoxLayout(self.gesture_box)
            self.g_box_layout.addWidget(QLabel("ACTIVE GESTURE"))
            self.label_gesture = QLabel("INITIALIZING...", objectName="ValueText")
            self.g_box_layout.addWidget(self.label_gesture)
            self.side_layout.addWidget(self.gesture_box)

            # --- Current Mode Display (Matched Styling) ---
            self.mode_box = QWidget(objectName="StatBox")
            self.m_box_layout = QVBoxLayout(self.mode_box)
            self.m_box_layout.addWidget(QLabel("CURRENT MODE"))
            self.label_mode = QLabel("STANDBY", objectName="ValueText")
            self.label_mode.setStyleSheet("color: #00d4ff;") # Cyber blue color
            self.m_box_layout.addWidget(self.label_mode)
            self.side_layout.addWidget(self.mode_box)

            # --- Mouse Position ---
            self.pos_box = QWidget(objectName="StatBox")
            self.p_box_layout = QVBoxLayout(self.pos_box)
            self.p_box_layout.addWidget(QLabel("CURSOR COORDINATES"))
            self.label_coords = QLabel("X: 0 | Y: 0", objectName="ValueText")
            self.p_box_layout.addWidget(self.label_coords)
            self.side_layout.addWidget(self.pos_box)

            # --- Performance Stats ---
            self.stats_row = QHBoxLayout()
            self.conf_bar = QLabel("CONF: 0.00")
            self.label_fps = QLabel("FPS: 0")
            self.label_fps.setStyleSheet("color: #00ff88; font-weight: bold;")
            self.stats_row.addWidget(self.conf_bar)
            self.stats_row.addWidget(self.label_fps)
            self.side_layout.addLayout(self.stats_row)

            # --- Log Window ---
            self.side_layout.addSpacing(10)
            self.side_layout.addWidget(QLabel("EVENT LOG"))
            self.event_log = QTextEdit()
            self.event_log.setReadOnly(True)
            self.event_log.setObjectName("LogWindow")
            self.side_layout.addWidget(self.event_log)

            self.layout.addWidget(self.sidebar)

            # Start Worker Thread
            self.worker = AdvancedWorkerThread(self.controller)
            self.worker.frame_signal.connect(self.update_screen)
            self.worker.start()

    def toggle_control(self):
        is_on = self.btn_toggle.isChecked()
        self.worker.control_enabled = is_on
        if is_on:
            self.btn_toggle.setText("DISENGAGE MOUSE CONTROL")
            self.btn_toggle.setObjectName("Active")
            self.log_event("SYSTEM: Mouse control authorized.")
        else:
            self.btn_toggle.setText("ENGAGE MOUSE CONTROL")
            self.btn_toggle.setObjectName("")
            self.log_event("SYSTEM: Mouse control suspended.")
        self.btn_toggle.setStyle(self.btn_toggle.style())

    def update_screen(self, frame, data):
        # Update Image
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        # Update Text Data
        self.label_gesture.setText(data["gesture"])
        self.label_mode.setText(data["active_function"])
        self.label_coords.setText(f"X: {data['mouse_pos'][0]} | Y: {data['mouse_pos'][1]}")
        self.conf_bar.setText(f"CONF: {data['confidence']:.2f}")
        self.label_fps.setText(f"SYSTEM FPS: {data['fps']}")
        
        # Log auto-scroll
        if data["gesture"] != "IDLE" and data["gesture"] != "UNKNOWN":
            # Just as an example, you could log every gesture change here
            pass

    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedGUIMouse()
    window.show()
    sys.exit(app.exec())