import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# Importing your logic from playground or gesture_mouse_advanced
from playground import DualHandController

# --- SCI-FI STYLESHEET ---
STYLESHEET = """
QMainWindow {
    background-color: #0a0b10;
}
QLabel {
    color: #00f2ff;
    font-family: 'Segoe UI', sans-serif;
}
#TitleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #00f2ff;
    border-bottom: 2px solid #00f2ff;
    padding-bottom: 5px;
    margin-bottom: 10px;
}
#StatLabel {
    font-size: 14px;
    color: #a0a0a0;
}
#ValueLabel {
    font-size: 18px;
    font-weight: bold;
    color: #00f2ff;
}
QFrame#SidePanel {
    background-color: #161b22;
    border-left: 2px solid #00f2ff;
    min-width: 250px;
}
QPushButton {
    background-color: #161b22;
    color: #00f2ff;
    border: 1px solid #00f2ff;
    padding: 10px;
    font-weight: bold;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #00f2ff;
    color: #0a0b10;
}
QPushButton#QuitBtn {
    border: 1px solid #ff4b2b;
    color: #ff4b2b;
}
QPushButton#QuitBtn:hover {
    background-color: #ff4b2b;
    color: white;
}
"""

class VideoThread(QThread):
    # Updated signal to send both the frame and detection data
    update_data_signal = pyqtSignal(np.ndarray, list)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._run_flag = True
        self.gesture_enabled = False
    

    def run(self):
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # 1. Generate timestamp (Fixes the ValueError crash)
                    timestamp_ms = int(time.time() * 1000)
                    
                    # 2. Detect hands
                    hand_landmarks_list = self.controller.hand_tracker.detect(frame, timestamp_ms)
                    gesture_data = []

                    if hand_landmarks_list:
                        for i, landmarks in enumerate(hand_landmarks_list):
                            # Always draw the basic skeleton (dots/lines)
                            self.controller.hand_tracker.draw_landmarks(frame, landmarks)
                            
                            # --- THE MASTER SWITCH ---
                            # If enabled, this call draws the SHIELD and moves the MOUSE
                            if self.gesture_enabled:
                                self.controller.process_hand_logic(landmarks, i, w, h, frame)
                            
                            # 3. Collect data for the Sidebar Telemetry
                            gesture = self.controller.get_hand_gesture(landmarks)
                            gesture_data.append({
                                "id": i + 1,
                                "gesture": gesture,
                                "pos": (int(landmarks[0].x * w), int(landmarks[0].y * h))
                            })

                    # 4. Send the frame (with shield) and the text data to the UI
                    self.update_data_signal.emit(frame, gesture_data)
            
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Gesture Hub | System OS")
        self.setFixedSize(1550, 800) # Wider to accommodate sidebar
        self.setStyleSheet(STYLESHEET)
        self.is_system_active = False
        
        # Main Widget & Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_horizontal = QHBoxLayout(self.central_widget)

        # --- LEFT: VIDEO FEED ---
        self.video_container = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setStyleSheet("border: 1px solid #333; background-color: black;")
        self.video_container.addWidget(self.image_label)
        
        # Bottom Controls
        self.btn_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("ENABLE MOUSE CONTROL")
        self.btn_toggle.clicked.connect(self.toggle_system)
        self.btn_quit = QPushButton("SYSTEM SHUTDOWN")
        self.btn_quit.setObjectName("QuitBtn")
        self.btn_quit.clicked.connect(self.close)
        
        self.btn_layout.addWidget(self.btn_toggle)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_quit)
        self.video_container.addLayout(self.btn_layout)
        
        self.layout_horizontal.addLayout(self.video_container)

        # --- RIGHT: SIDE PANEL (SYSTEM STATS) ---
        self.side_panel = QFrame()
        self.side_panel.setObjectName("SidePanel")
        self.side_layout = QVBoxLayout(self.side_panel)

        self.title = QLabel("TELEMETRY")
        self.title.setObjectName("TitleLabel")
        self.side_layout.addWidget(self.title)

        # Hand 1 Info
        self.side_layout.addWidget(QLabel("PRIMARY INPUT (HAND 1)", objectName="StatLabel"))
        self.h1_gesture = QLabel("NO SIGNAL")
        self.h1_gesture.setObjectName("ValueLabel")
        self.side_layout.addWidget(self.h1_gesture)
        
        self.h1_pos = QLabel("X: 000 | Y: 000")
        self.h1_pos.setObjectName("StatLabel")
        self.side_layout.addWidget(self.h1_pos)

        self.side_layout.addSpacing(20)

        # Hand 2 Info
        self.side_layout.addWidget(QLabel("SECONDARY INPUT (HAND 2)", objectName="StatLabel"))
        self.h2_gesture = QLabel("NO SIGNAL")
        self.h2_gesture.setObjectName("ValueLabel")
        self.side_layout.addWidget(self.h2_gesture)
        
        self.h2_pos = QLabel("X: 000 | Y: 000")
        self.h2_pos.setObjectName("StatLabel")
        self.side_layout.addWidget(self.h2_pos)

        self.side_layout.addStretch()
        
        # System FPS
        self.fps_label = QLabel("SYSTEM FPS: --")
        self.fps_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.side_layout.addWidget(self.fps_label)

        self.layout_horizontal.addWidget(self.side_panel)

        # Initialization
        self.last_time = time.time()
        self.controller = DualHandController()
        self.thread = VideoThread(self.controller)
        self.thread.update_data_signal.connect(self.update_ui)
        self.thread.start()

    def toggle_system(self):
            """Toggles the hand gesture control on/off"""
            self.is_system_active = not self.is_system_active
            
            # Tell the video thread to start/stop processing gestures
            self.thread.gesture_enabled = self.is_system_active
            
            if self.is_system_active:
                # ACTIVE STATE
                self.btn_toggle.setText("DISABLE GESTURE CONTROL")
                self.btn_toggle.setStyleSheet("background-color: #238636; color: white; border: 1px solid #00ff00;")
                self.title.setText("TELEMETRY [SYSTEM ACTIVE]")
                self.title.setStyleSheet("color: #00ff00; border-bottom: 2px solid #00ff00;")
            else:
                # STANDBY STATE
                self.btn_toggle.setText("ENABLE GESTURE CONTROL")
                self.btn_toggle.setStyleSheet("") # Resets to your CSS style
                self.title.setText("TELEMETRY [STANDBY]")
                self.title.setStyleSheet("")
                
    def update_ui(self, cv_img, gesture_data):
        """Main update loop triggered by VideoThread"""
        # 1. Update Video
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        # 2. Update Stats Sidebar
        self.update_telemetry(gesture_data)

        # 3. Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_time)
        self.last_time = curr_time
        self.fps_label.setText(f"SYSTEM FPS: {int(fps)}")

    def update_telemetry(self, data):
        """Updates the labels on the right panel"""
        # Reset labels
        self.h1_gesture.setText("NO SIGNAL")
        self.h2_gesture.setText("NO SIGNAL")
        
        for i, hand in enumerate(data):
            if i == 0:
                self.h1_gesture.setText(hand['gesture'])
                self.h1_pos.setText(f"X: {hand['pos'][0]} | Y: {hand['pos'][1]}")
            elif i == 1:
                self.h2_gesture.setText(hand['gesture'])
                self.h2_pos.setText(f"X: {hand['pos'][0]} | Y: {hand['pos'][1]}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())