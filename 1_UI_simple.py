import sys
import cv2
import numpy as np
import time

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

# Now importing from your main control script
from gesture_mouse_control import GestureMouseController

STYLESHEET = """
QMainWindow { background-color: #0d1117; }
QLabel { color: #58a6ff; font-family: 'Consolas', monospace; }
#TitleLabel { font-size: 20px; font-weight: bold; color: #58a6ff; border-bottom: 1px solid #30363d; }
#SidePanel { background-color: #161b22; border-left: 1px solid #30363d; min-width: 280px; }
#ValueLabel { font-size: 16px; color: #79c0ff; font-weight: bold; }
QPushButton { 
    background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; 
    padding: 12px; border-radius: 6px; font-weight: bold; 
}
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton#ActiveBtn { background-color: #238636; color: white; border: none; }
QPushButton#ActiveBtn:hover { background-color: #2ea043; }
QPushButton#QuitBtn { color: #f85149; }
QPushButton#QuitBtn:hover { background-color: #da3633; color: white; }
"""

class VideoThread(QThread):
    update_data_signal = pyqtSignal(np.ndarray, str, tuple)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._run_flag = True
        self.mouse_enabled = False # The master switch

    def run(self):
        cap = cv2.VideoCapture(self.controller.camera_index)
        # Setting resolution based on your script's defaults
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                hand_landmarks_list = self.controller.hand_tracker.detect(frame)
                gesture_name = "NO HAND"
                coords = (0, 0)

                if hand_landmarks_list:
                    # Logic pulled from gesture_mouse_control.py
                    for landmarks in hand_landmarks_list:
                        self.controller.hand_tracker.draw_landmarks(frame, landmarks)
                        gesture_name = self.controller.get_hand_gesture(landmarks)
                        
                        # EXECUTE MOUSE CONTROL ONLY IF ENABLED
                        if self.mouse_enabled:
                            self.controller.control_mouse(landmarks, w, h, gesture_name)
                        
                        coords = (self.controller.prev_x, self.controller.prev_y)

                self.update_data_signal.emit(frame, gesture_name, coords)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Core Gesture Engine")
        self.setFixedSize(1600, 850)
        self.setStyleSheet(STYLESHEET)

        # UI State
        self.mouse_active = False

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Video Section
        self.video_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_layout.addWidget(self.image_label)

        # Controls
        self.btn_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("ENABLE MOUSE CONTROL")
        self.btn_toggle.clicked.connect(self.toggle_mouse)
        
        self.btn_quit = QPushButton("EXIT SYSTEM")
        self.btn_quit.setObjectName("QuitBtn")
        self.btn_quit.clicked.connect(self.close)

        self.btn_layout.addWidget(self.btn_toggle)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_quit)
        self.video_layout.addLayout(self.btn_layout)
        
        self.main_layout.addLayout(self.video_layout)

        # Sidebar
        self.side_panel = QFrame()
        self.side_panel.setObjectName("SidePanel")
        self.side_layout = QVBoxLayout(self.side_panel)
        
        self.side_layout.addWidget(QLabel("ENGINE STATUS", objectName="TitleLabel"))
        
        self.label_gesture = QLabel("GESTURE: READY")
        self.label_gesture.setObjectName("ValueLabel")
        self.side_layout.addWidget(self.label_gesture)

        self.label_pos = QLabel("CURSOR: 0, 0")
        self.side_layout.addWidget(self.label_pos)
        
        self.side_layout.addStretch()
        self.main_layout.addWidget(self.side_panel)

        # Initialize Logic
        self.controller = GestureMouseController()
        self.thread = VideoThread(self.controller)
        self.thread.update_data_signal.connect(self.update_ui)
        self.thread.start()

    def toggle_mouse(self):
        """Toggles mouse control on/off in the worker thread"""
        self.mouse_active = not self.mouse_active
        self.thread.mouse_enabled = self.mouse_active
        
        if self.mouse_active:
            self.btn_toggle.setText("DISABLE MOUSE CONTROL")
            self.btn_toggle.setObjectName("ActiveBtn")
        else:
            self.btn_toggle.setText("ENABLE MOUSE CONTROL")
            self.btn_toggle.setObjectName("") # Reset style
        
        # Force stylesheet refresh to apply the new ObjectName style
        self.btn_toggle.setStyle(self.btn_toggle.style())

    def update_ui(self, cv_img, gesture, coords):
        # Update Image
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        # Update Sidebar
        self.label_gesture.setText(f"GESTURE: {gesture}")
        self.label_pos.setText(f"CURSOR: {coords[0]}, {coords[1]}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_format)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())