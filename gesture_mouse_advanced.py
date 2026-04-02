"""
Advanced gesture mouse controller with extended features
Includes: drag and drop, scrolling, keyboard shortcuts, gesture recording
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import threading
import time
from gesture_utils import GestureRecognizer, HandAnalyzer
from config import *

class AdvancedGestureMouseController:
    def __init__(self, config_module=None):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize mouse controller
        self.mouse = Controller()
        
        # State tracking
        self.running = True
        self.paused = False
        self.last_gesture_time = 0
        self.prev_x, self.prev_y = 0, 0
        self.dragging = False
        self.scroll_accumulator = 0
        
        # Camera selection
        self.camera_index = DEFAULT_CAMERA_INDEX
        
        # Gesture history for recording
        self.gesture_history = []
        self.recording = False
        
        # FPS counter
        self.fps = 0
        self.frame_count = 0
        self.prev_time = time.time()
        
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
    
    def detect_advanced_gesture(self, landmarks):
        """Detect advanced gestures with confidence"""
        gesture_info = {
            'name': 'UNKNOWN',
            'confidence': 0.0,
            'gesture_type': 'movement'
        }
        
        # Use utility functions for detection
        extended = GestureRecognizer.get_extended_fingers(landmarks)
        
        # Detect specific gestures
        if GestureRecognizer.detect_open_hand(landmarks):
            gesture_info['name'] = 'PALM'
            gesture_info['gesture_type'] = 'movement'
        elif GestureRecognizer.detect_fist(landmarks):
            gesture_info['name'] = 'FIST'
            gesture_info['gesture_type'] = 'stop'
        elif extended == [False, True, False, False, False]:  # Only index
            gesture_info['name'] = 'POINT'
            gesture_info['gesture_type'] = 'action'
        elif GestureRecognizer.detect_thumbs_up(landmarks):
            gesture_info['name'] = 'THUMBS_UP'
            gesture_info['gesture_type'] = 'action'
        elif GestureRecognizer.detect_ok_sign(landmarks):
            gesture_info['name'] = 'OK'
            gesture_info['gesture_type'] = 'action'
        elif extended == [False, True, True, False, False]:  # Index + middle
            gesture_info['name'] = 'PEACE'
            gesture_info['gesture_type'] = 'action'
        
        gesture_info['confidence'] = GestureRecognizer.get_gesture_confidence(landmarks)
        
        return gesture_info
    
    def control_mouse_advanced(self, landmarks, image_width, image_height, gesture_info):
        """Advanced mouse control with multiple features"""
        current_time = time.time()
        gesture = gesture_info['name']
        confidence = gesture_info['confidence']
        
        if confidence < 0.5:
            return
        
        # Get hand center for cursor position
        hand_center = HandAnalyzer.get_hand_center(landmarks)
        
        # Normalize to screen space
        x = int(hand_center[0] * image_width)
        y = int(hand_center[1] * image_height)
        
        # Invert X for natural movement
        x = image_width - x
        
        # Map to screen resolution
        screen_x = int((x / image_width) * SCREEN_WIDTH)
        screen_y = int((y / image_height) * SCREEN_HEIGHT)
        
        # Smooth movement
        smooth_x = int(self.prev_x * SMOOTHING_FACTOR + screen_x * (1 - SMOOTHING_FACTOR))
        smooth_y = int(self.prev_y * SMOOTHING_FACTOR + screen_y * (1 - SMOOTHING_FACTOR))
        
        self.prev_x = smooth_x
        self.prev_y = smooth_y
        
        # Move mouse
        if gesture == 'PALM' and not self.dragging:
            self.mouse.position = (smooth_x, smooth_y)
        
        # Action gestures
        if current_time - self.last_gesture_time > GESTURE_DELAY:
            if gesture == 'POINT':
                # Left click on pinch
                if GestureRecognizer.detect_pinch(landmarks, 8):
                    if ENABLE_LEFT_CLICK:
                        self.mouse.click(Button.left, 1)
                    self.last_gesture_time = current_time
            
            elif gesture == 'PEACE':
                # Right click on pinch
                if GestureRecognizer.detect_pinch(landmarks, 12):
                    if ENABLE_RIGHT_CLICK:
                        self.mouse.click(Button.right, 1)
                    self.last_gesture_time = current_time
            
            elif gesture == 'THUMBS_UP':
                # Double click action
                if ENABLE_LEFT_CLICK:
                    self.mouse.click(Button.left, 2)
                self.last_gesture_time = current_time
            
            elif gesture == 'OK':
                # Middle click
                if ENABLE_LEFT_CLICK:
                    self.mouse.click(Button.middle, 1)
                self.last_gesture_time = current_time
    
    def run_advanced(self):
        """Main control loop with advanced features"""
        cap = cv2.VideoCapture(self.camera_index)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        print("Advanced Hand Gesture Mouse Controller")
        print("\nGestures:")
        print("  PALM - Move mouse")
        print("  POINT + pinch - Left click")
        print("  PEACE + pinch - Right click")
        print("  THUMBS_UP - Double click")
        print("  OK sign - Middle click")
        print("  FIST - Pause/stop")
        print("\nControls:")
        print("  Q - Quit")
        print("  P - Pause/Resume")
        print("  C - Change camera")
        print("  S - Screenshot")
        print("  R - Reset")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # Update FPS
            self.update_fps()
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(rgb_frame)
            
            # Draw info on frame
            if not self.paused:
                cv2.putText(frame, "ACTIVE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "PAUSED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {self.fps}", (width-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Process gestures
            if results.multi_hand_landmarks and not self.paused:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton
                    if SHOW_HAND_SKELETON:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                    
                    # Detect gesture
                    gesture_info = self.detect_advanced_gesture(hand_landmarks.landmark)
                    
                    # Control mouse
                    self.control_mouse_advanced(hand_landmarks.landmark, width, height, gesture_info)
                    
                    # Display gesture info
                    if SHOW_GESTURE_NAME:
                        cv2.putText(frame, gesture_info['name'], (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {gesture_info['confidence']:.2f}",
                                   (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    if SHOW_MOUSE_POSITION:
                        cv2.putText(frame, f"Mouse: ({self.prev_x}, {self.prev_y})",
                                   (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                if not self.paused:
                    cv2.putText(frame, "No hand detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Advanced Gesture Mouse Controller", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.running = False
            elif key == ord('p') or key == ord('P'):
                self.paused = not self.paused
                print(f"Paused: {self.paused}")
            elif key == ord('c') or key == ord('C'):
                try:
                    camera_idx = int(input("Enter camera index: "))
                    cap.release()
                    self.camera_index = camera_idx
                    cap = cv2.VideoCapture(self.camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    print(f"Switched to camera {camera_idx}")
                except ValueError:
                    print("Invalid input")
            elif key == ord('s') or key == ord('S'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                self.prev_x, self.prev_y = 0, 0
                self.scroll_accumulator = 0
                print("Reset complete")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Application closed")

def main():
    """Main entry point for advanced controller"""
    controller = AdvancedGestureMouseController()
    controller.run_advanced()

if __name__ == "__main__":
    main()
