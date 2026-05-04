"""
Advanced gesture mouse controller with extended features
Includes: drag and drop, scrolling, keyboard shortcuts, gesture recording
"""

import cv2
import math
import numpy as np
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import threading
import time
from gesture_utils import GestureRecognizer, HandAnalyzer
from config import *
from mediapipe_compat import HandTracker
from pynput.keyboard import Key, Controller as KeyboardController

class AdvancedGestureMouseController:
    def __init__(self, config_module=None):
        # Initialize MediaPipe hand tracking via Tasks API
        self.hand_tracker = HandTracker(
            num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        # buffer for gesture stability
        self.gesture_buffer = []
        self.buffer_size = 5 # Number of frames to stay consistent
        self.stable_gesture = "UNKNOWN"

        self.current_angle = 0 # For rotation-based volume control

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

        self.keyboard = KeyboardController()
        self.prev_vol_y = 0 # To track movement direction
        self.prev_scroll_y = 0
        self.last_logged_mode = "IDLE"

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
    
    def detect_advanced_gesture(self, landmarks):
        gesture_info = {'name': 'UNKNOWN', 'confidence': 0.0}
        extended = GestureRecognizer.get_extended_fingers(landmarks)
        
        # 1. SPECIAL CASE: OK SIGN (Uses distance logic)
        if GestureRecognizer.detect_ok_sign(landmarks):
            gesture_info['name'] = 'OK'
            
        # 2. THUMBS_UP vs PINKY_UP (The Conflict Zone)
        # Check: Is ONLY the thumb up and physically higher than the index base?
        elif extended == [True, False, False, False, False]:
            if landmarks[4].y < landmarks[5].y: # Thumb tip higher than Index base
                gesture_info['name'] = 'THUMBS_UP'
        
        # Check: Is ONLY the pinky up and physically higher than its own joint?
        elif extended == [False, False, False, False, True]:
            if landmarks[20].y < landmarks[18].y:
                gesture_info['name'] = 'PINKY_UP'

        # 3. ARRAY MATCHING (Numbers and Nav)
        elif extended == [True, True, True, True, True]:
            gesture_info['name'] = 'PALM'
        elif extended == [False, True, False, False, False]:
            gesture_info['name'] = 'POINT'
        elif extended == [False, True, True, False, False]:
            gesture_info['name'] = 'PEACE'
        elif extended == [False, True, True, True, False]:
            gesture_info['name'] = 'THREE'
        elif extended == [False, False, False, False, False]:
            gesture_info['name'] = 'FIST'
        elif extended == [False, True, False, False, True]:
            gesture_info['name'] = 'LOVE'
            
        gesture_info['confidence'] = GestureRecognizer.get_gesture_confidence(landmarks)

        current_raw_gesture = gesture_info['name']
    
        # Add to buffer
        self.gesture_buffer.append(current_raw_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
            
        # Only update stable_gesture if the buffer is unanimous
        if len(set(self.gesture_buffer)) == 1:
            self.stable_gesture = self.gesture_buffer[0]
            
        gesture_info['name'] = self.stable_gesture
        return gesture_info
    
    def control_mouse_advanced(self, landmarks, image_width, image_height, gesture_info):
        """Advanced mouse control with multiple features"""
        current_time = time.time()
        gesture = gesture_info['name']
        confidence = gesture_info['confidence']
        
        if confidence < 0.5:
            return
        
        # Track the palm, not the fingertips
        palm_x, palm_y = HandAnalyzer.get_palm_center(landmarks)

        # Volume control with pinky up
        if gesture == 'PINKY_UP':
            if abs(palm_y - self.prev_vol_y) > 0.05:
                if palm_y < self.prev_vol_y:
                    self.keyboard.press(Key.media_volume_up)
                    self.keyboard.release(Key.media_volume_up)
                else:
                    self.keyboard.press(Key.media_volume_down)
                    self.keyboard.release(Key.media_volume_down)
                self.prev_vol_y = palm_y
            return # Block mouse movement while adjusting volume
        
        # Volume control with rotation (Love sign)
        
        if gesture == 'LOVE':
            # 1. Get coordinates for the Index and Pinky knuckles
            p5 = landmarks[5]
            p17 = landmarks[17]
            
            # 2. Calculate the angle of the hand in degrees
            # We use atan2(dy, dx) to get the rotation relative to the horizontal axis
            self.current_angle = math.degrees(math.atan2(p17.y - p5.y, p17.x - p5.x))
            
            # 3. Initialize prev_angle if it doesn't exist
            if not hasattr(self, 'prev_angle'):
                self.prev_angle = self.current_angle
                return

            # 4. Calculate the rotation difference (Delta)
            angle_delta = self.current_angle - self.prev_angle
            
            # Sensitivity: Only trigger if rotated more than 5 degrees
            if abs(angle_delta) > 5:
                if angle_delta > 0:
                    # Rotated Right (Clockwise)
                    self.keyboard.press(Key.media_volume_up)
                    self.keyboard.release(Key.media_volume_up)
                    print("Rotate Right: Volume Up")
                else:
                    # Rotated Left (Counter-Clockwise)
                    self.keyboard.press(Key.media_volume_down)
                    self.keyboard.release(Key.media_volume_down)
                    print("Rotate Left: Volume Down")
                    
                # Update reference for next frame
                self.prev_angle = self.current_angle
            
            return # Lock cursor movement

        # Scrolling with three fingers up
        if gesture == 'THREE':
            # Use a default if prev_scroll_y isn't set yet
            if not hasattr(self, 'prev_scroll_y'):
                self.prev_scroll_y = palm_y
                
            y_delta = self.prev_scroll_y - palm_y 
            
            # Only calculate and use scroll_amount if movement is significant
            if abs(y_delta) > 0.01:
                scroll_amount = int(y_delta * 150) # Increased multiplier for smoother feel
                self.mouse.scroll(0, scroll_amount)
                
                # FIX: Print must be INSIDE this block to avoid UnboundLocalError
                print(f"DEBUG: Scrolling {'Up' if scroll_amount > 0 else 'Down'}")
            
            # Always update the previous position to track movement
            self.prev_scroll_y = palm_y
            return # Block mouse movement while scrolling

        # Scale palm position to give more usable screen range without leaving the camera frame
        scaled_x = 0.5 + ((palm_x - 0.5) * PALM_CONTROL_GAIN)
        scaled_y = 0.5 + ((palm_y - 0.5) * PALM_CONTROL_GAIN)
        scaled_x = max(0.0, min(1.0, scaled_x))
        scaled_y = max(0.0, min(1.0, scaled_y))

        screen_x = int(scaled_x * SCREEN_WIDTH)
        screen_y = int(scaled_y * SCREEN_HEIGHT)
        
        # Smooth movement
        smooth_x = int(self.prev_x * SMOOTHING_FACTOR + screen_x * (1 - SMOOTHING_FACTOR))
        smooth_y = int(self.prev_y * SMOOTHING_FACTOR + screen_y * (1 - SMOOTHING_FACTOR))
        
        self.prev_x = smooth_x
        self.prev_y = smooth_y
        
        # Move mouse with the palm position regardless of finger gesture
        if not self.dragging:
            self.mouse.position = (smooth_x, smooth_y)
        
        # Action gestures
        if current_time - self.last_gesture_time > GESTURE_DELAY:
            if gesture == 'POINT':
                # Precision Left Click
                if GestureRecognizer.detect_pinch(landmarks, 8, threshold=PINCH_SENSITIVITY):
                    self.mouse.click(Button.left, 1)
                    self.last_gesture_time = current_time
            
            elif gesture == 'PEACE':
                # Precision Right Click
                if GestureRecognizer.detect_pinch(landmarks, 12, threshold=PINCH_SENSITIVITY):
                    self.mouse.click(Button.right, 1)
                    self.last_gesture_time = current_time
            
            elif gesture == 'THUMBS_UP':
                # Power Action: Double Click
                self.mouse.click(Button.left, 2)
                self.last_gesture_time = current_time
            
            elif gesture == 'OK':
                # Utility: Middle Click
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
            
            # Process hand detection
            hand_landmarks_list = self.hand_tracker.detect(frame)
            
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
            if hand_landmarks_list and not self.paused:
                for hand_landmarks in hand_landmarks_list:
                    # Draw hand skeleton
                    if SHOW_HAND_SKELETON:
                        self.hand_tracker.draw_landmarks(frame, hand_landmarks)
                    
                    # Detect gesture
                    gesture_info = self.detect_advanced_gesture(hand_landmarks)
                    
                    # Control mouse
                    self.control_mouse_advanced(hand_landmarks, width, height, gesture_info)
                    
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
        self.hand_tracker.close()
        print("Application closed")

def main():
    """Main entry point for advanced controller"""
    controller = AdvancedGestureMouseController()
    controller.run_advanced()

if __name__ == "__main__":
    main()