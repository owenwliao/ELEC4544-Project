import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import threading
import time

# Configuration
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SMOOTHING_FACTOR = 0.7  # Smooth mouse movement (0-1, higher = smoother)
CLICK_THRESHOLD = 0.05  # Distance threshold for click detection
GESTURE_DELAY = 0.2  # Delay between gesture actions

class GestureMouseController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize mouse controller
        self.mouse = Controller()
        
        # State tracking
        self.running = True
        self.last_gesture_time = 0
        self.prev_x, self.prev_y = 0, 0
        self.left_click_hold = False
        self.right_click_hold = False
        
        # Camera selection
        self.camera_index = 0
        
    def get_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_finger_up(self, landmarks, finger_tip_idx, finger_pip_idx):
        """Check if a finger is pointing up"""
        return landmarks[finger_tip_idx].y < landmarks[finger_pip_idx].y
    
    def is_finger_down(self, landmarks, finger_tip_idx, finger_pip_idx):
        """Check if a finger is pointing down"""
        return landmarks[finger_tip_idx].y > landmarks[finger_pip_idx].y
    
    def get_hand_gesture(self, landmarks):
        """
        Recognize hand gestures
        - PALM: All fingers extended (move mode)
        - POINT: Only index finger up (click mode)
        - PEACE: Index and middle fingers up (right click)
        - FIST: All fingers closed (stop/standby)
        - THUMBS_UP: Thumb pointing up
        """
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Count extended fingers
        fingers_up = 0
        
        # Thumb (special case - check if to the right of index)
        thumb_extended = thumb_tip.x > index_pip.x
        if thumb_extended:
            fingers_up += 1
        
        # Other fingers
        index_up = self.is_finger_up(landmarks, 8, 6)
        middle_up = self.is_finger_up(landmarks, 12, 10)
        ring_up = self.is_finger_up(landmarks, 16, 14)
        pinky_up = self.is_finger_up(landmarks, 20, 18)
        
        if index_up:
            fingers_up += 1
        if middle_up:
            fingers_up += 1
        if ring_up:
            fingers_up += 1
        if pinky_up:
            fingers_up += 1
        
        # Gesture recognition
        if fingers_up == 0:
            return "FIST"
        elif fingers_up == 5:
            return "PALM"
        elif fingers_up == 1 and index_up and not thumb_extended:
            return "POINT"
        elif fingers_up == 2 and index_up and middle_up and not thumb_extended:
            return "PEACE"
        else:
            return "PALM"  # Default to palm for ambiguous gestures
    
    def control_mouse(self, landmarks, image_width, image_height, gesture):
        """
        Control mouse based on hand position and gesture
        """
        current_time = time.time()
        
        # Use index finger tip for cursor position in POINT mode
        if gesture == "POINT":
            cursor_landmark = landmarks[8]  # Index finger tip
        else:
            cursor_landmark = landmarks[9]  # Index finger PIP as default
        
        # Normalize coordinates to screen space
        x = int(cursor_landmark.x * image_width)
        y = int(cursor_landmark.y * image_height)
        
        # Invert X for natural movement (mirrored camera view)
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
        self.mouse.position = (smooth_x, smooth_y)
        
        # Handle clicks based on gesture
        if current_time - self.last_gesture_time > GESTURE_DELAY:
            if gesture == "POINT":
                # Check for click by distance between index and thumb
                if self.get_distance(
                    (landmarks[8].x, landmarks[8].y),
                    (landmarks[4].x, landmarks[4].y)
                ) < CLICK_THRESHOLD:
                    self.mouse.click(Button.left, 1)
                    self.last_gesture_time = current_time
            
            elif gesture == "PEACE":
                # Right click
                if self.get_distance(
                    (landmarks[12].x, landmarks[12].y),
                    (landmarks[4].x, landmarks[4].y)
                ) < CLICK_THRESHOLD:
                    self.mouse.click(Button.right, 1)
                    self.last_gesture_time = current_time
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(self.camera_index)
        
        # Try to set higher resolution for better hand detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Hand Gesture Mouse Controller")
        print("Gestures:")
        print("  PALM - Move mouse")
        print("  POINT + pinch index/thumb - Left click")
        print("  PEACE + pinch middle/thumb - Right click")
        print("  FIST - Stop")
        print("\nPress 'Q' to quit, 'C' to change camera, 'R' to reset smoothing")
        print("Starting camera feed...")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(rgb_frame)
            
            # Draw on frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get gesture and control mouse
                    gesture = self.get_hand_gesture(hand_landmarks.landmark)
                    self.control_mouse(hand_landmarks.landmark, width, height, gesture)
                    
                    # Display gesture on frame
                    cv2.putText(
                        frame,
                        gesture,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display current mouse position
                    cv2.putText(
                        frame,
                        f"Mouse: ({self.prev_x}, {self.prev_y})",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            # Display frame
            cv2.imshow("Hand Gesture Mouse Controller", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.running = False
            elif key == ord('c') or key == ord('C'):
                print("Available cameras:")
                print("  0 - Built-in camera")
                print("  1 - DroidCam or external camera")
                try:
                    camera_idx = int(input("Enter camera index: "))
                    cap.release()
                    self.camera_index = camera_idx
                    cap = cv2.VideoCapture(self.camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print(f"Switched to camera {camera_idx}")
                except ValueError:
                    print("Invalid input")
            elif key == ord('r') or key == ord('R'):
                self.prev_x, self.prev_y = 0, 0
                print("Smoothing reset")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Application closed")

def main():
    controller = GestureMouseController()
    controller.run()

if __name__ == "__main__":
    main()
