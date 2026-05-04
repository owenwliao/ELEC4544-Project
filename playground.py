import cv2
import numpy as np
from pynput.mouse import Controller, Button
import time
from mediapipe_compat import HandTracker
from gesture_utils import HandAnalyzer

class DualHandController:
    def __init__(self):
        # CHANGED: set num_hands to 2
        self.hand_tracker = HandTracker(
            num_hands=2, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mouse = Controller()
        self.running = True
        self.prev_x, self.prev_y = 0, 0
        self.control_gain = 2.0
        
    def process_hand_logic(self, landmarks, hand_idx, w, h, frame):
        gesture = self.get_hand_gesture(landmarks)
        palm_x, palm_y = HandAnalyzer.get_palm_center(landmarks)
        px, py = int(palm_x * w), int(palm_y * h)

        # Hand Color: Hand 1 (Cyan/Blue), Hand 2 (Purple/Magenta)
        color = (255, 255, 0) if hand_idx == 0 else (255, 0, 255)

        # Trigger Shield on PALM
        if gesture == "PALM":
            self.draw_shield(frame, px, py, color)
        
        # Keep other gesture feedback
        else:
            cv2.putText(frame, f"H{hand_idx+1}: {gesture}", (px - 20, py - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_shield(self, frame, x, y, color):
            """Draws a pulsing sci-fi shield effect at (x, y)"""
            # Create a pulsing effect based on time
            pulse = int(np.sin(time.time() * 12) * 8)
            
            # 1. Draw the 'Hex' Core
            cv2.circle(frame, (x, y), 10, color, -1) # Solid center
            
            # 2. Main Shield Rings
            # Outer thick ring
            cv2.circle(frame, (x, y), 80 + pulse, color, 2)
            # Inner thin ring
            cv2.circle(frame, (x, y), 65, color, 1)
            
            # 3. Geometric 'Support' Lines (makes it look like a hologram)
            for i in range(0, 360, 60):
                # Calculate points for a hexagon around the palm
                angle = np.deg2rad(i)
                line_x = int(x + (80 + pulse) * np.cos(angle))
                line_y = int(y + (80 + pulse) * np.sin(angle))
                cv2.line(frame, (x, y), (line_x, line_y), color, 1)

            # 4. Label
            cv2.putText(frame, "ACTIVE SHIELD", (x - 60, y - 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def get_hand_gesture(self, landmarks):
        """Standard gesture recognition logic"""
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Fist
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST"
        # Peace
        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"
        # Palm
        if index_up and middle_up and ring_up and pinky_up:
            return "PALM"
        # Point
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "POINT"
            
        return "UNKNOWN"

    def process_hand_logic(self, landmarks, hand_idx, w, h, frame):
        """Logic applied to each hand independently"""
        gesture = self.get_hand_gesture(landmarks)
        palm_x, palm_y = HandAnalyzer.get_palm_center(landmarks)
        px, py = int(palm_x * w), int(palm_y * h)

        # Visual Feedback for Hand ID
        color = (0, 255, 0) if hand_idx == 0 else (255, 0, 255) # Green for Hand 1, Purple for Hand 2
        cv2.putText(frame, f"HAND {hand_idx+1}: {gesture}", (px - 50, py - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # HOLOGRAM: If Palm is detected on either hand
        if gesture == "PALM":
            cv2.circle(frame, (px, py), 50, color, 2)
            cv2.circle(frame, (px, py), 60, color, 1)

        # MOUSE CONTROL: Only Hand 1 controls the cursor (to avoid conflict)
        if hand_idx == 0 and gesture == "POINT":
            # Simple direct mapping for playground testing
            self.mouse.position = (int(palm_x * 1920), int(palm_y * 1080))
            
        # SCROLL CONTROL: Hand 2 controls scrolling
        if hand_idx == 1 and gesture == "FIST":
            self.mouse.scroll(0, -2) # Scroll down

    def run(self):
            # 1. Start the camera
            cap = cv2.VideoCapture(self.camera_index)
            
            while self.running:
                ret, frame = cap.read()
                if not ret: break
                
                # 2. Mirror the frame (makes it more intuitive to move your hands)
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # 3. Create a unique timestamp (Required for Mediapipe Video Mode)
                timestamp_ms = int(time.time() * 1000)
                
                # 4. Ask the AI to find hands
                hand_landmarks_list = self.hand_tracker.detect(frame, timestamp_ms)
                
                if hand_landmarks_list:
                    # 5. Loop through every hand found (Hand 0, Hand 1, etc.)
                    for i, landmarks in enumerate(hand_landmarks_list):
                        
                        # Draw the "Skeleton" on the frame
                        self.hand_tracker.draw_landmarks(frame, landmarks)
                        
                        # 6. CALL THE LOGIC: This is where the shield gets drawn
                        # We pass 'frame' so the function can draw the circles on it
                        self.process_hand_logic(landmarks, i, w, h, frame)
                
                # 7. Show the result!
                cv2.imshow("Dual Hand Shield System", frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = DualHandController()
    engine.run()