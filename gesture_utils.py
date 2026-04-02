"""
Utility functions for gesture recognition and hand tracking
"""

import numpy as np
from typing import List, Tuple

class Vector2D:
    """2D vector utility class"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def angle_between(p1: Tuple[float, float], p2: Tuple[float, float], 
                      p3: Tuple[float, float]) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3
        Returns angle in radians
        """
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (
            (np.sqrt(v1[0]**2 + v1[1]**2) * np.sqrt(v2[0]**2 + v2[1]**2)) + 1e-6
        )
        
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

class HandAnalyzer:
    """Analyze hand pose and extract meaningful metrics"""
    
    @staticmethod
    def get_hand_center(landmarks) -> Tuple[float, float]:
        """Get approximate center of hand"""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (np.mean(xs), np.mean(ys))
    
    @staticmethod
    def is_palm_facing_camera(landmarks) -> bool:
        """Detect if palm is facing towards camera"""
        # Check if middle finger is above wrist
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        wrist = landmarks[0]
        
        return middle_tip.y < wrist.y and middle_pip.y < wrist.y
    
    @staticmethod
    def get_hand_orientation(landmarks) -> str:
        """
        Returns hand orientation:
        - "FACING_CAMERA" - palm towards camera
        - "FACING_AWAY" - back of hand towards camera
        - "NEUTRAL" - perpendicular to camera
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]
        
        # Simple heuristic based on finger positions
        fingers_below_wrist = sum([
            landmarks[8].y > wrist.y,   # index tip
            landmarks[12].y > wrist.y,  # middle tip
            landmarks[16].y > wrist.y,  # ring tip
            landmarks[20].y > wrist.y   # pinky tip
        ])
        
        if fingers_below_wrist >= 3:
            return "FACING_CAMERA"
        elif fingers_below_wrist <= 1:
            return "FACING_AWAY"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def calculate_hand_size(landmarks) -> float:
        """Calculate approximate hand size in image normalized units"""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        return np.sqrt(width**2 + height**2)

class GestureRecognizer:
    """Advanced gesture recognition with confidence scores"""
    
    @staticmethod
    def get_extended_fingers(landmarks) -> List[bool]:
        """
        Returns list of 5 booleans for each finger (thumb to pinky)
        indicating if finger is extended
        """
        extended = [False] * 5
        
        # PIP joints (knuckles)
        pips = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
        
        # Finger tips
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        # Check each finger (excluding thumb initially)
        for i in range(4):
            if tips[i].y < pips[i].y:
                extended[i + 1] = True
        
        # Thumb (check x position relative to index)
        if landmarks[4].x > landmarks[3].x:  # Right hand
            extended[0] = landmarks[4].x < landmarks[2].x
        else:  # Left hand (mirror consideration)
            extended[0] = landmarks[4].x > landmarks[2].x
        
        return extended
    
    @staticmethod
    def detect_pinch(landmarks, finger_tip_idx: int, thumb_idx: int = 4,
                      threshold: float = 0.05) -> bool:
        """
        Detect if a finger is pinched with thumb
        finger_tip_idx: index of finger tip (8, 12, 16, 20)
        returns: True if distance is below threshold
        """
        finger_tip = landmarks[finger_tip_idx]
        thumb_tip = landmarks[thumb_idx]
        
        distance = np.sqrt((finger_tip.x - thumb_tip.x)**2 + 
                          (finger_tip.y - thumb_tip.y)**2)
        
        return distance < threshold
    
    @staticmethod
    def detect_thumbs_up(landmarks) -> bool:
        """Detect thumbs up gesture"""
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Thumb should point up
        if thumb_tip.y >= thumb_pip.y:
            return False
        
        # Other fingers should be closed
        fingers_closed = (
            landmarks[8].y > landmarks[6].y and   # index closed
            landmarks[12].y > landmarks[10].y and # middle closed
            landmarks[16].y > landmarks[14].y and # ring closed
            landmarks[20].y > landmarks[18].y     # pinky closed
        )
        
        return fingers_closed
    
    @staticmethod
    def detect_ok_sign(landmarks) -> bool:
        """
        Detect OK sign:
        - Thumb and index finger forming circle
        - Other fingers extended
        """
        # Index and thumb close together
        index_thumb_distance = np.sqrt(
            (landmarks[8].x - landmarks[4].x)**2 +
            (landmarks[8].y - landmarks[4].y)**2
        )
        
        if index_thumb_distance > 0.08:
            return False
        
        # Other fingers extended
        other_extended = (
            landmarks[12].y < landmarks[10].y and  # middle extended
            landmarks[16].y < landmarks[14].y and  # ring extended
            landmarks[20].y < landmarks[18].y      # pinky extended
        )
        
        return other_extended
    
    @staticmethod
    def detect_open_hand(landmarks) -> bool:
        """Detect fully open hand with all fingers extended"""
        extended = GestureRecognizer.get_extended_fingers(landmarks)
        return sum(extended) >= 4
    
    @staticmethod
    def detect_fist(landmarks) -> bool:
        """Detect closed fist"""
        extended = GestureRecognizer.get_extended_fingers(landmarks)
        return sum(extended) <= 1
    
    @staticmethod
    def get_gesture_confidence(landmarks) -> float:
        """
        Estimate confidence of gesture detection
        Based on hand clarity and finger separation
        Returns: 0.0 to 1.0
        """
        # Calculate hand span
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        
        hand_width = max(xs) - min(xs)
        hand_height = max(ys) - min(ys)
        
        # Penalize very small or very large hands
        hand_size = np.sqrt(hand_width**2 + hand_height**2)
        
        # Ideal hand size in normalized coordinates
        ideal_size = 0.3
        size_confidence = 1.0 - abs(hand_size - ideal_size) / ideal_size
        size_confidence = max(0.3, min(1.0, size_confidence))
        
        # Check finger spread (open hands should have spread fingers)
        finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        distances = []
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                d = np.sqrt(
                    (finger_tips[i].x - finger_tips[j].x)**2 +
                    (finger_tips[i].y - finger_tips[j].y)**2
                )
                distances.append(d)
        
        finger_spread = np.mean(distances) if distances else 0
        spread_confidence = min(1.0, finger_spread * 5)
        
        return (size_confidence + spread_confidence) / 2

def test_gesture_recognition():
    """Test gesture recognition functions"""
    print("Gesture recognition utilities loaded successfully")
    print("Available functions:")
    print("  - GestureRecognizer: detect pinch, thumbs up, OK sign, fist, open hand")
    print("  - HandAnalyzer: analyze hand orientation and size")
    print("  - Vector2D: distance and angle calculations")
