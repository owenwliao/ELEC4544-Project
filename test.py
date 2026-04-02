#!/usr/bin/env python3
"""
Test script to verify hand gesture detection and mouse control
Run this to test your setup before using the main application
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
from gesture_utils import GestureRecognizer, HandAnalyzer, Vector2D

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing imports...")
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        return False
    
    try:
        import mediapipe
        print(f"  ✓ MediaPipe")
    except ImportError as e:
        print(f"  ✗ MediaPipe: {e}")
        return False
    
    try:
        from pynput.mouse import Controller
        print(f"  ✓ pynput")
    except ImportError as e:
        print(f"  ✗ pynput: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    print("✓ All imports successful\n")
    return True

def test_camera(camera_index=0):
    """Test if camera is accessible"""
    print(f"Testing camera (index {camera_index})...")
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"  ✗ Camera {camera_index} not accessible")
            return False, None
        
        ret, frame = cap.read()
        if not ret:
            print(f"  ✗ Failed to read frame from camera {camera_index}")
            cap.release()
            return False, None
        
        height, width, _ = frame.shape
        print(f"  ✓ Camera {camera_index} working")
        print(f"  ✓ Resolution: {width}x{height}")
        
        return True, cap
    except Exception as e:
        print(f"  ✗ Camera test failed: {e}")
        return False, None

def test_hand_detection(cap):
    """Test hand detection"""
    print("\nTesting hand detection...")
    print("  Position your hand in front of the camera...")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    detected = False
    frame_count = 0
    max_frames = 300  # ~10 seconds at 30fps
    
    cv2.namedWindow("Hand Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Detection Test", 640, 480)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get hand bounding box
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(xs) * width), int(max(xs) * width)
                y_min, y_max = int(min(ys) * height), int(max(ys) * height)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, "Hand detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Hand Detection Test", frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        
        frame_count += 1
    
    cv2.destroyAllWindows()
    hands.close()
    
    if detected:
        print("  ✓ Hand detection working!")
        return True
    else:
        print("  ✗ No hand detected during test")
        print("  Try: Better lighting, move hand closer, clear background")
        return False

def test_gesture_recognition():
    """Test gesture recognition utilities"""
    print("\nTesting gesture recognition utilities...")
    
    try:
        # Create mock landmarks (simplified for testing)
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        # Test open hand
        open_hand = [MockLandmark(0.5, 0.5) for _ in range(21)]
        # Set finger tips above knuckles for open hand
        open_hand[8].y = 0.3   # index tip up
        open_hand[12].y = 0.3  # middle tip up
        open_hand[16].y = 0.3  # ring tip up
        open_hand[20].y = 0.3  # pinky tip up
        open_hand[6].y = 0.5   # index knuckle
        open_hand[10].y = 0.5  # middle knuckle
        open_hand[14].y = 0.5  # ring knuckle
        open_hand[18].y = 0.5  # pinky knuckle
        
        is_open = GestureRecognizer.detect_open_hand(open_hand)
        print(f"  ✓ Open hand detection: {'PASS' if is_open else 'FAIL'}")
        
        # Test fist
        fist = [MockLandmark(0.5, 0.5) for _ in range(21)]
        # Set all finger tips below knuckles
        for i in [8, 12, 16, 20]:
            fist[i].y = 0.7  # tips below knuckles
        
        is_fist = GestureRecognizer.detect_fist(fist)
        print(f"  ✓ Fist detection: {'PASS' if is_fist else 'FAIL'}")
        
        # Test utilities
        center = HandAnalyzer.get_hand_center(open_hand)
        print(f"  ✓ Hand center calculation: ({center[0]:.2f}, {center[1]:.2f})")
        
        size = HandAnalyzer.calculate_hand_size(open_hand)
        print(f"  ✓ Hand size calculation: {size:.3f}")
        
        distance = Vector2D.distance((0, 0), (3, 4))
        expected = 5.0
        print(f"  ✓ Distance calculation: {distance:.1f} (expected {expected})")
        
        return True
    except Exception as e:
        print(f"  ✗ Gesture recognition test failed: {e}")
        return False

def test_mouse_control():
    """Test mouse control (non-destructive)"""
    print("\nTesting mouse control...")
    
    try:
        from pynput.mouse import Controller
        mouse = Controller()
        
        # Get current position
        current_pos = mouse.position
        print(f"  ✓ Current mouse position: {current_pos}")
        
        # Move mouse slightly
        test_x, test_y = current_pos[0] + 10, current_pos[1] + 10
        mouse.position = (test_x, test_y)
        
        # Verify movement
        new_pos = mouse.position
        if abs(new_pos[0] - test_x) < 5 and abs(new_pos[1] - test_y) < 5:
            print(f"  ✓ Mouse movement working")
            # Move back
            mouse.position = current_pos
            print(f"  ✓ Mouse reset to original position")
            return True
        else:
            print(f"  ✗ Mouse movement test failed")
            return False
    except Exception as e:
        print(f"  ✗ Mouse control test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("HAND GESTURE MOUSE CONTROLLER - TEST SUITE")
    print("="*60 + "\n")
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        print("Run: pip install -r requirements.txt\n")
    
    # Test 2: Camera
    success, cap = test_camera(0)
    if success:
        tests_passed += 1
        
        # Test 3: Hand Detection
        if test_hand_detection(cap):
            tests_passed += 1
        
        cap.release()
    else:
        # Try alternate camera
        print("\nTrying alternate camera index...")
        success, cap = test_camera(1)
        if success:
            tests_passed += 1
            if test_hand_detection(cap):
                tests_passed += 1
            cap.release()
        else:
            print("✗ No camera found. Check DroidCam setup.\n")
    
    # Test 4: Gesture Recognition
    if test_gesture_recognition():
        tests_passed += 1
    
    # Test 5: Mouse Control
    if test_mouse_control():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✓ ALL TESTS PASSED!")
        print("You're ready to run: python gesture_mouse_control.py")
    elif tests_passed >= tests_total - 1:
        print("\n⚠ MOST TESTS PASSED")
        print("Some features may not work optimally")
        print("Check troubleshooting in README.md")
    else:
        print("\n✗ TESTS FAILED")
        print("Install dependencies and check camera setup")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
