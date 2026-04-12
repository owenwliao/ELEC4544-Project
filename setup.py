#!/usr/bin/env python3
"""
Setup and diagnostic script for Hand Gesture Mouse Controller
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_opencv():
    """Check if OpenCV is installed"""
    print("\nChecking OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def check_mediapipe():
    """Check if MediaPipe is installed"""
    print("\nChecking MediaPipe...")
    try:
        import mediapipe as mp
        has_tasks = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")
        has_hand_landmarker = has_tasks and hasattr(mp.tasks.vision, "HandLandmarker")

        if not has_hand_landmarker:
            print("❌ MediaPipe Tasks HandLandmarker API not available")
            return False

        print(f"✓ MediaPipe installed")
        print("✓ HandLandmarker Tasks API available")
        return True
    except ImportError:
        print("❌ MediaPipe not installed")
        return False

def check_pynput():
    """Check if pynput is installed"""
    print("\nChecking pynput...")
    try:
        from pynput import mouse
        print(f"✓ pynput installed")
        return True
    except ImportError:
        print("❌ pynput not installed")
        return False

def check_camera():
    """Check if camera is available"""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected on index 0")
            cap.release()
            return True
        else:
            print("⚠ No camera detected on index 0")
            print("  Note: This might be normal if using DroidCam")
            return True
    except Exception as e:
        print(f"⚠ Camera check failed: {e}")
        return True

def install_requirements():
    """Install required packages"""
    print("\n" + "="*50)
    print("Installing requirements...")
    print("="*50)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ requirements.txt not found at {requirements_file}")
        return False
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✓ All requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_diagnostics():
    """Run full diagnostic"""
    print("\n" + "="*50)
    print("HAND GESTURE MOUSE CONTROLLER - DIAGNOSTICS")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version),
        ("OpenCV", check_opencv),
        ("MediaPipe", check_mediapipe),
        ("pynput", check_pynput),
        ("Camera", check_camera),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result and name != "Camera":
            all_passed = False
    
    return all_passed

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup and diagnostic tool for Hand Gesture Mouse Controller"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install requirements"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostics"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full setup (diagnose + install if needed)"
    )
    
    args = parser.parse_args()
    
    if args.full or (not args.install and not args.diagnose):
        # Default: run diagnostics
        if not run_diagnostics():
            print("\n⚠ Some checks failed. Attempting to install requirements...")
            if install_requirements():
                print("\n✓ Setup complete! Run: python gesture_mouse_control.py")
            else:
                print("\n❌ Setup failed. Please install requirements manually:")
                print("   pip install -r requirements.txt")
        else:
            print("\n✓ All checks passed! Ready to run.")
            print("   Run: python gesture_mouse_control.py")
    
    elif args.diagnose:
        run_diagnostics()
    
    elif args.install:
        if check_python_version():
            install_requirements()
        else:
            print("❌ Python version check failed. Please upgrade Python to 3.8+")

if __name__ == "__main__":
    main()
