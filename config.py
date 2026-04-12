"""
Configuration settings for Hand Gesture Mouse Controller
"""

# Screen settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Hand detection settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# Mouse control settings
SMOOTHING_FACTOR = 0.7  # 0-1, higher = smoother but slower response
CLICK_THRESHOLD = 0.08  # Distance threshold for pinch click
GESTURE_DELAY = 0.2  # Minimum delay between gesture actions (seconds)
PALM_CONTROL_GAIN = 1.8  # Higher = more screen coverage from the same hand motion

# Camera settings
DEFAULT_CAMERA_INDEX = 0  # 0 for built-in, 1+ for DroidCam or external
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Gesture sensitivity
PINCH_SENSITIVITY = 0.08  # How close fingers need to be for click
POINT_DEADZONE = 50  # Pixels of movement required to register

# Enable/disable features
ENABLE_LEFT_CLICK = True
ENABLE_RIGHT_CLICK = True
ENABLE_SCROLL = False  # Experimental
ENABLE_DRAG = False    # Experimental

# Display settings
SHOW_HAND_SKELETON = True
SHOW_GESTURE_NAME = True
SHOW_MOUSE_POSITION = True
SHOW_FPS = True
