"""
MediaPipe compatibility layer for Python 3.13 using the Tasks API.

This module replaces legacy mp.solutions.hands usage with
mediapipe.tasks.vision.HandLandmarker.
"""

from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class HandTracker:
    """Hand tracker backed by MediaPipe Tasks HandLandmarker."""

    def __init__(
        self,
        num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        min_presence_confidence=0.5,
        model_path=None,
    ):
        model_file = self._ensure_model(model_path)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _ensure_model(self, model_path=None):
        """Ensure hand_landmarker.task exists locally, downloading if needed."""
        if model_path:
            resolved = Path(model_path)
            if not resolved.exists():
                raise FileNotFoundError(f"Model file not found: {resolved}")
            return resolved

        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "hand_landmarker.task"

        if not model_file.exists():
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(model_file))

        return model_file

    def detect(self, frame_bgr):
        """Return a list of detected hand landmark lists."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.monotonic() * 1000)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks if result.hand_landmarks else []

    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0)):
        """Draw landmarks and hand connections on an OpenCV frame."""
        height, width, _ = frame.shape

        for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    def close(self):
        """Release model resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
