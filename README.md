# Hand Gesture Mouse Controller

Control your computer mouse using hand gestures from a webcam or DroidCam video feed.

## Features

- **Hand pose detection** using MediaPipe for robust gesture recognition
- **Mouse movement** tracking based on hand position
- **Gesture-based clicks**: Pinch gesture for left/right clicks
- **Smooth cursor movement** with configurable smoothing factor
- **Real-time feedback** showing detected gestures and cursor position
- **Camera switching** - easily switch between built-in camera and DroidCam
- **Customizable sensitivity** for different hand sizes and preferences

## Requirements

- Python 3.8+
- Webcam or DroidCam for Android
- Windows, Mac, or Linux

## Installation

### 1. Set up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install DroidCam

- Download DroidCam from: https://www.dev47apps.com/
- Install on your Android phone
- Install DroidCam client on your PC
- Run DroidCam to stream your phone camera

## Usage

### Basic Usage

```bash
python gesture_mouse_control.py
```

### Gesture Controls

| Gesture | Action |
|---------|--------|
| **PALM** (all fingers open) | Move mouse cursor |
| **POINT** (only index finger up) | Point mode (cursor follows fingertip) |
| **PINCH INDEX** (thumb + index) | Left click |
| **PINCH MIDDLE** (thumb + middle) | Right click |
| **FIST** (all fingers closed) | Stop/standby mode |

### Keyboard Controls

- **Q** - Quit application
- **C** - Change camera (switch between built-in and DroidCam)
- **R** - Reset mouse smoothing

## Configuration

Edit `config.py` to customize:

- **SMOOTHING_FACTOR** - Adjust cursor smoothness (0.5-1.0)
- **CLICK_THRESHOLD** - Adjust pinch click sensitivity
- **SCREEN_WIDTH/HEIGHT** - Set to your monitor resolution
- **MIN_DETECTION_CONFIDENCE** - Hand detection confidence level (0.5-1.0)

## Using with DroidCam

1. **Start DroidCam on your phone** with WiFi or USB connection
2. **Open DroidCam client on PC** - note which camera index it shows (usually index 1)
3. **Run the application**
4. **Press 'C'** to change camera and enter the DroidCam index
5. Alternatively, modify `config.py` and set `DEFAULT_CAMERA_INDEX = 1`

## Advanced Features

### Adjust Sensitivity

To make clicks easier:
- Increase `CLICK_THRESHOLD` in config.py (larger = easier to trigger)
- Increase `MIN_DETECTION_CONFIDENCE` for more stable detection

To improve tracking:
- Increase `SMOOTHING_FACTOR` for smoother movement (0.8-0.95)
- Ensure adequate lighting

### Camera Selection

```python
# In gesture_mouse_control.py
self.camera_index = 0  # 0 = built-in, 1+ = external/DroidCam
```

## Troubleshooting

### Hand not detected
- Ensure adequate lighting (face the light source)
- Keep hand visible in frame
- Make sure background is clear and uncluttered

### Cursor movement is jerky
- Increase `SMOOTHING_FACTOR` in config.py
- Reduce sudden hand movements
- Ensure stable frame rate (check FPS display)

### Clicks not working
- Make the pinch stronger (bring thumb and finger closer)
- Increase `CLICK_THRESHOLD` value
- Ensure detection confidence is high (watch gesture feedback)

### How clicking works
- Basic mode: pinch thumb + index for a left click, or pinch thumb + middle for a right click.
- Advanced mode: use `POINT` + pinch for left click, or `PEACE` + pinch for right click.
- If clicks are hard to trigger, raise `CLICK_THRESHOLD` in `config.py` a bit more.

### DroidCam not connecting
- Verify DroidCam client is running on PC
- Check that phone is on same network
- Restart DroidCam if connection drops
- Try USB connection if WiFi is unstable

### High CPU usage
- Reduce `CAMERA_FPS` in config.py
- Use lower `CAMERA_WIDTH` and `CAMERA_HEIGHT`
- Run on machine with better GPU support for MediaPipe

## Performance Tips

1. **Good Lighting** - Bright, consistent lighting improves detection
2. **Contrast** - Avoid wearing colors that match your background
3. **Distance** - Keep hand 30-100cm from camera
4. **Single Hand** - The app works best with one hand visible
5. **Quick Movements** - Smooth, deliberate gestures work better than jerky ones

## File Structure

```
├── gesture_mouse_control.py    # Main application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Technical Details

### Hand Tracking
- Uses MediaPipe Hands for ML-based hand detection
- Detects 21 hand landmarks in real-time
- Supports single hand tracking

### Gesture Recognition
- Custom gesture classifier based on finger positions
- Real-time classification with smoothing
- Configurable sensitivity thresholds

### Mouse Control
- Uses pynput library for cross-platform mouse control
- Exponential smoothing for natural cursor movement
- Screen coordinate mapping for multi-monitor support

## Limitations

- Single hand tracking only
- Requires visible hand in camera frame
- Performance depends on computer hardware
- Gesture recognition accuracy varies with lighting
- Cannot perform certain complex gestures

## Future Enhancements

- [ ] Multi-hand support
- [ ] Scroll wheel control
- [ ] Drag and drop functionality
- [ ] Keyboard input gestures
- [ ] Custom gesture training
- [ ] Machine learning model optimization
- [ ] GPU acceleration improvements

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Ensure all dependencies are installed correctly
3. Test with the built-in webcam first before trying DroidCam
4. Verify Python version is 3.8 or higher

---

**Enjoy hands-free mouse control!**
