# Quick Start Guide - Hand Gesture Mouse Controller with DroidCam

## 5-Minute Setup

### Step 1: Install Python Dependencies (2 min)

```bash
# Navigate to project folder
cd ELEC4544-Project

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Setup DroidCam (2 min)

#### On Your Android Phone:
1. Download **DroidCam** from Google Play Store or from https://www.dev47apps.com/droidcam/
2. Install the app
3. Open DroidCam
4. Note the IP address and port shown on screen
5. Allow permission to access camera

#### On Your PC:
1. Download **DroidCam Client** from https://www.dev47apps.com/droidcam/
2. Install the client
3. Open DroidCam Client on PC
4. Enter the IP address from your phone
5. Click "Connect" or "Start"
6. Verify camera feed appears

### Step 3: Run the Application (1 min)

```bash
# Basic version
python gesture_mouse_control.py

# OR Advanced version with more gestures
python gesture_mouse_advanced.py
```

### Step 4: Select Camera

1. Press **C** to change camera
2. Enter **1** (for DroidCam usually camera index 1)
3. You should see your phone's camera feed

## Hand Gestures Quick Reference

| Gesture | How to Do It | Action |
|---------|------------|--------|
| **PALM** | Open hand | Move mouse cursor |
| **POINT** | Index finger only | Point mode |
| **PINCH INDEX** | Touch thumb to index tip | Left click |
| **PEACE** | Index + middle up | Right click mode |
| **THUMBS UP** | Thumb pointing up, fingers closed | Double click |
| **FIST** | All fingers closed | Stop/Pause |

## Keyboard Shortcuts

- **Q** - Quit application
- **C** - Change camera
- **P** - Pause/Resume (advanced version only)
- **R** - Reset mouse position
- **S** - Take screenshot (advanced version only)

## Troubleshooting Quick Fixes

### "No hand detected"
- ✓ Ensure hand is fully visible
- ✓ Increase lighting
- ✓ Move hand closer (30-80cm from camera)

### "DroidCam won't connect"
- ✓ Ensure phone and PC are on same WiFi network
- ✓ Restart DroidCam application
- ✓ Check IP address matches between phone and PC
- ✓ Try USB connection instead (requires USB cable)

### "Clicks not working"
- ✓ Make pinch gesture more pronounced
- ✓ Decrease `CLICK_THRESHOLD` in `config.py` (try 0.03)
- ✓ Hold pinch longer (1-2 seconds)

### "Cursor movement is jerky"
- ✓ Increase `SMOOTHING_FACTOR` in `config.py` (try 0.85)
- ✓ Improve lighting conditions
- ✓ Move hand more slowly

### "Camera not found"
- ✓ Check camera index by running `python setup.py --diagnose`
- ✓ For DroidCam, try different indices: 1, 2, or 3
- ✓ Restart DroidCam client

## Camera Index Guide

Different systems and setups use different camera indices:

```
Index 0   - Usually built-in webcam
Index 1   - Usually external camera or DroidCam (common)
Index 1+  - Other external cameras or virtual cameras

If DroidCam doesn't work on index 1, try: 2, 3, 4, etc.
```

## Optimization Tips

### For Better Hand Detection:
1. **Lighting** - Position light source in front of your hand
2. **Background** - Use plain/neutral background
3. **Contrast** - Wear contrasting clothing (dark hand on light background)
4. **Distance** - Keep hand 40-60cm from camera

### For Smoother Movement:
1. Edit `config.py`:
   ```python
   SMOOTHING_FACTOR = 0.85  # Increase for smoother (0.5-0.95)
   ```

### For Faster Response:
1. Edit `config.py`:
   ```python
   SMOOTHING_FACTOR = 0.5   # Decrease for faster response
   GESTURE_DELAY = 0.1      # Reduce gesture detection delay
   ```

## File Reference

```
gesture_mouse_control.py       - Main application (basic)
gesture_mouse_advanced.py      - Advanced version (more features)
config.py                      - Configuration settings
gesture_utils.py              - Utility functions and gesture library
setup.py                      - Setup and diagnostics
requirements.txt              - Python dependencies
README.md                     - Full documentation
QUICKSTART.md                 - This file
```

## Next Steps

1. **Experiment with Gestures** - Try different hand positions to find what works best
2. **Fine-tune Sensitivity** - Adjust thresholds in `config.py` for your environment
3. **Explore Advanced Features** - Try `gesture_mouse_advanced.py` for more gestures
4. **Customize** - Modify code to add your own gestures or actions

## Performance Notes

- **CPU Usage**: Normal 10-20%, high CPU = reduce resolution in config.py
- **Latency**: ~100-200ms typical (depends on camera and PC performance)
- **FPS**: Target 30+ FPS for smooth tracking

## USB Connection Setup (Alternative to WiFi)

1. Connect Android phone to PC with USB cable
2. Enable USB debugging on phone (Settings > Developer Options)
3. Open DroidCam Client on PC
4. Select "USB" option instead of WiFi
5. Click "Connect"
6. Use camera index 1 or 2

## Tips & Tricks

**Pro Tip 1**: Mount your phone at eye level for best ergonomics
**Pro Tip 2**: Keep a solid background behind your hand
**Pro Tip 3**: Practice the gestures slowly first, then speed up
**Pro Tip 4**: Use POINT gesture for precision clicking on small targets
**Pro Tip 5**: Take breaks to avoid hand fatigue

---

**Enjoy hands-free computing!**

For detailed documentation, see README.md
