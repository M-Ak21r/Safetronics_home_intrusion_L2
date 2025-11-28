# SENTINEL Interior Guardian - Level 2

A high-performance Python application for theft detection and interior monitoring. The system fuses Object Detection, Pose/Hand Estimation, and Facial Recognition into a single pipeline, optimized for Intel hardware using OpenVINO.

## Features

- **Object Detection**: YOLOv8-based detection of protected assets (cell phones, laptops, TVs, etc.)
- **Facial Recognition**: Authorize known personnel and detect intruders
- **Hand Tracking**: MediaPipe-based fingertip detection for interaction monitoring
- **Theft Detection**: Triggers alerts when unauthorized persons interact with protected objects
- **Evidence Capture**: Automatically saves annotated snapshots during security events
- **Intel Optimization**: OpenVINO acceleration for Intel Core Ultra processors

## Architecture

The system implements a threaded Producer-Consumer pattern for optimal performance:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Frame Capture  │───▶│   Inference     │───▶│  Display/Alert  │
│    (Producer)   │    │   (Consumer/    │    │    (Consumer)   │
│                 │    │    Producer)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Requirements

- Python 3.8+
- Intel Core Ultra processor (recommended for OpenVINO optimization)
- Webcam or USB camera

## Installation

1. Clone the repository:
```bash
git clone https://github.com/M-Ak21r/Safetronics_home_intrusion_L2.git
cd Safetronics_home_intrusion_L2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add authorized personnel images to `authorized_personels/` directory (see [Authorized Personnel](#authorized-personnel))

## Usage

Run the SENTINEL Guardian:
```bash
python sentinel_guardian.py
```

### Controls
- **Q** - Quit the application
- **R** - Reset alarm

## Authorized Personnel

Add images of authorized personnel to the `authorized_personels/` directory:

```
authorized_personels/
├── john_smith.jpg
├── jane_doe.png
└── security_guard.jpg
```

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- The filename (without extension) will be used as the person's display name
- Use clear, front-facing photos for best recognition accuracy

## Configuration

Key configuration options can be modified in the `Config` class in `sentinel_guardian.py`:

```python
class Config:
    # Protected object classes
    PROTECTED_CLASSES = ['cell phone', 'laptop', 'tv', 'remote', 'keyboard', 'mouse']
    
    # Confidence thresholds
    OBJECT_CONFIDENCE_THRESHOLD = 0.5
    PERSON_CONFIDENCE_THRESHOLD = 0.5
    FACE_RECOGNITION_TOLERANCE = 0.6
    
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
```

## System Logic

### Authorization Logic
1. When a person is detected, their face is compared against known authorized personnel
2. **Match == True**: System enters "Passive Mode" (logging only)
3. **Match == False**: System enters "Active Defense Mode"

### Theft Detection Logic
1. The system creates a "Hitbox" (Region of Interest) around protected objects
2. Hand landmarks (fingertips) are tracked using MediaPipe
3. **Trigger Condition**: If `Person == Unauthorized` AND `Hand_Landmark_Coordinates` intersect with `Object_Hitbox`:
   - Alert is triggered
   - Evidence snapshot is saved
   - Alarm is activated

## Evidence

When a security event is detected, evidence is automatically saved to the `evidence/` directory:
- Annotated frames with bounding boxes
- Timestamp and event type in filename
- Events are logged to `sentinel_guardian.log`

## Technical Stack

- **Language**: Python 3.x
- **Vision**: OpenCV, Ultralytics YOLOv8, MediaPipe
- **Acceleration**: Intel OpenVINO Runtime
- **Face Recognition**: face_recognition library

## License

This project is part of the Safetronics security system suite.