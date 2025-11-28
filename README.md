# SENTINEL Interior Guardian - Level 2

A high-performance Python application for theft detection and interior monitoring. The system fuses Object Detection, Pose/Hand Estimation, and Facial Recognition into a single pipeline, with optimizations for both Intel (OpenVINO) and NVIDIA (CUDA/TensorRT) hardware.

## Features

- **Object Detection**: YOLOv8-based detection of protected assets (cell phones, laptops, TVs, etc.)
- **Facial Recognition**: Authorize known personnel and detect intruders
- **Hand Tracking**: MediaPipe-based fingertip detection for interaction monitoring
- **Theft Detection**: Triggers alerts when unauthorized persons interact with protected objects
- **Evidence Capture**: Automatically saves annotated snapshots during security events
- **Intel Optimization**: OpenVINO acceleration for Intel Core Ultra processors
- **NVIDIA Optimization**: CUDA Zero-Copy Pipeline with TensorRT for maximum FPS

## Architecture

### Intel/OpenVINO Version (`sentinel_guardian.py`)

The system implements a threaded Producer-Consumer pattern for optimal performance:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Frame Capture  │───▶│   Inference     │───▶│  Display/Alert  │
│    (Producer)   │    │   (Consumer/    │    │    (Consumer)   │
│                 │    │    Producer)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### NVIDIA/CUDA Version (`theft_detection_cuda.py`)

Zero-Copy Pipeline architecture for maximum FPS and low latency:

```
┌───────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Video Source  │───▶│ GPU Decoder  │───▶│ TensorRT Models │───▶│ GPU Tensor   │
│               │    │   (NVDEC)    │    │   (Dual-Model)  │    │    Logic     │
└───────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                           │                                            │
                     No CPU Transfer                              Alert? → CPU
```

**Key Optimizations:**
- **Zero-Copy Decoding**: Video decoded directly on GPU (NVDEC), no CPU-GPU transfer
- **TensorRT Inference**: YOLOv8 models exported to `.engine` format for optimized inference
- **Dual-Model Strategy**: Object detection every 30 frames (stationary) + Pose every frame
- **GPU Tensor Math**: Hitbox intersection calculated using PyTorch on GPU

## Requirements

### Intel/OpenVINO Version
- Python 3.8+
- Intel Core Ultra processor (recommended for OpenVINO optimization)
- Webcam or USB camera

### NVIDIA/CUDA Version
- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- (Optional) NVIDIA Video Codec SDK for Zero-Copy decoding

## Installation

### Standard Installation

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

### NVIDIA CUDA Installation

For the Zero-Copy CUDA pipeline, run the automated setup script:

```bash
chmod +x setup_cuda_environment.sh
./setup_cuda_environment.sh
```

This script will:
1. Verify NVIDIA GPU and CUDA availability
2. Install PyTorch with CUDA support
3. Install PyNvVideoCodec (if Video Codec SDK is available)
4. Install NVIDIA DALI as fallback
5. Export YOLOv8 models to TensorRT format

## Usage

### Intel/OpenVINO Version
```bash
python sentinel_guardian.py
```

### NVIDIA/CUDA Version
```bash
python theft_detection_cuda.py --source 0
```

#### CUDA Command Line Options
```
--source, -s      Video source: camera index (0) or video file path
--device, -d      CUDA device index (default: 0)
--detect-model    Path to object detection TensorRT engine
--pose-model      Path to pose estimation TensorRT engine
--detection-interval  Run object detection every N frames (default: 30)
--no-display      Disable video display (headless mode)
```

### Controls
- **Q** - Quit the application
- **R** - Reset alarm (Intel version)

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

For CUDA version, see `CUDAConfig` in `theft_detection_cuda.py`.

## System Logic

### Authorization Logic
1. When a person is detected, their face is compared against known authorized personnel
2. **Match == True**: System enters "Passive Mode" (logging only)
3. **Match == False**: System enters "Active Defense Mode"

### Theft Detection Logic
1. The system creates a "Hitbox" (Region of Interest) around protected objects
2. Hand/wrist keypoints are tracked using pose estimation
3. **Trigger Condition**: If `Person == Unauthorized` AND `Hand_Landmark_Coordinates` intersect with `Object_Hitbox`:
   - Alert is triggered
   - Evidence snapshot is saved
   - Alarm is activated

### CUDA Dual-Model Optimization
- **Model A (Object Detection)**: Runs every 30 frames since protected objects are stationary
- **Model B (Pose Estimation)**: Runs every frame for real-time hand tracking
- Hitbox coordinates cached between object detection runs

## Evidence

When a security event is detected, evidence is automatically saved to the `evidence/` directory:
- Annotated frames with bounding boxes
- Timestamp and event type in filename
- Events are logged to `sentinel_guardian.log` or `theft_detection_cuda.log`

## Technical Stack

- **Language**: Python 3.x
- **Vision**: OpenCV, Ultralytics YOLOv8
- **Acceleration**: 
  - Intel: OpenVINO Runtime
  - NVIDIA: CUDA, TensorRT, PyNvVideoCodec
- **Face Recognition**: face_recognition library
- **Hand Tracking**: MediaPipe (Intel) / YOLOv8-Pose (NVIDIA)

## License

This project is part of the Safetronics security system suite.