#!/bin/bash
# =============================================================================
# SENTINEL Interior Guardian - CUDA Environment Setup Script
# =============================================================================
# 
# This script installs the required dependencies for the Zero-Copy Pipeline
# theft detection system optimized for NVIDIA GPUs.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - CUDA Toolkit 11.8+ installed
#   - NVIDIA Video Codec SDK (for PyNvVideoCodec)
#
# Usage:
#   chmod +x setup_cuda_environment.sh
#   ./setup_cuda_environment.sh
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}SENTINEL Guardian - CUDA Environment Setup${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check NVIDIA GPU and CUDA availability
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking NVIDIA GPU and CUDA availability...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed.${NC}"
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}WARNING: nvcc not found. CUDA Toolkit may not be properly installed.${NC}"
    echo "Some features may not work without CUDA Toolkit."
else
    echo "CUDA Version:"
    nvcc --version | grep release
fi
echo ""

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Installing Python dependencies...${NC}"

pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA support
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install ultralytics for YOLOv8
pip install ultralytics>=8.2.0

# Install OpenCV with CUDA support (headless for server, regular for desktop)
pip install opencv-python>=4.8.0

# Install numpy
pip install numpy>=1.24.0

echo ""

# -----------------------------------------------------------------------------
# Install PyNvVideoCodec (GPU Video Decoding)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Installing PyNvVideoCodec for GPU video decoding...${NC}"

# PyNvVideoCodec requires NVIDIA Video Codec SDK
# Try to install via pip first
if pip install PyNvVideoCodec 2>/dev/null; then
    echo -e "${GREEN}PyNvVideoCodec installed successfully.${NC}"
else
    echo -e "${YELLOW}WARNING: PyNvVideoCodec installation failed.${NC}"
    echo "This is expected if NVIDIA Video Codec SDK is not installed."
    echo "The system will fall back to threaded CPU capture with GPU push."
    echo ""
    echo "To install PyNvVideoCodec manually:"
    echo "  1. Download NVIDIA Video Codec SDK from:"
    echo "     https://developer.nvidia.com/nvidia-video-codec-sdk"
    echo "  2. Follow the installation instructions in the SDK"
    echo "  3. Run: pip install PyNvVideoCodec"
fi
echo ""

# -----------------------------------------------------------------------------
# Install NVIDIA DALI (Fallback for video decoding)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Installing NVIDIA DALI as fallback...${NC}"

# NVIDIA DALI provides efficient GPU-accelerated data loading
# Detect CUDA version and install matching DALI
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')

if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="118"  # Default to CUDA 11.8 to match PyTorch installation
fi

# DALI package naming: nvidia-dali-cuda{major}{minor}0
DALI_CUDA_VERSION="${CUDA_VERSION:0:2}0"  # e.g., 118 -> 110, 120 -> 120

echo "Detected CUDA version for DALI: $CUDA_VERSION (using nvidia-dali-cuda$DALI_CUDA_VERSION)"

if pip install --extra-index-url https://developer.download.nvidia.com/compute/redist "nvidia-dali-cuda${DALI_CUDA_VERSION}" 2>/dev/null; then
    echo -e "${GREEN}NVIDIA DALI installed successfully.${NC}"
else
    # Try common CUDA versions as fallback
    echo "Trying alternative DALI versions..."
    if pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120 2>/dev/null; then
        echo -e "${GREEN}NVIDIA DALI (CUDA 12.0) installed successfully.${NC}"
    elif pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 2>/dev/null; then
        echo -e "${GREEN}NVIDIA DALI (CUDA 11.0) installed successfully.${NC}"
    else
        echo -e "${YELLOW}WARNING: NVIDIA DALI installation failed.${NC}"
        echo "The system will use threaded CPU capture with GPU push."
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Export YOLOv8 models to TensorRT
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Exporting YOLOv8 models to TensorRT...${NC}"

# Create models directory
mkdir -p models

echo "Downloading and exporting YOLOv8n detection model..."
python3 -c "
from ultralytics import YOLO

# Load YOLOv8n detection model
print('Loading YOLOv8n detection model...')
model_detect = YOLO('yolov8n.pt')

# Export to TensorRT
print('Exporting to TensorRT engine (this may take several minutes)...')
model_detect.export(format='engine', device=0, imgsz=640, half=True)
print('Detection model exported: yolov8n.engine')
"

echo ""
echo "Downloading and exporting YOLOv8n-pose model..."
python3 -c "
from ultralytics import YOLO

# Load YOLOv8n-pose model
print('Loading YOLOv8n-pose model...')
model_pose = YOLO('yolov8n-pose.pt')

# Export to TensorRT
print('Exporting to TensorRT engine (this may take several minutes)...')
model_pose.export(format='engine', device=0, imgsz=640, half=True)
print('Pose model exported: yolov8n-pose.engine')
"

echo ""

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Verifying installation...${NC}"

python3 << 'EOF'
import sys

print("Checking dependencies...")
errors = []

# Check PyTorch
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
    else:
        errors.append("PyTorch CUDA not available")
except ImportError:
    errors.append("PyTorch not installed")

# Check Ultralytics
try:
    from ultralytics import YOLO
    print(f"  [OK] Ultralytics YOLO")
except ImportError:
    errors.append("Ultralytics not installed")

# Check TensorRT models
import os
if os.path.exists("yolov8n.engine"):
    print(f"  [OK] YOLOv8n TensorRT engine")
else:
    errors.append("YOLOv8n TensorRT engine not found")

if os.path.exists("yolov8n-pose.engine"):
    print(f"  [OK] YOLOv8n-pose TensorRT engine")
else:
    errors.append("YOLOv8n-pose TensorRT engine not found")

# Check OpenCV
try:
    import cv2
    print(f"  [OK] OpenCV {cv2.__version__}")
except ImportError:
    errors.append("OpenCV not installed")

# Check NumPy
try:
    import numpy as np
    print(f"  [OK] NumPy {np.__version__}")
except ImportError:
    errors.append("NumPy not installed")

# Check PyNvVideoCodec (optional)
try:
    import PyNvVideoCodec as nvc
    print(f"  [OK] PyNvVideoCodec (Zero-Copy Pipeline enabled)")
except ImportError:
    print(f"  [--] PyNvVideoCodec not available (using fallback)")

# Check NVIDIA DALI (optional)
try:
    import nvidia.dali as dali
    print(f"  [OK] NVIDIA DALI (fallback available)")
except ImportError:
    print(f"  [--] NVIDIA DALI not available")

print("")
if errors:
    print("ERRORS:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("All critical dependencies installed successfully!")
    print("")
    print("To run the theft detection system:")
    print("  python theft_detection_cuda.py --source 0")
    print("")
EOF

echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}=================================================${NC}"
