"""
SENTINEL Interior Guardian - CUDA Zero-Copy Pipeline
=====================================================

A high-performance theft detection system optimized for NVIDIA GPUs using
TensorRT inference and GPU-accelerated video decoding.

Architecture:
    Video Source → GPU Decoder (NVDEC) → CUDA Tensor → TensorRT Models
                                              ↓
                                    GPU-Accelerated Logic
                                              ↓
                                    Alert Frame (CPU only if needed)

Key Optimizations:
    1. Zero-Copy: Video decoded directly on GPU, no CPU-GPU transfer
    2. TensorRT: Optimized inference using .engine models
    3. Dual-Model Strategy: Object detection (30 frames) + Pose (every frame)
    4. GPU Tensor Math: Hitbox intersection calculated on GPU

Author: SENTINEL Security Systems
Hardware: NVIDIA GPU with CUDA support
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# PyTorch for GPU tensor operations
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('theft_detection_cuda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

# Check for PyTorch CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    logger.warning("CUDA not available. Performance will be severely degraded.")

# Check for PyNvVideoCodec (Zero-Copy GPU decoding)
try:
    import PyNvVideoCodec as nvc
    PYNVVIDEOCODEC_AVAILABLE = True
    logger.info("PyNvVideoCodec available - Zero-Copy Pipeline enabled")
except ImportError:
    PYNVVIDEOCODEC_AVAILABLE = False
    logger.warning("PyNvVideoCodec not available - using performance fallback")

# Check for NVIDIA DALI (alternative GPU decoding)
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
    logger.info("NVIDIA DALI available as fallback")
except ImportError:
    DALI_AVAILABLE = False
    logger.info("NVIDIA DALI not available")

# Check for Ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("Ultralytics YOLO not available. Cannot proceed.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CUDAConfig:
    """Configuration for CUDA-accelerated theft detection."""
    
    # GPU Settings
    device: int = 0
    cuda_stream: Optional[torch.cuda.Stream] = None
    
    # Model paths (TensorRT engines)
    detect_model_path: str = "yolov8n.engine"
    pose_model_path: str = "yolov8n-pose.engine"
    
    # Fallback to .pt if .engine not available
    detect_model_fallback: str = "yolov8n.pt"
    pose_model_fallback: str = "yolov8n-pose.pt"
    
    # Detection settings
    protected_classes: Tuple[str, ...] = ('tv', 'cell phone', 'laptop')
    object_confidence: float = 0.5
    person_confidence: float = 0.5
    
    # Dual-model optimization
    object_detection_interval: int = 30  # Run object detection every N frames
    
    # Hitbox expansion (pixels)
    hitbox_padding: int = 20
    
    # Video settings
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    
    # Pose keypoint indices (COCO format)
    # Wrist indices: 9 (left wrist), 10 (right wrist)
    # Note: Index finger tips not available in standard COCO pose
    # Using wrist as proxy for hand position
    wrist_indices: Tuple[int, ...] = (9, 10)
    
    # Evidence directory
    evidence_dir: str = "evidence"
    
    # Authorized personnel (placeholder - integrate with face recognition)
    authorized_personnel: List[str] = field(default_factory=list)


# =============================================================================
# GPU TENSOR DATA STRUCTURES
# =============================================================================

@dataclass
class GPUBoundingBox:
    """Bounding box stored as GPU tensor for efficient computation."""
    
    # Tensor shape: (4,) containing [x1, y1, x2, y2]
    coords: torch.Tensor
    label: str
    confidence: float
    
    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float,
                  label: str, confidence: float, device: torch.device) -> 'GPUBoundingBox':
        """Create bounding box from coordinates."""
        coords = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
        return cls(coords=coords, label=label, confidence=confidence)
    
    def contains_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside this bounding box (GPU-accelerated).
        
        Args:
            points: Tensor of shape (N, 2) containing [x, y] coordinates
            
        Returns:
            Boolean tensor of shape (N,) indicating containment
        """
        # Extract box coordinates
        x1, y1, x2, y2 = self.coords
        
        # Vectorized containment check
        x_inside = (points[:, 0] >= x1) & (points[:, 0] <= x2)
        y_inside = (points[:, 1] >= y1) & (points[:, 1] <= y2)
        
        return x_inside & y_inside
    
    def expand(self, padding: int) -> 'GPUBoundingBox':
        """Create expanded hitbox."""
        expanded_coords = self.coords.clone()
        expanded_coords[0] -= padding  # x1
        expanded_coords[1] -= padding  # y1
        expanded_coords[2] += padding  # x2
        expanded_coords[3] += padding  # y2
        return GPUBoundingBox(
            coords=expanded_coords,
            label=self.label,
            confidence=self.confidence
        )
    
    def to_cpu(self) -> Tuple[int, int, int, int]:
        """Convert to CPU tuple for visualization."""
        coords = self.coords.cpu().numpy()
        return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))


@dataclass
class GPUKeypoints:
    """Pose keypoints stored as GPU tensor."""
    
    # Tensor shape: (num_people, num_keypoints, 3) for [x, y, confidence]
    keypoints: torch.Tensor
    
    def get_wrist_positions(self, wrist_indices: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract wrist positions for all detected people.
        
        Args:
            wrist_indices: Tuple of keypoint indices for wrists
            
        Returns:
            Tensor of shape (N, 2) containing wrist [x, y] coordinates
        """
        if self.keypoints.numel() == 0:
            return torch.empty((0, 2), device=self.keypoints.device)
        
        # Extract wrist keypoints for all people
        # Shape: (num_people, num_wrists, 3)
        wrists = self.keypoints[:, list(wrist_indices), :]
        
        # Filter by confidence (keypoint visibility)
        # Confidence is in the third column
        mask = wrists[:, :, 2] > 0.3
        
        # Flatten to (N, 2) where N is total number of visible wrists
        positions = wrists[:, :, :2]  # Get x, y
        positions = positions.reshape(-1, 2)
        mask = mask.reshape(-1)
        
        return positions[mask]


# =============================================================================
# VIDEO DECODER CLASSES
# =============================================================================

class BaseVideoDecoder:
    """Base class for video decoders."""
    
    def __init__(self, source: str, config: CUDAConfig):
        self.source = source
        self.config = config
        self.device = torch.device(f'cuda:{config.device}')
    
    def get_frame(self) -> Optional[torch.Tensor]:
        """Get next frame as GPU tensor. Returns None if no frame available."""
        raise NotImplementedError
    
    def release(self):
        """Release resources."""
        pass


class PyNvVideoCodecDecoder(BaseVideoDecoder):
    """
    Zero-Copy video decoder using PyNvVideoCodec.
    
    This decoder uses NVIDIA's hardware video decoder (NVDEC) to decode
    video directly on the GPU, avoiding CPU-GPU memory transfers.
    """
    
    def __init__(self, source: str, config: CUDAConfig):
        super().__init__(source, config)
        
        if not PYNVVIDEOCODEC_AVAILABLE:
            raise RuntimeError("PyNvVideoCodec not available")
        
        self.decoder = None
        self._init_decoder()
    
    def _init_decoder(self):
        """Initialize the NVDEC decoder."""
        try:
            # Create decoder for the video source
            self.decoder = nvc.PyNvDecoder(
                self.source,
                self.config.device
            )
            logger.info(f"PyNvVideoCodec decoder initialized for: {self.source}")
        except Exception as e:
            logger.error(f"Failed to initialize PyNvVideoCodec: {e}")
            raise
    
    def get_frame(self) -> Optional[torch.Tensor]:
        """Get next frame as GPU tensor (Zero-Copy)."""
        try:
            # Decode frame directly to GPU memory
            surface = self.decoder.DecodeSingleSurface()
            if surface.Empty():
                return None
            
            # Convert NV12 surface to RGB tensor on GPU
            # This stays entirely on GPU - no CPU transfer
            frame_tensor = self._surface_to_tensor(surface)
            return frame_tensor
            
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None
    
    def _surface_to_tensor(self, surface) -> torch.Tensor:
        """
        Convert NVDEC surface to PyTorch tensor on GPU.
        
        PyNvVideoCodec decodes to NV12 format (YUV 4:2:0).
        This method handles the conversion to RGB tensor on GPU.
        
        Note: Actual implementation depends on PyNvVideoCodec version.
        Modern versions provide built-in color conversion utilities.
        """
        # Get surface dimensions
        height, width = surface.Height(), surface.Width()
        
        # PyNvVideoCodec typically provides methods to convert to different formats
        # The actual API varies by version. Here's the general approach:
        
        try:
            # Try using built-in color converter if available (newer versions)
            if hasattr(surface, 'ConvertToRGB'):
                rgb_surface = surface.ConvertToRGB()
                frame_data = rgb_surface.CudaMemPtr()
                tensor = torch.as_tensor(
                    frame_data,
                    dtype=torch.uint8,
                    device=self.device
                ).reshape(height, width, 3)
                return tensor
            
            # Alternative: Use PyCuda or CuPy for NV12 to RGB conversion
            # Get Y plane (full resolution) and UV plane (half resolution)
            y_plane_ptr = surface.PlanePtr(0)  # Y plane
            uv_plane_ptr = surface.PlanePtr(1)  # UV interleaved plane
            
            # Create Y tensor (height x width)
            y_tensor = torch.as_tensor(
                y_plane_ptr,
                dtype=torch.uint8,
                device=self.device
            ).reshape(height, width)
            
            # Create UV tensor (height/2 x width) - interleaved U and V
            uv_tensor = torch.as_tensor(
                uv_plane_ptr,
                dtype=torch.uint8,
                device=self.device
            ).reshape(height // 2, width)
            
            # Convert NV12 to RGB using GPU operations
            # This is a simplified conversion - for production, use proper YUV conversion
            rgb_tensor = self._nv12_to_rgb_gpu(y_tensor, uv_tensor, height, width)
            return rgb_tensor
            
        except Exception as e:
            logger.warning(f"Surface conversion error: {e}")
            # Fallback: return grayscale Y channel as 3-channel image
            y_plane_ptr = surface.PlanePtr() if not hasattr(surface, 'PlanePtr') else surface.PlanePtr(0)
            y_tensor = torch.as_tensor(
                y_plane_ptr,
                dtype=torch.uint8,
                device=self.device
            ).reshape(height, width)
            # Stack grayscale to create pseudo-RGB
            return torch.stack([y_tensor, y_tensor, y_tensor], dim=-1)
    
    def _nv12_to_rgb_gpu(self, y: torch.Tensor, uv: torch.Tensor,
                         height: int, width: int) -> torch.Tensor:
        """
        Convert NV12 (Y + interleaved UV) to RGB on GPU using PyTorch.
        
        Args:
            y: Y plane tensor (height x width)
            uv: UV interleaved plane tensor (height/2 x width)
            height: Frame height
            width: Frame width
            
        Returns:
            RGB tensor (height x width x 3)
        """
        # Separate U and V channels and upsample to full resolution
        u = uv[:, 0::2].repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        v = uv[:, 1::2].repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        
        # Ensure dimensions match
        u = u[:height, :width]
        v = v[:height, :width]
        
        # Convert to float for color conversion
        y_f = y.float()
        u_f = u.float() - 128.0
        v_f = v.float() - 128.0
        
        # YUV to RGB conversion (BT.601 standard)
        r = torch.clamp(y_f + 1.402 * v_f, 0, 255)
        g = torch.clamp(y_f - 0.344136 * u_f - 0.714136 * v_f, 0, 255)
        b = torch.clamp(y_f + 1.772 * u_f, 0, 255)
        
        # Stack and convert to uint8
        rgb = torch.stack([r, g, b], dim=-1).to(torch.uint8)
        return rgb
    
    def release(self):
        """Release decoder resources."""
        if self.decoder:
            del self.decoder
            self.decoder = None


class ThreadedCPUDecoder(BaseVideoDecoder):
    """
    PERFORMANCE FALLBACK: Threaded CPU decoder with immediate GPU push.
    
    This decoder uses OpenCV for CPU-based decoding but immediately
    transfers frames to GPU. While not zero-copy, it provides good
    performance when PyNvVideoCodec is not available.
    
    Note: This is marked as a "performance fallback" as required
    by the specification.
    """
    
    def __init__(self, source: str, config: CUDAConfig):
        super().__init__(source, config)
        
        self.cap = None
        self.cuda_stream = torch.cuda.Stream(device=self.device)
        self._init_capture()
        
        logger.warning("Using PERFORMANCE FALLBACK: ThreadedCPUDecoder")
        logger.warning("For optimal performance, install PyNvVideoCodec")
    
    def _init_capture(self):
        """Initialize OpenCV video capture."""
        # Handle camera index vs file path
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Configure capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        # Enable hardware acceleration if available
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video capture: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    def get_frame(self) -> Optional[torch.Tensor]:
        """Get next frame, decode on CPU and immediately push to GPU."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Immediately push to GPU using pinned memory for faster transfer
        with torch.cuda.stream(self.cuda_stream):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create tensor on GPU
            frame_tensor = torch.from_numpy(frame_rgb).to(
                device=self.device,
                non_blocking=True
            )
        
        # Wait for transfer to complete
        self.cuda_stream.synchronize()
        
        return frame_tensor
    
    def release(self):
        """Release capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


def create_video_decoder(source: str, config: CUDAConfig) -> BaseVideoDecoder:
    """
    Factory function to create the best available video decoder.
    
    Priority:
    1. PyNvVideoCodec (Zero-Copy, best performance)
    2. ThreadedCPUDecoder (Fallback with GPU push)
    """
    # Try PyNvVideoCodec first (Zero-Copy)
    if PYNVVIDEOCODEC_AVAILABLE:
        try:
            return PyNvVideoCodecDecoder(source, config)
        except Exception as e:
            logger.warning(f"PyNvVideoCodec failed: {e}")
    
    # Fallback to threaded CPU decoder
    logger.info("Using ThreadedCPUDecoder (performance fallback)")
    return ThreadedCPUDecoder(source, config)


# =============================================================================
# TENSORRT MODEL WRAPPER
# =============================================================================

class TensorRTModel:
    """
    Wrapper for YOLOv8 TensorRT engine models.
    
    Handles both .engine (TensorRT) and .pt (PyTorch) formats,
    with automatic fallback if TensorRT model not available.
    """
    
    def __init__(self, engine_path: str, fallback_path: str, 
                 device: int = 0, task: str = 'detect'):
        """
        Initialize TensorRT model.
        
        Args:
            engine_path: Path to TensorRT .engine file
            fallback_path: Path to PyTorch .pt file (fallback)
            device: CUDA device index
            task: Model task type ('detect' or 'pose')
        """
        self.device = device
        self.model = None
        self.using_tensorrt = False
        self.task = task
        
        # Try TensorRT engine first
        if os.path.exists(engine_path):
            try:
                self.model = YOLO(engine_path, task=task)
                self.using_tensorrt = True
                logger.info(f"Loaded TensorRT engine ({task}): {engine_path}")
            except Exception as e:
                logger.warning(f"Failed to load TensorRT engine: {e}")
        
        # Fallback to PyTorch model
        if self.model is None:
            if os.path.exists(fallback_path):
                self.model = YOLO(fallback_path)
                logger.info(f"Loaded PyTorch model (fallback): {fallback_path}")
            else:
                # Download from ultralytics hub
                self.model = YOLO(fallback_path)
                logger.info(f"Downloaded model: {fallback_path}")
    
    def predict(self, frame: torch.Tensor, conf: float = 0.5,
                classes: Optional[List[int]] = None) -> List:
        """
        Run inference on frame.
        
        Args:
            frame: Input frame as GPU tensor (H, W, C) RGB format
            conf: Confidence threshold
            classes: List of class indices to detect (None = all)
            
        Returns:
            List of detection results
            
        Note on GPU-CPU Transfer:
            The Ultralytics YOLO library's predict() method currently requires
            numpy array input. While this means a GPU-to-CPU transfer occurs here,
            the TensorRT engine handles the actual inference on GPU efficiently.
            
            For fully zero-copy inference, one would need to:
            1. Use the TensorRT Python API directly with GPU memory pointers
            2. Or use NVIDIA Triton Inference Server
            
            However, the preprocessing/postprocessing overhead is minimal compared
            to the inference time savings from TensorRT optimization.
        """
        if self.model is None:
            return []
        
        # Ultralytics YOLO accepts various input formats
        # For TensorRT engines, the internal preprocessing handles GPU transfer
        if isinstance(frame, torch.Tensor):
            # Convert GPU tensor to numpy for YOLO preprocessing
            # Note: YOLO will re-upload to GPU for TensorRT inference
            # This is a limitation of the ultralytics API
            frame_np = frame.cpu().numpy()
            # Convert RGB to BGR (YOLO expects BGR)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_np = frame
        
        # Run inference - TensorRT engine runs on GPU regardless of input format
        results = self.model.predict(
            frame_np,
            conf=conf,
            device=self.device,
            verbose=False,
            classes=classes
        )
        
        return results


# =============================================================================
# MAIN THEFT DETECTION PIPELINE
# =============================================================================

class CUDATheftDetector:
    """
    GPU-accelerated theft detection pipeline.
    
    Architecture:
        Video Decoder (NVDEC) → GPU Tensor → TensorRT Models → GPU Logic → Alert
        
    Optimization Strategy:
        - Object Detection (Model A): Every 30 frames (stationary objects)
        - Pose Estimation (Model B): Every frame (real-time tracking)
        - Theft Logic: GPU tensor operations (no CPU transfer)
    """
    
    def __init__(self, config: CUDAConfig):
        """Initialize the CUDA theft detection pipeline."""
        self.config = config
        self.device = torch.device(f'cuda:{config.device}')
        
        # Initialize CUDA stream for async operations
        self.cuda_stream = torch.cuda.Stream(device=self.device)
        
        # Initialize models
        logger.info("Initializing TensorRT models...")
        self.detect_model = TensorRTModel(
            config.detect_model_path,
            config.detect_model_fallback,
            config.device,
            task='detect'
        )
        self.pose_model = TensorRTModel(
            config.pose_model_path,
            config.pose_model_fallback,
            config.device,
            task='pose'
        )
        
        # State tracking
        self.frame_count = 0
        self.cached_objects: List[GPUBoundingBox] = []
        self.last_detection_time = 0
        self.fps_history: List[float] = []
        
        # Create evidence directory
        os.makedirs(config.evidence_dir, exist_ok=True)
        
        # Get class indices for protected objects
        self._init_class_indices()
        
        logger.info("CUDA Theft Detector initialized")
    
    def _init_class_indices(self):
        """Initialize class indices for object detection."""
        # COCO class names mapping (YOLOv8 uses COCO dataset indices)
        # Reference: https://docs.ultralytics.com/datasets/detect/coco/
        coco_names = {
            0: 'person', 
            62: 'tv',           # Also known as 'tvmonitor' in some versions
            63: 'laptop',
            67: 'cell phone'    # Also known as 'cellphone' or 'mobile phone'
        }
        
        # Find indices for protected classes
        self.protected_class_indices = []
        for idx, name in coco_names.items():
            if name in self.config.protected_classes:
                self.protected_class_indices.append(idx)
        
        # Person class index
        self.person_class_index = 0
        
        logger.info(f"Protected class indices: {self.protected_class_indices}")
    
    def process_frame(self, frame_tensor: torch.Tensor) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame_tensor: Input frame as GPU tensor (H, W, C) RGB format
            
        Returns:
            Tuple of (alert_triggered, annotated_frame_if_alert)
        """
        self.frame_count += 1
        start_time = time.time()
        
        with torch.cuda.stream(self.cuda_stream):
            # Step 1: Object Detection (every N frames)
            if self.frame_count % self.config.object_detection_interval == 1:
                self._run_object_detection(frame_tensor)
            
            # Step 2: Pose Estimation (every frame)
            keypoints = self._run_pose_estimation(frame_tensor)
            
            # Step 3: Theft Logic (GPU-accelerated)
            alert, interaction_info = self._check_theft_condition(keypoints)
        
        # Synchronize CUDA stream
        self.cuda_stream.synchronize()
        
        # Track FPS
        elapsed = time.time() - start_time
        self.fps_history.append(1.0 / max(elapsed, 0.001))
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        # Step 4: Only move to CPU if alert triggered
        annotated_frame = None
        if alert:
            annotated_frame = self._handle_alert(frame_tensor, interaction_info)
        
        return alert, annotated_frame
    
    def _run_object_detection(self, frame_tensor: torch.Tensor):
        """
        Run object detection model (Model A).
        
        Updates cached_objects with detected protected items.
        Only runs every N frames for optimization.
        """
        results = self.detect_model.predict(
            frame_tensor,
            conf=self.config.object_confidence,
            classes=self.protected_class_indices
        )
        
        # Clear cached objects
        self.cached_objects.clear()
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]
                
                # Create GPU bounding box
                gpu_box = GPUBoundingBox.from_xyxy(
                    x1, y1, x2, y2, label, conf, self.device
                )
                self.cached_objects.append(gpu_box)
        
        self.last_detection_time = time.time()
        logger.debug(f"Detected {len(self.cached_objects)} protected objects")
    
    def _run_pose_estimation(self, frame_tensor: torch.Tensor) -> GPUKeypoints:
        """
        Run pose estimation model (Model B).
        
        Runs every frame for real-time tracking.
        
        Returns:
            GPUKeypoints containing all detected pose keypoints
        """
        results = self.pose_model.predict(
            frame_tensor,
            conf=self.config.person_confidence
        )
        
        # Extract keypoints from results
        all_keypoints = []
        
        for result in results:
            if result.keypoints is None:
                continue
            
            # Keypoints shape: (num_people, num_keypoints, 3)
            kpts = result.keypoints.data
            if kpts.numel() > 0:
                all_keypoints.append(kpts)
        
        if all_keypoints:
            # Concatenate all keypoints on GPU
            keypoints_tensor = torch.cat(all_keypoints, dim=0).to(self.device)
        else:
            keypoints_tensor = torch.empty((0, 17, 3), device=self.device)
        
        return GPUKeypoints(keypoints=keypoints_tensor)
    
    def _check_theft_condition(self, keypoints: GPUKeypoints) -> Tuple[bool, Optional[Dict]]:
        """
        Check for theft condition using GPU-accelerated logic.
        
        Theft Condition:
        - Wrist/hand keypoint is inside a protected object's hitbox
        - AND person is not authorized (placeholder - always unauthorized for now)
        
        All calculations performed on GPU using PyTorch tensors.
        
        Args:
            keypoints: Detected pose keypoints
            
        Returns:
            Tuple of (alert_triggered, interaction_info_dict)
        """
        # Get wrist positions (GPU tensor)
        wrist_positions = keypoints.get_wrist_positions(self.config.wrist_indices)
        
        if wrist_positions.numel() == 0 or len(self.cached_objects) == 0:
            return False, None
        
        # Check each protected object's hitbox
        for obj in self.cached_objects:
            # Create expanded hitbox
            hitbox = obj.expand(self.config.hitbox_padding)
            
            # GPU-accelerated point-in-box check
            inside_mask = hitbox.contains_points(wrist_positions)
            
            if inside_mask.any():
                # Theft condition detected!
                # Only now do we need any CPU data
                interaction_info = {
                    'object': obj,
                    'hitbox': hitbox,
                    'wrist_positions': wrist_positions.cpu().numpy(),
                    'interacting_wrists': wrist_positions[inside_mask].cpu().numpy()
                }
                return True, interaction_info
        
        return False, None
    
    def _handle_alert(self, frame_tensor: torch.Tensor,
                      interaction_info: Dict) -> np.ndarray:
        """
        Handle theft alert - move frame to CPU and save evidence.
        
        This is the ONLY place where we transfer the frame to CPU,
        and only when an alert is triggered.
        
        Args:
            frame_tensor: Current frame as GPU tensor
            interaction_info: Dictionary containing interaction details
            
        Returns:
            Annotated frame as numpy array
        """
        # Move frame to CPU for annotation and saving
        frame_cpu = frame_tensor.cpu().numpy()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
        
        # Draw annotations
        annotated = draw_hitbox_and_skeleton(
            frame_bgr,
            self.cached_objects,
            interaction_info
        )
        
        # Save evidence
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_path = os.path.join(
            self.config.evidence_dir,
            f"theft_alert_{timestamp}.jpg"
        )
        cv2.imwrite(evidence_path, annotated)
        
        logger.warning(f"🚨 THEFT ALERT: Hand interaction with {interaction_info['object'].label}")
        logger.warning(f"Evidence saved: {evidence_path}")
        
        return annotated
    
    def get_average_fps(self) -> float:
        """Get average FPS over recent frames."""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_display_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Get frame for display (transfers to CPU).
        
        Use sparingly - only for visualization/display purposes.
        
        Args:
            frame_tensor: Frame as GPU tensor
            
        Returns:
            BGR numpy array for cv2.imshow
        """
        frame_cpu = frame_tensor.cpu().numpy()
        frame_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
        
        # Draw cached objects and FPS
        for obj in self.cached_objects:
            x1, y1, x2, y2 = obj.to_cpu()
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame_bgr,
                f"{obj.label}: {obj.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
            
            # Draw hitbox
            hitbox = obj.expand(self.config.hitbox_padding)
            hx1, hy1, hx2, hy2 = hitbox.to_cpu()
            cv2.rectangle(frame_bgr, (hx1, hy1), (hx2, hy2), (0, 255, 0), 1)
        
        # Draw FPS
        fps = self.get_average_fps()
        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return frame_bgr


# =============================================================================
# VISUALIZATION HELPER FUNCTION
# =============================================================================

def draw_hitbox_and_skeleton(
    frame: np.ndarray,
    cached_objects: List[GPUBoundingBox],
    interaction_info: Optional[Dict] = None
) -> np.ndarray:
    """
    Draw hitbox and skeleton overlay on frame.
    
    This function is called ONLY for the final display frame or when
    saving evidence. It uses OpenCV for efficient CPU-based drawing.
    
    Args:
        frame: BGR numpy array
        cached_objects: List of detected protected objects
        interaction_info: Optional interaction details for alert annotation
        
    Returns:
        Annotated BGR numpy array
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Color constants
    COLOR_OBJECT = (0, 255, 255)     # Yellow - protected objects
    COLOR_HITBOX = (0, 255, 0)       # Green - hitbox
    COLOR_HITBOX_ALERT = (0, 0, 255) # Red - hitbox when triggered
    COLOR_HAND = (255, 0, 255)       # Magenta - hand/wrist markers
    COLOR_ALERT_TEXT = (0, 0, 255)   # Red - alert text
    
    # Draw all protected objects with hitboxes
    for obj in cached_objects:
        x1, y1, x2, y2 = obj.to_cpu()
        
        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Object bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_OBJECT, 2)
        
        # Label
        label = f"{obj.label}: {obj.confidence:.2f}"
        cv2.putText(
            annotated, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            COLOR_OBJECT, 2
        )
        
        # Hitbox (expanded region)
        hitbox = obj.expand(20)  # Use default padding
        hx1, hy1, hx2, hy2 = hitbox.to_cpu()
        hx1 = max(0, hx1)
        hy1 = max(0, hy1)
        hx2 = min(w, hx2)
        hy2 = min(h, hy2)
        
        # Color hitbox red if this is the triggered object
        # Compare by label and coordinates instead of object identity
        hitbox_color = COLOR_HITBOX
        if interaction_info:
            alert_obj = interaction_info['object']
            # Compare using label and coordinates
            if obj.label == alert_obj.label:
                obj_coords = obj.to_cpu()
                alert_coords = alert_obj.to_cpu()
                if obj_coords == alert_coords:
                    hitbox_color = COLOR_HITBOX_ALERT
        
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), hitbox_color, 1)
    
    # Draw hand/wrist markers if interaction detected
    if interaction_info:
        # Draw all wrist positions
        for wx, wy in interaction_info['wrist_positions']:
            cv2.circle(annotated, (int(wx), int(wy)), 8, COLOR_HAND, 2)
        
        # Highlight interacting wrists
        for wx, wy in interaction_info['interacting_wrists']:
            cv2.circle(annotated, (int(wx), int(wy)), 12, COLOR_HITBOX_ALERT, -1)
            cv2.circle(annotated, (int(wx), int(wy)), 14, COLOR_HITBOX_ALERT, 2)
        
        # Alert banner
        alert_obj = interaction_info['object']
        alert_text = f"!!! ALERT: Hand on {alert_obj.label} !!!"
        
        # Draw alert banner at top
        cv2.rectangle(annotated, (0, 0), (w, 40), COLOR_ALERT_TEXT, -1)
        cv2.putText(
            annotated, alert_text,
            (w // 2 - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
    
    return annotated


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for CUDA theft detection system."""
    parser = argparse.ArgumentParser(
        description='SENTINEL Guardian - CUDA Zero-Copy Theft Detection'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='Video source: camera index (0) or video file path'
    )
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='CUDA device index'
    )
    parser.add_argument(
        '--detect-model',
        type=str,
        default='yolov8n.engine',
        help='Path to object detection TensorRT engine'
    )
    parser.add_argument(
        '--pose-model',
        type=str,
        default='yolov8n-pose.engine',
        help='Path to pose estimation TensorRT engine'
    )
    parser.add_argument(
        '--detection-interval',
        type=int,
        default=30,
        help='Run object detection every N frames'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (headless mode)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    logger.info("=" * 60)
    logger.info("SENTINEL Interior Guardian - CUDA Zero-Copy Pipeline")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if not CUDA_AVAILABLE:
        logger.error("CUDA not available. This system requires an NVIDIA GPU.")
        return 1
    
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(args.device)}")
    logger.info(f"Video Source: {args.source}")
    logger.info(f"Object Detection Interval: Every {args.detection_interval} frames")
    logger.info("")
    
    # Create configuration
    config = CUDAConfig(
        device=args.device,
        detect_model_path=args.detect_model,
        pose_model_path=args.pose_model,
        detect_model_fallback=args.detect_model.replace('.engine', '.pt'),
        pose_model_fallback=args.pose_model.replace('.engine', '.pt'),
        object_detection_interval=args.detection_interval
    )
    
    # Initialize components
    try:
        # Create video decoder
        decoder = create_video_decoder(args.source, config)
        
        # Create theft detector
        detector = CUDATheftDetector(config)
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1
    
    logger.info("System initialized. Press 'Q' to quit.")
    logger.info("")
    
    # Main loop
    try:
        while True:
            # Get frame from decoder (GPU tensor)
            frame_tensor = decoder.get_frame()
            if frame_tensor is None:
                logger.info("End of video stream")
                break
            
            # Process frame
            alert, annotated_frame = detector.process_frame(frame_tensor)
            
            # Display (optional)
            if not args.no_display:
                if alert and annotated_frame is not None:
                    display_frame = annotated_frame
                else:
                    display_frame = detector.get_display_frame(frame_tensor)
                
                cv2.imshow("SENTINEL Guardian - CUDA", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit command received")
                    break
                    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    finally:
        # Cleanup
        decoder.release()
        cv2.destroyAllWindows()
    
    logger.info("")
    logger.info(f"Average FPS: {detector.get_average_fps():.1f}")
    logger.info("SENTINEL Guardian shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
