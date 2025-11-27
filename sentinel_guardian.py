"""
SENTINEL Interior Guardian - Level 2
=====================================

A high-performance Python application for theft detection and interior monitoring.
Fuses Object Detection, Pose/Hand Estimation, and Facial Recognition into a single pipeline.

Target Hardware: Intel Core Ultra 5 (optimized via OpenVINO)

Architecture: Producer-Consumer pattern with threaded design
- Producer Thread: Captures frames from camera
- Inference Thread: Processes frames through detection pipeline
- Alert Thread: Handles display and alert logic

Author: SENTINEL Security Systems
"""

import cv2
import numpy as np
import threading
import queue
import logging
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Third-party imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Object detection will be disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Hand tracking will be disabled.")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not available. Facial recognition will be disabled.")

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.warning("OpenVINO not available. Hardware acceleration will be disabled.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinel_guardian.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR TYPE SAFETY AND CLARITY
# =============================================================================

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this bounding box (hitbox)."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return coordinates as tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectionResult:
    """Contains all detection results for a single frame."""
    frame: np.ndarray
    timestamp: datetime
    protected_objects: List[BoundingBox]
    persons: List[BoundingBox]
    hand_landmarks: List[Tuple[int, int]]  # Fingertip coordinates
    face_encodings: List[Any]
    face_locations: List[Tuple[int, int, int, int]]


@dataclass
class AlertEvent:
    """Represents a security alert event."""
    timestamp: datetime
    event_type: str
    frame: np.ndarray
    description: str
    annotated_frame: Optional[np.ndarray] = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration constants."""
    # Protected object classes (COCO dataset labels)
    PROTECTED_CLASSES = ['cell phone', 'laptop', 'tv', 'remote', 'keyboard', 'mouse']
    
    # Confidence thresholds
    OBJECT_CONFIDENCE_THRESHOLD = 0.5
    PERSON_CONFIDENCE_THRESHOLD = 0.5
    FACE_RECOGNITION_TOLERANCE = 0.6
    
    # Frame queue sizes (for threading)
    CAPTURE_QUEUE_SIZE = 5
    INFERENCE_QUEUE_SIZE = 3
    ALERT_QUEUE_SIZE = 10
    
    # Paths
    AUTHORIZED_PERSONNEL_DIR = "authorized_personels"
    EVIDENCE_DIR = "evidence"
    MODELS_DIR = "models"
    
    # OpenVINO device (AUTO will select best available: GPU > NPU > CPU)
    OPENVINO_DEVICE = "AUTO"
    
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    
    # Hitbox expansion (pixels around detected object)
    HITBOX_PADDING = 20


# =============================================================================
# OPENVINO OPTIMIZATION MODULE
# =============================================================================

class OpenVINOOptimizer:
    """
    Handles OpenVINO optimization for Intel hardware acceleration.
    
    This class provides utilities to:
    1. Export YOLO models to OpenVINO IR format
    2. Configure optimal inference device (CPU/GPU/NPU)
    3. Quantize models for INT8 inference
    
    Threading Note: This class is thread-safe for inference operations.
    The Core object is created once and shared across threads.
    """
    
    def __init__(self, device: str = "AUTO"):
        """
        Initialize OpenVINO runtime.
        
        Args:
            device: Target device for inference. "AUTO" selects best available.
                   Options: "CPU", "GPU", "NPU", "AUTO"
        """
        self.device = device
        self.core = None
        self._lock = threading.Lock()
        
        if OPENVINO_AVAILABLE:
            try:
                self.core = Core()
                available_devices = self.core.available_devices
                logger.info(f"OpenVINO initialized. Available devices: {available_devices}")
                
                # Log device properties for debugging
                for dev in available_devices:
                    try:
                        full_name = self.core.get_property(dev, "FULL_DEVICE_NAME")
                        logger.info(f"  {dev}: {full_name}")
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed to initialize OpenVINO: {e}")
                self.core = None
    
    def export_yolo_to_openvino(self, model_path: str) -> Optional[str]:
        """
        Export YOLO model to OpenVINO IR format for optimized inference.
        
        The exported model will be quantized to INT8 for faster inference
        on Intel Core Ultra processors.
        
        Args:
            model_path: Path to the YOLO .pt model file
            
        Returns:
            Path to the exported OpenVINO model, or None if export fails
        """
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available for export")
            return None
            
        try:
            model = YOLO(model_path)
            # Export to OpenVINO format with INT8 quantization
            # This optimizes for Intel Core Ultra architecture
            export_path = model.export(
                format='openvino',
                imgsz=640,
                half=False,  # INT8 quantization is handled separately
                int8=True if self.core else False,  # Enable INT8 if OpenVINO available
                device=self.device
            )
            logger.info(f"Model exported to OpenVINO format: {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Failed to export model to OpenVINO: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if OpenVINO optimization is available."""
        return self.core is not None


# =============================================================================
# OBJECT DETECTION MODULE
# =============================================================================

class ObjectDetector:
    """
    YOLOv8-based object detector optimized with OpenVINO.
    
    This module detects:
    1. Protected assets (cell phone, laptop, TV, etc.)
    2. Persons in the frame
    
    Threading Note: The YOLO model is thread-safe for inference.
    Multiple threads can call detect() concurrently.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", use_openvino: bool = True):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLO model (will auto-download if not present)
            use_openvino: Whether to use OpenVINO optimization
        """
        self.model = None
        self.use_openvino = use_openvino and OPENVINO_AVAILABLE
        self._lock = threading.Lock()
        
        if YOLO_AVAILABLE:
            try:
                # If OpenVINO is available and requested, try to use optimized model
                if self.use_openvino:
                    openvino_path = model_path.replace('.pt', '_openvino_model')
                    if os.path.exists(openvino_path):
                        self.model = YOLO(openvino_path)
                        logger.info(f"Loaded OpenVINO-optimized model: {openvino_path}")
                    else:
                        # Load standard model with OpenVINO backend
                        self.model = YOLO(model_path)
                        logger.info(f"Loaded YOLO model: {model_path}")
                else:
                    self.model = YOLO(model_path)
                    logger.info(f"Loaded YOLO model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Detect protected objects and persons in the frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (protected_objects, persons) as lists of BoundingBox
        """
        protected_objects = []
        persons = []
        
        if self.model is None:
            return protected_objects, persons
        
        try:
            # Run inference (verbose=False to reduce console output)
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    # Get class name from COCO labels
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=cls_name,
                        confidence=confidence
                    )
                    
                    # Categorize detection
                    if cls_name == 'person' and confidence >= Config.PERSON_CONFIDENCE_THRESHOLD:
                        persons.append(bbox)
                    elif cls_name in Config.PROTECTED_CLASSES and confidence >= Config.OBJECT_CONFIDENCE_THRESHOLD:
                        protected_objects.append(bbox)
                        
        except Exception as e:
            logger.error(f"Object detection error: {e}")
        
        return protected_objects, persons
    
    def create_hitbox(self, bbox: BoundingBox, padding: int = Config.HITBOX_PADDING) -> BoundingBox:
        """
        Create an expanded hitbox (ROI) around a detected object.
        
        The hitbox is used to detect hand interactions with protected objects.
        
        Args:
            bbox: Original bounding box
            padding: Pixels to expand in each direction
            
        Returns:
            Expanded BoundingBox
        """
        return BoundingBox(
            x1=max(0, bbox.x1 - padding),
            y1=max(0, bbox.y1 - padding),
            x2=bbox.x2 + padding,
            y2=bbox.y2 + padding,
            label=bbox.label,
            confidence=bbox.confidence
        )


# =============================================================================
# HAND TRACKING MODULE
# =============================================================================

class HandTracker:
    """
    MediaPipe-based hand tracking for detecting hand landmarks.
    
    This module tracks:
    1. Hand presence and position
    2. Fingertip coordinates (used for interaction detection)
    
    Threading Note: MediaPipe Hands is NOT thread-safe.
    Access to the Hands object is protected by a lock.
    """
    
    # MediaPipe fingertip landmark indices
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    
    def __init__(self, max_hands: int = 4, detection_confidence: float = 0.5):
        """
        Initialize hand tracker.
        
        Args:
            max_hands: Maximum number of hands to track
            detection_confidence: Minimum confidence for detection
        """
        self.hands = None
        self._lock = threading.Lock()
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Hands initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Hands: {e}")
    
    def get_fingertips(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect and return fingertip coordinates in the frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of (x, y) tuples for each detected fingertip
        """
        fingertips = []
        
        if self.hands is None:
            return fingertips
        
        # MediaPipe requires RGB input
        # Lock ensures thread safety for MediaPipe processing
        with self._lock:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    h, w = frame.shape[:2]
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        for tip_idx in self.FINGERTIP_INDICES:
                            landmark = hand_landmarks.landmark[tip_idx]
                            # Convert normalized coordinates to pixel coordinates
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            fingertips.append((x, y))
                            
            except Exception as e:
                logger.error(f"Hand tracking error: {e}")
        
        return fingertips
    
    def close(self):
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()


# =============================================================================
# FACIAL RECOGNITION MODULE
# =============================================================================

class FacialRecognizer:
    """
    Face recognition for authorization logic.
    
    This module:
    1. Loads known faces from authorized_personels directory
    2. Compares detected faces against known faces
    3. Returns authorization status
    
    Threading Note: face_recognition library is thread-safe for encoding
    and comparison operations.
    """
    
    def __init__(self, authorized_dir: str = Config.AUTHORIZED_PERSONNEL_DIR):
        """
        Initialize facial recognizer and load authorized personnel.
        
        Args:
            authorized_dir: Directory containing authorized personnel images
        """
        self.known_encodings = []
        self.known_names = []
        self._lock = threading.Lock()
        
        if FACE_RECOGNITION_AVAILABLE:
            self._load_authorized_personnel(authorized_dir)
    
    def _load_authorized_personnel(self, directory: str):
        """
        Load face encodings from authorized personnel directory.
        
        Expected directory structure:
        authorized_personels/
            person1_name.jpg
            person2_name.png
            ...
        """
        if not os.path.exists(directory):
            logger.warning(f"Authorized personnel directory not found: {directory}")
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(directory, filename)
                try:
                    # Load image and extract face encoding
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        # Extract name from filename (without extension)
                        name = os.path.splitext(filename)[0]
                        self.known_names.append(name)
                        logger.info(f"Loaded authorized face: {name}")
                    else:
                        logger.warning(f"No face found in: {filename}")
                        
                except Exception as e:
                    logger.error(f"Failed to load face from {filename}: {e}")
        
        logger.info(f"Loaded {len(self.known_encodings)} authorized personnel faces")
    
    def check_authorization(self, frame: np.ndarray) -> Tuple[bool, List[str], List[Tuple[int, int, int, int]]]:
        """
        Check if faces in frame are authorized.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of:
            - is_authorized: True if any authorized face is detected
            - names: List of recognized names
            - face_locations: List of face bounding boxes (top, right, bottom, left)
        """
        is_authorized = False
        recognized_names = []
        face_locations = []
        
        if not FACE_RECOGNITION_AVAILABLE:
            return is_authorized, recognized_names, face_locations
        
        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and get encodings
            # Using 'hog' model for CPU, 'cnn' for GPU
            locations = face_recognition.face_locations(rgb_frame, model='hog')
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            
            face_locations = locations
            
            for encoding in encodings:
                if len(self.known_encodings) > 0:
                    # Compare against known faces
                    matches = face_recognition.compare_faces(
                        self.known_encodings, 
                        encoding, 
                        tolerance=Config.FACE_RECOGNITION_TOLERANCE
                    )
                    
                    if True in matches:
                        # Find best match
                        face_distances = face_recognition.face_distance(
                            self.known_encodings, encoding
                        )
                        best_match_idx = np.argmin(face_distances)
                        
                        if matches[best_match_idx]:
                            recognized_names.append(self.known_names[best_match_idx])
                            is_authorized = True
                    else:
                        recognized_names.append("Unknown")
                else:
                    recognized_names.append("Unknown")
                    
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
        
        return is_authorized, recognized_names, face_locations


# =============================================================================
# ALERT AND EVIDENCE MODULE
# =============================================================================

class AlertSystem:
    """
    Handles security alerts and evidence capture.
    
    This module:
    1. Captures evidence snapshots
    2. Logs security events
    3. Triggers alarms (placeholder)
    
    Threading Note: This class is thread-safe. File operations are
    protected by a lock to prevent race conditions.
    """
    
    def __init__(self, evidence_dir: str = Config.EVIDENCE_DIR):
        """
        Initialize alert system.
        
        Args:
            evidence_dir: Directory to save evidence snapshots
        """
        self.evidence_dir = evidence_dir
        self._lock = threading.Lock()
        self.alarm_active = False
        
        # Create evidence directory if it doesn't exist
        os.makedirs(evidence_dir, exist_ok=True)
        logger.info(f"Evidence directory: {evidence_dir}")
    
    def save_evidence(self, event: AlertEvent) -> str:
        """
        Save evidence snapshot and log the event.
        
        Args:
            event: AlertEvent containing frame and details
            
        Returns:
            Path to saved evidence file
        """
        with self._lock:
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            filename = f"evidence_{event.event_type}_{timestamp_str}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            
            try:
                # Save annotated frame if available, otherwise original
                frame_to_save = event.annotated_frame if event.annotated_frame is not None else event.frame
                cv2.imwrite(filepath, frame_to_save)
                
                # Log the event
                logger.warning(f"SECURITY ALERT: {event.event_type}")
                logger.warning(f"  Timestamp: {event.timestamp}")
                logger.warning(f"  Description: {event.description}")
                logger.warning(f"  Evidence saved: {filepath}")
                
                return filepath
                
            except Exception as e:
                logger.error(f"Failed to save evidence: {e}")
                return ""
    
    def trigger_alarm(self):
        """
        Trigger audible alarm (placeholder implementation).
        
        In a production system, this would:
        - Activate physical alarm devices
        - Send notifications to security personnel
        - Trigger smart home integrations
        """
        if not self.alarm_active:
            self.alarm_active = True
            logger.warning("🚨 ALARM TRIGGERED - UNAUTHORIZED ACCESS DETECTED 🚨")
            # Placeholder: In production, implement actual alarm trigger
            # Examples: GPIO pin for alarm, HTTP request to home automation, etc.
    
    def reset_alarm(self):
        """Reset the alarm state."""
        self.alarm_active = False
        logger.info("Alarm reset")


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class Visualizer:
    """
    Handles frame annotation and visualization.
    
    This module draws:
    1. Object bounding boxes (hitboxes)
    2. Person bounding boxes
    3. Hand landmarks
    4. Face recognition results
    5. System status overlay
    """
    
    # Color constants (BGR format)
    COLOR_PROTECTED = (0, 255, 255)    # Yellow - protected objects
    COLOR_HITBOX = (0, 255, 0)         # Green - hitbox around objects
    COLOR_PERSON_AUTH = (0, 255, 0)    # Green - authorized person
    COLOR_PERSON_UNAUTH = (0, 0, 255)  # Red - unauthorized person
    COLOR_HAND = (255, 0, 255)         # Magenta - hand landmarks
    COLOR_ALERT = (0, 0, 255)          # Red - alert
    COLOR_INFO = (255, 255, 255)       # White - info text
    
    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        protected_objects: List[BoundingBox],
        persons: List[BoundingBox],
        fingertips: List[Tuple[int, int]],
        authorized: bool,
        recognized_names: List[str],
        face_locations: List[Tuple[int, int, int, int]],
        alert_active: bool = False
    ) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Original BGR frame
            protected_objects: List of protected object bounding boxes
            persons: List of person bounding boxes
            fingertips: List of fingertip coordinates
            authorized: Whether authorized person is detected
            recognized_names: Names of recognized faces
            face_locations: Face bounding box locations
            alert_active: Whether an alert is currently active
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw protected objects with hitboxes
        for obj in protected_objects:
            # Object bounding box
            cv2.rectangle(
                annotated,
                (obj.x1, obj.y1), (obj.x2, obj.y2),
                Visualizer.COLOR_PROTECTED, 2
            )
            
            # Hitbox (expanded region)
            hitbox = BoundingBox(
                x1=max(0, obj.x1 - Config.HITBOX_PADDING),
                y1=max(0, obj.y1 - Config.HITBOX_PADDING),
                x2=min(w, obj.x2 + Config.HITBOX_PADDING),
                y2=min(h, obj.y2 + Config.HITBOX_PADDING),
                label=obj.label,
                confidence=obj.confidence
            )
            cv2.rectangle(
                annotated,
                (hitbox.x1, hitbox.y1), (hitbox.x2, hitbox.y2),
                Visualizer.COLOR_HITBOX, 1
            )
            
            # Label
            label = f"{obj.label}: {obj.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (obj.x1, obj.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                Visualizer.COLOR_PROTECTED, 2
            )
        
        # Draw persons
        person_color = Visualizer.COLOR_PERSON_AUTH if authorized else Visualizer.COLOR_PERSON_UNAUTH
        for person in persons:
            cv2.rectangle(
                annotated,
                (person.x1, person.y1), (person.x2, person.y2),
                person_color, 2
            )
        
        # Draw face recognition results
        for i, (top, right, bottom, left) in enumerate(face_locations):
            color = Visualizer.COLOR_PERSON_AUTH if authorized else Visualizer.COLOR_PERSON_UNAUTH
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            
            if i < len(recognized_names):
                name = recognized_names[i]
                cv2.putText(
                    annotated, name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2
                )
        
        # Draw fingertips
        for x, y in fingertips:
            cv2.circle(annotated, (x, y), 8, Visualizer.COLOR_HAND, -1)
            cv2.circle(annotated, (x, y), 10, Visualizer.COLOR_HAND, 2)
        
        # Draw status overlay
        status_text = "AUTHORIZED" if authorized else "UNAUTHORIZED"
        status_color = Visualizer.COLOR_PERSON_AUTH if authorized else Visualizer.COLOR_PERSON_UNAUTH
        cv2.putText(
            annotated, f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            status_color, 2
        )
        
        # Alert indicator
        if alert_active:
            cv2.rectangle(annotated, (0, 0), (w, 10), Visualizer.COLOR_ALERT, -1)
            cv2.putText(
                annotated, "!!! ALERT - UNAUTHORIZED INTERACTION !!!",
                (w // 2 - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                Visualizer.COLOR_ALERT, 2
            )
        
        return annotated


# =============================================================================
# MAIN SENTINEL GUARDIAN CLASS
# =============================================================================

class SentinelGuardian:
    """
    Main SENTINEL Interior Guardian system.
    
    Implements a threaded Producer-Consumer architecture:
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Frame Capture  │───▶│   Inference     │───▶│  Display/Alert  │
    │    (Producer)   │    │   (Consumer/    │    │    (Consumer)   │
    │                 │    │    Producer)    │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
           │                       │                      │
           ▼                       ▼                      ▼
     capture_queue           inference_queue         alert_queue
     
    Threading Design:
    1. Capture Thread: Continuously grabs frames from camera
       - Runs at maximum camera FPS
       - Drops frames if queue is full to maintain real-time operation
       
    2. Inference Thread: Processes frames through detection pipeline
       - Object detection (YOLO + OpenVINO)
       - Face recognition
       - Hand tracking
       - Produces detection results for display
       
    3. Display Thread: Renders annotated frames and handles alerts
       - Draws annotations on frame
       - Checks for theft conditions
       - Triggers alerts and saves evidence
    """
    
    def __init__(self, camera_index: int = Config.CAMERA_INDEX):
        """
        Initialize the SENTINEL Guardian system.
        
        Args:
            camera_index: Camera device index (0 for default camera)
        """
        self.camera_index = camera_index
        self.running = False
        
        # Initialize queues for thread communication
        # Using bounded queues to prevent memory overflow
        self.capture_queue = queue.Queue(maxsize=Config.CAPTURE_QUEUE_SIZE)
        self.inference_queue = queue.Queue(maxsize=Config.INFERENCE_QUEUE_SIZE)
        self.alert_queue = queue.Queue(maxsize=Config.ALERT_QUEUE_SIZE)
        
        # Initialize detection modules
        logger.info("Initializing SENTINEL Guardian modules...")
        self.openvino_optimizer = OpenVINOOptimizer(device=Config.OPENVINO_DEVICE)
        self.object_detector = ObjectDetector(use_openvino=self.openvino_optimizer.is_available())
        self.hand_tracker = HandTracker()
        self.facial_recognizer = FacialRecognizer()
        self.alert_system = AlertSystem()
        
        # Camera and threading
        self.camera = None
        self.capture_thread = None
        self.inference_thread = None
        self.display_thread = None
        
        # State tracking
        self.current_mode = "PASSIVE"  # PASSIVE or ACTIVE_DEFENSE
        self.last_alert_time = None
        self.alert_cooldown = 5.0  # Seconds between alerts
        
        logger.info("SENTINEL Guardian initialized")
    
    def start(self):
        """Start the SENTINEL Guardian system."""
        logger.info("Starting SENTINEL Guardian...")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        self.running = True
        
        # Start threads
        # Threads are daemon threads so they will exit when main thread exits
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        
        self.capture_thread.start()
        self.inference_thread.start()
        self.display_thread.start()
        
        logger.info("All threads started")
        return True
    
    def stop(self):
        """Stop the SENTINEL Guardian system."""
        logger.info("Stopping SENTINEL Guardian...")
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        # Release resources
        if self.camera:
            self.camera.release()
        if self.hand_tracker:
            self.hand_tracker.close()
        
        cv2.destroyAllWindows()
        logger.info("SENTINEL Guardian stopped")
    
    def _capture_loop(self):
        """
        Frame capture thread (Producer).
        
        Continuously captures frames from the camera and places them
        in the capture queue for processing.
        
        Threading Notes:
        - This thread runs at camera FPS
        - If queue is full, oldest frames are dropped to maintain real-time
        - Uses non-blocking put with timeout to prevent deadlock
        """
        logger.info("Capture thread started")
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                timestamp = datetime.now()
                
                # Try to put frame in queue, drop if full (real-time operation)
                try:
                    # Non-blocking put - drops frame if queue is full
                    self.capture_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    # Queue full - drop this frame to maintain real-time
                    pass
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.01)
        
        logger.info("Capture thread stopped")
    
    def _inference_loop(self):
        """
        Inference thread (Consumer/Producer).
        
        Consumes frames from capture queue, runs detection pipeline,
        and produces detection results for the display thread.
        
        Threading Notes:
        - This is the main processing thread
        - Uses OpenVINO-optimized models for Intel hardware
        - Balances between capture and display threads
        """
        logger.info("Inference thread started")
        
        while self.running:
            try:
                # Get frame from capture queue (blocking with timeout)
                try:
                    frame, timestamp = self.capture_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Run detection pipeline
                # Step 1: Detect objects and persons
                protected_objects, persons = self.object_detector.detect(frame)
                
                # Step 2: Get hand landmarks (fingertips)
                fingertips = self.hand_tracker.get_fingertips(frame)
                
                # Step 3: Check face authorization (only if persons detected)
                is_authorized = False
                recognized_names = []
                face_locations = []
                
                if len(persons) > 0:
                    is_authorized, recognized_names, face_locations = \
                        self.facial_recognizer.check_authorization(frame)
                
                # Create detection result
                result = DetectionResult(
                    frame=frame,
                    timestamp=timestamp,
                    protected_objects=protected_objects,
                    persons=persons,
                    hand_landmarks=fingertips,
                    face_encodings=[],  # Not needed for display
                    face_locations=face_locations
                )
                
                # Package result with authorization status
                inference_result = (result, is_authorized, recognized_names)
                
                # Put result in inference queue
                try:
                    self.inference_queue.put_nowait(inference_result)
                except queue.Full:
                    # Queue full - drop oldest result
                    try:
                        self.inference_queue.get_nowait()
                        self.inference_queue.put_nowait(inference_result)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                time.sleep(0.01)
        
        logger.info("Inference thread stopped")
    
    def _display_loop(self):
        """
        Display and alert thread (Consumer).
        
        Consumes detection results, renders annotated frames,
        checks for theft conditions, and handles alerts.
        
        Threading Notes:
        - Handles all visualization and user interaction
        - Triggers alerts when unauthorized interaction detected
        - Manages OpenCV window (must be in main thread for some platforms)
        """
        logger.info("Display thread started")
        
        while self.running:
            try:
                # Get inference result (blocking with timeout)
                try:
                    result, is_authorized, recognized_names = \
                        self.inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Update system mode based on authorization
                self.current_mode = "PASSIVE" if is_authorized else "ACTIVE_DEFENSE"
                
                # Check for theft condition
                alert_triggered = False
                if self.current_mode == "ACTIVE_DEFENSE":
                    alert_triggered = self._check_theft_condition(
                        result.protected_objects,
                        result.hand_landmarks
                    )
                
                # Annotate frame
                annotated_frame = Visualizer.annotate_frame(
                    frame=result.frame,
                    protected_objects=result.protected_objects,
                    persons=result.persons,
                    fingertips=result.hand_landmarks,
                    authorized=is_authorized,
                    recognized_names=recognized_names,
                    face_locations=result.face_locations,
                    alert_active=alert_triggered
                )
                
                # Handle alert if triggered
                if alert_triggered:
                    self._handle_alert(result, annotated_frame)
                
                # Display frame
                cv2.imshow("SENTINEL Guardian - Level 2", annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit command received")
                    self.running = False
                elif key == ord('r'):
                    self.alert_system.reset_alarm()
                    
            except Exception as e:
                logger.error(f"Display loop error: {e}")
                time.sleep(0.01)
        
        logger.info("Display thread stopped")
    
    def _check_theft_condition(
        self, 
        protected_objects: List[BoundingBox],
        fingertips: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if theft condition is met.
        
        Theft Condition:
        - Person is unauthorized (ACTIVE_DEFENSE mode)
        - AND hand landmark (fingertip) is inside a protected object's hitbox
        
        Args:
            protected_objects: List of protected object bounding boxes
            fingertips: List of fingertip coordinates
            
        Returns:
            True if theft condition is detected
        """
        for obj in protected_objects:
            # Create expanded hitbox
            hitbox = self.object_detector.create_hitbox(obj)
            
            # Check if any fingertip is inside the hitbox
            for x, y in fingertips:
                if hitbox.contains_point(x, y):
                    logger.warning(f"Hand interaction detected with {obj.label}!")
                    return True
        
        return False
    
    def _handle_alert(self, result: DetectionResult, annotated_frame: np.ndarray):
        """
        Handle security alert.
        
        Args:
            result: Detection result containing frame and detections
            annotated_frame: Frame with annotations
        """
        current_time = time.time()
        
        # Check cooldown to prevent alert spam
        if self.last_alert_time and (current_time - self.last_alert_time) < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        # Create alert event
        event = AlertEvent(
            timestamp=result.timestamp,
            event_type="THEFT_ATTEMPT",
            frame=result.frame,
            description="Unauthorized person interacting with protected object",
            annotated_frame=annotated_frame
        )
        
        # Save evidence
        self.alert_system.save_evidence(event)
        
        # Trigger alarm
        self.alert_system.trigger_alarm()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for SENTINEL Guardian."""
    logger.info("=" * 60)
    logger.info("SENTINEL Interior Guardian - Level 2")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Controls:")
    logger.info("  Q - Quit")
    logger.info("  R - Reset alarm")
    logger.info("")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    logger.info(f"  YOLO available: {YOLO_AVAILABLE}")
    logger.info(f"  MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    logger.info(f"  Face Recognition available: {FACE_RECOGNITION_AVAILABLE}")
    logger.info(f"  OpenVINO available: {OPENVINO_AVAILABLE}")
    logger.info("")
    
    # Create and start guardian
    guardian = SentinelGuardian()
    
    if not guardian.start():
        logger.error("Failed to start SENTINEL Guardian")
        return 1
    
    # Run until stopped
    try:
        while guardian.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        guardian.stop()
    
    logger.info("SENTINEL Guardian shutdown complete")
    return 0


if __name__ == "__main__":
    exit(main())
