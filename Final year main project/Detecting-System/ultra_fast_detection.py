#!/usr/bin/env python3
"""
Ultra-Fast YOLOv8 Detection - Maximum Speed
Using YOLOv8-N nano model for minimal latency
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class UltraFastDetection:
    def __init__(self, confidence_threshold=0.5, camera_index=0, brightness_adjustment=-30, contrast_adjustment=1.2):
        """Initialize ultra-fast detection system"""
        self.confidence_threshold = confidence_threshold
        self.camera_index = camera_index
        self.brightness_adjustment = brightness_adjustment
        self.contrast_adjustment = contrast_adjustment
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Load YOLOv8-S small model (better accuracy than nano)
        print("Loading YOLOv8-S small model...")
        try:
            self.model = YOLO('yolov8s.pt')
            print("✓ Loaded YOLOv8-S small")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Complete COCO dataset categories
        self.coco_categories = {
            # Humans
            'person': 'human',
            
            # Animals
            'bird': 'animal', 'cat': 'animal', 'dog': 'animal', 'horse': 'animal', 
            'sheep': 'animal', 'cow': 'animal', 'elephant': 'animal', 'bear': 'animal', 
            'zebra': 'animal', 'giraffe': 'animal',
            
            # Vehicles
            'bicycle': 'vehicle', 'car': 'vehicle', 'motorcycle': 'vehicle', 
            'airplane': 'vehicle', 'bus': 'vehicle', 'train': 'vehicle', 
            'truck': 'vehicle', 'boat': 'vehicle',
            
            # Objects
            'traffic light': 'object', 'fire hydrant': 'object', 'stop sign': 'object',
            'parking meter': 'object', 'bench': 'object', 'backpack': 'object', 
            'umbrella': 'object', 'handbag': 'object', 'tie': 'object', 'suitcase': 'object',
            'frisbee': 'object', 'skis': 'object', 'snowboard': 'object',
            'sports ball': 'object', 'kite': 'object', 'baseball bat': 'object',
            'baseball glove': 'object', 'skateboard': 'object', 'surfboard': 'object',
            'tennis racket': 'object', 'bottle': 'object', 'wine glass': 'object',
            'cup': 'object', 'fork': 'object', 'knife': 'object', 'spoon': 'object',
            'bowl': 'object', 'banana': 'object', 'apple': 'object', 'sandwich': 'object',
            'orange': 'object', 'broccoli': 'object', 'carrot': 'object', 'hot dog': 'object',
            'pizza': 'object', 'donut': 'object', 'cake': 'object', 'chair': 'object',
            'couch': 'object', 'potted plant': 'object', 'bed': 'object', 'dining table': 'object',
            'toilet': 'object', 'tv': 'object', 'laptop': 'object', 'mouse': 'object',
            'remote': 'object', 'keyboard': 'object', 'cell phone': 'object', 'microwave': 'object',
            'oven': 'object', 'toaster': 'object', 'sink': 'object', 'refrigerator': 'object',
            'book': 'object', 'clock': 'object', 'vase': 'object', 'scissors': 'object',
            'teddy bear': 'object', 'hair drier': 'object', 'toothbrush': 'object'
        }
        
        # Animals list for quick checking
        self.animals = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}
        
        print("Ultra-fast detection system initialized!")
        print("Using YOLOv8-S small for better accuracy")
        print(f"Detecting all {len(self.coco_categories)} COCO classes")
    
    def get_category(self, class_name):
        """Get category for any COCO class"""
        return self.coco_categories.get(class_name.lower(), 'object')
    
    def is_animal(self, class_name):
        """Check if detected object is an animal"""
        return class_name.lower() in self.animals
    
    def detect_objects(self, frame):
        """Ultra-fast object detection with better accuracy"""
        try:
            # Run detection with balanced settings for speed and accuracy
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.4,  # Higher IoU for better accuracy
                verbose=False,
                stream=False  # Disable streaming for better performance
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.model.names[class_id]
                        category = self.get_category(class_name)
                        is_animal = self.is_animal(class_name)
                        
                        detections.append({
                            'box': box,
                            'confidence': conf,
                            'class_name': class_name,
                            'category': category,
                            'is_animal': is_animal
                        })
            
            return detections
        
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def adjust_brightness_contrast(self, frame, brightness=0, contrast=1.0):
        """Adjust brightness and contrast to compensate for lighting conditions"""
        # Apply brightness and contrast adjustments
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # Additional processing for bright lighting conditions
        # Apply slight gamma correction to reduce washed-out appearance
        gamma = 0.8  # Values < 1 make the image darker
        gamma_corrected = np.power(adjusted / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        return gamma_corrected
    
    def draw_detections(self, frame, detections):
        """Minimal visualization for speed with brightness adjustment"""
        height, width = frame.shape[:2]
        
        # Adjust brightness and contrast for bright lighting conditions
        # Reduce brightness and increase contrast to compensate for bright environment
        output_frame = self.adjust_brightness_contrast(frame, brightness=self.brightness_adjustment, contrast=self.contrast_adjustment)
        
        # FPS display
        cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Count different types
        humans_count = 0
        animals_count = 0
        vehicles_count = 0
        objects_count = 0
        
        # Draw detections
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            category = detection['category']
            is_animal = detection['is_animal']
            
            # Count by category
            if category == 'human':
                humans_count += 1
                color = (255, 0, 0)  # Blue for humans
            elif category == 'animal':
                animals_count += 1
                color = (0, 0, 255)  # Red for animals
            elif category == 'vehicle':
                vehicles_count += 1
                color = (0, 255, 255)  # Yellow for vehicles
            else:  # object
                objects_count += 1
                color = (0, 0, 0)  # Black for objects
            
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box with margin
            margin = 3
            cv2.rectangle(output_frame, (x1-margin, y1-margin), (x2+margin, y2+margin), color, 2)
            
            # Simple label
            label = f"{class_name} {confidence:.1f}"
            cv2.putText(output_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Detailed counters
        y_offset = height - 80
        cv2.putText(output_frame, f"Humans: {humans_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # Blue
        y_offset += 20
        cv2.putText(output_frame, f"Animals: {animals_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red
        y_offset += 20
        cv2.putText(output_frame, f"Objects: {objects_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black
        y_offset += 20
        cv2.putText(output_frame, f"Vehicles: {vehicles_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow
        
        return output_frame
    
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def start_camera_detection(self):
        """Start ultra-fast camera detection"""
        print(f"Starting Ultra-Fast Detection (Camera {self.camera_index})...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        # Ultra-fast camera settings for minimal latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Camera settings for bright lighting conditions
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)  # Reduce brightness
        cap.set(cv2.CAP_PROP_CONTRAST, 120)   # Increase contrast
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)    # Reduce exposure for bright light
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure control
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        self.is_running = True
        print("Ultra-fast detection started!")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Ultra-fast detection
            detections = self.detect_objects(frame)
            
            # Minimal visualization
            annotated_frame = self.draw_detections(frame, detections)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display frame
            cv2.imshow('Ultra-Fast YOLOv8 Detection', annotated_frame)
            
            # Handle key presses with minimal delay
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting detection...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        print("Detection stopped.")

def main():
    """Main function"""
    print("=== Ultra-Fast YOLOv8 Detection System ===")
    print("Using YOLOv8-S small for better accuracy")
    print("Auto-detecting available camera")
    print()
    
    # Auto-configured settings
    confidence = 0.4  # Auto-selected confidence threshold
    
    # Auto-detect available camera
    camera_index = 0  # Start with default camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera {camera_index} not available, trying camera 1...")
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("No cameras available. Using default camera 0.")
            camera_index = 0
    cap.release()
    
    print(f"Using camera: {camera_index}")
    print(f"Confidence threshold: {confidence}")
    print("Brightness adjustment: -40 (reduced for bright lighting)")
    print("Contrast adjustment: 1.3x (increased for better visibility)")
    print("Starting detection...")
    
    # Initialize and start detection
    try:
        detector = UltraFastDetection(
            confidence_threshold=confidence,
            camera_index=camera_index,
            brightness_adjustment=-40,  # Reduce brightness for bright lighting
            contrast_adjustment=1.3     # Increase contrast for better visibility
        )
        detector.start_camera_detection()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
