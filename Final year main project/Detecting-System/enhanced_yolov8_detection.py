#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
import time

class FastYOLOv8Detection:
    def __init__(self, confidence_threshold=0.3, camera_index=0, model_size='l'):
        """Initialize the fast YOLOv8 detection system"""
        self.confidence_threshold = confidence_threshold
        self.camera_index = camera_index
        self.model_size = model_size
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8-{model_size.upper()} model...")
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"✓ Loaded YOLOv8-{model_size.upper()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Simple class mapping
        self.class_categories = {
            'person': 'human',  
            'bird': 'animal', 'cat': 'animal', 'dog': 'animal', 'horse': 'animal', 
            'sheep': 'animal', 'cow': 'animal', 'elephant': 'animal', 'bear': 'animal', 
            'zebra': 'animal', 'giraffe': 'animal', 
        }
        
        # Simple color mapping
        self.colors = {
            'human': (255, 0, 0),      # Blue for humans
            'animal': (0, 255, 0),     # Green for animals
            'vehicle': (0, 0, 255),    # Red for vehicles
            'object': (255, 255, 0),   # Cyan for objects
            'default': (255, 255, 255) # White for others
        }
        
        # Animals list
        self.animals = {
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        }
        
        print("Fast YOLOv8 detection system initialized!")
        print(f"Detection capabilities:")
        print(f"  👤 Humans: 1 type (person)")
        print(f"  🦁 Animals: {len(self.animals)} species")
        print(f"  🚗 Vehicles: 8 types")
        print(f"  🏠 Objects: 50+ categories")
    
    def get_object_category(self, class_name):
        """Get category for detected object"""
        class_name_lower = class_name.lower()
        
        # Check if it's an animal
        if class_name_lower in self.animals:
            return 'animal'
        
        return self.class_categories.get(class_name_lower, 'object')
    
    def is_animal(self, class_name):
        """Check if detected object is an animal"""
        return class_name.lower() in self.animals
    
    def detect_objects(self, frame):
        """Fast object detection with YOLOv8"""
        all_detections = []
        
        try:
            # Run detection with optimized settings
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.4,
                verbose=False
            )
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.model.names[class_id]
                        category = self.get_object_category(class_name)
                        
                        # Skip unwanted objects
                        if category == 'skip':
                            continue
                        
                        # Check if animal
                        is_animal = self.is_animal(class_name)
                        
                        detection = {
                            'box': box,
                            'confidence': conf,
                            'class_name': class_name,
                            'category': category,
                            'is_animal': is_animal
                        }
                        all_detections.append(detection)
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return all_detections
    
    def draw_detections(self, frame, detections):
        """Simple visualization"""
        height, width = frame.shape[:2]
        output_frame = frame.copy()
        
        # FPS display
        cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (width - 120, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detections
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            category = detection['category']
            is_animal = detection.get('is_animal', False)
            
            # Get color
            if is_animal:
                color = self.colors['animal']
            else:
                color = self.colors.get(category, self.colors['default'])
            
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if is_animal:
                label = f"🦁 {class_name} ({confidence:.2f})"
            else:
                label = f"{class_name} ({confidence:.2f})"
            
            # Simple label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Simple summary
        animals_found = sum(1 for d in detections if d.get('is_animal', False))
        total_detections = len(detections)
        
        cv2.putText(output_frame, f"Animals: {animals_found}", (20, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Total: {total_detections}", (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def start_camera_detection(self):
        """Start fast real-time camera detection"""
        print(f"Starting Fast YOLOv8 Detection (Camera {self.camera_index})...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        # Fast camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        self.is_running = True
        print("Fast detection started!")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Fast detection
            detections = self.detect_objects(frame)
            
            # Simple visualization
            annotated_frame = self.draw_detections(frame, detections)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display frame
            cv2.imshow('Fast YOLOv8 Detection', annotated_frame)
            
            # Handle key presses
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
    """Main function to run fast detection"""
    print("=== Fast YOLOv8 Detection System ===")
    print("Optimized for low latency")
    print()
    
    # Model selection
    print("Available YOLOv8 models:")
    print("1. YOLOv8-N (nano) - Fastest")
    print("2. YOLOv8-S (small) - Balanced")
    print("3. YOLOv8-M (medium) - Good accuracy")
    print("4. YOLOv8-L (large) - Highest accuracy")
    print("5. YOLOv8-X (xlarge) - Maximum accuracy")
    
    model_choice = input("\nSelect model (1-5) [4]: ").strip()
    model_sizes = ['n', 's', 'm', 'l', 'x']
    model_size = model_sizes[int(model_choice) - 1] if model_choice.isdigit() and 1 <= int(model_choice) <= 5 else 'l'
    
    # Camera selection
    print("\nCamera options:")
    print("1. Default camera (0)")
    print("2. External camera (1)")
    
    choice = input("\nSelect camera option (1-2) [1]: ").strip()
    camera_index = 0 if choice != "2" else 1
    
    # Confidence threshold
    conf_input = input("Confidence threshold (0.1-1.0) [0.3]: ").strip()
    confidence = float(conf_input) if conf_input else 0.3
    confidence = max(0.1, min(1.0, confidence))
    
    # Initialize and start detection
    try:
        detector = FastYOLOv8Detection(
            confidence_threshold=confidence,
            camera_index=camera_index,
            model_size=model_size
        )
        detector.start_camera_detection()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
