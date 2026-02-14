#!/usr/bin/env python3
"""
Enhanced YOLOv5 Detection System with CLAHE + Retinex Pre-processing
Provides superior image enhancement for better detection accuracy
"""

import cv2
import numpy as np
import torch
import yolov5
import os
import time
from skimage import exposure
from skimage.filters import unsharp_mask

class EnhancedYOLOv5Detection:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        self.setup_colors()
        
    def load_model(self):
        """Load YOLOv5 model"""
        print("🔄 Loading YOLOv5 model...")
        try:
            # Load YOLOv5 model
            self.model = yolov5.load('yolov5s.pt')
            self.model.to(self.device)
            print(f"✅ YOLOv5 model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📥 Downloading YOLOv5 model...")
            self.model = yolov5.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
            self.model.to(self.device)
            print("✅ YOLOv5 model downloaded and loaded")
    
    def setup_colors(self):
        """Setup color scheme for different object categories"""
        self.colors = {
            'person': (255, 0, 0),      # Blue (BGR)
            'animal': (0, 0, 255),      # Red (BGR)
            'vehicle': (0, 255, 0),     # Green (BGR)
            'object': (0, 0, 0),        # Black (BGR)
            'default': (128, 128, 128)  # Gray (BGR)
        }
        
        # COCO classes
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Animal classes for special handling
        self.animal_classes = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]
        
        # Vehicle classes
        self.vehicle_classes = [
            'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'
        ]
    
    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def apply_retinex(self, image, sigma=30):
        """Apply Retinex algorithm for illumination correction"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        
        # Apply Retinex
        retinex = np.log(img_float + 1) - np.log(blurred + 1)
        
        # Normalize
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min())
        
        # Convert back to uint8
        enhanced = (retinex * 255).astype(np.uint8)
        return enhanced
    
    def apply_unsharp_mask(self, image, radius=1, amount=1.5):
        """Apply unsharp mask for edge enhancement"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply unsharp mask
        sharpened = unsharp_mask(img_float, radius=radius, amount=amount)
        
        # Convert back to uint8
        enhanced = (sharpened * 255).astype(np.uint8)
        return enhanced
    
    def enhance_image(self, image):
        """Apply comprehensive image enhancement pipeline"""
        # Step 1: CLAHE for contrast enhancement
        clahe_enhanced = self.apply_clahe(image)
        
        # Step 2: Retinex for illumination correction
        retinex_enhanced = self.apply_retinex(clahe_enhanced)
        
        # Step 3: Unsharp mask for edge enhancement
        final_enhanced = self.apply_unsharp_mask(retinex_enhanced)
        
        return final_enhanced
    
    def get_object_category(self, class_name):
        """Get object category for color coding"""
        class_lower = class_name.lower()
        
        if class_lower == 'person':
            return 'person'
        elif class_lower in self.animal_classes:
            return 'animal'
        elif class_lower in self.vehicle_classes:
            return 'vehicle'
        else:
            return 'object'
    
    def get_emoji(self, class_name):
        """Get emoji for object class"""
        class_lower = class_name.lower()
        
        if class_lower == 'person':
            return '👤'
        elif class_lower in ['cat', 'dog', 'horse', 'cow', 'sheep']:
            return '🐾'
        elif class_lower in ['elephant', 'bear', 'zebra', 'giraffe']:
            return '🦁'
        elif class_lower == 'bird':
            return '🐦'
        elif class_lower in ['car', 'truck', 'bus', 'motorcycle']:
            return '🚗'
        elif class_lower in ['bicycle', 'airplane', 'train', 'boat']:
            return '🚲'
        else:
            return '📦'
    
    def detect_objects(self, image, confidence_threshold=0.25):
        """Detect objects in image with enhanced pre-processing"""
        # Apply image enhancement
        enhanced_image = self.enhance_image(image)
        
        # Run YOLOv5 detection
        results = self.model(enhanced_image, size=640)
        
        # Process results
        detections = []
        if results.pred[0] is not None:
            for *xyxy, conf, cls in results.pred[0]:
                if conf >= confidence_threshold:
                    class_id = int(cls)
                    class_name = self.coco_classes[class_id] if class_id < len(self.coco_classes) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class': class_name,
                        'category': self.get_object_category(class_name)
                    })
        
        return detections, enhanced_image
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            category = det['category']
            emoji = self.get_emoji(class_name)
            
            # Get color based on category
            color = self.colors.get(category, self.colors['default'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{emoji} {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def camera_detection(self):
        """Real-time camera detection with enhanced pre-processing"""
        print("📹 Starting camera detection...")
        print("🎨 Enhanced with CLAHE + Retinex + Unsharp Mask")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for better performance
                if frame_count % 3 == 0:
                    try:
                        # Detect objects
                        detections, enhanced_frame = self.detect_objects(frame)
                        
                        # Draw detections
                        annotated_frame = self.draw_detections(enhanced_frame, detections)
                        
                        # Calculate FPS
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        
                        # Draw FPS and detection info
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "Enhanced: CLAHE + Retinex + Unsharp", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display frame
                        cv2.imshow('Enhanced YOLOv5 Detection', annotated_frame)
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                        cv2.imshow('Enhanced YOLOv5 Detection', frame)
                else:
                    # Show original frame for skipped frames
                    cv2.imshow('Enhanced YOLOv5 Detection', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"❌ Camera detection error: {e}")
            print("🔄 Trying alternative display method...")
            
            # Alternative: Save frames instead of displaying
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 10 == 0:  # Process every 10th frame
                    try:
                        detections, enhanced_frame = self.detect_objects(frame)
                        annotated_frame = self.draw_detections(enhanced_frame, detections)
                        
                        # Save frame instead of displaying
                        output_path = f"camera_frame_{frame_count}.jpg"
                        cv2.imwrite(output_path, annotated_frame)
                        print(f"📸 Saved frame {frame_count} with {len(detections)} detections")
                        
                        # Print detections to console
                        for det in detections:
                            emoji = self.get_emoji(det['class'])
                            print(f"   {emoji} {det['class']}: {det['confidence']:.2f}")
                            
                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                
                # Check for quit (simulate with keyboard input)
                if frame_count > 100:  # Stop after 100 frames
                    break
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("✅ Camera detection stopped")
    
    def image_detection(self, image_path):
        """Detect objects in a single image"""
        print(f"🖼️ Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image {image_path}")
            return
        
        # Detect objects
        detections, enhanced_image = self.detect_objects(image)
        
        # Draw detections
        annotated_image = self.draw_detections(enhanced_image, detections)
        
        # Save results
        output_path = f"enhanced_detected_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated_image)
        
        # Print results
        print(f"✅ Detection completed")
        print(f"📊 Found {len(detections)} objects")
        print(f"💾 Saved as: {output_path}")
        
        for det in detections:
            emoji = self.get_emoji(det['class'])
            print(f"   {emoji} {det['class']}: {det['confidence']:.2f}")
        
        return annotated_image

def main():
    """Main function"""
    print("🚀 Enhanced YOLOv5 Detection System")
    print("🎨 Features: CLAHE + Retinex + Unsharp Mask Pre-processing")
    print("=" * 50)
    
    detector = EnhancedYOLOv5Detection()
    
    while True:
        print("\nOptions:")
        print("1. Camera detection (real-time)")
        print("2. Image detection")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            detector.camera_detection()
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                detector.image_detection(image_path)
            else:
                print("❌ Error: Image file not found")
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
