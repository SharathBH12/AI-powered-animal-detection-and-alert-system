# Multi-Model Animal Detection System

A comprehensive object detection system that can detect animals, humans, and objects in both day and night conditions using multiple YOLOv8 models with advanced features.

## Features

### 🎯 **Multi-Model Detection**
- **YOLOv8 Models**: Nano, Small, Medium, Large, and Extra Large variants
- **Model Fusion**: Combines results from multiple models for better accuracy
- **Custom Models**: Support for custom-trained models on your dataset

### 🌙 **Night Detection**
- **Automatic Night Detection**: Detects low-light conditions automatically
- **Image Enhancement**: Advanced algorithms for night image processing
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
- **Noise Reduction**: Denoising for better night detection
- **Gamma Correction**: Brightness adjustment for low-light images

### 🐾 **Animal Classes Supported**
The system can detect the following animals:
- **Wild Animals**: Bear, Deer, Elephant, Fox, Giraffe, Kangaroo, Leopard, Lion, Tiger, Wolf, Hyena, Rhinoceros, Panda
- **Domestic Animals**: Cow, Dog, Donkey, Goat, Horse, Pig, Sheep, Ox, Bison
-

### 👤 **Human and Object Detection**
- **Humans**: Person detection
- **Vehicles**: Car, Bicycle, Motorcycle, Bus, Train, Truck, Boat, Airplane
- **Electronics**: TV, Laptop, Mouse, Remote, Keyboard, Cell Phone
- **Kitchen Items**: Microwave, Oven, Toaster, Sink, Refrigerator
- **Utensils**: Bottle, Wine Glass, Cup, Fork, Knife, Spoon, Bowl
- **Traffic Objects**: Traffic Light, Fire Hydrant, Stop Sign, Parking Meter, Bench

### 📊 **Advanced Features**
- **Real-time Processing**: Live video and webcam support
- **Detection History**: Tracks detections across frames
- **Enhanced Visualization**: Rich annotations with confidence scores and model information
- **Performance Metrics**: FPS tracking and progress monitoring
- **Result Export**: Save detection results in JSON format

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- At least 8GB RAM

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download models** (optional - models will be downloaded automatically on first use):
```bash
python download_models.py
```

## Usage

### Quick Start

1. **Run the main detection system**:
```bash
python animal_detection_system.py
```

2. **Or use the enhanced version**:
```bash
python enhanced_detection_system.py
```

3. **Follow the interactive menu**:
   - Process single images
   - Process video files
   - Use webcam for real-time detection
   - Process your animals dataset

### Training Custom Models

If you want to train custom models on your animals dataset:

```bash
python train_custom_model.py
```

This will:
- Prepare your dataset in YOLO format
- Train multiple YOLOv8 model sizes
- Evaluate model performance
- Test models on sample images

### Advanced Usage

#### Process Images
```python
from enhanced_detection_system import EnhancedDetectionSystem

# Initialize detector
detector = EnhancedDetectionSystem(confidence_threshold=0.5)

# Process single image
image = cv2.imread("path/to/image.jpg")
detections = detector.detect_objects_multi_model(image, enhance_night=True)
annotated_image = detector.draw_enhanced_detections(image, detections)
cv2.imwrite("output.jpg", annotated_image)
```

#### Process Video
```python
# Process video with night enhancement
detector.process_video_enhanced(
    video_path="input_video.mp4",
    output_path="output_video.mp4",
    enhance_night=True,
    save_detections=True
)
```

#### Webcam Detection
```python
# Real-time webcam detection
detector.process_webcam(camera_id=0, enhance_night=True)
```

## File Structure

```
part2/
├── animals/                    # Your animals dataset
│   ├── bison/
│   ├── cow/
│   ├── deer/
│   └── ...
├── animal_detection_system.py  # Basic detection system
├── enhanced_detection_system.py # Advanced detection system
├── download_models.py          # Model downloader
├── train_custom_model.py       # Custom model trainer
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── models/                     # Downloaded models (created automatically)
```

## Configuration

### Detection Parameters

You can adjust detection parameters when initializing the system:

```python
detector = EnhancedDetectionSystem(
    confidence_threshold=0.5,  # Minimum confidence (0.0-1.0)
    nms_threshold=0.4          # Non-maximum suppression threshold
)
```

### Night Detection Parameters

Adjust night detection settings:

```python
detector.night_detection_params = {
    'brightness_threshold': 50,    # Brightness threshold for night detection
    'contrast_enhancement': True,  # Enable CLAHE enhancement
    'noise_reduction': True,       # Enable noise reduction
    'gamma_correction': 1.3        # Gamma correction value
}
```

## Performance Tips

### For Better Speed
- Use YOLOv8n (nano) model for fastest inference
- Reduce input image size
- Use GPU acceleration if available
- Lower confidence threshold

### For Better Accuracy
- Use YOLOv8x (extra large) model
- Increase input image size
- Use model fusion (combines multiple models)
- Higher confidence threshold

### For Night Detection
- Enable all night enhancement features
- Use larger models for better night performance
- Consider custom training on night images

## Training Custom Models

### Dataset Preparation

The system automatically prepares your animals dataset in YOLO format:

1. **Organize your dataset** in folders by class:
```
animals/
├── bison/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── cow/
│   ├── image1.jpg
│   └── ...
└── ...
```

2. **Run training**:
```bash
python train_custom_model.py
```

3. **Choose training options**:
   - Model size (n/s/m/l/x)
   - Number of epochs
   - Batch size
   - Image size

### Training Results

Trained models will be saved in:
```
custom_models/
├── yolov8n_animals/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   └── results/
├── yolov8s_animals/
└── ...
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Use smaller model (YOLOv8n)
   - Reduce input image size

2. **Slow performance**:
   - Use GPU acceleration
   - Use smaller model
   - Reduce input resolution

3. **Poor night detection**:
   - Enable all night enhancement features
   - Use larger models
   - Train on night images

4. **Model download issues**:
   - Check internet connection
   - Run `python download_models.py` manually
   - Download models from Ultralytics website

### System Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, CUDA GPU
- **Optimal**: 32GB RAM, RTX 3080 or better

## Examples

### Example 1: Process Animals Dataset
```bash
python enhanced_detection_system.py
# Choose option 4: Process animals dataset
```

### Example 2: Real-time Webcam Detection
```bash
python enhanced_detection_system.py
# Choose option 3: Use webcam
```

### Example 3: Train Custom Model
```bash
python train_custom_model.py
# Choose option 5: Full training pipeline
```

## Contributing

Feel free to contribute to this project by:
- Adding new animal classes
- Improving night detection algorithms
- Optimizing performance
- Adding new features

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with different parameters
4. Ensure all dependencies are installed

---

**Happy detecting! 🐾🔍**
