# YOLO Object Detection Project - Red Cup, Blue Bottle, and Phone Detection

## Project Overview

This project implements real-time object detection using YOLOv8 to detect three classes of everyday objects:

- **Red Cup** 🔴
- **Blue Bottle** 🔵
- **Phone** 📱

The project demonstrates Joseph Redmon's groundbreaking YOLO (You Only Look Once) architecture, which performs object detection in a single pass through a neural network, enabling real-time performance.

## Dataset

### Collection Method

The dataset was collected using publicly available sources:

1. **Bing Image Search API** - Automated download of labeled images
2. **Manual annotation** - Using YOLO format (normalized bounding box coordinates)
3. **Data organization** - Structure in `images/` and `labels/` directories

### Dataset Statistics

- **Total Images**: ~150 images (before split)
- **Classes**: 3 (red_cup, blue_bottle, phone)
- **Train/Val Split**: 80% training (120 images), 20% validation (30 images)
- **Image Format**: JPG/PNG
- **Label Format**: YOLO `.txt` format (class_id, x_center, y_center, width, height - normalized to 0-1)

### Directory Structure

```:
3.9_lab_yolo_object_detection/
├── images/              # Original images
├── labels/              # YOLO format labels
├── train/               # Training dataset
│   ├── images/
│   └── labels/
├── val/                 # Validation dataset
│   ├── images/
│   └── labels/
├── runs/                # Training outputs
│   └── detect/train/weights/best.pt
├── data.yaml            # YOLO dataset configuration
├── yolo.ipynb           # Main notebook
└── requirements.txt     # Python dependencies
└── README.md
└── lab_report.md
└── download_dataset.py
└── organize_images.py
└── test_image.jpg
└── test_video.mp4
```

## Environment Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

**Option 1: Using pip**

```bash
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection
pip install -r requirements.txt
```

**Option 2: Using conda** (Recommended)

```bash
conda create -n yolo python=3.9
conda activate yolo
pip install -r requirements.txt
```

### Key Dependencies

- **ultralytics** >= 8.0.0 - YOLO implementation
- **torch** >= 1.9.0 - Deep learning framework
- **torchvision** >= 0.10.0 - Computer vision utilities
- **opencv-python** >= 4.5.0 - Image processing
- **numpy** >= 1.21.0 - Numerical computing
- **matplotlib** >= 3.3.0 - Visualization
- **scikit-learn** >= 0.24.0 - ML utilities

## Usage Guide

### 1. Data Preparation

```python
# Cell 1: Set working directory
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection

# Cells 2-3: Check dataset and split into train/val
# Run cells 2-3 to organize data (already done if using provided structure)
```

### 2. Training

```python
# Cell 4: Run YOLO training
# The model trains for 10 epochs on CPU
# Takes approximately 2-4 hours depending on hardware
```

**Training Parameters:**

- **Model**: YOLOv8n (Nano - fastest, smallest)
- **Epochs**: 10
- **Image Size**: 416x416
- **Batch Size**: 8
- **Device**: CPU
- **Early Stopping**: 10 epochs patience

### 3. Evaluation

```python
# Cell 5: Evaluate trained model on validation set
# Displays mAP50, mAP50-95, recall, precision metrics
```

### 4. Single Image Detection

```python
# Cell 6: Run detection on sample image
# Visualizes detections with bounding boxes and confidence scores
```

### 5. Real-time Detection

```python
# Cell 7: Real-time detection function
# Option 1: Webcam detection
run_detector_on_video(model, video_source=0, confidence_threshold=0.5)

# Option 2: Video file detection
run_detector_on_video(model, video_source='./test_video.mp4', output_path='output.mp4')

# Option 3: Single image detection
detect_on_image(model, image_path='./test_image.jpg')
```

## YOLO Architecture Advantages

### Single-Pass Detection (The Key Innovation)

YOLO's fundamental advantage over previous methods like R-CNN:

- **Traditional approaches** (R-CNN): Generate region proposals → Classify each region → Multiple passes = Slow
- **YOLO**: Split image into grid → Predict bounding boxes & class probabilities in single pass = Fast

### Speed vs Accuracy Trade-offs

- **YOLOv8n (Nano)**: ~15-25 FPS on CPU, suitable for resource-constrained environments
- **YOLOv8m (Medium)**: ~5-10 FPS on CPU, better accuracy
- **YOLOv8x (Extra-Large)**: Higher accuracy on GPU, slower on CPU

## Real-time Performance Metrics

When running real-time detection, the notebook displays:

- **FPS**: Frames per second
- **Inference Time**: Time per frame (ms)
- **Detection Count**: Number of objects detected
- **Confidence Scores**: Probability of each detection

Example Output:
```
FPS: 12.5 | Detections: 3 | Time: 80.0ms
```

## Example Labeled Images

### Image Format

```
Image: example.jpg
Label: example.txt

Label file content (example):
0 0.45 0.50 0.30 0.40  # red_cup, center at (0.45, 0.50), width=0.30, height=0.40
1 0.70 0.60 0.25 0.35  # blue_bottle, ...
2 0.20 0.30 0.15 0.20  # phone, ...
```

### Visualizing Labeled Data

To visualize labeled images with annotations:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='val/images/', conf=0.25)
```

## Training Results

### Training Outputs

The training process generates:

- **Best weights**: `runs/detect/train/weights/best.pt`
- **Last weights**: `runs/detect/train/weights/last.pt`
- **Logs**: Training curves and metrics
- **Results.csv**: Epoch-by-epoch metrics

### Key Metrics

| Metric         | Definition                            |
|----------------|---------------------------------------|
| **mAP50**      | Mean Average Precision at IoU=0.5     |
| **mAP50-95**   | Mean AP across IoU 0.5-0.95           |
| **Precision**  | TP / (TP + FP)                        |
| **Recall**     | TP / (TP + FN)                        |
| **F1 Score**   | Harmonic mean of precision and recall |

### Expected Results

On this small dataset (~150 images):

- **mAP50**: 60-75%
- **Precision**: 70-80%
- **Recall**: 65-75%

Note: Performance varies based on:

- Image quality and diversity
- Annotation accuracy
- Training epochs
- Hardware (CPU vs GPU)

## Real-World Performance Analysis

### Strengths of the Detector

1. **Speed**: Single-pass detection enables real-time processing
2. **Simplicity**: One network for bounding box and classification
3. **Global Context**: Sees entire image at once (vs regional methods)

### Limitations Observed

1. **Small Dataset**: Only 150 images may lead to overfitting
2. **Background Variation**: Performance varies with different lighting/angles
3. **Small Objects**: May struggle with small instances
4. **False Positives**: Can misidentify objects in clutter

### Improvement Strategies

1. **Data Augmentation**: Apply rotation, scaling, brightness adjustments
2. **More Training Data**: Collect 500-1000+ images per class
3. **Larger Model**: Use YOLOv8m or YOLOv8l for better accuracy
4. **Fine-tuning**: Train longer (30+ epochs) with learning rate scheduling
5. **Hard Negative Mining**: Include challenging negative samples

## Troubleshooting

### Issue: "RuntimeError: Numpy is not available"

**Solution**: Restart kernel and ensure numpy is imported in training cell

```python
import numpy as np  # Add this line before model.train()
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use CPU

```python
device='cpu'  # or reduce batch from 16 to 8
```

### Issue: No validation data loading

**Solution**: Verify data.yaml path and structure

```bash
# Check file structure
ls -la train/images/ val/images/
```

## References

1. **YOLO Paper**: "You Only Look Once: Unified, Real-Time Object Detection" - Joseph Redmon et al. (2015)
2. **YOLOv8 Documentation**: https://docs.ultralytics.com/
3. **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics

## Project Timeline

| Phase             | Time      | Tasks                           |
|-------------------|-----------|---------------------------------|
| Data Collection   | 2-3 hours | Download and organize images    |
| Annotation        | 3-4 hours | Label objects in YOLO format    |
| Preprocessing     | 30 min    | Split train/val, create config  |
| Training          | 2-4 hours | Train on CPU                    |
| Evaluation        | 30 min    | Test on validation set          |
| Real-time Testing | 1-2 hours | Webcam/video detection          |

## Conclusion

This project successfully implements a real-time object detector for three common objects using YOLOv8's efficient single-pass architecture. While the current dataset is small, it demonstrates the core concepts of modern object detection and highlights the practical advantages of YOLO's design over traditional region-based approaches.

Future improvements would focus on expanding the dataset, utilizing GPU acceleration, and implementing advanced techniques like ensemble methods and test-time augmentation.

---

**Author**: Siying Rong Deep Learning Lab
**Date**: March 2026  
**Framework**: YOLOv8 (Ultralytics)  
**License**: MIT
