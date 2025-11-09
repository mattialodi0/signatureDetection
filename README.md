# 🖋️ Signature Detection
Handwritten Signature Detection using Computer Vision 
## 📚 Overview
This project focuses on detecting **handwritten signatures** in scanned documents using modern object detection models.
The dataset and models were designed to generalize well to real-world documents similar to those encountered in production use cases.

## 📦 Dataset
To achieve the best results, the training dataset combines **publicly available datasets** with **custom data** reflecting the target domain.
Each image is labeled with bounding boxes identifying signature regions (when present).
The script `dataset.py` handles:
- Dataset assembly from multiple sources
- Optional data augmentation
- Encoding and conversion to efficient training formats

## 🗂️ Data Sources
- scanned-images-dataset-for-ocr-and-vlm-finetuning
- tobacco-800-dataset
- nist-special-database-2
- Proprietary real-use-case data

### 🧠 Data Augmentation
To mitigate bias due to consistent layout patterns in some datasets (e.g., NIST, where all signatures appear in the same position), light augmentation was applied:
**Horizontal mirroring** (Y-axis)
**Vertical shifting**
These transformations help the model generalize across different document formats and signature placements.

### 🧩 Dataset Format
Training efficiency was significantly improved by converting the dataset to the **WebDataset** format, which reduced training time by approximately **50%**.
<hr/>

## 🧠 Models
### 🔹 Base Classifier
A simple classifier built from a **pretrained CNN**.
- **Accuracy:** ~60%
- **Advantages:** Very fast to train
- **Limitations:** Low overall accuracy
---
### 🔹 Faster R-CNN Detector
Object detection model based on **Faster R-CNN**.
**Training note:** Including images without signatures caused the model to receive false rewards — hence, only labeled images were used for training.
**Backbone:** MobileNet
**Performance:**
- Precision: **0.765**
- Recall: **1.000**
---
### 🔹 RetinaNet Detector
Implementation planned / under development.
---
### 🔹 YOLOv8 Detector
A **YOLOv8**-based model showed the best overall performance.
- **Confidence threshold:** 0.4
**Performance metrics:**
- Precision: **1.000**
- Recall: **0.895**
- mAP@50: **0.857**
- mAP@95: **0.548**
## ⚙️ Future Work
- Add RetinaNet implementation
- Explore transformer-based detectors (e.g., DETR)
- Build a simple model from scratch