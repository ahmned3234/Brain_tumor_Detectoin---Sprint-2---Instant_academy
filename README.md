[brain-tumor-classification-segmentation-README.md](https://github.com/user-attachments/files/31123816/brain-tumor-classification-segmentation-README.md)
# Brain Tumor Classification & Segmentation

A deep learning computer vision application that performs both brain tumor classification and tumor-region segmentation from MRI scans.

## Overview

The project combines two deep learning approaches:

1. **ResNet50** for classifying MRI images as healthy or tumor.
2. **U-Net** for identifying and segmenting the tumor region within an MRI image.

The models were integrated into an interactive **Streamlit** application where users can upload an MRI scan and view the corresponding predictions.

## System Workflow

```text
MRI Scan
   │
   ├──────────────► ResNet50
   │                    │
   │                    ▼
   │              Healthy / Tumor
   │
   └──────────────► U-Net
                        │
                        ▼
                 Tumor Segmentation
```

## Models

### Classification — ResNet50

A pre-trained **ResNet50** architecture was used for binary image classification:

- Healthy
- Tumor

Transfer learning was used to adapt the model to the MRI classification task.

### Segmentation — U-Net

A **U-Net** architecture was used to produce a pixel-level segmentation mask identifying the tumor region.

The segmentation training used a combination of:

- Binary Cross-Entropy loss
- Dice loss

This combination was selected to support both pixel-level classification and overlap quality for the tumor region.

## Application

The trained models were integrated into a Streamlit interface.

Users can:

1. Upload an MRI image.
2. Run the classification model.
3. View whether the image is classified as healthy or tumor.
4. Run the segmentation model.
5. Visualize the predicted tumor region.

## Tech Stack

- Python
- TensorFlow
- Keras
- ResNet50
- U-Net
- OpenCV / image preprocessing
- Streamlit
- NumPy

## What I Learned

This project provided hands-on experience with computer vision pipelines, transfer learning, deep learning model training, image preprocessing, classification, semantic segmentation, and deploying multiple AI models through an interactive application.

## Project Scope

This project was developed as an AI/Computer Vision learning project. The model outputs are intended for demonstration and educational purposes and are **not a substitute for professional medical diagnosis**.
