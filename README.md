# Deep-Fake-Detection

DeepFake Detection using CNN + LSTM | A PyTorch-based model to detect DeepFakes in videos by analyzing spatial and temporal features. Includes real-time Streamlit interface, data preprocessing, and model evaluation. Promotes awareness against misinformation and digital manipulation.

---

## Project Overview

The goal of this project is to develop a reliable system to classify videos as either real or DeepFake. The system processes video frames using a pretrained convolutional neural network (CNN) for feature extraction and uses a Long Short-Term Memory (LSTM) network to capture temporal dependencies across frames.

---

## Model Architecture

- **Feature Extractor**: ResNeXt50 (pretrained on ImageNet)
- **Sequence Model**: Two-layer LSTM
- **Classifier**: Fully connected layer with binary output
- **Loss Function**: Binary Cross Entropy with Logits
- **Optimizer**: AdamW with learning rate scheduling
- **Training**: Mixed precision (AMP) for efficiency

---

## Dataset and Preprocessing

- Collected real and fake videos, converted into tensors (.pt format)
- Preprocessing includes resizing, normalization, and basic augmentations
- **Augmentations applied**: random crop, jitter, blur, flip, rotation
- Used `train_test_split()` for balanced training and validation sets
- Extracted up to 30 frames per video for temporal modeling

---

## Evaluation

### Test Accuracy: 81%

### Confusion Matrix

![Confusion Matrix](https://github.com/user-attachments/assets/cc4d6c90-f74f-48b0-a987-f0350081f4c8)

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Real  | 0.76      | 0.90   | 0.83     |
| Fake  | 0.88      | 0.72   | 0.79     |

**Overall Accuracy**: 0.81

![Training and Validation plots](https://github.com/user-attachments/assets/7847cdc3-bffd-40a4-b5b9-786d963d33ec)


---

## Streamlit Interface

A web-based interface is developed using Streamlit to make predictions on uploaded video files.

### Features

- Upload `.mp4`, `.avi`, or `.mov` videos
- Frame extraction and preprocessing in real-time
- Display of prediction result: **Real** or **DeepFake**
- Styled UI with responsive layout

### Streamlit UI Screenshots

![Streamlit UI](https://github.com/user-attachments/assets/5e8be826-2d37-41ff-98b4-aef3a6af66fe)

![Streamlit UI](https://github.com/user-attachments/assets/5708dba4-f1b5-4587-9200-d03e301c018b)

---

