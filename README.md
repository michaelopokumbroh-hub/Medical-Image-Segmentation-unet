# Medical-Image-Segmentation-unet
Binary semantic segmentation using UNet and ResNet34+UNet on the Oxford-IIIT Pet Dataset with PyTorch.

# 🧠 Medical Image Segmentation (UNet & ResNet34+UNet)

## 📌 Overview
This project implements binary semantic segmentation on the Oxford-IIIT Pet Dataset using deep learning techniques.

Two architectures were developed and compared:
- UNet
- ResNet34 + UNet (Hybrid Encoder-Decoder)

## 🎯 Objectives
- Perform binary segmentation (pet vs background)
- Compare deep learning architectures
- Optimize Dice Score for segmentation accuracy

## 🧠 Methodology

### Data Preparation
- Dataset: Oxford-IIIT Pet Dataset
- Image resizing: 128×128
- Mask conversion: Trimap → Binary mask

### Models
- UNet (baseline encoder-decoder)
- ResNet34 encoder + UNet decoder

### Training
- Loss: Binary Cross Entropy (BCEWithLogitsLoss)
- Optimizer: Adam (lr = 3e-4)
- Epochs: 30 (UNet), 40 (ResNet34+UNet)

### Evaluation Metric
- Dice Score (primary metric)

## 📊 Results

| Model | Best Dice Score |
|------|---------------|
| UNet | **0.899** |
| ResNet34+UNet | ~0.874 |

- Based on validation results, the standard UNet achieved a superior dice score of 0.8957, compared to 0.8738 for the ResNet34-UNet. This shows that the standard UNet's architecture was more effective at capturing the specific spatial features of the pet dataset at this resolution. However the ResNet34-UNet was more than twice as fast, processing at 9.12 it/s compared to the UNet's 4.17 it/s.

  
## 🚀 Inference
- Generated segmentation masks on test set
- Converted masks to RLE format
- Submitted to Kaggle leaderboard. UNet = 0.89905 | ResNet34+UNet = 0.86686

## 🛠 Technologies
- Python
- PyTorch
- NumPy
- PIL
- tqdm

## 📷 Sample Output
- For UNet
<img width="1485" height="541" alt="frame_029" src="https://github.com/user-attachments/assets/568d6372-1496-4593-8f06-a3f7ee369324" />

- For ResNet34+UNet
<img width="500" height="376" alt="american_bulldog_84" src="https://github.com/user-attachments/assets/79767026-daa6-4d56-a785-2d48342331c5" />

## 🎯 Future Improvements
- Use Dice Loss or BCE + Dice Loss
- Data augmentation
- Transfer learning with pretrained ResNet
- Apply to medical imaging (MRI/CT)

## 👨‍⚕️ Author
Michael Opoku Mbroh  
MSc Biomedical Imaging & Radiological Science
