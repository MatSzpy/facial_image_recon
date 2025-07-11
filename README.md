# Face Deblur and Reconstruction

## Overview
A Python-based project focused on detecting facial regions degraded by **Gaussian blur** and **pixelation**. The system aims to reconstruct blurred faces in images using image processing and machine learning techniques.

## Features

### `dataset_create`
- Automatically generates a dataset of blurred faces using the [IMDB-Wiki image dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
- Detects faces in the original images using the [YuNet face detection model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- Applies **Gaussian blur** and **pixelation** to facial regions.

### `blurred_faces_detect`
- Identifies and extracts blurred face regions from the processed images using image processing techniques.

### `face_deblur_model`
- Uses a neural network based on an **encoder-decoder** architecture.
- Reconstructs the original facial features using deep learning methods trained to reverse the degradation.

## Getting Started

### Requirements

- Python 3.9.2  
- OpenCV 4.9.0.80  
- NumPy 1.26.2  
- Matplotlib 3.3.4  
- scikit-image 0.24.0  
- PyTorch 2.7.1 + CUDA 11.8
