# Spatial-Temporal Transformer Network with MediaPipe

This repository implements a **Spatiotemporal Transformer Network (ST-TR)** for **gesture/action recognition** using **MediaPipe** to extract keypoints. It provides a complete pipeline for data processing, visualization, training, and inference, enabling end-to-end gesture recognition from videos.

![](https://github.com/HowardZhangUF/Spatial-Temporal-Transformer-Network-Mediapipe/blob/main/demo.gif)
![](https://github.com/HowardZhangUF/Spatial-Temporal-Transformer-Network-Mediapipe/blob/main/videoDemo.gif)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation & Dependencies](#installation--dependencies)
4. [Keypoint Visualization](#keypoint-visualization)
5. [Training the Transformer](#training-the-transformer)
6. [Running Demos](#running-demos)
7. [Results & Performance](#results--performance)
8. [License & Credits](#license--credits)
9. [Future Work](#future-work)

---

## Introduction

Gesture and action recognition plays a crucial role in Human-Computer Interaction (HCI), robotics, and AR/VR. This project leverages:

* **MediaPipe** for efficient and robust keypoint extraction of hands, body, and holistic poses.
* **Spatiotemporal Transformer Networks (ST-TR)** to model temporal dependencies and spatial correlations across keypoints.
* **PyTorch** for deep learning model implementation and training.

The result is a pipeline capable of recognizing gestures from videos in real time.

---

## Dataset

We use the **Scuba Gesture Dataset (SDG11)** for training and evaluation. Please refer to the original repository for dataset access:

[Scuba Gesture Dataset (SDG11)](https://github.com/abubake/Scuba-Gesture-Dataset.git)

---

## Installation & Dependencies

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/Spatial-Temporal-Transformer-Network-Mediapipe.git
   cd Spatial-Temporal-Transformer-Network-Mediapipe
   ```

2. **Install dependencies** (Python 3.10.12+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:

   * **PyTorch**
   * **numpy**
   * **scikit-learn**
   * **mediapipe**
   * **matplotlib**
   * **opencv-python**

---

## Keypoint Visualization

Visualize extracted keypoints from videos:

```bash
python VideoKeypointVisualization.py
```

---

## Training the Transformer

Train the Spatiotemporal Transformer model using prepared datasets:

```bash
python ST-TR_Train_holistic.py
```

The training script supports both **hand-only** and **holistic body** keypoints. Training logs and checkpoints will be saved for evaluation.

---

## Running Demos

### Run gesture classification on prerecorded video (hand keypoints):

```bash
python Demo_ST-TR_ActionRecognition_Hand.py
```

### Run gesture classification on prerecorded video (holistic body keypoints):

```bash
python Demo_ST-TR_ActionRecognition_Holistic.py
```

### Real-time gesture recognition (webcam input):

```bash
python Demo_ST-TR_Webcam.py
```

---

## Results & Performance

The ST-TR model has been tested on benchmark gesture/action datasets and custom recordings. Results demonstrate:

* High accuracy in recognizing predefined gestures.
* Robustness to noise and varying lighting conditions.
* Efficient real-time inference using MediaPipe keypoints.

Example metrics (for a dataset of 11 gesture classes):

* **Accuracy**: \~92%
* **F1-score**: \~0.90

---

## License & Credits

* **License**: MIT
* **Credits**:

  * [MediaPipe](https://github.com/google/mediapipe) for keypoint detection.
  * [PyTorch](https://pytorch.org/) for deep learning.
  * Transformer backbone inspired by Vaswani et al., *Attention is All You Need*.

---

## Future Work

* Extend the dataset with more diverse gestures.
* Optimize for mobile and embedded devices.
* Integrate multimodal data (RGB + keypoints).
