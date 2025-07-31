# Spatial-Temporal Transformer Network with MediaPipe
<<<<<<< HEAD

This repository implements a **Spatiotemporal Transformer Network (ST-TR)** for **gesture/action recognition** using **MediaPipe** to extract keypoints. It provides a complete pipeline for data processing, visualization, training, and inference, enabling end-to-end gesture recognition from videos.

=======

This repository implements a **Spatiotemporal Transformer Network (ST-TR)** for **gesture/action recognition** using **MediaPipe** to extract keypoints. It provides a complete pipeline for data processing, visualization, training, and inference, enabling end-to-end gesture recognition from videos.

![](https://github.com/HowardZhangUF/Spatial-Temporal-Transformer-Network-Mediapipe/blob/main/videoDemo.gif)
>>>>>>> 76bcc66ac4e619c9b4f0c28e35e13ba3be8d387c
![](https://github.com/HowardZhangUF/Spatial-Temporal-Transformer-Network-Mediapipe/blob/main/demo.gif)
![](https://github.com/HowardZhangUF/Spatial-Temporal-Transformer-Network-Mediapipe/blob/main/videoDemo.gif)

---

## Table of Contents

1. [Introduction](#introduction)
<<<<<<< HEAD
2. [Dataset](#dataset)
3. [Installation & Dependencies](#installation--dependencies)
4. [Keypoint Visualization](#keypoint-visualization)
5. [Training the Transformer](#training-the-transformer)
6. [Running Demos](#running-demos)
6. [License & Credits](#license--credits)

---

## Dataset
=======
2. [Installation & Dependencies](#installation--dependencies)
3. [Dataset Preparation](#dataset-preparation)
4. [Keypoint Visualization](#keypoint-visualization)
5. [Training the Transformer](#training-the-transformer)
6. [Running Demos](#running-demos)
7. [Results & Performance](#results--performance)
8. [License & Credits](#license--credits)

---

## Introduction

Gesture and action recognition plays a crucial role in Human-Computer Interaction (HCI), robotics, and AR/VR. This project leverages:

* **MediaPipe** for efficient and robust keypoint extraction of hands, body, and holistic poses.
* **Spatiotemporal Transformer Networks (ST-TR)** to model temporal dependencies and spatial correlations across keypoints.
>>>>>>> 76bcc66ac4e619c9b4f0c28e35e13ba3be8d387c

We use the **Scuba Gesture Dataset (SDG11)** for training and evaluation. Please refer to the original repository for dataset access:

<<<<<<< HEAD
[Scuba Gesture Dataset (SDG11)](https://github.com/abubake/Scuba-Gesture-Dataset.git)
=======
The result is a pipeline that can recognize gestures from videos in real time.
>>>>>>> 76bcc66ac4e619c9b4f0c28e35e13ba3be8d387c

---
## Introduction

Gesture and action recognition plays a crucial role in Human-Computer Interaction (HCI), robotics, and AR/VR. This project leverages:

* **MediaPipe** for efficient and robust keypoint extraction of hands, body, and holistic poses.
* **Spatiotemporal Transformer Networks (ST-TR)** to model temporal dependencies and spatial correlations across keypoints.


The result is a pipeline that can recognize gestures from videos in real time.

---

## Dataset

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

Train the Spatiotemporal Transformer model using prepared datasets.

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




## License & Credits

* **License**: MIT
* **Credits**:

  * [MediaPipe](https://github.com/google/mediapipe) for keypoint detection.
  * [PyTorch](https://pytorch.org/) for deep learning.
  
---

## Future Work

* Extend dataset with more diverse gestures.
* Optimize for mobile/embedded devices.
* Integrate multimodal data (RGB + keypoints). polish the  
