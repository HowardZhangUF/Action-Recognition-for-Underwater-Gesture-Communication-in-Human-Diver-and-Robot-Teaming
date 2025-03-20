# SPATIAL TEMPORAL TRANSFORMER NETWORK MEDIAPIPE

This repository implements a **Spatiotemporal Transformer** for **gesture/action recognition** using **MediaPipe** to extract keypoints. It includes scripts for **data processing**, **visualization**, **training**, and an **end-to-end pipeline**.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Installation & Dependencies](#installation--dependencies)  
3. [Keypoint Data Processing](#keypoint-data-processing)  
4. [Keypoint Visualization](#keypoint-visualization)  
5. [Training the Transformer](#training-the-transformer)  
6. [Full Pipeline](#full-pipeline)  
7. [Parameters & Command-Line Arguments](#parameters--command-line-arguments)  
8. [License & Credits](#license--credits)  

---

## Project Structure

A recommended layout is:

```
SPATIAL_TEMPORAL_TRANSFORMER_NETWORK_MEDIAPIPE/
├── scripts/
│   ├── process_keypoints.sh
│   ├── run_pipeline.sh
│   ├── train.sh
│   └── visualize_keypoints.sh
│
├── data/
│   ├── ExampleVideo/
│   └── Keypoint_Data/
│       ├── MP_Data_3030_HAND_OFFICE/
│       ├── MP_Data_6060_HOLISTIC_OFFICE/
│       └── MP_Data_6060+60_HOLISTIC_OFFICETankMix/
│
├── models/
│   ├── 0306transformer_action_recognition_holistic_6060.pth
│   ├── ...
│
├── src/
│   ├── ST-TR_Train_holistic.py
│   ├── KeypointDataProcess.py
│   ├── VideoKeypointVisualization.py
│   ├── Demo_ST-TR_ActionRecognition_Hand.py
│   └── Demo_ST-TR_ActionRecognition_Holistic.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Dependencies

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/Spatial-Temporal-Transformer-Network-Mediapipe.git
   cd Spatial-Temporal-Transformer-Network-Mediapipe
   ```

2. **Install dependencies** (Python 3.7+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have **PyTorch**, **numpy**, **scikit-learn**, **mediapipe**, **matplotlib**, etc.

3. *(Optional)* **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux/macOS)
   # or
   .\venv\Scripts\activate   # (Windows)
   pip install -r requirements.txt
   ```

---

## Keypoint Data Processing

You can extract keypoints from your raw video files via **MediaPipe** using `KeypointDataProcess.py`.  
For convenience, there’s a shell script **`process_keypoints.sh`** that automates this process.

**Example Command**:
```bash
bash scripts/process_keypoints.sh \
  "data/ExampleVideo/sample.mp4" \
  "data/Keypoint_Data/MP_Data_6060_HOLISTIC_OFFICE" \
  60 \
  "ASCEND DESCEND ME STOP RIGHT BUDDY_UP FOLLOW_ME OKAY LEFT YOU LEVEL"
```

- **Argument 1**: *Video path* (`"data/ExampleVideo/sample.mp4"`)
- **Argument 2**: *Data path* for saving `.npy` files (`"data/Keypoint_Data/..."`)
- **Argument 3**: *Sequence length* (number of frames per sample, e.g. `60`)
- **Argument 4**: *Actions list* (space-separated actions)

### GIF Demo

Below is a GIF illustrating the concept of capturing keypoints:

![Scuba Diving Keypoint Processing](Scuba%20Diving%20GIF%20by%20Girls%20that%20Scuba.gif)

*(If the GIF does not render in GitHub, make sure this file is committed and the path is correct.)*

---

## Keypoint Visualization

Use **`VideoKeypointVisualization.py`** (or the helper script **`visualize_keypoints.sh`**) to **visualize** the skeleton overlays on your video.

```bash
bash scripts/visualize_keypoints.sh "data/ExampleVideo/sample.mp4"
```

---

## Training the Transformer

Train the **Transformer** model via **`ST-TR_Train_holistic.py`**:

```bash
bash scripts/train.sh \
  "data/Keypoint_Data/MP_Data_6060_HOLISTIC_OFFICE" \
  60 \
  "ASCEND DESCEND ME STOP RIGHT BUDDY_UP FOLLOW_ME OKAY LEFT YOU LEVEL" \
  "models/my_custom_model.pth" \
  100 \
  32
```

---

## Full Pipeline

To run **all stages** (keypoint extraction → visualization → training) in one go:

```bash
bash scripts/run_pipeline.sh \
  "data/ExampleVideo/sample.mp4" \
  "data/Keypoint_Data/MP_Data_6060_HOLISTIC_OFFICE" \
  60 \
  "ASCEND DESCEND ME STOP RIGHT BUDDY_UP FOLLOW_ME OKAY LEFT YOU LEVEL" \
  "models/my_custom_model.pth" \
  100 \
  32 \
  "true"
```

---

## Parameters & Command-Line Arguments

| Script | Parameter | Description | Default Value |
|--------|-----------|-------------|---------------|
| `process_keypoints.sh` | `VIDEO_PATH` | Input video path | `data/ExampleVideo/sample.mp4` |
| | `DATA_PATH` | Output `.npy` keypoints path | `data/Keypoint_Data/...` |
| `visualize_keypoints.sh` | `VIDEO_PATH` | Path to visualize | `data/ExampleVideo/sample.mp4` |
| `train.sh` | `DATA_PATH` | Path to `.npy` keypoints | `data/Keypoint_Data/...` |
| | `SEQUENCE_LENGTH` | Frames per sample | `60` |
| | `EPOCHS` | Training epochs | `100` |
| `run_pipeline.sh` | `VISUALIZE` | `"true"` or `"false"` | `"false"` |

---

## License & Credits

- **License**: MIT (or whichever license you prefer).
- **Credits**:
  - [MediaPipe](https://github.com/google/mediapipe) for keypoint detection.
  - [PyTorch](https://pytorch.org/) for deep learning.
  - The scuba GIF courtesy of [Girls that Scuba](https://giphy.com/girlsthatscuba).

Enjoy exploring **Spatiotemporal Transformers** with MediaPipe! 🚀

