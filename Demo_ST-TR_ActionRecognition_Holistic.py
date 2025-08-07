
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# 1) MODEL + ACTION LABELS + COLORS
# --------------------------------------
MODEL_PATH = "action_transformer_v2.pth"

actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'RIGHT', 'BUDDY_UP',
    'FOLLOW_ME', 'OKAY', 'LEFT', 'YOU', 'LEVEL'
]
NUM_CLASSES = len(actions)

# A color for each class (11 total)
colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (50, 100, 50),
    (100, 50, 150)
]

def prob_viz(probs, actions, input_frame, colors):
    
    output_frame = input_frame.copy()
    limit = min(len(probs), len(actions), len(colors))

    for i in range(limit):
        action = actions[i]
        c = colors[i]
        prob = probs[i]  # Probability for the i-th action
        prob_percent = int(prob * 100)

        # Draw a filled rectangle proportional to the probability
        start_point = (0, 60 + i * 40)
        end_point = (prob_percent, 90 + i * 40)  # prob * 100
        cv2.rectangle(output_frame, start_point, end_point, c, -1)

        # Label: "ACTION_NAME XX%"
        label_text = f"{action} {prob_percent}%"
        cv2.putText(
            output_frame,
            label_text,
            (0, 85 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    return output_frame

# --------------------------------------
# 2) DEFINE THE TRANSFORMER MODEL
# --------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(ActionTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)         # (batch_size, seq_len, embed_dim)
        x = self.pos_encoding(x)       # (batch_size, seq_len, embed_dim)
        x = self.transformer_encoder(x)# (batch_size, seq_len, embed_dim)
        x = torch.mean(x, dim=1)       # Global average pooling -> (batch_size, embed_dim)
        x = self.dropout(x)
        logits = self.fc_out(x)        # (batch_size, num_classes)
        return logits

# --------------------------------------
# 3) LOAD MODEL & SETUP MEDIAPIPE
# --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActionTransformer(
    feature_dim=225,   # 33 pose x 3 + 21 L-hand x 3 + 21 R-hand x 3
    num_classes=NUM_CLASSES,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model Loaded Successfully!")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------
# 4) EXTRACT LANDMARKS
# --------------------------------------
def extract_landmarks(results):
    """
    Extract pose (33x3=99), left hand (21x3=63), right hand (21x3=63).
    Total 225-dim vector per frame.
    """
    pose = (np.array([[res.x, res.y, res.z] 
             for res in results.pose_landmarks.landmark])
             if results.pose_landmarks else np.zeros((33, 3)))
    left_hand = (np.array([[res.x, res.y, res.z]
                 for res in results.left_hand_landmarks.landmark])
                 if results.left_hand_landmarks else np.zeros((21, 3)))
    right_hand = (np.array([[res.x, res.y, res.z]
                  for res in results.right_hand_landmarks.landmark])
                  if results.right_hand_landmarks else np.zeros((21, 3)))
    
    return np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
    
# --------------------------------------
# 5) REAL-TIME ACTION RECOGNITION
# --------------------------------------
sequence = []         # Rolling window of frames
sentence = []         # Store predicted actions for display
threshold = 0.6       # Confidence threshold
# Replace 0 with the path to your video file
VIDEO_PATH = "FOLLOWME_SPRINGS_VERTICAL_0030ab1096 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("▶️ End of video.")
            break

        # Convert frame for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract and store the landmarks in a rolling sequence
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-60:]  # Keep last 60 frames

        # When we have enough frames, run inference
        if len(sequence) == 60:
            # Prepare input for the model: (1, 60, 225)
            input_data = torch.tensor([sequence], dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_data)  # (1, num_classes)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                pred_label = np.argmax(probs)
                confidence = probs[pred_label]

            # Probability bars (with % for each action)
            image = prob_viz(probs, actions, image, colors)

            # If highest confidence above threshold, update the displayed sentence
            if confidence > threshold:
                predicted_action = actions[pred_label]
                if not sentence or (predicted_action != sentence[-1]):
                    sentence.append(predicted_action)

            # Keep only last 5 predictions in 'sentence'
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Draw the current "sentence" at the top
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                " | ".join(sentence),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255),
                2, cv2.LINE_AA
            )

        # Show webcam feed
        cv2.imshow("Real-Time Action Recognition", image)

        # Quit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
