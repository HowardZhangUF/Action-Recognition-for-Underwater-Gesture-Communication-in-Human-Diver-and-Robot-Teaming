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
MODEL_PATH = "model/transformer_action_recognition_hand3030.pth"

actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'RIGHT', 'BUDDY_UP',
    'FOLLOW_ME', 'OKAY', 'LEFT', 'YOU', 'LEVEL'
]
NUM_CLASSES = len(actions)

# A color for each class
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
        prob = probs[i]  
        prob_percent = int(prob * 100)

        cv2.rectangle(output_frame, (0, 60 + i * 40), (prob_percent, 90 + i * 40), c, -1)
        cv2.putText(output_frame, f"{action} {prob_percent}%", (0, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# --------------------------------------
# 2) TRANSFORMER MODEL
# --------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(ActionTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  
        x = self.dropout(x)
        return self.fc_out(x)

# --------------------------------------
# 3) LOAD MODEL & SETUP MEDIAPIPE HANDS
# --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActionTransformer(
    feature_dim=126,   # left hand (63) + right hand (63)
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model Loaded Successfully!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------
# 4) EXTRACT LANDMARKS (hands only)
# --------------------------------------
def extract_landmarks(results):
    """
    Extract left and right hand landmarks (21x3 each).
    Total = 126-dim vector (if both hands found, else zeros filled).
    """
    # Initialize with zeros
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Convert landmarks to numpy (21x3)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Check if left or right
            if handedness.classification[0].label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    return np.concatenate([left_hand.flatten(), right_hand.flatten()])

# --------------------------------------
# 5) REAL-TIME ACTION RECOGNITION
# --------------------------------------
sequence, sentence = [], []
threshold = 0.6
VIDEO_PATH = "Screencast from 07-22-2025 10:57:37 AM.webm"
cap = cv2.VideoCapture(VIDEO_PATH)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-60:]

        # Run inference
        if len(sequence) == 60:
            input_data = torch.tensor([sequence], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_data)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                pred_label = np.argmax(probs)
                confidence = probs[pred_label]

            image = prob_viz(probs, actions, image, colors)

            if confidence > threshold:
                predicted_action = actions[pred_label]
                if not sentence or (predicted_action != sentence[-1]):
                    sentence.append(predicted_action)

            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, " | ".join(sentence), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Action Recognition (Hands)", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

