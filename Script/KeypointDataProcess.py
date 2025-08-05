import cv2
import numpy as np
import os
import mediapipe as mp

# ----- Mediapipe Setup -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----- Define Paths -----
DATA_PATH = "Blue_Grutto_HandPose"  # Changed to reflect pose data
VIDEO_PATH = "C:/Users/zhangzihao/Action-Recognition-for-Underwater-Gesture-Communication-in-Human-Diver-and-Robot-Teaming/Public_Dataset"  

# Actions array (11 actions)
actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'RIGHT', 'BUDDY_UP',
    'FOLLOW_ME', 'OKAY', 'LEFT', 'YOU', 'LEVEL'
]

# Number of videos per action
no_sequences = 60  
# We want exactly 60 frames per 2-second video @ 30 FPS
sequence_length = 60

# ----- Optional Visualization -----
def draw_styled_landmarks(image, results):
    """Draws landmarks on the given image."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

def mediapipe_detection(image, model):
    """
    Converts image BGR->RGB, processes with model,
    returns the results (hand landmarks, etc.).
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = model.process(rgb_image)
    return results

def extract_keypoints(results):
    """
    Returns a 126-dim array:
    [21 pts (x,y,z) for left hand] + [21 pts (x,y,z) for right hand].
    """
    lh = np.zeros(63)  # 21*3
    rh = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label  # "Left"/"Right"
            hand_landmarks = results.multi_hand_landmarks[idx]
            coords = np.array([[lm.x, lm.y, lm.z] 
                               for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = coords
            else:
                rh = coords

    return np.concatenate([lh, rh])

# ----- Create Directory Structure -----
os.makedirs(DATA_PATH, exist_ok=True)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    for sequence in range(1, no_sequences + 1):
        os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)

print("✅ Directory structure created/verified successfully!")

# ----- Process Pre-recorded Videos -----
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
) as hands:

    # Loop over each action
    for action in actions:
        print(f"🔄 Processing action: {action}")
        action_folder = os.path.join(VIDEO_PATH, action)
        if not os.path.isdir(action_folder):
            print(f"❌ Action folder not found: {action_folder}")
            continue
            
        # Get all video files in the action folder
        video_files = [f for f in os.listdir(action_folder) if f.lower().endswith(('.mp4', '.MP4'))]
        print(f"  📁 Found {len(video_files)} videos for action {action}")
        
        # Process each video file
        for sequence, video_filename in enumerate(video_files, 1):
            if sequence > no_sequences:  # Limit to no_sequences videos
                break
                
            video_file = os.path.join(action_folder, video_filename)
            print(f"  🎥 Processing video {sequence}/{min(len(video_files), no_sequences)}: {video_filename}")

            cap = cv2.VideoCapture(video_file)
            last_valid_frame = None

            
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if not ret:
                    # If video ended early, use the last valid frame again
                    if last_valid_frame is None:
                        print(f"⚠️ No frames found in {video_file}, skipping...")
                        break
                    else:
                        frame = last_valid_frame
                else:
                    last_valid_frame = frame

                # Process with MediaPipe
                results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(frame, results)

                # Extract keypoints
                keypoints = extract_keypoints(results)

                # Save the keypoints
                npy_path = os.path.join(
                    DATA_PATH, 
                    action, 
                    str(sequence), 
                    f"{frame_num}.npy"
                )
                np.save(npy_path, keypoints)

                try:
                    cv2.imshow('OpenCV Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # Skip display if OpenCV GUI is not available
                    pass

            cap.release()

try:
    cv2.destroyAllWindows()
except cv2.error:
    # Skip if OpenCV GUI is not available
    pass
print("✅ Landmark extraction using MediaPipe Hands complete!")
