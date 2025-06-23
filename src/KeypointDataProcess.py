import cv2
import numpy as np
import os
import mediapipe as mp
import argparse
# ----- Mediapipe Setup -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----- Define Paths -----
DATA_PATH = "0228_lmData_mpHand"
VIDEO_PATH = "/home/zhangzihao@ad.ufl.edu/Downloads/Public_Dataset"  

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




def process_keypoints(video_path, data_path, sequence_length, actions):
    print(f"Processing video: {video_path}")
    print(f"Saving keypoints to: {data_path}")
    print(f"Sequence length: {sequence_length}")
    print(f"Actions: {actions}")

    # Your keypoint processing code here...
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
            # Loop over each video for the current action
            for sequence in range(1, no_sequences + 1):
                video_file = os.path.join(VIDEO_PATH, action, f"{sequence}.MP4")
                if not os.path.isfile(video_file):
                    print(f"❌ Video file not found: {video_file}")
                    continue

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

                    # Optional: Display the feed for debugging
                    cv2.imshow('OpenCV Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()

    cv2.destroyAllWindows()
    print("✅ Landmark extraction using MediaPipe Hands complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--data_path", type=str, required=True, help="Where to save extracted keypoints")
    parser.add_argument("--sequence_length", type=int, default=60, help="Frames per sequence")
    parser.add_argument("--actions", nargs="+", required=True, help="List of action labels")

    args = parser.parse_args()

    process_keypoints(args.video_path, args.data_path, args.sequence_length, args.actions)
