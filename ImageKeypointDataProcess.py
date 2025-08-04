import cv2
import numpy as np
import os
import mediapipe as mp

# ----- Mediapipe Setup -----
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----- Define Paths -----
DATA_PATH = "MP_Data_raspberrypi_vertical"  # Changed to reflect pose-only data
IMAGE_PATH = "Image_data/vertical"  # Changed from VIDEO_PATH to IMAGE_PATH

# Actions array (11 actions) - Updated to match the folder names in Image_data
actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'ToRight', 'BUDDY_UP',
    'FOLLOW_ME', 'OK', 'ToLeft', 'YOU', 'STAY'
]

# Number of videos per action
no_sequences = 60  
# We want exactly 60 frames per 2-second video @ 30 FPS
sequence_length = 60

# ----- Optional Visualization -----
def draw_styled_landmarks(image, results):
    """Draws pose landmarks on the given image."""
    # Draw pose connections only
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
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
    Returns pose landmarks only:
    - Pose: 33 landmarks * 3 (x,y,z) = 99 values
    Total: 99 values
    """
    # Extract pose landmarks (only x,y,z - excluding visibility)
    pose = np.zeros(33*3)  # 33 landmarks * 3 (x,y,z)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] 
                        for res in results.pose_landmarks.landmark]).flatten()
    
    return pose

# ----- Create Directory Structure -----
os.makedirs(DATA_PATH, exist_ok=True)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)
    for sequence in range(1, no_sequences + 1):
        os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)

print("✅ Directory structure created/verified successfully!")

# ----- Process Pre-recorded Images -----
with mp_holistic.Holistic(
    static_image_mode=True,  # Changed to True for image processing
    model_complexity=2,      # 0, 1, or 2. Higher = more accurate but slower
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
) as holistic:

    # Loop over each action
    for action in actions:
        print(f"🔄 Processing action: {action}")
        # Loop over each sequence for the current action
        for sequence in range(1, no_sequences + 1):
            sequence_folder = os.path.join(IMAGE_PATH, action, str(sequence))
            if not os.path.isdir(sequence_folder):
                print(f"❌ Sequence folder not found: {sequence_folder}")
                continue

            print(f"  📁 Processing sequence {sequence}")
            
            # Get all PNG files in the sequence folder and sort them numerically
            image_files = [f for f in os.listdir(sequence_folder) if f.endswith('.png')]
            # Sort by numeric value (0.png, 1.png, ..., 59.png)
            image_files.sort(key=lambda x: int(x.split('.')[0]))
            
            # Process each image in the sequence (up to sequence_length frames)
            for frame_num in range(min(len(image_files), sequence_length)):
                image_file = os.path.join(sequence_folder, image_files[frame_num])
                
                # Read the image
                frame = cv2.imread(image_file)
                if frame is None:
                    print(f"❌ Could not read image: {image_file}")
                    continue

                # Process with MediaPipe
                results = mediapipe_detection(frame, holistic)
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

                # Optional: Display the feed for debugging (commented out for batch processing)
                # cv2.imshow('OpenCV Feed', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

print("✅ Pose landmark extraction using MediaPipe Holistic from images complete!")
