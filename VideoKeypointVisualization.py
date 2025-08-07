import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_3d_landmarks(results):
    """
    Collect 3D (x, y, z) landmarks from:python ST-TR_Train_holistic.py
      - results.pose_landmarks (33 points)
      - results.left_hand_landmarks (21 points)
      - results.right_hand_landmarks (21 points)
    Returns a NumPy array of shape (N, 3),
    where N=75 if all sets are present.
    """
    all_landmarks = []

    # Pose Landmarks
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])

    # Left Hand Landmarks
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])

    # Right Hand Landmarks
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])

    if len(all_landmarks) == 0:
        return None
    return np.array(all_landmarks)  # shape = (N,3)

def live_3d_plot(video_path=None, use_webcam=False):
    """
    Reads frames either from a given `video_path` or from the webcam,
    processes them with MediaPipe Holistic, and plots 3D landmarks
    in real-time using Matplotlib. 
    Also saves:
      1) A processed video with 2D skeleton on the original frames
      2) A second video with 2D skeleton on a white background
      3) A final 3D snapshot as a PNG.
    """

    # -------------------------------------------
    # 1) Video capture
    # -------------------------------------------
    if use_webcam:
        cap = cv2.VideoCapture(0) 
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # MediaPipe Holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    # -------------------------------------------
    # 2) Setup Video Writers
    # -------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_folder = "." 
    # (A) Processed video (original background + skeleton)
    out_file_processed = os.path.join(out_folder, "processed_output.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_processed = cv2.VideoWriter(out_file_processed, fourcc, fps, (w, h))
    print(f"Saving processed video to: {out_file_processed}")

    # (B) 2D skeleton
    out_file_white = os.path.join(out_folder, "2d_white_bg_skeleton.mp4")
    writer_white = cv2.VideoWriter(out_file_white, fourcc, fps, (w, h))
    print(f"Saving 2D skeleton on white bg to: {out_file_white}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_count += 1

        # BGR -> RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # -------------------------------------------
        # 4) 2D skeleton on the ORIGINAL frame
        # -------------------------------------------
        processed_frame = frame.copy()  # BGR
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        
        cv2.imshow("2D Holistic Landmarks (Processed)", processed_frame)

        writer_processed.write(processed_frame)

        # -------------------------------------------
        # 5) 2D skeleton on a WHITE background (Thicker Lines)
        # -------------------------------------------
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Custom thicker line specs
        landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4)
        connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=4)

       
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                white_bg, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_style, connection_style
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                white_bg, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_style, connection_style
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                white_bg, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_style, connection_style
            )

        cv2.imshow("2D Holistic (White BG)", white_bg)
        writer_white.write(white_bg)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------------------------------------------
    # 6) Cleanup
    # -------------------------------------------
    cap.release()
    holistic.close()
    writer_processed.release()
    writer_white.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames.")
    print("Done.")

if __name__ == "__main__":
    # Example usage:
    live_3d_plot(
        video_path="media/example_video.webm",
        use_webcam=False
    )