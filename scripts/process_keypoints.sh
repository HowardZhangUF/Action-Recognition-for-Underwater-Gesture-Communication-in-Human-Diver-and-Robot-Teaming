#!/usr/bin/env bash

# Set default values (if arguments are not provided)
VIDEO_PATH=${1:-"data/ExampleVideo/sample.mp4"}
DATA_PATH=${2:-"data/Keypoint_Data/MP_Data_6060_HOLISTIC_OFFICE"}
SEQUENCE_LENGTH=${3:-60}
ACTIONS=${4:-"ASCEND DESCEND ME STOP RIGHT BUDDY_UP FOLLOW_ME OKAY LEFT YOU LEVEL"}

echo "🚀 Extracting keypoints from video..."
echo "🔹 Video Path: $VIDEO_PATH"
echo "🔹 Data Path: $DATA_PATH"
echo "🔹 Sequence Length: $SEQUENCE_LENGTH"
echo "🔹 Actions: $ACTIONS"

# Run the keypoint extraction script
python src/KeypointDataProcess.py \
    --video_path "$VIDEO_PATH" \
    --data_path "$DATA_PATH" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --actions "$ACTIONS"

echo "✅ Keypoint extraction complete!"
