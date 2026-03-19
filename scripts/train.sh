#!/usr/bin/env bash

# Default values
DATA_PATH=${1:-"data/Keypoint_Data/MP_Data_6060_HOLISTIC_OFFICE"}
SEQUENCE_LENGTH=${2:-60}
ACTIONS=${3:-"ASCEND DESCEND ME STOP RIGHT BUDDY_UP FOLLOW_ME OKAY LEFT YOU LEVEL"}
SAVE_PATH=${4:-"models/transformer_action_recognition.pth"}
EPOCHS=${5:-100}
BATCH_SIZE=${6:-32}

echo "🚀 Training model..."
echo "🔹 Data Path: $DATA_PATH"
echo "🔹 Sequence Length: $SEQUENCE_LENGTH"
echo "🔹 Actions: $ACTIONS"
echo "🔹 Model Save Path: $SAVE_PATH"
echo "🔹 Epochs: $EPOCHS"
echo "🔹 Batch Size: $BATCH_SIZE"

# Run the training script
python src/ST-TR_Train_holistic.py \
    --data_path "$DATA_PATH" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --actions "$ACTIONS" \
    --save_path "$SAVE_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo "✅ Model training complete! Model saved at: $SAVE_PATH"
