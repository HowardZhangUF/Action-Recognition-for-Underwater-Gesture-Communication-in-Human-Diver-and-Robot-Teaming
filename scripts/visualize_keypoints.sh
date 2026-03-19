#!/usr/bin/env bash

# Set default video path if not provided
VIDEO_PATH=${1:-"data/ExampleVideo/sample.mp4"}

echo "📊 Visualizing keypoints from video..."
echo "🔹 Video Path: $VIDEO_PATH"

# Run the visualization script
python src/VideoKeypointVisualization.py --video_path "$VIDEO_PATH"

echo "✅ Keypoint visualization complete!"
