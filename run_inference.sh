#!/bin/bash
# run_inference.sh
# Bash script to run the video inference Python script.

# Define paths to the required files.
VIDEO_PATH="/fs/nexus-projects/AerialAI/datasets/KINETICS400/videos_train/mCaf7Zo1q3E.mp4"
CHECKPOINT_PATH="./vits_checkpoint.pth"
LABEL_CSV_PATH="./kinetics_400_labels.csv"
OUTPUT_VIDEO="demo.mp4"

# Optionally, specify the Python interpreter (e.g., python3)
PYTHON=python3

# Run the Python inference script with the provided arguments.
$PYTHON inference.py \
    --video "$VIDEO_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --label_csv "$LABEL_CSV_PATH" \
    --output "$OUTPUT_VIDEO"

# Optional: print a message when done.
echo "Inference complete. Annotated video saved as $OUTPUT_VIDEO"