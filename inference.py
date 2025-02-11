#!/usr/bin/env python3
import argparse
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import os

# ---------------------------
# Import your ViT model architecture.
# Here we assume your model code is in a module named "models"
# and you want to use the vit_small_patch16_224 variant.
# ---------------------------
from models import vit_small_patch16_224  # adjust as necessary

# ---------------------------
# Function to load label mapping from a CSV file.
# The CSV is assumed to have a header row: "id,name"
# and then rows like: 0,abseiling
# ---------------------------
def load_label_map(csv_file):
    label_map = {}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label_id = int(row['id'])
            except ValueError:
                continue  # skip header or invalid rows
            label_name = row['name']
            label_map[label_id] = label_name
    print(f"[DEBUG] Loaded label map with {len(label_map)} entries from {csv_file}")
    return label_map

# ---------------------------
# Preprocess a single video frame.
# ---------------------------
def preprocess_frame(frame, target_size=224):
    """
    Preprocess a single video frame:
      - Convert from BGR (OpenCV) to RGB.
      - Resize to (target_size, target_size).
      - Convert to float in [0, 1], then normalize with mean=0.5 and std=0.5.
      - Return a torch.Tensor of shape (3, target_size, target_size).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (target_size, target_size))
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    frame_rgb = (frame_rgb - 0.5) / 0.5
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
    return tensor

# ---------------------------
# Inference function on a clip.
# ---------------------------
def infer_clip(model, clip, device):
    """
    Given a clip tensor of shape (T, 3, H, W),
    rearrange it to (1, 3, T, H, W) and run inference.
    Returns:
      - scores: the raw output logits (as a torch.Tensor)
    """
    clip = clip.unsqueeze(0).to(device)  # shape: (1, T, 3, H, W)
    # Rearrange to (1, 3, T, H, W) as expected by the model.
    clip = clip.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        outputs = model(clip)
        scores = outputs.squeeze(0)  # shape: (num_classes,)
    return scores

# ---------------------------
# Overlay text lines on an image.
# ---------------------------
def overlay_text(frame, text_lines, start_x=10, start_y=30, line_height=30, font_scale=0.8, color=(0, 255, 0), thickness=2):
    """
    Overlay multiple lines of text on the frame.
    """
    for i, line in enumerate(text_lines):
        y = start_y + i * line_height
        cv2.putText(frame, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

# ---------------------------
# Main function: process video, run inference, annotate, and save output video.
# ---------------------------
def main(args):
    # Load label map from CSV.
    label_map = load_label_map(args.label_csv)
    
    # Set up device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    
    # Initialize the model.
    print("[DEBUG] Initializing model...")
    model = vit_small_patch16_224(num_classes=400, all_frames=16, tubelet_size=2)
    
    # Load checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"[DEBUG] Loaded checkpoint from {args.checkpoint}")
    if isinstance(checkpoint, dict) and 'module' in checkpoint:
        state_dict = checkpoint['module']
        print("[DEBUG] Extracted state_dict from checkpoint['module']")
    else:
        state_dict = checkpoint
        print("[DEBUG] Using checkpoint as state_dict directly")
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("[DEBUG] model.load_state_dict completed")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()
    model.to(device)
    
    # Open the input video.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Error opening video file: {args.video}")
        return

    # Get video properties.
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] Original video FPS: {orig_fps:.2f}, resolution: {width}x{height}")
    
    # Define target FPS (for inference) and compute frame skipping.
    target_fps = 6
    skip_frames = max(int(round(orig_fps / target_fps)), 1)
    print(f"[DEBUG] Sampling every {skip_frames} frames to get ~{target_fps} FPS.")
    
    # Set up VideoWriter for saving the annotated video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, orig_fps, (width, height))
    if not out.isOpened():
        print(f"[ERROR] VideoWriter could not be opened. Check codec and output path: {args.output}")
        return
    else:
        print(f"[DEBUG] VideoWriter opened for output: {args.output}")
    
    # Prepare a sliding window for inference.
    clip_len = 16  # number of frames per clip expected by the model
    frame_queue = deque(maxlen=clip_len)
    
    # Variable to hold current prediction text (update every inference).
    current_prediction_text = ["No prediction yet"]

    target_size = 224  # for preprocessing frames
    frame_count = 0
    start_time = time.time()
    
    print("[DEBUG] Starting video processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] No more frames to read; exiting loop.")
            break  # End of video

        frame_count += 1

        # Sample frames at target FPS.
        if frame_count % skip_frames != 0:
            overlay_text(frame, current_prediction_text)
            out.write(frame)
            continue

        # Preprocess frame and add to sliding window.
        frame_tensor = preprocess_frame(frame, target_size)
        if frame_tensor is None:
            print(f"[WARNING] Skipping frame {frame_count} due to preprocessing error.")
            continue
        frame_queue.append(frame_tensor)

        # Run inference when a full clip is ready.
        if len(frame_queue) == clip_len:
            clip_tensor = torch.stack(list(frame_queue), dim=0)  # shape: (T, 3, H, W)
            scores = infer_clip(model, clip_tensor, device)
            if scores is not None:
                probs = torch.softmax(scores, dim=0)
                topk = torch.topk(probs, k=3)
                top_indices = topk.indices.cpu().numpy()
                top_probs = topk.values.cpu().numpy()
                
                new_text_lines = []
                for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
                    label_name = label_map.get(int(idx), f"ID {idx}")
                    new_text_lines.append(f"{rank}. {label_name}: {prob:.2f}")
                current_prediction_text = new_text_lines
                print(f"[DEBUG] Inference at frame {frame_count}: {current_prediction_text}")
            else:
                print(f"[WARNING] Inference failed at frame {frame_count}")
        
        overlay_text(frame, current_prediction_text)
        out.write(frame)

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    print(f"[DEBUG] Processed {frame_count} frames in {elapsed:.2f} seconds.")
    print(f"[DEBUG] Output video saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Inference with ViT-small and Annotated Top-3 Labels (No Display)")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to the label mapping CSV file.")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Path for the output annotated video.")
    args = parser.parse_args()
    main(args)