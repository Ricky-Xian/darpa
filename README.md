# Video Action Recognition

This repository provides code for performing action recognition on video data using a pretrained Vision Transformer (ViT-Small) model. The inference script processes an input video, runs action recognition on a sliding window of frames, overlays the top-3 predicted action labels (with probabilities) on each frame, and saves the annotated video.

> **Note:**  
> This code has been developed and tested with the following versions:  
> - **PyTorch:** 2.1.2  
> - **torchvision:** 0.16.2  
> - **timm:** 0.4.12  
> - **opencv-python**  
> - **numpy**

---

## Features

- **Video Inference:** Processes an input video and outputs an annotated video with action predictions.
- **Sliding Window Processing:** Uses a sliding window of 16 frames to perform inference at approximately 6 FPS.
- **Top-3 Predictions:** Overlays the top-3 predicted action labels along with their probabilities on each video frame.
- **Label Mapping:** Maps numeric label IDs to human-readable action names using a CSV file (e.g., `kinetics_400_labels.csv`).

---

## Repository Structure
.
├── models.py                   # Contains the ViT-Small model architecture.
├── inference.py                # Main inference script.
├── requirements.txt            # Pip dependencies.
├── environment.yml             # (Optional) Conda environment configuration.
├── kinetics_400_labels.csv     # CSV file mapping label IDs to action names.
└── README.md                   # This file.

---

## Installation

You can install the required dependencies using either pip or Conda.

### Using pip

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ricky-Xian/darpa.git
   cd darpa

2. **Create New Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate videomae_env

3.	**Install the dependencies:**
   ```bash
   pip install -r requirements.txt
