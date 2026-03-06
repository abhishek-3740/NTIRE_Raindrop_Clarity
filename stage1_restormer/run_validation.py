# -*- coding: utf-8 -*-
import os
import sys
import torch
import cv2
import numpy as np
import time
from tqdm import tqdm
from torch.cuda.amp import autocast

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Correct import based on your project structure
from stage1_restormer.model_restormer import build_restormer

# --- CONFIGURATION ---
# Update these paths to match your local setup
VAL_ROOT = os.getenv("VAL_ROOT", "data/test_input")  # Path to validation/test images
OUT_DIR = "submission_stage1_results"
CKPT_PATH = "checkpoints/stage1_epoch_60_loss_0.0247.pth"   # Update if your best checkpoint name changes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 [Submission] Loading model from {CKPT_PATH}...")
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found at: {CKPT_PATH}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load Model 
    model = build_restormer().to(DEVICE)
    
    # 2. Safely Load Weights (Prioritize EMA for higher accuracy)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    
    if "ema_model" in checkpoint:
        print("🌟 Loading EMA model weights for maximum performance...")
        model.load_state_dict(checkpoint["ema_model"])
    elif "model" in checkpoint:
        print("⚠️ Loading standard model weights (EMA not found)...")
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Get Images
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(VAL_ROOT) if f.lower().endswith(valid_exts)]
    
    if len(image_files) == 0:
        print(f"❌ Error: No images found directly inside {VAL_ROOT}")
        return

    print(f"📦 Found {len(image_files)} images. Starting inference...")

    # --- START TIMER ---
    total_time = 0.0
    
    # 4. Inference Loop
    for filename in tqdm(image_files):
        full_path = os.path.join(VAL_ROOT, filename)
        img_bgr = cv2.imread(full_path)
        if img_bgr is None: continue

        # Preprocess: Convert BGR to RGB, scale to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Padding: Restormer needs H and W to be multiples of 8
        h, w = input_tensor.shape[2:]
        ph = ((h + 7) // 8) * 8
        pw = ((w + 7) // 8) * 8
        padding = (0, pw - w, 0, ph - h)
        input_tensor = torch.nn.functional.pad(input_tensor, padding, mode='reflect')

        # Measure Pure Inference Time
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start_t = time.time()
        
        with torch.no_grad():
            with autocast(enabled=(DEVICE.type == "cuda")): 
                output = model(input_tensor)
                
            # Crop padding out to match exact input size
            output = output[:, :, :h, :w]
            
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end_t = time.time()
        total_time += (end_t - start_t)

        # Save Result with the exact same filename
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR) 
        
        cv2.imwrite(os.path.join(OUT_DIR, filename), output_bgr)

    # 5. Generate Readme.txt (Correctly outside the loop)
    avg_time = total_time / len(image_files)
    print(f"⏱️  Average Runtime: {avg_time:.4f} seconds/image")

    readme_content = f"""runtime per image [s] : {avg_time:.4f}
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description: PyTorch implementation of Stage 1 Restormer for the Raindrop Removal challenge. Trained on the provided Day and Night datasets. Inference executed on a single NVIDIA RTX 6000 GPU."""
    
    with open(os.path.join(OUT_DIR, "readme.txt"), "w") as f:
        f.write(readme_content.strip() + "\n")
        
    print(f"✅ readme.txt created in {OUT_DIR}")
    print("Now you can zip the folder using: zip -j submission_stage1.zip *.png readme.txt")

if __name__ == "__main__":
    main()