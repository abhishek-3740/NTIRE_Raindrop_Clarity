# -*- coding: utf-8 -*-
import os
import sys
import torch
import cv2
import numpy as np
import time
from tqdm import tqdm

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import Stage 1 Model
from stage1_restormer.model_restormer import build_restormer

# --- CONFIGURATION ---
# Update VAL_ROOT to your test image directory
VAL_ROOT = os.getenv("VAL_ROOT", "data/test_input")
OUT_DIR = "submission_final_tta8_results"
CKPT_PATH = "checkpoints/stage1_epoch_84_loss_0.0222.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tta_forward(model, x):
    """
    8-Way Test-Time Augmentation (TTA)
    Systematically applies Transpose, Vertical Flip, and Horizontal Flip.
    """
    preds = []
    for k in range(8):
        x_aug = x.clone()
        
        # 1. Apply augmentations
        if k >= 4:
            x_aug = x_aug.transpose(2, 3)  # Transpose (simulates 90/270 rotations)
        if k % 4 >= 2:
            x_aug = torch.flip(x_aug, [2]) # Vertical flip
        if k % 2 == 1:
            x_aug = torch.flip(x_aug, [3]) # Horizontal flip
            
        # 2. Forward pass (Pure FP32)
        out_aug = model(x_aug)
        
        # 3. Reverse augmentations (in strict reverse order)
        if k % 2 == 1:
            out_aug = torch.flip(out_aug, [3])
        if k % 4 >= 2:
            out_aug = torch.flip(out_aug, [2])
        if k >= 4:
            out_aug = out_aug.transpose(2, 3)
            
        preds.append(out_aug)
        
    # Average all 8 predictions
    return sum(preds) / 8.0

def main():
    print(f"🚀 [Final Submission] Loading Stage 1 Restormer from {CKPT_PATH}...")
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found at {CKPT_PATH}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load Model 
    model = build_restormer().to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # STRICTLY ENFORCE EMA WEIGHTS
    if "ema_model" in checkpoint:
        print("🌟 SUCCESS: Loading EMA model weights for maximum leaderboard score...")
        model.load_state_dict(checkpoint["ema_model"])
    else:
        print("⚠️ Warning: EMA not found. Loading standard weights...")
        model.load_state_dict(checkpoint.get("model", checkpoint))
        
    model.eval()

    # 2. Get Images
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(VAL_ROOT) if f.lower().endswith(valid_exts)]
    
    if len(image_files) == 0:
        print(f"❌ Error: No test images found in {VAL_ROOT}")
        return

    print(f"📦 Found {len(image_files)} images. Starting 8-Way TTA Inference...")
    total_time = 0.0
    
    # 3. Inference Loop 
    for filename in tqdm(image_files):
        full_path = os.path.join(VAL_ROOT, filename)
        img_bgr = cv2.imread(full_path)
        if img_bgr is None: continue

        # Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Pad to multiple of 8
        h, w = input_tensor.shape[2:]
        ph = ((h + 7) // 8) * 8
        pw = ((w + 7) // 8) * 8
        padding = (0, pw - w, 0, ph - h)
        input_tensor = torch.nn.functional.pad(input_tensor, padding, mode='reflect')

        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        start_t = time.time()
        
        with torch.no_grad():
            # Pass through the 8-way TTA function
            output = tta_forward(model, input_tensor)
            output = output[:, :, :h, :w] # Crop padding back to exact original size
            
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        total_time += (time.time() - start_t)

        # Save Result
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(os.path.join(OUT_DIR, filename), output_bgr)

    # 4. Generate Strict Readme.txt
    avg_time = total_time / len(image_files)
    print(f"⏱️  Average Runtime: {avg_time:.4f} seconds/image")

    readme_content = f"""runtime per image [s] : {avg_time:.4f}
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description: Stage 1 Restormer (Physics-based). Utilizing EMA weights, FP32 precision inference, and an 8-way geometric Test-Time Augmentation (TTA) ensemble on a single NVIDIA RTX 6000 GPU."""

    with open(os.path.join(OUT_DIR, "readme.txt"), "w") as f:
        f.write(readme_content.strip() + "\n")

    print(f"✅ readme.txt created in {OUT_DIR}")
    print("Now you can zip the folder using: zip -j submission_final_tta8.zip *.png readme.txt")

if __name__ == "__main__":
    main()
