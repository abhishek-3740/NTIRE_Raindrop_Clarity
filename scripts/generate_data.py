import os
import sys
import cv2
import shutil
import numpy as np
import torch
from tqdm import tqdm

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
pipeline_dir = os.path.join(current_dir, "Raindrop_pipeline")
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

from stage1_restormer.model_restormer import build_restormer

def main():
    # ==========================================
    # 1. PATH CONFIGURATION
    # ==========================================
    # Where your original dataset lives
    DATASET_ROOT = os.path.join(current_dir, "dataset")
    
    # The subfolders you want to process. 
    # If you only want the "Day Expert", just leave ["DayRainDrop_Train"] here!
    SUBSETS_TO_PROCESS = ["DayRainDrop_Train", "NightRainDrop_Train"]

    # Where NAFNet expects the data to be
    STAGE1_OUT_DIR = os.path.join(pipeline_dir, "data", "stage1", "train")
    GT_OUT_DIR = os.path.join(pipeline_dir, "data", "gt", "train")

    os.makedirs(STAGE1_OUT_DIR, exist_ok=True)
    os.makedirs(GT_OUT_DIR, exist_ok=True)

    # ==========================================
    # 2. LOAD RESTORMER MODEL
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Restormer model on {device}...")

    model = build_restormer().to(device)
    ckpt_path = os.path.join(pipeline_dir, "checkpoints", "stage1_epoch_84_loss_0.0222.pth")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Model loaded successfully!")

    # ==========================================
    # 3. BULK INFERENCE
    # ==========================================
    for subset in SUBSETS_TO_PROCESS:
        gt_dir = os.path.join(DATASET_ROOT, subset, "Clear")
        drop_dir = os.path.join(DATASET_ROOT, subset, "Drop")

        if not os.path.exists(gt_dir) or not os.path.exists(drop_dir):
            print(f"Warning: Missing paths for {subset}. Skipping...")
            continue

        # Gather all images using os.walk (to handle your 00001/ subfolders)
        image_paths = []
        for root, _, files in os.walk(gt_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, file), gt_dir)
                    image_paths.append(rel_path)

        print(f"\nGenerating Stage-2 data from: {subset} ({len(image_paths)} images)")
        
        for rel_path in tqdm(image_paths, unit="img"):
            gt_path = os.path.join(gt_dir, rel_path)
            
            # Handle extensions for Drop path
            base_name, _ = os.path.splitext(rel_path)
            drop_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                temp_path = os.path.join(drop_dir, base_name + ext)
                if os.path.exists(temp_path):
                    drop_path = temp_path
                    break
            
            if not drop_path:
                continue

            # Read images
            drop_img = cv2.imread(drop_path)
            if drop_img is None:
                continue

            # --- MODEL INFERENCE ---
            drop_rgb = cv2.cvtColor(drop_img, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(drop_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                with torch.cuda.amp.autocast(): 
                    pred_tensor = model(input_tensor)
            
            pred_tensor = torch.clamp(pred_tensor, 0, 1)
            pred_rgb = (pred_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

            # --- FLATTEN FILENAMES ---
            # To make Stage2Dataset easy, we flatten "00001/00001.png" -> "Day_00001_00001.png"
            prefix = "Day" if "Day" in subset else "Night"
            flat_filename = f"{prefix}_{base_name.replace(os.sep, '_')}.png"

            out_stage1_path = os.path.join(STAGE1_OUT_DIR, flat_filename)
            out_gt_path = os.path.join(GT_OUT_DIR, flat_filename)

            # Save the Restormer output
            cv2.imwrite(out_stage1_path, pred_bgr)
            
            # Copy the original clear image
            shutil.copy(gt_path, out_gt_path)

    print(f"\n✅ All Stage-2 training data successfully generated in: {os.path.join(pipeline_dir, 'data')}")

if __name__ == "__main__":
    main()