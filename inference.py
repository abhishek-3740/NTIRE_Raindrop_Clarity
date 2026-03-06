import os
import cv2
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set path for model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, "stage1_restormer"))
sys.path.append(os.path.join(current_dir, "stage2_Refiner"))

from stage1_restormer.model_restormer import Restormer
from stage2_Refiner.models.nafnet_refiner import NAFNet

def setup_directories(out_dir):
    """Ensures directories exist based on your specific structure."""
    dirs = [out_dir, "checkpoints/stage1", "checkpoints/stage2_day", "checkpoints/stage2_night"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_weights_safe(model, weight_path, device):
    """Safely extracts 'model_state_dict', handles EMA, and strips 'module.' prefixes."""
    if not os.path.exists(weight_path):
        print(f"⚠️  Weight file not found at {weight_path}")
        return False
    
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = {}

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'ema_model' in checkpoint:
            state_dict = checkpoint['ema_model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DataParallel 'module.' prefix
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Loaded (Strict): {os.path.basename(weight_path)}")
    except RuntimeError:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded (Non-Strict): {os.path.basename(weight_path)}")
        
    return True

def calculate_brightness(image_rgb):
    """Calculates luminance to determine routing."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)

def tta_forward(forward_fn, x):
    """
    8-Way Test-Time Augmentation (TTA)
    Systematically applies Transpose, Vertical Flip, and Horizontal Flip.
    """
    preds = []
    for k in range(8):
        x_aug = x.clone()
        
        # 1. Apply augmentations
        if k >= 4:
            x_aug = x_aug.transpose(2, 3)  # Transpose
        if k % 4 >= 2:
            x_aug = torch.flip(x_aug, [2]) # Vertical flip
        if k % 2 == 1:
            x_aug = torch.flip(x_aug, [3]) # Horizontal flip
            
        # 2. Forward pass through the 2-stage pipeline
        out_aug = forward_fn(x_aug)
        
        # 3. Reverse augmentations (strict reverse order)
        if k % 2 == 1:
            out_aug = torch.flip(out_aug, [3])
        if k % 4 >= 2:
            out_aug = torch.flip(out_aug, [2])
        if k >= 4:
            out_aug = out_aug.transpose(2, 3)
            
        preds.append(out_aug)
        
    # Average all 8 predictions
    return sum(preds) / 8.0

def main(args):
    # Auto-detect GPU or use specified device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"\n🚀 Running on CPU")
    
    print(f"Running Inference Pipeline with 8-Way TTA")
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        print(f"   Please create 'inputs' directory or specify --input_dir <path>")
        return
    
    setup_directories(args.output_dir)

    # Initialize Architectures (Width 64 for NAFNet)
    restormer = Restormer().to(device)
    day_nafnet = NAFNet(img_channel=3, width=64, middle_blk_num=1, 
                        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)
    night_nafnet = NAFNet(img_channel=3, width=64, middle_blk_num=1, 
                          enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)

    # Load Weights directly from arguments (No YAML overrides)
    print("\n📥 Loading Model Weights...")
    stage1_loaded = load_weights_safe(restormer, args.restormer_weights, device)
    day_loaded = load_weights_safe(day_nafnet, args.day_weights, device)
    night_loaded = load_weights_safe(night_nafnet, args.night_weights, device)
    
    if not (stage1_loaded and day_loaded and night_loaded):
        print("\n⚠️  Warning: Some weights could not be loaded.")
        print("   Models will use random initialization.")
        print("   Download pre-trained weights and update checkpoint paths in .env or CLI arguments.")
        print(f"   Expected paths:")
        print(f"     - Stage 1: {args.restormer_weights}")
        print(f"     - Day: {args.day_weights}")
        print(f"     - Night: {args.night_weights}")
        input("   Press Enter to continue with random weights (not recommended)...")
    
    restormer.eval()
    day_nafnet.eval()
    night_nafnet.eval()

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"❌ No images found in {args.input_dir}")
        return

    print(f"📦 Found {len(image_files)} images. Starting 8-Way TTA Inference...")

    # Define the 2-stage pipeline logic for a single pass
    def pipeline_forward(x_in):
        s1_out = torch.clamp(restormer(x_in), 0, 1)
        s1_np = (s1_out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        brightness = calculate_brightness(s1_np)
        
        if brightness > args.brightness_threshold:
            return day_nafnet(s1_out)
        else:
            return night_nafnet(s1_out)

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(args.input_dir, filename)
            img_bgr = cv2.imread(img_path)
            
            # Safety check for corrupted images
            if img_bgr is None: 
                print(f"⚠️  Skipping unreadable file: {filename}")
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad to multiple of 16 (Required for NAFNet architectures)
            h, w = input_tensor.shape[2:]
            ph = ((h + 15) // 16) * 16
            pw = ((w + 15) // 16) * 16
            padding = (0, pw - w, 0, ph - h)
            input_tensor = torch.nn.functional.pad(input_tensor, padding, mode='reflect')
            
            # Pass through 8-way TTA
            output = tta_forward(pipeline_forward, input_tensor)
            
            # Crop padding back to exact original size
            output = output[:, :, :h, :w]

            # Save Result
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, filename), cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))

    print(f"\n🎉 Done! Cleaned images saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raindrop Clarity Inference - Two-stage image restoration pipeline"
    )
    
    # Get defaults from environment variables or use fallback
    default_input = os.getenv("INPUT_DIR", "inputs")
    default_output = os.getenv("OUTPUT_DIR", "results")
    default_restormer = os.getenv("STAGE1_CKPT_PATH", "checkpoints/stage1/stage1_epoch_84_loss_0.0222.pth")
    default_day = os.getenv("FINAL_MODEL_PATH_DAY", "checkpoints/stage2_day/best_day_expert.pth")
    default_night = os.getenv("FINAL_MODEL_PATH_NIGHT", "checkpoints/stage2_night/best_stage2_night.pth")
    default_device = os.getenv("DEVICE", "cuda")
    
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help=f"Input directory with images (default: {default_input})")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help=f"Output directory for results (default: {default_output})")
    
    parser.add_argument("--restormer_weights", type=str, default=default_restormer,
                        help=f"Path to Stage 1 Restormer weights (default: {default_restormer})")
    parser.add_argument("--day_weights", type=str, default=default_day,
                        help=f"Path to Day-time NAFNet weights (default: {default_day})")
    parser.add_argument("--night_weights", type=str, default=default_night,
                        help=f"Path to Night-time NAFNet weights (default: {default_night})")
    parser.add_argument("--brightness_threshold", type=float, default=65.0,
                        help="Brightness threshold to route between day/night model (default: 65.0)")
    parser.add_argument("--device", type=str, default=default_device,
                        help=f"Device to use: cuda or cpu (default: {default_device})")
    
    args = parser.parse_args()
    
    # Override device if CLI argument was provided
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.set_device(0)
    
    main(args)