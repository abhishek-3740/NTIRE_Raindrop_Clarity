import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  

# ------------------------------------------------------------
# DYNAMIC PATH RESOLUTION (Fixes the rename crash)
# ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) 
pipeline_dir = os.path.abspath(os.path.join(current_dir, ".."))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

import configs.nafnet_night_config as cfg
from models.nafnet_refiner import NAFNet  
from datasets.stage2_night_dataset import Stage2NightDataset
from losses.charbonnier import CharbonnierLoss
from losses.edge_loss import EdgeLoss
from utils.psnr import psnr
from utils.misc import (
    set_seed,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    clamp_image,
)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # -------------------------
    # Reproducibility & Device
    # -------------------------
    set_seed(cfg.SEED)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # 🛡️ DYNAMIC SAVE DIRECTORY
    # This guarantees the folder exists inside your CURRENT project folder
    save_dir = os.path.join(pipeline_dir, "checkpoints", "stage2_night")
    ensure_dir(save_dir)
    resume_path = os.path.join(save_dir, "best_stage2_night.pth")

    print("\n========================================================")
    print("🌙  STARTING/RESUMING STAGE-2 NAFNET REFINER (NIGHT) 🌙")
    print("========================================================")
    print("STRATEGY: Pre-trained 'GoPro' Brain -> 'Night-Expert'")
    print(f"DEVICE:   {device}")
    print(f"SAVE DIR: {save_dir}")
    print("========================================================\n")

    # -------------------------
    # Model Setup
    # -------------------------
    model = NAFNet(
        img_channel=cfg.IMG_CHANNEL,
        width=cfg.WIDTH,
        middle_blk_num=cfg.MIDDLE_BLK_NUM,
        enc_blk_nums=cfg.ENC_BLK_NUMS,
        dec_blk_nums=cfg.DEC_BLK_NUMS
    ).to(device)

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,
        eta_min=cfg.MIN_LR,
    )

    # -------------------------
    # Resume Logic OR Fresh Start
    # -------------------------
    best_psnr = 0.0
    start_epoch = 1

    if os.path.exists(resume_path):
        print(f"🔄 Found expert checkpoint at {resume_path}. Resuming...")
        last_epoch, saved_psnr = load_checkpoint(model, optimizer, resume_path, device)
        start_epoch = last_epoch + 1
        
        if saved_psnr is not None:
            best_psnr = saved_psnr
            
        # Fast-forward scheduler to maintain the correct learning rate curve
        for _ in range(last_epoch):
            scheduler.step()
            
        print(f"⏩ Resuming from Epoch {start_epoch} with Best PSNR: {best_psnr:.2f} dB\n")
    else:
        # Fallback to GoPro weights if no night expert exists yet
        pretrained_path = os.path.join(pipeline_dir, "checkpoints", "NAFNet-GoPro-width64.pth")
        print(f"🧠 Injecting pre-trained GoPro weights from: {pretrained_path}")
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
            print("✅ Pre-trained GoPro weights loaded!\n")
        else:
            print(f"⚠️ Warning: Base GoPro weights not found at {pretrained_path}. Starting from scratch.\n")

    # -------------------------
    # Losses & Datasets
    # -------------------------
    loss_char = CharbonnierLoss().to(device)
    loss_edge = EdgeLoss().to(device)

    train_dataset = Stage2NightDataset(stage1_dir=cfg.TRAIN_STAGE1_DIR, gt_dir=cfg.TRAIN_GT_DIR, patch_size=cfg.PATCH_SIZE, train=True)
    val_dataset = Stage2NightDataset(stage1_dir=cfg.VAL_STAGE1_DIR, gt_dir=cfg.VAL_GT_DIR, train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Loaded {len(train_dataset)} Night training images and {len(val_dataset)} Night validation images.\n")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS} [Train]", unit="batch")

        for x, y in pbar_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            loss = loss_char(out, y) + cfg.EDGE_WEIGHT * loss_edge(out, y)
            loss.backward()

            if cfg.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

            optimizer.step()
            epoch_loss += loss.item()
            pbar_train.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_psnr = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS} [Val]", unit="img")

        with torch.no_grad():
            for x, y in pbar_val:
                out = clamp_image(model(x.to(device)))
                val_psnr += psnr(out, y.to(device))

        val_psnr /= len(val_loader)

        print(f"\n> Epoch {epoch:03d} Summary | Avg Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        # -------------------------
        # Checkpointing
        # -------------------------
        if val_psnr > best_psnr:
            print(f"   🌟 New best PSNR! ({best_psnr:.2f} -> {val_psnr:.2f}). Saving checkpoint...")
            best_psnr = val_psnr
            # We explicitly pass our dynamic resume_path here
            save_checkpoint(model, optimizer, epoch, best_psnr, resume_path)
            
        print("-" * 60)

    print(f"\n🎉 Night Training finished. Best Validation PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    main()