import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  

current_dir = os.path.dirname(os.path.abspath(__file__)) 
pipeline_dir = os.path.abspath(os.path.join(current_dir, ".."))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

import configs.nafnet_config as cfg
from models.nafnet_refiner import NAFNet  # <--- CHANGED: Now importing the official NAFNet
from datasets.stage2_dataset import Stage2Dataset
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

    ensure_dir(cfg.SAVE_DIR)

    print("\n========================================================")
    print("☀️  STARTING STAGE-2 NAFNET REFINER (FINE-TUNING) ☀️")
    print("========================================================")
    print("STRATEGY: Pre-trained 'GoPro' Brain -> 'Day-Expert'")
    print("DATASET:  Strictly loading 'Day_' images (Night images ignored)")
    print(f"DEVICE:   {device}")
    print("========================================================\n")

    # -------------------------
    # Model (Fine-Tuning NAFNet-GoPro)
    # -------------------------
    model = NAFNet(
        img_channel=cfg.IMG_CHANNEL,
        width=cfg.WIDTH,
        middle_blk_num=cfg.MIDDLE_BLK_NUM,
        enc_blk_nums=cfg.ENC_BLK_NUMS,
        dec_blk_nums=cfg.DEC_BLK_NUMS
    ).to(device)

    # 🧠 Load the pre-trained weights
    pretrained_path = os.getenv("PRETRAINED_PATH", "checkpoints/NAFNet-GoPro-width64.pth")
    
    print(f"🧠 Injecting pre-trained weights from: {pretrained_path}")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        # Handle different saved dictionary formats (official NAFNet uses 'params')
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print("✅ Pre-trained weights successfully loaded! Ready to fine-tune.\n")
    else:
        raise FileNotFoundError(f"❌ Could not find weights at {pretrained_path}. Did you download them?")

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
    # Losses
    # -------------------------
    loss_char = CharbonnierLoss().to(device)
    loss_edge = EdgeLoss().to(device)

    # -------------------------
    # Datasets & Loaders
    # -------------------------
    train_dataset = Stage2Dataset(
        stage1_dir=cfg.TRAIN_STAGE1_DIR,
        gt_dir=cfg.TRAIN_GT_DIR,
        patch_size=cfg.PATCH_SIZE,
        train=True,
    )

    val_dataset = Stage2Dataset(
        stage1_dir=cfg.VAL_STAGE1_DIR,
        gt_dir=cfg.VAL_GT_DIR,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.\n")

    # -------------------------
    # Training Loop Setup & Resume Logic
    # -------------------------
    best_psnr = 0.0
    start_epoch = 1
    resume_path = os.path.join(cfg.SAVE_DIR, "best_stage2.pth")

    if os.path.exists(resume_path):
        print(f"🔄 Found checkpoint at {resume_path}. Resuming training...")
        last_epoch, saved_psnr = load_checkpoint(model, optimizer, resume_path, device)
        start_epoch = last_epoch + 1
        
        if saved_psnr is not None:
            best_psnr = saved_psnr
            
        for _ in range(last_epoch):
            scheduler.step()
            
        print(f"⏩ Resuming from Epoch {start_epoch} with Best PSNR: {best_psnr:.2f} dB\n")
    else:
        print(f"✨ Starting fresh fine-tuning for {cfg.NUM_EPOCHS} epochs...\n")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS} [Train]", unit="batch")

        for x, y in pbar_train:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = (
                loss_char(out, y)
                + cfg.EDGE_WEIGHT * loss_edge(out, y)
            )

            loss.backward()

            if cfg.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.GRAD_CLIP,
                )

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
                x = x.to(device)
                y = y.to(device)

                out = clamp_image(model(x))
                val_psnr += psnr(out, y)

        val_psnr /= len(val_loader)

        print(
            f"\n> Epoch {epoch:03d} Summary | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )

        # -------------------------
        # Checkpointing
        # -------------------------
        if val_psnr > best_psnr:
            print(f"   🌟 New best PSNR! ({best_psnr:.2f} -> {val_psnr:.2f}). Saving checkpoint...")
            best_psnr = val_psnr
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                resume_path,
            )
        print("-" * 60)

    print(
        f"\n🎉 Training finished. "
        f"Best Validation PSNR: {best_psnr:.2f} dB"
    )

if __name__ == "__main__":
    main()