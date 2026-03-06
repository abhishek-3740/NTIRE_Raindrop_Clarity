import os
import sys
import copy
import random
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import configs.restormer as cfg
from stage1_restormer.dataset_stage1_patch import Stage1RainDatasetPatch
from stage1_restormer.model_restormer import build_restormer

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class Stage1Loss(nn.Module):
    def __init__(self, w_clean=1.0, w_blur=0.3):
        super().__init__()
        self.w_clean = w_clean
        self.w_blur = w_blur
        self.criterion = CharbonnierLoss()

    def forward(self, pred, clean, blur):
        loss_clean = self.criterion(pred, clean)
        total_loss = self.w_clean * loss_clean
        return total_loss, {"loss_clean": loss_clean.item(), "loss_blur": 0.0}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def update_ema(model, ema_model, decay):
    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

# Track top checkpoints safely
best_checkpoints = [] 

def save_checkpoint(epoch, model, ema_model, optimizer, scaler, scheduler, current_loss):
    global best_checkpoints
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    
    # Safe string formatting
    ckpt_name = "stage1_epoch_{}_loss_{:.4f}.pth".format(epoch, current_loss)
    ckpt_path = os.path.join(cfg.CKPT_DIR, ckpt_name)
    
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(), # Saving scheduler state
        "loss": current_loss,
        "best_checkpoints": best_checkpoints 
    }
    
    torch.save(save_dict, ckpt_path)
    torch.save(save_dict, os.path.join(cfg.CKPT_DIR, "latest.pth"))
    
    # Manage tracking (DELETION LOGIC COMPLETELY REMOVED FOR SAFETY)
    best_checkpoints.append((current_loss, epoch, ckpt_path))
    best_checkpoints.sort(key=lambda x: x[0])
            
    print("\n[Checkpoint] Saved Epoch {}. Loss: {:.4f}.".format(epoch, current_loss))

def load_checkpoint(path, model, ema_model, optimizer, scaler, device):
    global best_checkpoints
    
    print("\n[Resume] Successfully loading checkpoint -> {}".format(path))
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "ema_model" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    
    if "best_checkpoints" in ckpt:
        best_checkpoints = ckpt["best_checkpoints"]
        
    return ckpt["epoch"]

def main():
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[Device] Using {}".format(device))

    dataset = Stage1RainDatasetPatch(
        roots=[
            {"name": "day", "path": cfg.DAY_ROOT, "clean_dir": "Clear"},
            {"name": "night", "path": cfg.NIGHT_ROOT, "clean_dir": "Clear"},
        ],
        patch_size=cfg.PATCH_SIZE,
        debug=False,
    )

    print("[Data] Found {} training samples.".format(len(dataset)))

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = build_restormer().to(device)
    model.train()

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.WEIGHT_DECAY,
    )
    
    criterion = Stage1Loss(w_clean=cfg.W_CLEAN, w_blur=cfg.W_BLUR).to(device)
    scaler = GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    # --- SMART RESUME LOGIC ---
    start_epoch = 0
    epoch84_ckpt = os.path.join(cfg.CKPT_DIR, "stage1_epoch_84_loss_0.0247.pth")
    latest_ckpt = os.path.join(cfg.CKPT_DIR, "latest.pth")
    
    if os.path.exists(epoch84_ckpt):
        start_epoch = load_checkpoint(epoch84_ckpt, model, ema_model, optimizer, scaler, device)
    elif os.path.exists(latest_ckpt):
        start_epoch = load_checkpoint(latest_ckpt, model, ema_model, optimizer, scaler, device)


    # --- INITIALIZE SCHEDULER SAFELY ---
    # We must manually inject 'initial_lr' into the optimizer because 
    # the older checkpoints were saved without a scheduler.
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = cfg.LR

    # last_epoch tells the scheduler we are already X epochs into the training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.EPOCHS, 
        eta_min=1e-6, 
        last_epoch=start_epoch - 1
    )

    print("[Start] Training resuming from epoch {} to {} epochs...".format(start_epoch + 1, cfg.EPOCHS))
    
    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        epoch_total_loss = 0.0
        
        current_lr = scheduler.get_last_lr()[0]
        
        pbar = tqdm(loader, desc="Epoch {}/{} [LR: {:.6f}]".format(epoch+1, cfg.EPOCHS, current_lr), unit="batch")
        
        for i, (drop, blur, clean) in enumerate(pbar):
            drop = drop.to(device, non_blocking=True)
            blur = blur.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(cfg.USE_AMP and device.type == "cuda")):
                pred = model(drop)
                loss, loss_dict = criterion(pred, clean, blur)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if cfg.USE_EMA:
                update_ema(model, ema_model, cfg.EMA_DECAY)

            epoch_total_loss += loss.item()
            pbar.set_postfix({
                "Loss": "{:.4f}".format(loss.item()), 
                "Clean": "{:.4f}".format(loss_dict['loss_clean'])
            })

        # Step the scheduler at the end of the epoch
        scheduler.step()

        avg_loss = epoch_total_loss / len(loader)
        save_checkpoint(epoch + 1, model, ema_model, optimizer, scaler, scheduler, avg_loss)

    print("✅ Stage-1 training completed successfully")

if __name__ == "__main__":
    main()