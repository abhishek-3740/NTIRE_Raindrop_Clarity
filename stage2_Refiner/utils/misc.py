import os
import random
import numpy as np
import torch

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Directory utilities
# ============================================================

def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================
# Model checkpointing
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    psnr: float,
    path: str
):
    """
    Save training checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "psnr": psnr,
        },
        path
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: str = "cuda"
):
    """
    Load training checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt.get("psnr", None)


# ============================================================
# Image clamping
# ============================================================

def clamp_image(x: torch.Tensor) -> torch.Tensor:
    """
    Clamp tensor to valid image range [0, 1].
    """
    return torch.clamp(x, 0.0, 1.0)