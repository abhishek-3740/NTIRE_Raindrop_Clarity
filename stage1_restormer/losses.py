import torch
import torch.nn.functional as F

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def stage1_loss(pred, clean, blur, w_clean=1.0, w_blur=0.3):
    """
    Stage-1 loss:
    - Main supervision: Clean
    - Auxiliary supervision: Blur
    """
    loss_clean = charbonnier_loss(pred, clean)
    loss_blur  = charbonnier_loss(pred, blur)

    total_loss = w_clean * loss_clean + w_blur * loss_blur

    return total_loss, {
        "loss_clean": loss_clean.item(),
        "loss_blur": loss_blur.item()
    }
