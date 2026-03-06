import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.charbonnier import CharbonnierLoss

class EdgeLoss(nn.Module):
    """
    Sobel Edge Loss for Stage-2 Refinement

    Purpose:
    - Preserve edges & fine structures
    - Prevent over-smoothing by Stage-2
    - Especially helpful for Night images
    """
    def __init__(self):
        super().__init__()

        # Sobel kernels
        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

        self.criterion = CharbonnierLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W), range [0, 1]
        """

        # Convert RGB → luminance (important for night scenes)
        pred_y = (
            0.299 * pred[:, 0:1] +
            0.587 * pred[:, 1:2] +
            0.114 * pred[:, 2:3]
        )

        target_y = (
            0.299 * target[:, 0:1] +
            0.587 * target[:, 1:2] +
            0.114 * target[:, 2:3]
        )

        # Sobel gradients
        pred_dx = F.conv2d(pred_y, self.kx, padding=1)
        pred_dy = F.conv2d(pred_y, self.ky, padding=1)

        target_dx = F.conv2d(target_y, self.kx, padding=1)
        target_dy = F.conv2d(target_y, self.ky, padding=1)

        loss_x = self.criterion(pred_dx, target_dx)
        loss_y = self.criterion(pred_dy, target_dy)

        return loss_x + loss_y