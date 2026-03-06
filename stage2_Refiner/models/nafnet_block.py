import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Official NAFNet LayerNorm (2D optimized)
# ============================================================

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        N, C, H, W = grad_output.size()

        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = (g - mean_g - y * mean_gy) / torch.sqrt(var + eps)

        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """Official NAFNet LayerNorm for image tensors (N, C, H, W)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# ============================================================
# SimpleGate (NAFNet core idea)
# ============================================================

class SimpleGate(nn.Module):
    """Splits channels into two halves and multiplies."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# ============================================================
# Simplified Channel Attention (SCA)
# ============================================================

class SCA(nn.Module):
    """Simplified Channel Attention used in NAFNet."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))

# ============================================================
# Official NAFBlock
# ============================================================

class NAFBlock(nn.Module):
    """
    Official NAFNet Block
    Paper: Simple Baselines for Image Restoration (NAFNet)
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channels = c * DW_Expand
        ffn_channels = c * FFN_Expand

        # --- First branch ---
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, kernel_size=3,
            padding=1, groups=dw_channels, bias=True
        )
        self.sg = SimpleGate()
        self.sca = SCA(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, c, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # --- Second branch ---
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, c, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Learnable residual scaling
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        # --- First residual path ---
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        # --- Second residual path ---
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return x + y * self.gamma