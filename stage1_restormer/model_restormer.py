import torch
# This line works because 'basicsr' is now in the top folder
from basicsr.models.archs.restormer_arch import Restormer

def build_restormer():
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    return model