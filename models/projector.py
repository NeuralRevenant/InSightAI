import torch, torch.nn as nn
from ..config import IMG_SIZE

class Projector(nn.Module):
    """
    Align visual CLS embedding (1024) to Llama token dim (4096) with MLP.
    """
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):          # x : (B, 1024)
        return self.proj(x)        # returns (B, 4096)
