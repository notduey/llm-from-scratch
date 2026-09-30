import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

class RMSNorm(nn.Module):
    """
    Root mean square normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model) # initial weights 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt( #root
            x.pow(2).mean(dim=-1, keepdim=True) # mean squared
            + self.eps # added for num stability
        )

        return (x / rms) * self.weight # normalized * learned featurewise scale

class SwiGLU(nn.Module):
    """
    Swish gated linear unit feedforward network.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # linear projection layers
        self.gate_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )

        self.up_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )

        self.down_proj = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x)) # activation to gate

        value = self.up_proj(x)

        return self.down_proj(gate * value) # down_proj(silu(gate) * value)