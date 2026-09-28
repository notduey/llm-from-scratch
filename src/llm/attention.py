import math

import torch
import torch.nn as nn

from .config import ModelConfig

# TODO: Precompute/cache RoPE values and causal mask, once max context length is configured.

# build the RoPE angles
def build_rope_angles(sequence_length, head_dim, device, base=10000):
    pair_indices = torch.arange(
        0,
        head_dim,
        2, # pairs
        dtype=torch.float32,
        device=device
    )

    inverse_frequencies = 1.0 / (
        base ** (pair_indices / head_dim)
    )

    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
        device=device
    )

    angles = (
        positions[:, None] # P
        * inverse_frequencies[None, :] # 1 / base^(i/d)
    )

    return torch.cos(angles), torch.sin(angles)

# Apply the rotations
def apply_rope(x, cos, sin):
    # Split into even and odd indices
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Add batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotations
    rotated_even = (
        x_even * cos
        - x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        + x_odd * cos
    )

    rotated = torch.empty_like(x)

    # Fill tensor's even/odd indices
    rotated[..., 0::2] = rotated_even
    rotated[..., 1::2] = rotated_odd

    return rotated

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with rotary positional embeddings.
    """
    def __init__(self, config: ModelConfig):
        super().__init__() # call parent constructor

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(
            self.d_model,
            3 * self.d_model,
            bias=False # no bias for now
        )

        self.out_proj = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False
        )

    def forward(self, x):
        _, sequence_length, _ = x.shape # only need second dim

        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        cos, sin = build_rope_angles(
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            device=x.device
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled attention scores
        scores = q @ k.transpose(-2, -1) # transpose last 2 dimensions
        scores = scores / math.sqrt(self.head_dim)

        # Causally mask future positions
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool, # boolean mask
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~mask, #inverse booleans
            float('-inf')
        )

        # Softmax across columns in each row
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of V
        head_outputs = attention_weights @ v

        # Concatenate heads
        combined = self._combine_heads(head_outputs)

        # Mix information across heads with output projection
        output = self.out_proj(combined)

        return output

    # helper that splits matrix representation across heads
    def _split_heads(self, x):
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size, # B
            sequence_length, # T
            self.num_heads, # H
            self.head_dim # d_head
        )

        return x.transpose(1, 2) # transpose to [B, H, T, d_head]

    # helper that concatenates the heads
    def _combine_heads(self, x):
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2) # transpose back to [B, T, H, d_head]

        return x.reshape(
            batch_size, # B
            sequence_length, # T
            self.d_model # d_model
        )