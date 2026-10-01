import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock


class DecoderTransformer(nn.Module):
    "Decoder-only Transformer model."

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # Input embedding table
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Transformer block stacks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        # Final RMSNorm
        self.final_norm = RMSNorm(config.d_model)

        # Initialize output projection
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Tie input embedding and output projection
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed token IDs
        x = self.token_embedding(x)

        # Iterate over blocks, passing each output to the next
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Output projection
        logits = self.lm_head(x)

        return logits