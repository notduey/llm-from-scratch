from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for the Transformer language model
    """

    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int

    num_layers: int

    # Validate config after dataclass initialization
    def __post_init__(self) -> None:

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    # Derived properties
    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads
