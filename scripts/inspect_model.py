import torch.nn as nn

from llm.config import ModelConfig
from llm.model import DecoderTransformer

def count_parameters(module: nn.Module) -> int:
    """
    Return the total number of unique parameters in the model.
    """
    return sum(
        p.numel()
        for p in module.parameters()
    )

def count_trainable_parameters(module: nn.Module) -> int:
    """
    Return the total number of parameters that require gradients.
    """

    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

def percent_of_total(params: int, total: int) -> float:
    """
    Return a parameter count as a percentage of the model total.
    """
    return (params / total) * 100

def print_model_config(model: DecoderTransformer, config: ModelConfig) -> None:
    """
    Display the model configurations.
    """

    print("\nModel Configurations")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Model dimension: {config.d_model:,}")
    print(f"Attention heads: {config.num_heads:,}")
    print(f"Head dimension: {config.head_dim:,}")
    print(f"Feedforward dimension: {config.d_ff:,}")
    print(f"Number of layers/blocks: {config.num_layers:,}")

    print("\nEmbedding and LM projection weights tied:")
    print(model.token_embedding.weight is model.lm_head.weight)

def print_parameter_breakdown(model: DecoderTransformer) -> None:
    """
    Display parameter counts for specific model components.
    """

    first_block = model.blocks[0]

    embedding_params = count_parameters(model.token_embedding)

    attn_norm_params = count_parameters(first_block.attn_norm)
    attention_params = count_parameters(first_block.attn)

    mlp_norm_params = count_parameters(first_block.mlp_norm)
    mlp_params = count_parameters(first_block.mlp)

    block_params = count_parameters(first_block)
    all_blocks_params = count_parameters(model.blocks)

    final_norm_params = count_parameters(model.final_norm)

    total_params = count_parameters(model)

    print("\nParameter Breakdown")
    print(
        f"Token embedding parameters: "
        f"{embedding_params:,} "
        f"({percent_of_total(embedding_params, total_params):.2f}%)"
    )

    print(
        f"Attention parameters: "
        f"{attention_params:,} "
        f"({percent_of_total(attention_params, block_params):.2f}% of one block)"
    )

    print(
        f"MLP parameters: "
        f"{mlp_params:,} "
        f"({percent_of_total(mlp_params, block_params):.2f}% of one block)"
    )

    print(
        f"Attention norm parameters: "
        f"{attn_norm_params:,}"
    )

    print(
        f"MLP norm parameters: "
        f"{mlp_norm_params:,}"
    )

    print(
        f"Single block parameters: "
        f"{block_params:,}"
        f" ({percent_of_total(block_params, total_params):.2f}%)"
    )

    print(
        f"All blocks parameters: "
        f"{all_blocks_params:,} "
        f"({percent_of_total(all_blocks_params, total_params):.2f}%)"
    )

    print(
        f"Final norm parameters: "
        f"{final_norm_params:,}"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

def parameter_memory_mb(num_parameters: int, bytes_per_parameter: int) -> float:
    """
    Estimate raw parameter storage in MiB.
    """
    bytes_total = (num_parameters * bytes_per_parameter)

    return bytes_total / (1024 ** 2)

def print_parameter_memory(model: DecoderTransformer) -> None:
    """
    Display estimated raw parameter storage.
    """

    total_params = count_parameters(model)

    fp32_mb = parameter_memory_mb(
        total_params,
        bytes_per_parameter=4
    )

    fp16_mb = parameter_memory_mb(
        total_params,
        bytes_per_parameter=2
    )

    print("\nRaw Parameter Memory")
    print(f"FP32:      {fp32_mb:.2f} MiB")
    print(f"FP16/BF16: {fp16_mb:.2f} MiB")

def main() -> None:

    base_vocab_size = 16000
    base_d_model = 384
    base_num_heads = 6 # so head_dim = 64
    base_d_ff = 1024
    base_num_layers = 6

    
    config = ModelConfig(
        vocab_size=32000,
        d_model=512,
        num_heads=8,
        d_ff=1408,
        num_layers=12
    )

    model = DecoderTransformer(config)

    print_model_config(model, model.config)
    print_parameter_breakdown(model)
    print_parameter_memory(model)

    print(f"\nTrainable parameters: {count_trainable_parameters(model):,}")

if __name__ == "__main__":
    main()

