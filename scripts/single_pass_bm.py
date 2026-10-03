"""
$env:PYTHONPATH="src" in terminal to run
"""

import time

import torch
import torch.nn.functional as F

from llm.config import ModelConfig
from llm.model import DecoderTransformer

# Proposed configuration
config = ModelConfig(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    d_ff=1408,
    num_layers=12
)

batch_size = 2
sequence_length = 512

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for GPU benchmarking."
    )

# Move model to GPU
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

model = DecoderTransformer(config).to(device)

token_ids = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length + 1), # one extra token so targets can be shifted
    device=device
)

inputs = token_ids[:, :-1]
targets = token_ids[:, 1:]

# Forward/backward pass benchmarking
model.zero_grad(set_to_none=True) # reset gradients
torch.cuda.reset_peak_memory_stats() # reset peak memory-usage

torch.cuda.synchronize() # force CPU to wait until previous GPU operations finish
start = time.perf_counter()

logits = model(inputs) # forward pass

loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    targets.reshape(-1)
)

loss.backward() # backprop

torch.cuda.synchronize() # make CPU wait until benchmarked GPU operations finish
elapsed = time.perf_counter() - start # calculated elapsed time

# Get peak GPU memory usage from latest forward/backward pass
peak_allocated = torch.cuda.max_memory_allocated() # max memory actually used from GPU
peak_reserved = torch.cuda.max_memory_reserved() # max reserved in advance

num_tokens = batch_size * sequence_length
tokens_per_second = num_tokens / elapsed

# Convert bytes to GiB
bytes_per_gib = 1024 ** 3

peak_allocated_gib = peak_allocated / bytes_per_gib
peak_reserved_gib = peak_reserved / bytes_per_gib

print("\nBenchmark")
print(f"Batch size: {batch_size}")
print(f"Sequence length: {sequence_length}")
print(f"Tokens/batch: {num_tokens:,}")

print(f"\nElapsed time: {elapsed:.4f} seconds")
print(f"Tokens/second: {tokens_per_second:,.0f}")
print(f"Peak allocated VRAM: {peak_allocated_gib:.2f} GiB")
print(f"Peak reserved VRAM: {peak_reserved_gib:.2f} GiB")
print(f"Loss: {loss.item():.4f}")