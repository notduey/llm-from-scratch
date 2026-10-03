"""
$env:PYTHONPATH="src" in terminal to run
"""

import time

import torch
import torch.nn.functional as F

from llm.config import ModelConfig
from llm.model import DecoderTransformer

# Configurations
config = ModelConfig(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    d_ff=1408,
    num_layers=12
)

# Input dimensions [B,T]
batch_size = 8
sequence_length = 1024

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for GPU benchmarking."
    )

# Move model to GPU
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

model = DecoderTransformer(config).to(device)

# Token ID inputs and targets
token_ids = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length + 1), # one extra token so targets can be shifted
    device=device
)

inputs = token_ids[:, :-1] # omit last token since there's no next-token target
targets = token_ids[:, 1:] # omit first token since it's not a prediction target

# Warmup/benchmark cycles
warmup_iterations = 3
benchmark_iterations = 10

# Warmup
for _ in range(warmup_iterations):
    model.zero_grad(set_to_none=True)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1)
    )

    loss.backward()

torch.cuda.synchronize()

# Reset benchmark state
model.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# Actual Benchmarks
start = time.perf_counter()

for _ in range(benchmark_iterations):
    model.zero_grad(set_to_none=True)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1)
    )

    loss.backward()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# Calculate average across the interations
average_time = elapsed / benchmark_iterations
tokens_per_iteration = (batch_size * sequence_length)

tokens_per_second = (tokens_per_iteration / average_time)

# Get peak GPU memory usage across the iterations
peak_allocated = torch.cuda.max_memory_allocated() # max memory actually used from GPU
peak_reserved = torch.cuda.max_memory_reserved() # max reserved in advance

# Convert memory usage from bytes to GiB
bytes_per_gib = 1024 ** 3

peak_allocated_gib = peak_allocated / bytes_per_gib
peak_reserved_gib = peak_reserved / bytes_per_gib

# Display results
print("\nBenchmark")
print("-" * 35)
print(f"Batch size: {batch_size}")
print(f"Sequence length: {sequence_length}")
print(f"Tokens/batch: {tokens_per_iteration:,}")
print(f"Measured iterations: {benchmark_iterations}")

print(f"\nAverage iteration time: {average_time:.4f} s")
print(f"Tokens/second: {tokens_per_second:,.0f}")
print(f"Peak allocated VRAM: {peak_allocated_gib:.2f} GiB")
print(f"Peak reserved VRAM: {peak_reserved_gib:.2f} GiB")

print(f"\nLoss: {loss.item():.4f}")