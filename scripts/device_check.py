import torch

print(f"PyTorch version: [{torch.__version__}]")
print(f"CUDA available: [{torch.cuda.is_available()}]")
print(f"MPS available: [{torch.backends.mps.is_available()}]")

if torch.cuda.is_available():
    print("\nGPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        "GiB"
    )