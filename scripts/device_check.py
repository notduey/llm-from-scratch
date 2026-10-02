import torch

print(f"PyTorch version: [{torch.__version__}]")
print(f"CUDA available: [{torch.cuda.is_available()}]")
print(f"MPS available: [{torch.backends.mps.is_available()}]")