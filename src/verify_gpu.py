import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

if torch.cuda.is_available():
    x = torch.rand(5, 3)
    print("Tensor en CPU:", x)
    x = x.cuda()
    print("Tensor en GPU:", x) 