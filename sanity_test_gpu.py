import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

# Simple tensor test
x = torch.randn(10_000, 10_000, device=device)
y = torch.randn(10_000, 10_000, device=device)

torch.cuda.synchronize()
t0 = time.time()
z = x @ y
torch.cuda.synchronize()
t1 = time.time()

print("Matrix multiply time (s):", round(t1 - t0, 3))
print("Result device:", z.device)
