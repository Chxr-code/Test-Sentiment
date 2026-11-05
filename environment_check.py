import platform
import sys

print("===== THAI SENTIMENT.AI =====")

print("\n[Python]")
print(sys.version)

print("\n[Operating System]")
print(platform.system(), platform.release())

print("\n[Processor]")
print(platform.processor())

print("\n[RAM]")
try:
    import psutil
    ram = psutil.virtual_memory()
    print("Total:", round(ram.total / (1024**3), 2), "GB")
    print("Available:", round(ram.available / (1024**3), 2), "GB")
except ImportError:
    print("psutil not installed")

print("\n[PyTorch]")
try:
    import torch
    print("Version:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch not installed")

print("\n===== CHECK COMPLETED =====")