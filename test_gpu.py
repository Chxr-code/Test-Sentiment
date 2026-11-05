
import torch

print("===== THAI SENTIMENT.AI =====")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

    x = torch.randn(3, 3).to("cuda")
    y = torch.randn(3, 3).to("cuda")

    result = torch.matmul(x, y)

    print("\nGPU Calculation:")
    print(result)

    print("\nDevice:", result.device)
    print("\nGPU TEST PASSED")

else:
    print("GPU NOT AVAILABLE")

print("==============================")
