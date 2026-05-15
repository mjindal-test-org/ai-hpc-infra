# PyTorch — most common way in ML code
import torch
print(torch.cuda.device_count())       # number of GPUs
print(torch.cuda.is_available())       # True/False — is CUDA working?

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))   # name of GPU 0, e.g. "NVIDIA A100 80GB"

    # Loop through all GPUs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} — {props.total_memory/1e9:.1f} GB")
else:
    print("CUDA is not available on this machine.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS (Metal Performance Shaders) is available.")