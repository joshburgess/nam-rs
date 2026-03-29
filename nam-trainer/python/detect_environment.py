import json, sys, subprocess, shutil
result = {
    "nam": False,
    "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "devices": [{"id": "cpu", "name": "CPU"}],
    "warnings": []
}
try:
    from nam.train import core
    result["nam"] = True
except ImportError:
    pass
try:
    import torch
    has_cuda_torch = torch.cuda.is_available()
    if has_cuda_torch:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            result["devices"].append({"id": f"cuda:{i}", "name": f"CUDA {i}: {name}"})
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        result["devices"].append({"id": "mps", "name": "Apple GPU (MPS)"})

    # Check for NVIDIA GPU hardware that PyTorch can't see
    if not has_cuda_torch and shutil.which("nvidia-smi"):
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if smi.returncode == 0 and smi.stdout.strip():
                gpu_names = [g.strip() for g in smi.stdout.strip().split("\n") if g.strip()]
                result["warnings"].append(
                    f"NVIDIA GPU detected ({', '.join(gpu_names)}) but PyTorch was installed without CUDA support. "
                    f"Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )
        except Exception:
            pass
except ImportError:
    # No torch at all — check if GPU hardware exists anyway
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if smi.returncode == 0 and smi.stdout.strip():
                gpu_names = [g.strip() for g in smi.stdout.strip().split("\n") if g.strip()]
                result["warnings"].append(
                    f"NVIDIA GPU detected ({', '.join(gpu_names)}) — install PyTorch with CUDA for GPU training: "
                    f"pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )
        except Exception:
            pass
print(json.dumps(result))
