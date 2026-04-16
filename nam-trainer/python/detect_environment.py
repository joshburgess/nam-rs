import json, sys, subprocess, shutil, re

result = {
    "nam": False,
    "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "devices": [{"id": "cpu", "name": "CPU"}],
    "warnings": [],
    "cuda_install": None,
}


def detect_nvidia():
    """Return (gpu_names, cuda_version_str) from nvidia-smi, or (None, None)."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if q.returncode != 0 or not q.stdout.strip():
            return None, None
        names = [g.strip() for g in q.stdout.strip().split("\n") if g.strip()]
        full = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5,
        )
        cuda_version = None
        if full.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", full.stdout)
            if m:
                cuda_version = f"{m.group(1)}.{m.group(2)}"
        return names, cuda_version
    except Exception:
        return None, None


def pick_wheel_index(cuda_version):
    """Driver-reported CUDA version to newest compatible PyTorch wheel index."""
    if not cuda_version:
        return None
    try:
        major, minor = (int(p) for p in cuda_version.split("."))
    except ValueError:
        return None
    base = "https://download.pytorch.org/whl"
    if (major, minor) >= (12, 4):
        return f"{base}/cu124"
    if (major, minor) >= (12, 1):
        return f"{base}/cu121"
    if (major, minor) >= (11, 8):
        return f"{base}/cu118"
    return None


try:
    from nam.train import core
    result["nam"] = True
except ImportError:
    pass

has_cuda_torch = False
try:
    import torch
    has_cuda_torch = torch.cuda.is_available()
    if has_cuda_torch:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            result["devices"].append({"id": f"cuda:{i}", "name": f"CUDA {i}: {name}"})
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        result["devices"].append({"id": "mps", "name": "Apple GPU (MPS)"})
except ImportError:
    pass

if not has_cuda_torch:
    names, cuda_version = detect_nvidia()
    if names:
        wheel_index = pick_wheel_index(cuda_version)
        if wheel_index:
            result["cuda_install"] = {
                "cuda_version": cuda_version,
                "wheel_index": wheel_index,
                "gpu_names": names,
            }
            # Only surface a visible warning when torch IS installed. If torch
            # is missing, the Install NAM flow will handle CUDA torch setup
            # automatically, so there is nothing for the user to do.
            try:
                import torch  # noqa: F401
                result["warnings"].append(
                    f"NVIDIA GPU detected ({', '.join(names)}) but PyTorch is installed without CUDA support. "
                    f"Click \"Install PyTorch with CUDA {cuda_version}\" below to fix it."
                )
            except ImportError:
                pass
        else:
            result["warnings"].append(
                f"NVIDIA GPU detected ({', '.join(names)}) but the driver (CUDA {cuda_version or 'unknown'}) "
                f"is too old for any supported PyTorch CUDA wheel. Update NVIDIA drivers to enable GPU training."
            )

print(json.dumps(result))
