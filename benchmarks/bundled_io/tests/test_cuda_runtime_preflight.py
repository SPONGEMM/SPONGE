from __future__ import annotations

import ctypes
import ctypes.util
import subprocess

CUDA_SUCCESS = 0


def _nvidia_smi_diagnostic() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        return "nvidia-smi was not found on PATH"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out"
    output = "\n".join(item for item in (result.stdout, result.stderr) if item)
    return f"nvidia-smi exited with {result.returncode}\n{output}".strip()


def test_cuda_runtime_is_available_for_production_smoke():
    library_path = ctypes.util.find_library("cudart")
    assert library_path is not None, (
        "CUDA runtime library libcudart was not found; dev-cuda13 production "
        "smoke requires an initialized CUDA runtime."
    )

    cudart = ctypes.CDLL(library_path)
    cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDeviceCount.restype = ctypes.c_int

    device_count = ctypes.c_int()
    rc = cudart.cudaGetDeviceCount(ctypes.byref(device_count))
    assert rc == CUDA_SUCCESS, (
        f"cudaGetDeviceCount failed with CUDA error code {rc}; dev-cuda13 "
        "production smoke requires a usable CUDA driver/runtime.\n"
        f"{_nvidia_smi_diagnostic()}"
    )
    assert device_count.value > 0, (
        "cudaGetDeviceCount reported zero CUDA devices; dev-cuda13 production "
        "smoke requires at least one visible GPU."
    )
