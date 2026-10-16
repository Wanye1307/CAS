"""
Utility functions for the Medical VQA framework.
"""

import os
import random
from typing import Union, Dict, List

import numpy as np


def set_gpu(gpu_id: Union[int, List[int], str]) -> None:
    """Set which GPU(s) to use before importing torch.

    IMPORTANT: Call this BEFORE importing torch or any torch-dependent modules.

    Args:
        gpu_id: GPU index (int), list of indices, or comma-separated string

    Examples:
        set_gpu(0)           # Use GPU 0
        set_gpu(5)           # Use GPU 5
        set_gpu([0, 1])      # Use GPUs 0 and 1
        set_gpu("0,5,6")     # Use GPUs 0, 5, and 6
    """
    if isinstance(gpu_id, int):
        gpu_str = str(gpu_id)
    elif isinstance(gpu_id, list):
        gpu_str = ",".join(str(g) for g in gpu_id)
    else:
        gpu_str = str(gpu_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={gpu_str}")


# Import torch after potential GPU setting
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information.

    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "num_gpus": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def print_gpu_memory(prefix: str = "") -> None:
    """Print current GPU memory usage.

    Args:
        prefix: Optional prefix for the output
    """
    info = get_gpu_memory_info()

    if not info["available"]:
        print(f"{prefix}No GPU available")
        return

    print(f"{prefix}GPU Memory: {info['allocated_gb']:.2f} GB allocated, "
          f"{info['reserved_gb']:.2f} GB reserved")
