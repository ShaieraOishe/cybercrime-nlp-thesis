"""Reproducibility & Compute Acceleration Utilities.

==============================================================================
THESIS CONTEXT:
Academic reproducibility is a foundational requirement for NLP thesis research.
Stochastic processes in Python, NumPy, and PyTorch (e.g. random weight initialization,
shuffling in data loaders, dropout layers) must be seeded deterministically so that
all reported evaluation metrics (Macro-F1, MRR@10, NDCG@10) are strictly replicable.

This module also provides automated device detection for Apple Silicon (Metal
Performance Shaders / MPS) to accelerate Transformer training on Mac GPUs.
==============================================================================
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across standard Python, NumPy, and PyTorch for deterministic runs.

    Args:
        seed (int): The integer seed value (default: 42).
    """
    # 1. Standard Python random module
    random.seed(seed)
    
    # 2. NumPy mathematical random generator
    np.random.seed(seed)
    
    # 3. Python environment hash seed (guarantees deterministic dict/set ordering)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 4. PyTorch CPU random seed
    torch.manual_seed(seed)
    
    # 5. CUDA GPU settings (if running on an Nvidia GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # 6. Apple Silicon GPU settings (Metal Performance Shaders / MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"[✓] Deterministic random seed successfully set to: {seed}")

def get_device() -> torch.device:
    """Detects and returns the best available compute hardware accelerator.

    Priority order:
    1. CUDA (Nvidia GPU cluster / Google Colab)
    2. MPS (Apple Silicon GPU: M1/M2/M3/M4 via Metal Performance Shaders)
    3. CPU (Fallback)

    Returns:
        torch.device: The selected PyTorch device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[✓] Active hardware acceleration: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[✓] Active hardware acceleration: Apple Silicon Metal GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[✓] Active hardware acceleration: Standard CPU")
        
    return device
