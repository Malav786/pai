"""Utility functions for reproducibility and random seed management"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all random number generators.

    Args:
        seed (int): Random seed value to use. Default is 42.
        deterministic (bool): If True, enables deterministic algorithms in PyTorch
            and disables cuDNN benchmarking for reproducibility. Default is True.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
