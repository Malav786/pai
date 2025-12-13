"""
Defense mechanisms for protecting models against attacks.
"""

import numpy as np


def defend_postprocess(probs, topk=None, decimals=2, add_noise=False):
    """
    Apply defense mechanisms to softmax probabilities.
    
    This function implements several defense strategies:
    - Top-k filtering: Only return top-k probabilities
    - Probability rounding: Round probabilities to reduce gradient resolution
    - Calibrated noise: Add small Dirichlet noise to probabilities
    
    Args:
        probs: Softmax probability vector
        topk: If not None, only return top-k probabilities (others set to 0)
        decimals: If not None, round probabilities to this many decimal places
        add_noise: If True, add small Dirichlet noise to probabilities
    
    Returns:
        Defended probability vector (normalized to sum to 1)
    """
    probs = probs.copy()
    
    # Top-k only: only return top-k probabilities
    if topk is not None:
        topk_idx = np.argsort(probs)[-topk:]
        newp = np.zeros_like(probs)
        newp[topk_idx] = probs[topk_idx]
        probs = newp / (newp.sum() + 1e-12)
    
    # Round probabilities to reduce gradient resolution
    if decimals is not None:
        probs = np.round(probs, decimals)
        probs = probs / (probs.sum() + 1e-12)
    
    # Add calibrated noise
    if add_noise:
        noise = np.random.dirichlet([0.1] * len(probs))
        probs = probs + 0.01 * noise
        probs = probs / probs.sum()
    
    return probs

