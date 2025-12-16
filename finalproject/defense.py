"""
Defense mechanisms for protecting models against attacks.
"""

import numpy as np


def defend_postprocess(
    probs, topk=None, decimals=2, add_noise=False, rng=None, eps=1e-12
):
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
    p = np.asarray(probs, dtype=np.float64).copy()
    p = np.clip(p, eps, None)
    p = p / p.sum()

    if topk is not None:
        idx = np.argsort(p)[-int(topk) :]
        masked = np.zeros_like(p)
        masked[idx] = p[idx]
        s = masked.sum()
        p = masked / (s if s > 0 else 1.0)

    if decimals is not None:
        p = np.round(p, int(decimals))
        s = p.sum()
        if s <= 0:
            # fallback: keep argmax only (nonzero)
            m = np.argmax(probs)
            p = np.zeros_like(p)
            p[m] = 1.0
        else:
            p = p / s

    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.dirichlet([0.1] * len(p))
        p = p + 0.01 * noise
        p = p / p.sum()

    return p
