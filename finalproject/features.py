"""
Feature extraction utilities for face recognition models.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_features(encoder, dataset, device='cpu'):
    """
    Extract features from images using the trained encoder.
    
    Args:
        encoder: Trained encoder model
        dataset: Dataset to extract features from
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        If dataset has labels: (features, labels) as numpy arrays
        Otherwise: features as numpy array
    """
    encoder.eval()
    features = []
    labels = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            # Handle batch - DataLoader returns tuple (images, labels) when dataset has labels
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                xb, yb = batch
            else:
                xb = batch
                yb = None
            
            # Ensure xb is a tensor
            if not isinstance(xb, torch.Tensor):
                xb = torch.tensor(xb) if isinstance(xb, (list, np.ndarray)) else xb
            
            xb = xb.to(device)
            z = encoder(xb)  # Extract latent features
            features.append(z.cpu().numpy())
            
            if yb is not None:
                # Convert labels to numpy if needed
                if isinstance(yb, torch.Tensor):
                    labels.append(yb.cpu().numpy())
                else:
                    labels.append(np.array(yb))
    
    features = np.concatenate(features, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0)
        return features, labels
    return features

