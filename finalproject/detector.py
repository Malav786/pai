"""
Query-pattern detector for detecting model inversion attacks.

This module implements a defense mechanism that monitors client query patterns
and flags anomalous behavior that might indicate a model inversion attack.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy
from collections import deque


def extract_window_features(probs_window):
    """
    Extract features from a window of softmax probability vectors.
    
    Features extracted:
    - mean top1 probability
    - std(top1 probability)
    - mean entropy
    - entropy variance
    - mean KL divergence between successive outputs
    - fraction of times top1 class stays same (stability)
    
    Args:
        probs_window: list of softmax arrays of length W (W x C)
        
    Returns:
        np.array: Feature vector of length 6
    """
    if len(probs_window) == 0:
        return np.zeros(6)
    
    arr = np.stack(probs_window)  # W x C
    top1 = arr.argmax(axis=1)
    top1_prob = arr.max(axis=1)
    
    # Entropy for each output
    ent = np.apply_along_axis(lambda p: entropy(p + 1e-12), 1, arr)
    
    # KL divergence between successive outputs
    kl = []
    for i in range(1, arr.shape[0]):
        p = arr[i-1]
        q = arr[i]
        kl.append(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))
    
    if len(kl) == 0:
        kl = [0.0]
    
    # Top1 class stability (fraction of times top1 class stays same)
    if len(top1) > 1:
        top1_stability = float((top1[:-1] == top1[1:]).mean())
    else:
        top1_stability = 1.0
    
    features = {
        'top1_mean': float(top1_prob.mean()),
        'top1_std': float(top1_prob.std()),
        'entropy_mean': float(ent.mean()),
        'entropy_std': float(ent.std()),
        'kl_mean': float(np.mean(kl)),
        'top1_stability': top1_stability
    }
    
    # Return features in fixed order
    feature_order = ['top1_mean', 'top1_std', 'entropy_mean', 'entropy_std', 'kl_mean', 'top1_stability']
    return np.array([features[k] for k in feature_order])


def train_detector(window_features_array, contamination=0.01, random_state=42):
    """
    Train an IsolationForest detector on benign user behavior.
    
    Args:
        window_features_array: (N_windows, n_features) array from benign users
        contamination: Expected proportion of outliers (default: 0.01)
        random_state: Random seed for reproducibility
        
    Returns:
        IsolationForest: Trained detector
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(window_features_array)
    return clf


class QueryPatternDetector:
    """
    Real-time query pattern detector for monitoring client behavior.
    
    Maintains a sliding window of recent queries for each client and
    flags anomalous patterns that might indicate an attack.
    """
    
    def __init__(self, window_size=50, detector=None):
        """
        Initialize the detector.
        
        Args:
            window_size: Size of sliding window for feature extraction
            detector: Pre-trained IsolationForest detector (or None to train later)
        """
        self.window_size = window_size
        self.detector = detector
        self.client_windows = {}  # client_id -> deque of recent probs
        
    def add_query(self, client_id, prob_vector):
        """
        Add a new query from a client.
        
        Args:
            client_id: Unique identifier for the client
            prob_vector: Softmax probability vector (1D array)
            
        Returns:
            tuple: (is_anomaly, features) or (None, None) if window not full
        """
        if client_id not in self.client_windows:
            self.client_windows[client_id] = deque(maxlen=self.window_size)
        
        self.client_windows[client_id].append(prob_vector)
        
        # Only check if window is full
        if len(self.client_windows[client_id]) >= self.window_size:
            window = list(self.client_windows[client_id])
            features = extract_window_features(window)
            
            if self.detector is not None:
                prediction = self.detector.predict([features])[0]
                # Ensure prediction is a scalar and convert to Python bool
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item()
                is_anomaly = bool(prediction == -1)
                return is_anomaly, features
            else:
                return None, features
        
        return None, None
    
    def get_client_features(self, client_id):
        """
        Get current window features for a client (if window is full).
        
        Args:
            client_id: Client identifier
            
        Returns:
            np.array or None: Feature vector if window is full, None otherwise
        """
        if client_id not in self.client_windows:
            return None
        
        window = list(self.client_windows[client_id])
        if len(window) < self.window_size:
            return None
        
        return extract_window_features(window)
    
    def reset_client(self, client_id):
        """Reset the query history for a specific client."""
        if client_id in self.client_windows:
            del self.client_windows[client_id]
    
    def reset_all(self):
        """Reset all client query histories."""
        self.client_windows = {}

