"""
Query-pattern detector for detecting model inversion attacks.

This module implements a defense mechanism that monitors client query patterns
and flags anomalous behavior that might indicate a model inversion attack.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from collections import deque
from typing import Deque, Dict, NamedTuple, Optional


class QueryResult(NamedTuple):
    """Return type for QueryPatternDetector.add_query.

    Supports both attribute access (res.is_anomaly) and tuple-unpacking
    (is_anomaly, features = res) for notebook/module-testing compatibility.
    """

    is_anomaly: Optional[bool]
    features: Optional[np.ndarray]


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
        p = arr[i - 1]
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
        "top1_mean": float(top1_prob.mean()),
        "top1_std": float(top1_prob.std()),
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "kl_mean": float(np.mean(kl)),
        "top1_stability": top1_stability,
    }

    # Return features in fixed order
    feature_order = [
        "top1_mean",
        "top1_std",
        "entropy_mean",
        "entropy_std",
        "kl_mean",
        "top1_stability",
    ]
    return np.array([features[k] for k in feature_order])


def train_detector(
    window_features_array, contamination=0.01, random_state=42, n_estimators=100
):
    """
    Train an IsolationForest detector on benign user behavior.

    Args:
        window_features_array: (N_windows, n_features) array from benign users
        contamination: Expected proportion of outliers (default: 0.01)
        random_state: Random seed for reproducibility

    Returns:
        sklearn Pipeline: StandardScaler + IsolationForest pipeline
    """
    detector = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    pipe = make_pipeline(StandardScaler(), detector)
    pipe.fit(window_features_array)
    return pipe


def calibrate_threshold(detector_model, benign_features, target_fpr=0.01):
    """Calibrate an anomaly threshold on benign data to target a desired false positive rate.

    We assume the detector exposes `decision_function(X)` where *lower* scores indicate
    more anomalous behavior (as in scikit-learn's IsolationForest and pipelines wrapping it).

    The returned threshold is the score at the `target_fpr` quantile of benign scores, so that
    approximately `target_fpr` fraction of benign windows fall below the threshold.

    Args:
        detector_model: Fitted detector or pipeline with `decision_function(X)` (preferred),
            or `score_samples(X)` as a fallback.
        benign_features: (N, n_features) array of benign window features.
        target_fpr: Desired false positive rate on benign data (e.g., 0.01 for ~1%).

    Returns:
        float: Threshold score; classify as anomaly if score < threshold.
    """
    X = np.asarray(benign_features)

    if hasattr(detector_model, "decision_function"):
        scores = detector_model.decision_function(X)
    elif hasattr(detector_model, "score_samples"):
        scores = detector_model.score_samples(X)
    else:
        raise TypeError(
            "detector_model must implement decision_function(X) or score_samples(X)."
        )

    scores = np.asarray(scores).reshape(-1)
    if scores.size == 0:
        raise ValueError("benign_features must contain at least one row to calibrate.")

    target_fpr = float(target_fpr)
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be between 0 and 1 (exclusive).")

    return float(np.quantile(scores, target_fpr))


class QueryPatternDetector:
    """
    Real-time query pattern detector for monitoring client behavior.

    Maintains a sliding window of recent queries for each client and
    flags anomalous patterns that might indicate an attack.
    """

    def __init__(
        self,
        window_size=50,
        detector=None,
        threshold: Optional[float] = None,
        persistence_k: int = 1,
        persistence_m: int = 1,
    ):
        """
        Initialize the detector.

        Args:
            window_size: Size of sliding window for feature extraction
            detector: Pre-trained IsolationForest detector (or None to train later)
            threshold: Optional decision threshold for anomaly scores. If provided, we
                compute a score using `detector.decision_function([features])` and flag
                anomaly when score < threshold. If None, we fall back to the detector's
                `predict` output (-1 indicates anomaly).
            persistence_k: Require at least k anomalies within the last m checks to
                flag a client (helps reduce single-window false positives).
            persistence_m: Window length m for persistence logic. If <= 1, persistence
                logic is disabled and we use the current window decision directly.
        """
        self.window_size = window_size
        self.detector = detector
        self.threshold = threshold

        self.persistence_k = int(persistence_k)
        self.persistence_m = int(persistence_m)
        if self.persistence_k < 1:
            raise ValueError("persistence_k must be >= 1")
        if self.persistence_m < 1:
            raise ValueError("persistence_m must be >= 1")
        if self.persistence_k > self.persistence_m:
            raise ValueError("persistence_k cannot be greater than persistence_m")

        self.client_windows = {}  # client_id -> deque of recent probs
        self.client_flags: Dict[str, Deque[bool]] = {}  # client_id -> deque of recent anomaly flags

    def add_query(self, client_id, prob_vector):
        """
        Add a new query from a client.

        Args:
            client_id: Unique identifier for the client
            prob_vector: Softmax probability vector (1D array)

        Returns:
            QueryResult: (is_anomaly, features)
                - is_anomaly: None until window is full, else bool if detector exists
                - features: None until window is full, else np.ndarray of shape (6,)
        """
        if client_id not in self.client_windows:
            self.client_windows[client_id] = deque(maxlen=self.window_size)
            self.client_flags[client_id] = deque(maxlen=self.persistence_m)

        self.client_windows[client_id].append(prob_vector)

        # Only check if window is full
        if len(self.client_windows[client_id]) >= self.window_size:
            window = list(self.client_windows[client_id])
            features = extract_window_features(window)

            if self.detector is not None:
                # Decide anomaly using either calibrated threshold or detector's predict()
                if self.threshold is not None and hasattr(self.detector, "decision_function"):
                    score = float(self.detector.decision_function([features])[0])
                    current_anomaly = score < float(self.threshold)
                else:
                    prediction = self.detector.predict([features])[0]
                    current_anomaly = prediction == -1

                # Persistence logic: require k anomalies within last m checks
                if self.persistence_m > 1:
                    self.client_flags[client_id].append(bool(current_anomaly))
                    is_anomaly = sum(self.client_flags[client_id]) >= self.persistence_k
                else:
                    is_anomaly = bool(current_anomaly)

                return QueryResult(is_anomaly=is_anomaly, features=features)
            else:
                return QueryResult(is_anomaly=None, features=features)

        return QueryResult(is_anomaly=None, features=None)
