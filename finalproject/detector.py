"""
Query-pattern detector for detecting model inversion attacks.

Industry-grade version:
- Uses Jensen–Shannon divergence (stable) and entropy statistics
- Scales features (StandardScaler) + IsolationForest in a pipeline
- Supports threshold calibration to target false-positive rate
- Provides consistent scoring and optional persistence logic
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Iterable

from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EPS = 1e-12

def _safe_normalize(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Clamp + normalize to sum to 1."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Probability vector sum must be finite and > 0.")
    return p / s

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = _EPS) -> float:
    """
    Jensen–Shannon divergence between two probability vectors.
    Stable, symmetric, bounded.
    """
    p = _safe_normalize(p, eps)
    q = _safe_normalize(q, eps)
    m = 0.5 * (p + q)
    # Use log safely; values are clipped > 0
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def extract_window_features(probs_window: Iterable[np.ndarray]) -> np.ndarray:
    """
    Extract features from a window of softmax probability vectors.

    Features extracted (6):
    - mean top1 probability
    - std(top1 probability)
    - mean entropy
    - entropy std
    - mean JS divergence between successive outputs
    - top1 stability (fraction of consecutive queries with same top1 class)
    """
    if probs_window is None:
        return np.zeros(6, dtype=np.float32)

    window_list = list(probs_window)
    if len(window_list) == 0:
        return np.zeros(6, dtype=np.float32)

    arr = np.stack(window_list).astype(np.float64)  # W x C

    # Defensive normalization per row
    arr = np.clip(arr, _EPS, None)
    arr = arr / arr.sum(axis=1, keepdims=True)

    top1 = arr.argmax(axis=1)
    top1_prob = arr.max(axis=1)

    # Entropy for each output (vectorized)
    # scipy.stats.entropy expects probabilities; add eps already ensured
    ent = np.array([entropy(row) for row in arr], dtype=np.float64)

    # JS divergence between successive outputs
    if arr.shape[0] > 1:
        js_vals = [js_divergence(arr[i - 1], arr[i]) for i in range(1, arr.shape[0])]
        js_mean = float(np.mean(js_vals))
        top1_stability = float(np.mean(top1[:-1] == top1[1:]))
    else:
        js_mean = 0.0
        top1_stability = 1.0

    feat = np.array([
        float(top1_prob.mean()),
        float(top1_prob.std()),
        float(ent.mean()),
        float(ent.std()),
        js_mean,
        top1_stability,
    ], dtype=np.float32)

    # Replace non-finite with zeros (hard safety)
    if not np.isfinite(feat).all():
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return feat

def build_detector(
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Any:
    """
    Build an industry-safe detector pipeline:
      StandardScaler -> IsolationForest
    """
    return make_pipeline(
        StandardScaler(),
        IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1,
        )
    )

def train_detector(
    window_features_array: np.ndarray,
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Any:
    """
    Train the detector on benign window features.
    Returns a fitted pipeline: StandardScaler + IsolationForest
    """
    X = np.asarray(window_features_array, dtype=np.float32)
    model = build_detector(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    model.fit(X)
    return model

def calibrate_threshold(
    detector: Any,
    benign_features: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """
    Calibrate a decision threshold using benign validation data.

    IsolationForest decision_function:
      higher score => more normal
      lower score  => more anomalous

    We choose threshold such that approx target_fpr of benign windows fall below it.
    """
    if not (0.0 < target_fpr < 0.5):
        raise ValueError("target_fpr must be in (0, 0.5) for sensible calibration.")

    X = np.asarray(benign_features, dtype=np.float32)
    scores = detector.decision_function(X)  # shape (N,)

    # threshold at the target_fpr quantile (lower tail)
    thr = float(np.quantile(scores, target_fpr))
    return thr



@dataclass
class DetectionResult:
    is_anomaly: Optional[bool]
    features: Optional[np.ndarray]
    score: Optional[float]
    threshold: Optional[float]
    client_id: Any
    window_full: bool


class QueryPatternDetector:
    """
    Real-time query pattern detector.

    Maintains a sliding window of recent prob vectors per client.
    Computes window features and applies anomaly detector.

    Optional: anomaly persistence (require K anomalies in last M windows).
    """

    def __init__(
        self,
        window_size: int = 50,
        detector: Optional[Any] = None,
        threshold: Optional[float] = None,
        *,
        persistence_k: int = 1,
        persistence_m: int = 1,
    ):
        """
        Args:
            window_size: sliding window length
            detector: fitted sklearn pipeline (recommended) or IsolationForest-like
            threshold: manual threshold on decision_function score (lower => anomaly)
            persistence_k: flag anomaly only if >=K of last M windows are anomalous
            persistence_m: number of recent windows to consider for persistence
        """
        self.window_size = int(window_size)
        if self.window_size <= 1:
            raise ValueError("window_size must be >= 2")

        self.detector = detector
        self.threshold = threshold

        self.client_windows: Dict[Any, deque] = {}
        self.client_flags: Dict[Any, deque] = {}  # store recent anomaly bools

        self.persistence_k = int(persistence_k)
        self.persistence_m = int(persistence_m)
        if self.persistence_k < 1 or self.persistence_m < 1 or self.persistence_k > self.persistence_m:
            raise ValueError("Require 1 <= persistence_k <= persistence_m")

    @staticmethod
    def _validate_prob_vector(prob_vector: np.ndarray) -> np.ndarray:
        """Validation + normalization (safe for production)."""
        p = np.asarray(prob_vector, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError("prob_vector must be 1D")
        if p.size < 2:
            raise ValueError("prob_vector must have at least 2 classes")
        if not np.isfinite(p).all():
            raise ValueError("prob_vector contains NaN/Inf")
        return _safe_normalize(p, _EPS).astype(np.float32)

    def _record_flag(self, client_id: Any, flag: bool) -> bool:
        """Update persistence history and return persisted anomaly decision."""
        if client_id not in self.client_flags:
            self.client_flags[client_id] = deque(maxlen=self.persistence_m)
        self.client_flags[client_id].append(bool(flag))
        # persisted anomaly if >=K in last M
        return (sum(self.client_flags[client_id]) >= self.persistence_k)

    def add_query(self, client_id: Any, prob_vector: np.ndarray) -> DetectionResult:
        """
        Add a new query from a client.
        Returns DetectionResult with:
          - window_full indicates if enough data exists
          - score is detector.decision_function (higher=normal)
        """
        p = self._validate_prob_vector(prob_vector)

        if client_id not in self.client_windows:
            self.client_windows[client_id] = deque(maxlen=self.window_size)
        self.client_windows[client_id].append(p)

        if len(self.client_windows[client_id]) < self.window_size:
            return DetectionResult(
                is_anomaly=None, features=None, score=None, threshold=self.threshold,
                client_id=client_id, window_full=False
            )

        window = list(self.client_windows[client_id])
        features = extract_window_features(window)

        if self.detector is None:
            return DetectionResult(
                is_anomaly=None, features=features, score=None, threshold=self.threshold,
                client_id=client_id, window_full=True
            )

        score = float(self.detector.decision_function([features])[0])

        # Decide anomaly
        if self.threshold is not None:
            raw_flag = (score < self.threshold)
        else:
            pred = int(self.detector.predict([features])[0])  # -1 anomaly, +1 normal
            raw_flag = (pred == -1)

        is_anomaly = self._record_flag(client_id, raw_flag)
        return DetectionResult(
            is_anomaly=bool(is_anomaly),
            features=features,
            score=score,
            threshold=self.threshold,
            client_id=client_id,
            window_full=True
        )

    def get_client_features(self, client_id: Any) -> Optional[np.ndarray]:
        """Get current window features for a client (if window is full)."""
        if client_id not in self.client_windows:
            return None
        window = list(self.client_windows[client_id])
        if len(window) < self.window_size:
            return None
        return extract_window_features(window)

    def reset_client(self, client_id: Any) -> None:
        """Reset query history + flags for a specific client."""
        self.client_windows.pop(client_id, None)
        self.client_flags.pop(client_id, None)

    def reset_all(self) -> None:
        """Reset all client query histories."""
        self.client_windows.clear()
        self.client_flags.clear()
