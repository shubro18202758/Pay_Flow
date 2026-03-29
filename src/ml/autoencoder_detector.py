"""
PayFlow — Autoencoder Anomaly Detector
=========================================
Reconstruction-based anomaly detection using a PyTorch autoencoder.
Transactions that reconstruct poorly (high MSE loss) are flagged as
anomalous — the autoencoder learns the manifold of *normal* transactions
and anomalies lie outside it.

Architecture:
    Encoder:  D → 64 → 32 → 16  (latent)
    Decoder: 16 → 32 → 64 → D

Training uses the full feature matrix where labels == 0 (legitimate
transactions only) so the autoencoder captures normal patterns.

Memory Budget:
    ~2 MB model parameters + ~50 MB training overhead.
    Runs on CPU by default (sub-millisecond inference per batch).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid torch overhead at module load
_torch = None
_nn = None


def _ensure_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


# ── Results ─────────────────────────────────────────────────────────────────

class ReconstructionScore(NamedTuple):
    """Score for a single transaction."""
    mse_loss: float            # raw reconstruction error
    is_anomaly: bool           # True if above threshold
    normalized_score: float    # ∈ [0.0, 1.0], higher = more anomalous


@dataclass
class AutoencoderMetrics:
    """Training and scoring counters."""
    fit_count: int = 0
    last_fit_samples: int = 0
    last_fit_epochs: int = 0
    last_fit_final_loss: float = 0.0
    last_fit_ms: float = 0.0
    anomaly_threshold: float = 0.0
    total_scored: int = 0
    total_anomalies: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def anomaly_rate(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.total_anomalies / self.total_scored

    def snapshot(self) -> dict:
        return {
            "fit_count": self.fit_count,
            "last_fit_samples": self.last_fit_samples,
            "last_fit_epochs": self.last_fit_epochs,
            "last_fit_final_loss": round(self.last_fit_final_loss, 6),
            "last_fit_ms": round(self.last_fit_ms, 2),
            "anomaly_threshold": round(self.anomaly_threshold, 6),
            "total_scored": self.total_scored,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "uptime_sec": round(time.monotonic() - self._start_time, 1),
        }


# ── Autoencoder Network ────────────────────────────────────────────────────

def _build_autoencoder(input_dim: int):
    """Build a symmetric encoder–decoder network."""
    _, nn = _ensure_torch()

    class TransactionAutoencoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, d),
            )

        def forward(self, x):
            latent = self.encoder(x)
            return self.decoder(latent)

    return TransactionAutoencoder(input_dim)


# ── Autoencoder Detector ───────────────────────────────────────────────────

class AutoencoderDetector:
    """
    Reconstruction-based anomaly detector.

    Parameters
    ----------
    epochs : int
        Training epochs (default 50).
    batch_size : int
        Mini-batch size for training (default 256).
    lr : float
        Learning rate (default 1e-3).
    threshold_percentile : float
        Reconstruction MSE percentile on training data to set the
        anomaly threshold (default 95.0 → top 5% = anomalous).
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        threshold_percentile: float = 95.0,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._threshold_percentile = threshold_percentile
        self._model = None
        self._threshold: float = 0.0
        self._fitted = False
        self._input_dim: int = 0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.metrics = AutoencoderMetrics()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Fit the autoencoder on *legitimate* transactions only.

        Parameters
        ----------
        features : np.ndarray
            Shape (N, D) float32 feature matrix.
        labels : np.ndarray, optional
            Shape (N,) labels. If provided, trains only on samples where
            label == 0 (legitimate). If None, trains on all data.
        """
        torch, nn = _ensure_torch()

        # Filter to legitimate transactions if labels available
        if labels is not None:
            mask = labels == 0
            train_data = features[mask]
        else:
            train_data = features

        if train_data.shape[0] < 20:
            logger.warning("AutoencoderDetector: too few samples (%d), skipping",
                           train_data.shape[0])
            return

        t0 = time.monotonic()
        self._input_dim = train_data.shape[1]

        # Standardize features (store mean/std for inference)
        self._mean = train_data.mean(axis=0)
        self._std = train_data.std(axis=0)
        self._std[self._std < 1e-8] = 1.0  # avoid division by zero
        normalized = (train_data - self._mean) / self._std

        # Build and train
        self._model = _build_autoencoder(self._input_dim)
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        tensor_data = torch.from_numpy(normalized.astype(np.float32))
        dataset = torch.utils.data.TensorDataset(tensor_data, tensor_data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True,
        )

        final_loss = 0.0
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstructed = self._model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        # Compute per-sample reconstruction error on training data
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor_data)
            per_sample_mse = torch.mean((recon - tensor_data) ** 2, dim=1).numpy()

        # Set threshold at configured percentile
        self._threshold = float(np.percentile(per_sample_mse, self._threshold_percentile))
        self._fitted = True

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.metrics.fit_count += 1
        self.metrics.last_fit_samples = train_data.shape[0]
        self.metrics.last_fit_epochs = self._epochs
        self.metrics.last_fit_final_loss = final_loss
        self.metrics.last_fit_ms = elapsed_ms
        self.metrics.anomaly_threshold = self._threshold

        logger.info(
            "AutoencoderDetector fitted on %d samples (D=%d), "
            "final_loss=%.6f, threshold=%.6f, %.1f ms",
            train_data.shape[0], self._input_dim, final_loss,
            self._threshold, elapsed_ms,
        )

    def score(self, features: np.ndarray) -> ReconstructionScore:
        """
        Score a single sample.

        Returns
        -------
        ReconstructionScore with MSE loss, is_anomaly flag, and
        normalized_score ∈ [0.0, 1.0] where 1.0 = highly anomalous.
        """
        if not self._fitted:
            return ReconstructionScore(mse_loss=0.0, is_anomaly=False, normalized_score=0.0)

        torch, _ = _ensure_torch()

        x = features.reshape(1, -1) if features.ndim == 1 else features[:1]
        normalized = (x - self._mean) / self._std
        tensor_x = torch.from_numpy(normalized.astype(np.float32))

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor_x)
            mse = float(torch.mean((recon - tensor_x) ** 2).item())

        is_anomaly = mse > self._threshold
        # Normalize: ratio of mse to threshold, clamped
        normalized_score = min(1.0, mse / self._threshold) if self._threshold > 0 else 0.0

        self.metrics.total_scored += 1
        if is_anomaly:
            self.metrics.total_anomalies += 1

        return ReconstructionScore(
            mse_loss=round(mse, 6),
            is_anomaly=is_anomaly,
            normalized_score=round(normalized_score, 6),
        )

    def score_batch(self, features: np.ndarray) -> list[ReconstructionScore]:
        """
        Score a batch of samples.

        Parameters
        ----------
        features : np.ndarray
            Shape (N, D) float32 matrix.

        Returns
        -------
        list[ReconstructionScore] — one per row.
        """
        if not self._fitted:
            return [ReconstructionScore(0.0, False, 0.0)] * features.shape[0]

        torch, _ = _ensure_torch()

        normalized = (features - self._mean) / self._std
        tensor_x = torch.from_numpy(normalized.astype(np.float32))

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor_x)
            per_sample_mse = torch.mean((recon - tensor_x) ** 2, dim=1).numpy()

        results: list[ReconstructionScore] = []
        for mse_val in per_sample_mse:
            mse_f = float(mse_val)
            is_anomaly = mse_f > self._threshold
            norm_score = min(1.0, mse_f / self._threshold) if self._threshold > 0 else 0.0
            results.append(ReconstructionScore(
                mse_loss=round(mse_f, 6),
                is_anomaly=is_anomaly,
                normalized_score=round(norm_score, 6),
            ))
            self.metrics.total_scored += 1
            if is_anomaly:
                self.metrics.total_anomalies += 1

        return results

    def snapshot(self) -> dict:
        """State snapshot for dashboard / debugging."""
        return {
            "fitted": self._fitted,
            "config": {
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "lr": self._lr,
                "threshold_percentile": self._threshold_percentile,
                "input_dim": self._input_dim,
            },
            "metrics": self.metrics.snapshot(),
        }
