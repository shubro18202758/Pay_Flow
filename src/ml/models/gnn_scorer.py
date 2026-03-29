"""
PayFlow — GNN Topological Risk Scorer
========================================
Lightweight Graph Attention Network for subgraph-level fraud risk scoring.
Operates on pre-extracted k-hop neighborhoods from TransactionGraph.

Architecture:
    TransactionGraph._extract_k_hop_subgraph()
         │
    nx.MultiDiGraph (5–50 nodes)
         │
    GNNScorer.score_subgraph(subgraph, center_nodes, timestamp)
         │
    GNNScoringResult(risk_score=0.87, inference_ms=2.3)

Model — FraudGAT (~278 K parameters, ~1.1 MB):
    3-layer GATConv with aggressive dimensionality reduction
    (512 → 128 → 32) and attention-based graph readout.

    Node features (7-dim): log_txn_count, activity_span, recency,
        log_in_degree, log_out_degree, degree_ratio, is_center
    Edge features (3-dim): log_amount, channel_norm, time_delta

GPU Strategy:
    - Primary: PyTorch CUDA on RTX 4070 (< 5 MB VRAM at inference)
    - Fallback: CPU inference (subgraphs are tiny, CPU is fine)
    - VRAM managed externally via ``with analysis_mode():``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature Dimensions ────────────────────────────────────────────────────────

NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 3


# ── Result Types ──────────────────────────────────────────────────────────────

class GNNScoringResult(NamedTuple):
    """Result of scoring a single subgraph."""
    risk_score: float          # [0.0, 1.0] topological fraud probability
    node_count: int
    edge_count: int
    inference_ms: float


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class GNNTrainingMetrics:
    """Training run diagnostics."""
    n_train_graphs: int = 0
    n_eval_graphs: int = 0
    epochs_completed: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    device_used: str = "unknown"
    training_ms: float = 0.0
    fell_back_to_cpu: bool = False
    total_parameters: int = 0

    def summary(self) -> dict:
        return {
            "n_train_graphs": self.n_train_graphs,
            "n_eval_graphs": self.n_eval_graphs,
            "epochs": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 6),
            "device": self.device_used,
            "training_ms": round(self.training_ms, 1),
            "cpu_fallback": self.fell_back_to_cpu,
            "parameters": self.total_parameters,
        }


@dataclass
class GNNValidationMetrics:
    """Hold-out evaluation metrics."""
    aucpr: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    eval_samples: int = 0

    def summary(self) -> dict:
        return {
            "aucpr": round(self.aucpr, 4),
            "auc_roc": round(self.auc_roc, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "eval_samples": self.eval_samples,
        }


# ── Model Factory (lazy-loaded) ──────────────────────────────────────────────

def _build_fraud_gat(cfg):
    """
    Construct the FraudGAT model.  Lazy-imports torch and torch_geometric.

    Architecture:
        GATConv(7->128*4=512) -> GATConv(512->128) -> GATConv(128->32)
        -> AttentionalAggregation -> MLP(32->16->1) -> Sigmoid
    """
    import torch.nn as nn
    from torch_geometric.nn import BatchNorm, GATConv
    from torch_geometric.nn.aggr import AttentionalAggregation

    out_channels = cfg.hidden_channels // 4    # 32

    class FraudGAT(nn.Module):
        """3-layer GAT with aggressive dimensionality reduction."""

        def __init__(self) -> None:
            super().__init__()
            # Layer 1: expand 7 → 128×4 = 512  (concat heads)
            self.conv1 = GATConv(
                NODE_FEATURE_DIM, cfg.hidden_channels,
                heads=cfg.heads, concat=True,
                dropout=cfg.dropout, edge_dim=EDGE_FEATURE_DIM,
            )
            self.bn1 = BatchNorm(cfg.hidden_channels * cfg.heads)

            # Layer 2: reduce 512 → 128  (average heads)
            self.conv2 = GATConv(
                cfg.hidden_channels * cfg.heads, cfg.hidden_channels,
                heads=cfg.heads, concat=False,
                dropout=cfg.dropout, edge_dim=EDGE_FEATURE_DIM,
            )
            self.bn2 = BatchNorm(cfg.hidden_channels)

            # Layer 3: compress 128 → 32  (single head)
            self.conv3 = GATConv(
                cfg.hidden_channels, out_channels,
                heads=1, concat=False,
                dropout=cfg.dropout, edge_dim=EDGE_FEATURE_DIM,
            )
            self.bn3 = BatchNorm(out_channels)

            # Graph-level readout via attention pooling
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(out_channels, 1))

            # Classification head
            self.head = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),   # 32 → 16
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(out_channels // 2, 1),              # 16 → 1
                nn.Sigmoid(),
            )

            self.act = nn.ELU()
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            # Layer 1
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = self.bn1(x)
            x = self.act(x)
            x = self.dropout(x)
            # Layer 2
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            x = self.bn2(x)
            x = self.act(x)
            x = self.dropout(x)
            # Layer 3
            x = self.conv3(x, edge_index, edge_attr=edge_attr)
            x = self.bn3(x)
            x = self.act(x)
            # Graph readout → (num_graphs, 32)
            x = self.pool(x, batch)
            # Classify → (num_graphs,)
            return self.head(x).squeeze(-1)

    return FraudGAT()


# ── nx → PyG Conversion ──────────────────────────────────────────────────────

def nx_to_pyg_data(subgraph, center_nodes, reference_timestamp):
    """
    Convert an ``nx.MultiDiGraph`` subgraph to a ``torch_geometric.data.Data``.

    Node features (7-dim):
        0  log1p(txn_count)
        1  activity_span_hours  (capped 168)
        2  recency_hours        (capped 168)
        3  log1p(in_degree)
        4  log1p(out_degree)
        5  degree_ratio  = in / (in + out + 1)
        6  is_center     (binary)

    Edge features (3-dim):
        0  log1p(amount_paisa / 100)   — log INR
        1  channel / 9.0               — normalised channel code
        2  time_delta_hours             (capped 168)

    Graph label (y):
        1.0 if any edge has fraud_label > 0, else 0.0.
    """
    import torch
    from torch_geometric.data import Data

    nodes = list(subgraph.nodes())
    node_map = {nid: i for i, nid in enumerate(nodes)}
    center_set = set(center_nodes)
    n = len(nodes)

    # ── node features (N, 7) ──────────────────────────────────────────────
    x = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    for i, nid in enumerate(nodes):
        nd = subgraph.nodes[nid]
        first_seen = nd.get("first_seen", reference_timestamp)
        last_seen = nd.get("last_seen", reference_timestamp)
        txn_count = nd.get("txn_count", 0)

        x[i, 0] = np.log1p(txn_count)
        x[i, 1] = min((last_seen - first_seen) / 3600.0, 168.0)
        x[i, 2] = min(max((reference_timestamp - last_seen) / 3600.0, 0.0), 168.0)

        in_deg = subgraph.in_degree(nid)
        out_deg = subgraph.out_degree(nid)
        x[i, 3] = np.log1p(in_deg)
        x[i, 4] = np.log1p(out_deg)
        x[i, 5] = in_deg / (in_deg + out_deg + 1)
        x[i, 6] = 1.0 if nid in center_set else 0.0

    # ── edge features ─────────────────────────────────────────────────────
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feats: list[list[float]] = []
    max_fraud_label = 0

    for u, v, _key, data in subgraph.edges(keys=True, data=True):
        src_list.append(node_map[u])
        dst_list.append(node_map[v])

        amt = data.get("amount_paisa", 0)
        ch = data.get("channel", 0)
        ts = data.get("timestamp", reference_timestamp)
        fl = data.get("fraud_label", 0)

        edge_feats.append([
            np.log1p(amt / 100.0),
            ch / 9.0,
            min(max((reference_timestamp - ts) / 3600.0, 0.0), 168.0),
        ])
        if fl > max_fraud_label:
            max_fraud_label = fl

    e = len(src_list)
    if e == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    y = torch.tensor([1.0 if max_fraud_label > 0 else 0.0], dtype=torch.float32)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


# ── GNN Scorer ────────────────────────────────────────────────────────────────

class GNNScorer:
    """
    Lightweight GNN-based topological risk scorer for transaction subgraphs.

    Follows the ``FraudClassifier`` convention:
    - Config injection via ``__init__(cfg=None)`` defaulting to ``GNN_CFG``
    - Lazy imports of torch / torch_geometric
    - GPU → CPU fallback
    - ``save()`` / ``load()`` persistence
    - VRAM managed externally by the caller

    Usage::

        from config import GNN_CFG, analysis_mode

        scorer = GNNScorer(GNN_CFG)
        scorer.load("artifacts/models/fraud_gat.pt")

        with analysis_mode():
            result = scorer.score_subgraph(subgraph, ["ACC1", "ACC2"], ts)
            # result.risk_score → 0.87
    """

    def __init__(self, cfg=None) -> None:
        from config.settings import GNN_CFG
        self._cfg = cfg or GNN_CFG
        self._model = None
        self._device: str = "cpu"
        self._is_loaded: bool = False
        self._training_metrics: GNNTrainingMetrics | None = None

    # ── Inference ──────────────────────────────────────────────────────────

    def score_subgraph(
        self,
        subgraph,
        center_nodes: list[str],
        reference_timestamp: int,
    ) -> GNNScoringResult:
        """
        Score a single subgraph for topological fraud risk.

        Args:
            subgraph: pre-extracted k-hop neighbourhood (nx.MultiDiGraph)
            center_nodes: [sender_id, receiver_id] of the flagged transaction
            reference_timestamp: Unix epoch of the flagged transaction

        Returns:
            GNNScoringResult with ``risk_score`` in [0.0, 1.0].
        """
        import torch

        if not self._is_loaded:
            raise RuntimeError("GNN model not loaded. Call load() or train() first.")

        t0 = time.perf_counter()
        data = nx_to_pyg_data(subgraph, center_nodes, reference_timestamp)

        # Empty graph → zero risk
        if data.x.shape[0] == 0:
            return GNNScoringResult(
                risk_score=0.0, node_count=0, edge_count=0,
                inference_ms=(time.perf_counter() - t0) * 1000,
            )

        data = data.to(self._device)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self._device)

        self._model.eval()
        with torch.no_grad():
            try:
                score = self._model(
                    data.x, data.edge_index,
                    edge_attr=data.edge_attr, batch=batch,
                )
                risk = float(score.cpu().item())
            except RuntimeError:
                risk = self._cpu_fallback_score(data)

        elapsed = (time.perf_counter() - t0) * 1000
        return GNNScoringResult(
            risk_score=risk,
            node_count=int(data.x.shape[0]),
            edge_count=int(data.edge_index.shape[1]),
            inference_ms=elapsed,
        )

    def _cpu_fallback_score(self, data) -> float:
        """Move model + data to CPU and retry inference."""
        import torch

        logger.warning("GNN GPU inference failed — falling back to CPU.")
        cpu_data = data.to("cpu")
        self._model.to("cpu")
        batch = torch.zeros(cpu_data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            score = self._model(
                cpu_data.x, cpu_data.edge_index,
                edge_attr=cpu_data.edge_attr, batch=batch,
            )
        self._model.to(self._device)
        return float(score.cpu().item())

    def score_subgraph_cpu(
        self,
        subgraph,
        center_nodes: list[str],
        reference_timestamp: int,
    ) -> GNNScoringResult:
        """
        Score a subgraph on CPU regardless of device setting.

        Used by the GPU concurrency manager to offload GNN inference to
        system RAM when VRAM pressure is critical, preserving real risk
        scores instead of returning sentinel values.
        """
        import torch

        if not self._is_loaded:
            raise RuntimeError("GNN model not loaded. Call load() or train() first.")

        t0 = time.perf_counter()
        data = nx_to_pyg_data(subgraph, center_nodes, reference_timestamp)

        if data.x.shape[0] == 0:
            return GNNScoringResult(
                risk_score=0.0, node_count=0, edge_count=0,
                inference_ms=(time.perf_counter() - t0) * 1000,
            )

        cpu_data = data.to("cpu")
        self._model.to("cpu")
        batch = torch.zeros(cpu_data.x.shape[0], dtype=torch.long)
        with torch.no_grad():
            score = self._model(
                cpu_data.x, cpu_data.edge_index,
                edge_attr=cpu_data.edge_attr, batch=batch,
            )
        risk = float(score.cpu().item())
        self._model.to(self._device)

        elapsed = (time.perf_counter() - t0) * 1000
        return GNNScoringResult(
            risk_score=risk,
            node_count=int(cpu_data.x.shape[0]),
            edge_count=int(cpu_data.edge_index.shape[1]),
            inference_ms=elapsed,
        )

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        train_graphs: list,
        eval_graphs: list | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> GNNTrainingMetrics:
        """
        Train the FraudGAT on a list of PyG ``Data`` objects (one per subgraph).

        Each ``Data`` must carry a graph-level label ``y`` (0 or 1).
        Uses PyG ``DataLoader`` for mini-batching and early stopping.
        """
        import torch
        from torch_geometric.loader import DataLoader

        self._model = _build_fraud_gat(self._cfg)
        self._device = self._resolve_device()
        self._model.to(self._device)

        metrics = GNNTrainingMetrics(
            n_train_graphs=len(train_graphs),
            n_eval_graphs=len(eval_graphs) if eval_graphs else 0,
            device_used=self._device,
            total_parameters=sum(p.numel() for p in self._model.parameters()),
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        train_loader = DataLoader(
            train_graphs, batch_size=self._cfg.batch_size, shuffle=True,
        )
        eval_loader = None
        if eval_graphs:
            eval_loader = DataLoader(
                eval_graphs, batch_size=self._cfg.batch_size,
            )

        t0 = time.perf_counter()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self._model.train()
            for batch_data in train_loader:
                batch_data = batch_data.to(self._device)
                optimizer.zero_grad()
                out = self._model(
                    batch_data.x, batch_data.edge_index,
                    edge_attr=batch_data.edge_attr, batch=batch_data.batch,
                )
                loss = criterion(out, batch_data.y.squeeze())
                loss.backward()
                optimizer.step()

            if eval_loader is not None:
                val_loss = self._evaluate_loss(eval_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    metrics.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            metrics.epochs_completed = epoch + 1

        metrics.best_val_loss = best_val_loss
        metrics.training_ms = (time.perf_counter() - t0) * 1000
        self._is_loaded = True
        self._training_metrics = metrics
        logger.info("GNN training complete: %s", metrics.summary())
        return metrics

    def validate(self, eval_graphs: list) -> GNNValidationMetrics:
        """Evaluate on a hold-out set, returning classification metrics."""
        import torch
        from torch_geometric.loader import DataLoader

        if not self._is_loaded:
            raise RuntimeError("GNN model not loaded.")

        loader = DataLoader(eval_graphs, batch_size=self._cfg.batch_size)
        all_scores: list[float] = []
        all_labels: list[float] = []

        self._model.eval()
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(self._device)
                out = self._model(
                    batch_data.x, batch_data.edge_index,
                    edge_attr=batch_data.edge_attr, batch=batch_data.batch,
                )
                all_scores.extend(out.cpu().tolist())
                all_labels.extend(batch_data.y.squeeze().cpu().tolist())

        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = np.array(all_labels)
        y_score = np.array(all_scores)
        y_pred = (y_score >= 0.5).astype(int)

        metrics = GNNValidationMetrics(eval_samples=len(y_true))
        if len(np.unique(y_true)) > 1:
            metrics.aucpr = float(average_precision_score(y_true, y_score))
            metrics.auc_roc = float(roc_auc_score(y_true, y_score))
        metrics.f1 = float(f1_score(y_true, y_pred, zero_division=0))
        metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
        return metrics

    def _evaluate_loss(self, loader, criterion) -> float:
        """Mean BCE loss over a DataLoader (for early stopping)."""
        import torch

        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(self._device)
                out = self._model(
                    batch_data.x, batch_data.edge_index,
                    edge_attr=batch_data.edge_attr, batch=batch_data.batch,
                )
                total += criterion(out, batch_data.y.squeeze()).item()
                count += 1
        return total / max(count, 1)

    def _resolve_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained model weights."""
        import torch

        if not self._is_loaded:
            raise RuntimeError("No trained GNN model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))
        logger.info("GNN model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load previously saved model weights."""
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GNN model file not found: {path}")
        self._model = _build_fraud_gat(self._cfg)
        self._device = self._resolve_device()
        state = torch.load(str(path), map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()
        self._is_loaded = True
        logger.info("GNN model loaded from %s (device=%s)", path, self._device)

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def training_info(self) -> dict | None:
        return self._training_metrics.summary() if self._training_metrics else None
