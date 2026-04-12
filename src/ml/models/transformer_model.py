"""
Temporal Fusion Transformer (Simplified) — Multi-horizon quantile prediction.
==============================================================================

Best at: combining static features (fundamentals) with temporal features
(price sequences), multi-step forecasting with confidence intervals.

Based on: Lim et al. 2021 (Google Research) — simplified for laptop training.

Outputs 3 quantiles: [0.1, 0.5, 0.9]
  - 0.1 quantile = pessimistic → natural stop loss
  - 0.5 quantile = expected → base prediction
  - 0.9 quantile = optimistic → target price

Architecture:
  Static features → Static Enrichment via GRN
  Temporal features → LSTM Encoder → Self-Attention → Quantile Head
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

if HAS_TORCH:
    class GatedResidualNetwork(nn.Module):
        """
        Gated Residual Network (GRN) — core building block of TFT.
        Linear → ELU → Linear → GLU → LayerNorm → Residual.
        """

        def __init__(self, input_size: int, hidden_size: int,
                     output_size: int, dropout: float = 0.1,
                     context_size: Optional[int] = None):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size

            self.fc1 = nn.Linear(input_size, hidden_size)
            if context_size is not None:
                self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
            else:
                self.context_fc = None

            self.fc2 = nn.Linear(hidden_size, output_size * 2)  # *2 for GLU
            self.layer_norm = nn.LayerNorm(output_size)
            self.dropout = nn.Dropout(dropout)

            # Skip connection (project if dimensions differ)
            if input_size != output_size:
                self.skip = nn.Linear(input_size, output_size)
            else:
                self.skip = None

        def forward(self, x, context=None):
            residual = self.skip(x) if self.skip else x

            out = self.fc1(x)
            if self.context_fc is not None and context is not None:
                out = out + self.context_fc(context)

            out = F.elu(out)
            out = self.fc2(out)

            # GLU (Gated Linear Unit)
            out1, out2 = out.chunk(2, dim=-1)
            out = out1 * torch.sigmoid(out2)

            out = self.dropout(out)
            out = self.layer_norm(out + residual)
            return out

    class VariableSelectionNetwork(nn.Module):
        """
        Variable Selection Network — learns which features are most relevant.
        Produces soft attention weights over input features.
        """

        def __init__(self, n_features: int, hidden_size: int,
                     dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features

            # Individual GRN per feature
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
                for _ in range(n_features)
            ])

            # Softmax weight generator
            self.weight_grn = GatedResidualNetwork(
                n_features * hidden_size, hidden_size, n_features, dropout
            )

            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            # x: [batch, n_features] or [batch, seq, n_features]
            is_temporal = x.ndim == 3

            # Process each feature through its own GRN
            processed = []
            for i in range(self.n_features):
                if is_temporal:
                    feat = x[:, :, i:i + 1]  # [batch, seq, 1]
                    # Reshape for GRN: [batch*seq, 1]
                    b, s, _ = feat.shape
                    feat_flat = feat.reshape(b * s, 1)
                    out = self.feature_grns[i](feat_flat)
                    out = out.reshape(b, s, -1)
                else:
                    feat = x[:, i:i + 1]
                    out = self.feature_grns[i](feat)
                processed.append(out)

            # Stack and compute weights
            stacked = torch.stack(processed, dim=-2)  # [batch, (seq), n_feat, hidden]

            if is_temporal:
                b, s, nf, h = stacked.shape
                concat = stacked.reshape(b, s, nf * h)
                weights = self.softmax(self.weight_grn(concat))  # [batch, seq, n_feat]
                weights = weights.unsqueeze(-1)  # [batch, seq, n_feat, 1]
            else:
                concat = stacked.reshape(stacked.shape[0], -1)
                weights = self.softmax(self.weight_grn(concat))
                weights = weights.unsqueeze(-1)

            # Weighted sum
            selected = (stacked * weights).sum(dim=-2)  # [batch, (seq), hidden]

            return selected, weights.squeeze(-1)

    class TFTLite(nn.Module):
        """
        Simplified Temporal Fusion Transformer.
        """

        def __init__(self, n_static: int, n_temporal: int,
                     seq_len: int = 60, hidden_size: int = 64,
                     n_heads: int = 4, dropout: float = 0.1,
                     quantiles: tuple = (0.1, 0.5, 0.9)):
            super().__init__()
            self.quantiles = quantiles
            self.hidden_size = hidden_size

            # Variable Selection
            self.static_vsn = VariableSelectionNetwork(
                max(n_static, 1), hidden_size, dropout
            ) if n_static > 0 else None

            self.temporal_vsn = VariableSelectionNetwork(
                n_temporal, hidden_size, dropout
            )

            # Static enrichment
            self.static_enrichment = GatedResidualNetwork(
                hidden_size, hidden_size, hidden_size, dropout,
                context_size=hidden_size if n_static > 0 else None
            )

            # Temporal processing
            self.lstm = nn.LSTM(
                hidden_size, hidden_size,
                num_layers=1, batch_first=True,
                dropout=0
            )

            # Self-attention
            self.attention = nn.MultiheadAttention(
                hidden_size, n_heads, batch_first=True, dropout=dropout
            )
            self.attn_norm = nn.LayerNorm(hidden_size)

            # Quantile prediction heads
            self.quantile_heads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in quantiles
            ])

            self.dropout = nn.Dropout(dropout)

        def forward(self, x_static, x_temporal):
            # Variable selection on temporal
            temporal_selected, temporal_weights = self.temporal_vsn(x_temporal)

            # Static enrichment
            if self.static_vsn is not None and x_static is not None:
                static_selected, static_weights = self.static_vsn(x_static)
                # Broadcast static context to temporal dimension
                static_ctx = static_selected.unsqueeze(1).expand_as(temporal_selected)
                enriched = self.static_enrichment(temporal_selected, static_ctx)
            else:
                enriched = self.static_enrichment(temporal_selected)
                static_weights = None

            # LSTM
            lstm_out, _ = self.lstm(enriched)

            # Self-attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            out = self.attn_norm(lstm_out + self.dropout(attn_out))

            # Last timestep
            final = out[:, -1, :]  # [batch, hidden]

            # Quantile outputs
            quantile_preds = []
            for head in self.quantile_heads:
                quantile_preds.append(head(final))

            output = torch.cat(quantile_preds, dim=-1)  # [batch, n_quantiles]

            return output, {
                'temporal_weights': temporal_weights,
                'static_weights': static_weights,
                'attention_weights': attn_weights,
            }


class TemporalFusionTransformer:
    """
    Simplified TFT for multi-horizon quantile prediction.

    Parameters
    ----------
    n_static_features : int
        Number of static (fundamental) features.
    n_temporal_features : int
        Number of temporal (technical) features.
    seq_len : int
        Sequence length for temporal input.
    hidden_size : int
        Hidden dimension throughout the model.
    n_heads : int
        Number of attention heads.
    model_path : str
        Directory for saving/loading.
    """

    def __init__(self, n_static_features: int = 10,
                 n_temporal_features: int = 50,
                 seq_len: int = 60, hidden_size: int = 64,
                 n_heads: int = 4,
                 model_path: str = 'models/transformer/'):
        self.n_static = n_static_features
        self.n_temporal = n_temporal_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.quantiles = (0.1, 0.5, 0.9)

        os.makedirs(model_path, exist_ok=True)

    @staticmethod
    def _get_device():
        if not HAS_TORCH:
            return 'cpu'
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def build_model(self):
        """Build TFTLite model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")

        self.model = TFTLite(
            n_static=self.n_static,
            n_temporal=self.n_temporal,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            quantiles=self.quantiles
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"TFT model built: {n_params} params on {self.device}")

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor
                        ) -> torch.Tensor:
        """
        Pinball loss for quantile regression.
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, i]
            error = targets - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)

    def train(self, X_static: np.ndarray, X_temporal: np.ndarray,
              y: np.ndarray, epochs: int = 100, batch_size: int = 64,
              lr: float = 0.001) -> Dict[str, Any]:
        """
        Train TFT with quantile (pinball) loss.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")

        if self.model is None:
            self.build_model()

        # Convert to tensors
        X_s = torch.FloatTensor(X_static).to(self.device)
        X_t = torch.FloatTensor(X_temporal).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        # Create sequences for temporal data
        X_static_seq, X_temporal_seq, y_seq = [], [], []
        for i in range(self.seq_len, len(X_t)):
            X_static_seq.append(X_s[i])
            X_temporal_seq.append(X_t[i - self.seq_len:i])
            y_seq.append(y_t[i])

        if not X_static_seq:
            return {'val_loss': float('inf')}

        X_static_seq = torch.stack(X_static_seq)
        X_temporal_seq = torch.stack(X_temporal_seq)
        y_seq = torch.stack(y_seq)

        # Split: 80% train, 20% val
        n = len(y_seq)
        split = int(n * 0.8)

        train_ds = TensorDataset(
            X_static_seq[:split], X_temporal_seq[:split], y_seq[:split]
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_static = X_static_seq[split:]
        val_temporal = X_temporal_seq[split:]
        val_y = y_seq[split:]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        patience, patience_counter = 15, 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for xs, xt, yt in train_loader:
                optimizer.zero_grad()
                pred, _ = self.model(xs, xt)
                loss = self._quantile_loss(pred, yt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred, _ = self.model(val_static, val_temporal)
                val_loss = self._quantile_loss(val_pred, val_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if (epoch + 1) % 20 == 0:
                logger.info(f"TFT Epoch {epoch + 1}: val_loss={val_loss:.4f}")

        return {'val_loss': best_val_loss, 'n_epochs': epoch + 1}

    def predict(self, X_static: np.ndarray, X_temporal: np.ndarray) -> Dict:
        """
        Predict quantiles for the latest data point.

        Returns
        -------
        dict: expected_move_pct, pessimistic_pct, optimistic_pct,
              confidence_interval, variable_importance
        """
        if self.model is None:
            return {
                'expected_move_pct': 0.0,
                'pessimistic_pct': -1.0,
                'optimistic_pct': 1.0,
                'confidence_interval': [-1.0, 1.0],
                'variable_importance': {},
            }

        self.model.eval()

        X_s = torch.FloatTensor(X_static).to(self.device)
        X_t = torch.FloatTensor(X_temporal).to(self.device)

        if X_s.ndim == 1:
            X_s = X_s.unsqueeze(0)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(0)

        with torch.no_grad():
            pred, weights = self.model(X_s, X_t)

        pred = pred.cpu().numpy()[0]

        # Extract variable importance from VSN weights
        var_importance = {}
        if weights.get('temporal_weights') is not None:
            tw = weights['temporal_weights'].cpu().numpy()
            if tw.ndim >= 2:
                avg_weights = tw.mean(axis=tuple(range(tw.ndim - 1)))
                for i, w in enumerate(avg_weights[:20]):
                    var_importance[f'temporal_{i}'] = float(w)

        return {
            'expected_move_pct': round(float(pred[1]), 4),  # 0.5 quantile
            'pessimistic_pct': round(float(pred[0]), 4),    # 0.1 quantile
            'optimistic_pct': round(float(pred[2]), 4),     # 0.9 quantile
            'confidence_interval': [round(float(pred[0]), 4), round(float(pred[2]), 4)],
            'variable_importance': var_importance,
        }

    def save(self) -> str:
        if self.model is None or not HAS_TORCH:
            return ''
        filepath = os.path.join(self.model_path, 'tft_model.pth')
        torch.save(self.model.state_dict(), filepath)
        meta = {
            'n_static': self.n_static, 'n_temporal': self.n_temporal,
            'seq_len': self.seq_len, 'hidden_size': self.hidden_size,
            'n_heads': self.n_heads, 'saved_at': datetime.now().isoformat(),
        }
        with open(os.path.join(self.model_path, 'tft_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        return filepath

    def load(self) -> bool:
        if not HAS_TORCH:
            return False
        filepath = os.path.join(self.model_path, 'tft_model.pth')
        meta_path = os.path.join(self.model_path, 'tft_meta.json')
        if not os.path.exists(filepath):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self.n_static = meta['n_static']
            self.n_temporal = meta['n_temporal']
            self.seq_len = meta['seq_len']
            self.hidden_size = meta['hidden_size']
            self.n_heads = meta['n_heads']
            self.build_model()
            self.model.load_state_dict(
                torch.load(filepath, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            return True
        except Exception as e:
            logger.warning(f"Failed to load TFT: {e}")
            return False
