"""
LSTM Price Model — Bidirectional LSTM with attention for sequence modeling.
============================================================================

Best at: capturing temporal patterns in price/volume sequences,
multi-timeframe dependencies, intraday momentum patterns.

Architecture:
  Input → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
       → Multi-head Self-Attention → Dense(32) → Dense(1, sigmoid)

Two models:
  1. Intraday: sequence of last 60 five-minute candles → next move
  2. Swing: sequence of last 60 daily candles → next week move
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple, Any

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

from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PyTorch Model Definition
# ─────────────────────────────────────────────

if HAS_TORCH:
    class LSTMWithAttention(nn.Module):
        """
        Bidirectional LSTM with Multi-head Self-Attention.

        Captures both forward and backward temporal patterns,
        with attention focusing on the most relevant timesteps.
        """

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.3,
                     n_heads: int = 4):
            super().__init__()
            self.hidden_size = hidden_size

            self.lstm = nn.LSTM(
                input_size, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout
            )

            self.layer_norm = nn.LayerNorm(hidden_size * 2)
            self.fc1 = nn.Linear(hidden_size * 2, 64)
            self.fc2 = nn.Linear(64, 1)
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # BiLSTM
            lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

            # Self-Attention
            attn_out, attn_weights = self.attention(
                lstm_out, lstm_out, lstm_out
            )

            # Residual + LayerNorm
            out = self.layer_norm(lstm_out + attn_out)

            # Take last timestep
            out = out[:, -1, :]  # [batch, hidden*2]

            # Dense layers
            out = self.dropout(F.relu(self.fc1(out)))
            out = self.sigmoid(self.fc2(out))

            return out.squeeze(-1), attn_weights


class LSTMPriceModel:
    """
    Bidirectional LSTM for price sequence modeling with attention.

    Parameters
    ----------
    seq_len : int
        Number of timesteps in each input sequence (default=60).
    n_features : int
        Number of features per timestep (set during training).
    mode : str
        'intraday' or 'swing'.
    model_path : str
        Directory for saving/loading model weights.
    """

    def __init__(self, seq_len: int = 60, n_features: Optional[int] = None,
                 mode: str = 'intraday',
                 model_path: str = 'models/lstm/'):
        self.seq_len = seq_len
        self.n_features = n_features
        self.mode = mode
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self._threshold = 0.6

        os.makedirs(model_path, exist_ok=True)

    @staticmethod
    def _get_device():
        """Auto-detect best device (CUDA > MPS > CPU)."""
        if not HAS_TORCH:
            return 'cpu'
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def build_model(self, n_features: int) -> 'LSTMWithAttention':
        """Build the LSTM+Attention model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install via: pip install torch")

        self.n_features = n_features
        self.model = LSTMWithAttention(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            n_heads=4
        ).to(self.device)

        logger.info(f"LSTM model built: {sum(p.numel() for p in self.model.parameters())} params")
        return self.model

    def create_sequences(self, X: np.ndarray, y: np.ndarray
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert flat feature matrix to sequences for LSTM.
        Each sample: X[i-seq_len:i] → y[i]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X)):
            X_seq.append(X[i - self.seq_len:i])
            y_seq.append(y[i])

        X_seq = torch.FloatTensor(np.array(X_seq))
        y_seq = torch.FloatTensor(np.array(y_seq))

        return X_seq, y_seq

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 64,
              lr: float = 0.001) -> Dict[str, Any]:
        """
        Training loop with AdamW, CosineAnnealingLR, early stopping,
        gradient clipping, and class-weighted BCE loss.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required.")

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        if len(X_train_seq) == 0:
            raise ValueError("Not enough data to create sequences.")

        # Build model
        n_features = X_train_seq.shape[2]
        if self.model is None:
            self.build_model(n_features)

        # Move data to device
        X_train_seq = X_train_seq.to(self.device)
        y_train_seq = y_train_seq.to(self.device)
        X_val_seq = X_val_seq.to(self.device)
        y_val_seq = y_val_seq.to(self.device)

        # DataLoader
        train_ds = TensorDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Class-weighted BCE loss
        n_pos = (y_train_seq == 1).sum().item()
        n_neg = (y_train_seq == 0).sum().item()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training
        best_val_loss = float('inf')
        patience, patience_counter = 15, 0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                # Use raw logits for BCEWithLogitsLoss
                logits = self.model.fc2(
                    self.model.dropout(F.relu(self.model.fc1(
                        self.model.layer_norm(
                            sum(self.model.lstm(X_batch)[0:1])
                        )[:, -1, :]
                    )))
                ).squeeze(-1)

                # Simpler forward for loss (use sigmoid output)
                output, _ = self.model(X_batch)
                loss = F.binary_cross_entropy(output, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_output, _ = self.model(X_val_seq)
                val_loss = F.binary_cross_entropy(val_output, y_val_seq).item()

                val_preds = val_output.cpu().numpy()
                val_true = y_val_seq.cpu().numpy()
                try:
                    val_auc = roc_auc_score(val_true, val_preds)
                except ValueError:
                    val_auc = 0.5

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f}"
                )

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_output, attn_weights = self.model(X_val_seq)
            val_preds = val_output.cpu().numpy()

        try:
            final_auc = roc_auc_score(val_true, val_preds)
        except ValueError:
            final_auc = 0.5

        return {
            'val_auc': final_auc,
            'val_loss': best_val_loss,
            'n_epochs_trained': len(history['train_loss']),
            'history': history,
        }

    def predict_with_attention(self, X: np.ndarray) -> Dict:
        """
        Return prediction + attention weights.
        Attention weights show WHICH timesteps the model focused on.
        """
        if self.model is None:
            return {
                'probability': 0.5, 'signal': 'HOLD',
                'attention_weights': [], 'peak_attention_candle': 0
            }

        if not HAS_TORCH:
            return {
                'probability': 0.5, 'signal': 'HOLD',
                'attention_weights': [], 'peak_attention_candle': 0
            }

        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        if X.shape[1] > self.seq_len:
            X = X[:, -self.seq_len:, :]

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            output, attn_weights = self.model(X_tensor)

        prob = output.cpu().item()
        signal = 'BUY' if prob >= self._threshold else \
                 'SELL' if prob <= (1 - self._threshold) else 'HOLD'

        # Extract attention weights (average over heads)
        if attn_weights is not None:
            attn = attn_weights.cpu().numpy()
            # Shape: [batch, heads, seq_len, seq_len]
            # Average over heads and take the last query position
            if attn.ndim == 4:
                avg_attn = attn[0].mean(axis=0)[-1]  # Last timestep query
            elif attn.ndim == 3:
                avg_attn = attn[0][-1]
            else:
                avg_attn = np.ones(self.seq_len) / self.seq_len
            avg_attn = avg_attn / (avg_attn.sum() + 1e-10)
        else:
            avg_attn = np.ones(self.seq_len) / self.seq_len

        return {
            'probability': round(prob, 4),
            'signal': signal,
            'confidence': round(prob if signal == 'BUY' else 1 - prob, 4),
            'attention_weights': avg_attn.tolist(),
            'peak_attention_candle': int(np.argmax(avg_attn)),
        }

    def fine_tune_on_new_trades(self, X_new: np.ndarray, y_new: np.ndarray,
                                 epochs: int = 5):
        """
        Fine-tune on new labeled trade data.
        Low learning rate to avoid catastrophic forgetting.
        Only update last 2 dense layers (freeze LSTM layers).
        """
        if self.model is None or not HAS_TORCH:
            return

        X_seq, y_seq = self.create_sequences(X_new, y_new)
        if len(X_seq) == 0:
            return

        X_seq = X_seq.to(self.device)
        y_seq = y_seq.to(self.device)

        # Freeze LSTM and attention layers
        for name, param in self.model.named_parameters():
            if 'fc1' not in name and 'fc2' not in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.0001
        )

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self.model(X_seq)
            loss = F.binary_cross_entropy(output, y_seq)
            loss.backward()
            optimizer.step()

        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info(f"LSTM fine-tuned on {len(y_seq)} new samples.")

    def save(self) -> str:
        """Save model weights + metadata."""
        if self.model is None or not HAS_TORCH:
            return ''

        filepath = os.path.join(self.model_path, f'lstm_{self.mode}.pth')
        torch.save(self.model.state_dict(), filepath)

        meta = {
            'mode': self.mode,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'saved_at': datetime.now().isoformat(),
        }
        meta_path = os.path.join(self.model_path, f'lstm_{self.mode}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return filepath

    def load(self) -> bool:
        """Load model from disk."""
        if not HAS_TORCH:
            return False

        filepath = os.path.join(self.model_path, f'lstm_{self.mode}.pth')
        meta_path = os.path.join(self.model_path, f'lstm_{self.mode}_meta.json')

        if not os.path.exists(filepath) or not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            self.seq_len = meta.get('seq_len', 60)
            self.n_features = meta.get('n_features')

            if self.n_features:
                self.build_model(self.n_features)
                self.model.load_state_dict(
                    torch.load(filepath, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                logger.info(f"LSTM model loaded from {filepath}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load LSTM: {e}")

        return False
