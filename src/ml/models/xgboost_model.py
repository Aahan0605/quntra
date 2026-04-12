"""
XGBoost Signal Model — Gradient boosting for trade signal generation.
======================================================================

Best at: handling tabular features (fundamental + technical ratios),
non-linear relationships, missing values, feature importance.

Two models trained separately:
  1. intraday_model: 5m timeframe, predicts 30-min price direction
  2. swing_model: daily timeframe, predicts 5-day price direction

Uses Optuna for hyperparameter optimization.
Tracks experiments with MLflow (optional).
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, f1_score
)

logger = logging.getLogger(__name__)


class XGBoostSignalModel:
    """
    XGBoost classifier for trade signal generation.

    Parameters
    ----------
    mode : str
        'intraday' or 'swing'. Determines model hyperparameter defaults.
    model_path : str
        Directory to save/load model files.
    """

    def __init__(self, mode: str = 'intraday',
                 model_path: str = 'models/xgboost/'):
        self.mode = mode
        self.model_path = model_path
        self.model = None
        self.best_params = None
        self.feature_names = None
        self._threshold = 0.6

        os.makedirs(model_path, exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_trials: int = 50) -> Dict[str, Any]:
        """
        Train XGBoost with Optuna hyperparameter optimization.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        n_trials : number of Optuna trials for HPO

        Returns
        -------
        dict with best_params, val_auc, val_accuracy, feature_importance
        """
        if not HAS_XGB:
            raise ImportError("xgboost is required. Install via: pip install xgboost")

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values
            X_val = X_val.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            y_val = y_val.values

        # Compute class weight for imbalanced data
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos = n_neg / max(n_pos, 1)

        if HAS_OPTUNA and n_trials > 0:
            # Optuna HPO
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'scale_pos_weight': scale_pos,
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'random_state': 42,
                    'verbosity': 0,
                }

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, y_pred_proba)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            self.best_params = study.best_params
            self.best_params['scale_pos_weight'] = scale_pos
            logger.info(f"Best Optuna AUC: {study.best_value:.4f}")
        else:
            # Default params
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 1.0,
                'scale_pos_weight': scale_pos,
            }

        # Train final model with best params
        self.best_params.update({
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42,
            'verbosity': 0,
        })

        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= self._threshold).astype(int)

        val_auc = roc_auc_score(y_val, y_pred_proba)
        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, zero_division=0)

        # Feature importance
        importance = self.get_feature_importance(top_n=20)

        # Log to MLflow if available
        if HAS_MLFLOW:
            try:
                with mlflow.start_run(run_name=f"xgb_{self.mode}_{datetime.now().strftime('%Y%m%d')}"):
                    mlflow.log_params(self.best_params)
                    mlflow.log_metrics({
                        'val_auc': val_auc, 'val_accuracy': val_acc, 'val_f1': val_f1
                    })
            except Exception:
                pass

        result = {
            'best_params': self.best_params,
            'val_auc': val_auc,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'feature_importance': importance,
        }

        logger.info(f"XGBoost {self.mode}: AUC={val_auc:.4f} ACC={val_acc:.4f} F1={val_f1:.4f}")
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class (bullish signal)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def predict_signal(self, X: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        Return structured signal with SHAP explainability.

        Returns
        -------
        dict: {signal, confidence, reasoning}
        """
        if self.model is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': []}

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        else:
            feature_names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = self.model.predict_proba(X)[:, 1][0]

        if proba >= threshold:
            signal = 'BUY'
        elif proba <= (1 - threshold):
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # SHAP explainability
        reasoning = []
        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 (bullish)

                # Top 5 features by absolute SHAP value
                abs_shap = np.abs(shap_values[0])
                top_idx = np.argsort(abs_shap)[-5:][::-1]

                for idx in top_idx:
                    fname = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                    direction = "↑" if shap_values[0][idx] > 0 else "↓"
                    reasoning.append(
                        f"{fname} {direction} (impact: {shap_values[0][idx]:.3f})"
                    )
            except Exception as e:
                logger.debug(f"SHAP failed: {e}")

        if not reasoning:
            # Fallback: use feature importance
            importance = self.get_feature_importance(top_n=5)
            reasoning = [f"{k}: importance={v:.3f}" for k, v in importance.items()]

        return {
            'signal': signal,
            'confidence': round(float(proba if signal == 'BUY' else 1 - proba), 4),
            'probability': round(float(proba), 4),
            'reasoning': reasoning,
        }

    def update_on_trade(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Online learning: retrain model incrementally with warm start.
        Only retrain if n_new_samples >= 10.
        """
        if self.model is None or len(y_new) < 10:
            return

        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values
        if isinstance(y_new, pd.Series):
            y_new = y_new.values

        try:
            # XGBoost warm start: use xgb_model parameter
            self.model.fit(
                X_new, y_new,
                xgb_model=self.model.get_booster(),
                verbose=False
            )
            logger.info(f"XGBoost updated with {len(y_new)} new samples.")
        except Exception as e:
            logger.warning(f"Online update failed: {e}")

    def save(self) -> str:
        """Save model + metadata to disk. Return save path."""
        if self.model is None:
            return ''

        filepath = os.path.join(self.model_path, f'xgb_{self.mode}.json')
        self.model.save_model(filepath)

        # Save metadata
        meta = {
            'mode': self.mode,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'threshold': self._threshold,
            'saved_at': datetime.now().isoformat(),
        }
        meta_path = os.path.join(self.model_path, f'xgb_{self.mode}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"XGBoost model saved to {filepath}")
        return filepath

    def load(self) -> bool:
        """Load model from disk. Return True if successful."""
        filepath = os.path.join(self.model_path, f'xgb_{self.mode}.json')
        meta_path = os.path.join(self.model_path, f'xgb_{self.mode}_meta.json')

        if not os.path.exists(filepath):
            logger.warning(f"No saved model at {filepath}")
            return False

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)

            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.best_params = meta.get('best_params')
                self.feature_names = meta.get('feature_names')
                self._threshold = meta.get('threshold', 0.6)

            logger.info(f"XGBoost model loaded from {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Return top N features by gain importance."""
        if self.model is None:
            return {}

        try:
            importance = self.model.feature_importances_
            names = self.feature_names or [f'f{i}' for i in range(len(importance))]

            paired = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
            return dict(paired[:top_n])
        except Exception:
            return {}
