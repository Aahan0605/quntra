"""Models sub-package — Ensemble prediction engine."""

try:
    from .xgboost_model import XGBoostSignalModel
    from .lstm_model import LSTMPriceModel
    from .transformer_model import TemporalFusionTransformer
    from .ensemble import QuantraEnsemble
except ImportError:
    pass  # Dependencies not yet installed
