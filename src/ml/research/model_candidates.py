"""
Model candidates for the research sweep — what beats XGBoost, if anything.
All scikit-learn (BSD) / xgboost (Apache-2.0). Research environment only.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_model_candidates() -> dict:
    return {
        "xgboost_tuned": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=3, gamma=0.1,
            eval_metric="logloss", n_jobs=2, random_state=42,
        ),
        "xgboost_shallow": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=2, random_state=42,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "logistic_l2": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, n_jobs=2, random_state=42,
        ),
    }
