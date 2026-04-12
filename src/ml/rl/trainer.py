"""
RL Trainer — Orchestrates the full RL training pipeline.
=========================================================

Manages training, evaluation, resumption, and report generation
for the Quantra PPO trading agent.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from .environment import QuantraTradingEnv
from .agent import QuantraPPOAgent
from ..features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class RLTrainer:
    """Orchestrates full RL training pipeline with logging."""

    def __init__(self, model_dir: str = 'models/rl_agent/',
                 results_dir: str = 'results/rl/'):
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def train_from_scratch(self, ticker: str, mode: str = 'intraday',
                           timesteps: int = 100000) -> Dict[str, Any]:
        """
        Full training pipeline for new ticker.

        Steps:
        1. Download and compute features
        2. Create trading environment
        3. Build and train PPO agent
        4. Evaluate and save results
        """
        logger.info(f"Training RL agent from scratch for {ticker} ({mode})")

        # Step 1: Build features
        timeframe = '5m' if mode == 'intraday' else '1d'
        pipeline = FeaturePipeline(
            ticker=ticker, exchange='NSE',
            timeframe=timeframe, mode=mode
        )

        try:
            X, y = pipeline.build_feature_matrix()
        except Exception as e:
            logger.error(f"Feature pipeline failed: {e}")
            return {'error': str(e)}

        if len(X) < 100:
            return {'error': f'Insufficient data: only {len(X)} rows'}

        # Step 2: Create environment
        env = QuantraTradingEnv(
            df=X,
            initial_capital=100000,
            mode=mode,
            transaction_cost=0.001,
        )

        # Step 3: Build and train agent
        agent = QuantraPPOAgent(env=env, model_path=self.model_dir)
        agent.build_agent()
        metrics = agent.train(total_timesteps=timesteps)

        # Step 4: Save results
        results = {
            'ticker': ticker,
            'mode': mode,
            'timesteps': timesteps,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
            'n_features': X.shape[1],
            'n_samples': len(X),
        }

        results_path = os.path.join(
            self.results_dir, f'training_{ticker}_{mode}.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Training complete. Results saved to {results_path}")
        return results

    def resume_training(self, ticker: str, additional_timesteps: int = 10000
                        ) -> Dict[str, Any]:
        """Continue training existing agent with fresh data."""
        logger.info(f"Resuming training for {ticker}")

        # Rebuild features
        pipeline = FeaturePipeline(ticker=ticker, exchange='NSE',
                                   timeframe='5m', mode='intraday')
        try:
            X, y = pipeline.build_feature_matrix()
        except Exception as e:
            return {'error': str(e)}

        env = QuantraTradingEnv(df=X, mode='intraday')
        agent = QuantraPPOAgent(env=env, model_path=self.model_dir)

        if not agent.load():
            logger.warning("No existing model found. Training from scratch.")
            agent.build_agent()

        metrics = agent.train(total_timesteps=additional_timesteps)
        return {'resumed': True, 'metrics': metrics}

    def evaluate(self, ticker: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate agent on test period, return performance metrics."""
        pipeline = FeaturePipeline(ticker=ticker, exchange='NSE',
                                   timeframe='5m', mode='intraday')
        try:
            X, y = pipeline.build_feature_matrix()
        except Exception as e:
            return {'error': str(e)}

        # Use last 20% as test
        test_start = int(len(X) * 0.8)
        X_test = X.iloc[test_start:]

        env = QuantraTradingEnv(df=X_test, mode='intraday')
        agent = QuantraPPOAgent(env=env, model_path=self.model_dir)

        if not agent.load():
            return {'error': 'No trained model found'}

        return agent.evaluate(n_episodes=n_episodes)

    def generate_training_report(self) -> Dict[str, Any]:
        """Compile training results from all saved runs."""
        reports = []
        for f in os.listdir(self.results_dir):
            if f.startswith('training_') and f.endswith('.json'):
                try:
                    with open(os.path.join(self.results_dir, f)) as fh:
                        reports.append(json.load(fh))
                except Exception:
                    pass

        return {
            'total_runs': len(reports),
            'reports': reports,
            'generated_at': datetime.now().isoformat(),
        }
