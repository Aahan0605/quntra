"""
Quantra PPO Agent — Proximal Policy Optimization for trading.
==============================================================

PPO chosen because:
  - More stable than vanilla policy gradient
  - Works well in financial environments (noisy rewards)
  - Better sample efficiency than DQN for continuous-ish states
  - Clipped objective prevents catastrophic policy updates

Self-learning loop:
  1. Agent trades on paper
  2. Every trade outcome logged to journal
  3. After N trades → re-train on updated data
  4. If improved → accept new policy
  5. If worse → keep old policy (conservative update)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class QuantraPPOAgent:
    """
    PPO agent using stable-baselines3.

    Parameters
    ----------
    env : QuantraTradingEnv
        The custom trading environment.
    model_path : str
        Directory for saving/loading agent weights.
    """

    def __init__(self, env=None, model_path: str = 'models/rl_agent/'):
        self.env = env
        self.model_path = model_path
        self.model = None
        self._old_model = None

        os.makedirs(model_path, exist_ok=True)

    def build_agent(self):
        """
        Build PPO agent with custom hyperparameters tuned for trading.
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required. pip install stable-baselines3")
        if self.env is None:
            raise ValueError("Environment must be set before building agent.")

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
        )
        if HAS_TORCH:
            policy_kwargs['activation_fn'] = nn.ReLU

        self.model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

        logger.info("PPO agent built with [256, 256, 128] architecture.")

    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """
        Train agent for total_timesteps environment steps.
        """
        if self.model is None:
            self.build_agent()

        logger.info(f"Training PPO for {total_timesteps} timesteps...")

        # Save old model for comparison
        old_path = os.path.join(self.model_path, 'ppo_old.zip')
        try:
            if os.path.exists(os.path.join(self.model_path, 'ppo_agent.zip')):
                self.model.save(old_path)
        except Exception:
            pass

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
        )

        # Evaluate
        metrics = self.evaluate(n_episodes=5)

        # Save best model
        self.save()

        logger.info(
            f"Training complete. Win rate: {metrics.get('win_rate', 0):.2%} | "
            f"Avg PnL: {metrics.get('avg_pnl', 0):.2f}"
        )

        return metrics

    def self_improve(self, new_trade_outcomes: list = None) -> Dict[str, Any]:
        """
        The core self-learning loop — called after every 10+ new trades.

        Steps:
        1. Continue PPO training for 10000 more steps
        2. Evaluate new policy on environment
        3. Compare: new_sharpe vs old_sharpe
        4. Accept if improved, revert if worse
        """
        if self.model is None:
            return {'improved': False, 'reason': 'No model loaded'}

        # Get old performance
        old_metrics = self.evaluate(n_episodes=5)
        old_sharpe = old_metrics.get('sharpe', 0)

        # Save current model as backup
        backup_path = os.path.join(self.model_path, 'ppo_backup.zip')
        self.model.save(backup_path)

        # Continue training
        try:
            self.model.learn(total_timesteps=10000, progress_bar=False)
        except Exception as e:
            logger.warning(f"Self-improvement training failed: {e}")
            return {'improved': False, 'reason': str(e)}

        # Evaluate new policy
        new_metrics = self.evaluate(n_episodes=5)
        new_sharpe = new_metrics.get('sharpe', 0)

        # Compare (conservative: accept only if >= 95% of old)
        improved = new_sharpe > old_sharpe * 0.95

        if improved:
            self.save()
            logger.info(f"Self-improvement accepted: Sharpe {old_sharpe:.3f} → {new_sharpe:.3f}")
        else:
            # Revert to backup
            try:
                self.model = PPO.load(backup_path, env=self.env)
                logger.info(f"Self-improvement rejected: {new_sharpe:.3f} < {old_sharpe:.3f} * 0.95")
            except Exception:
                pass

        return {
            'improved': improved,
            'old_sharpe': round(old_sharpe, 4),
            'new_sharpe': round(new_sharpe, 4),
            'delta_pct': round((new_sharpe - old_sharpe) / max(abs(old_sharpe), 1e-6) * 100, 2),
            'trades_analyzed': len(new_trade_outcomes) if new_trade_outcomes else 0,
        }

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent on environment for n_episodes.
        Return performance metrics.
        """
        if self.model is None or self.env is None:
            return {'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0}

        episode_rewards = []
        episode_pnls = []
        wins = 0
        total_trades = 0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                ep_reward += reward
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            pnl = info.get('total_pnl', 0)
            episode_pnls.append(pnl)
            trades = info.get('trades_today', 0)
            total_trades += trades

            if pnl > 0:
                wins += 1

        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        std_reward = np.std(episode_rewards) if len(episode_rewards) > 1 else 1
        sharpe = avg_reward / max(std_reward, 1e-6)

        return {
            'sharpe': round(float(sharpe), 4),
            'avg_reward': round(float(avg_reward), 4),
            'avg_pnl': round(float(np.mean(episode_pnls)), 2),
            'win_rate': round(wins / max(n_episodes, 1), 4),
            'total_trades': total_trades,
            'n_episodes': n_episodes,
        }

    def predict_action(self, observation: np.ndarray) -> Dict:
        """
        Get RL agent's recommended action.
        """
        if self.model is None:
            return {
                'action': 0, 'action_name': 'HOLD',
                'confidence': 0.0, 'value_estimate': 0.0,
            }

        obs = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        action, _states = self.model.predict(obs, deterministic=True)
        action = int(action)

        # Get value estimate from critic
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                value = self.model.policy.predict_values(obs_tensor)
            value_est = float(value.item())
        except Exception:
            value_est = 0.0

        # Confidence from action distribution
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().numpy()[0]
            confidence = float(probs[action])
        except Exception:
            confidence = 0.5

        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SHORT', 3: 'CLOSE_LONG', 4: 'CLOSE_SHORT'}

        return {
            'action': action,
            'action_name': action_names.get(action, 'HOLD'),
            'confidence': round(confidence, 4),
            'value_estimate': round(value_est, 4),
        }

    def save(self) -> str:
        """Save PPO model to disk."""
        if self.model is None:
            return ''
        filepath = os.path.join(self.model_path, 'ppo_agent')
        self.model.save(filepath)
        logger.info(f"PPO agent saved to {filepath}")
        return filepath

    def load(self) -> bool:
        """Load PPO model from disk."""
        if not HAS_SB3:
            return False
        filepath = os.path.join(self.model_path, 'ppo_agent.zip')
        if not os.path.exists(filepath):
            return False
        try:
            self.model = PPO.load(filepath, env=self.env)
            logger.info(f"PPO agent loaded from {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load PPO agent: {e}")
            return False
