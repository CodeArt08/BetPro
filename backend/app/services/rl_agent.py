"""
RL Agent (Q-Network) + UCB Contextual Bandit.
Module 3A + 3B de la spécification élite.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger
import math
import json


# ─────────────────────────────────────────────────────────────────────
# 3A — Agent RL Q-Learning (Q-table simplifié pour perf temps réel)
# ─────────────────────────────────────────────────────────────────────
ACTIONS = ['BET_HOME', 'BET_DRAW', 'BET_AWAY', 'NO_BET']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


class RLAgent:
    """
    Q-Learning agent qui optimise directement le ROI.
    State: vecteur réduit de features key (10 dims pour perf).
    Q-table: dictionnaire state_hash → Q-values[4].
    Update en background: ~0.1s.
    Lecture Q-values en inference: ~0.001s.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 0.30, epsilon_min: float = 0.05):
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995
        
        self.q_table: Dict[str, np.ndarray] = {}  # hash → Q[4]
        self.t = 0  # Total steps
    
    def _hash_state(self, state: np.ndarray, bins: int = 5) -> str:
        """Discretize state vector into a hashable key."""
        discretized = np.digitize(
            np.clip(state, -1, 1),
            bins=np.linspace(-1, 1, bins)
        )
        return ','.join(map(str, discretized))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state. Returns zeros if unseen."""
        key = self._hash_state(state)
        return self.q_table.get(key, np.zeros(4))
    
    def select_action(self, state: np.ndarray, greedy: bool = False) -> Tuple[str, float]:
        """
        Epsilon-greedy action selection.
        greedy=True → always select best Q (during inference).
        Returns (action_name, q_value).
        """
        q_values = self.get_q_values(state)
        
        if not greedy and np.random.random() < self.epsilon:
            idx = np.random.randint(4)
        else:
            idx = int(np.argmax(q_values))
        
        return ACTIONS[idx], float(q_values[idx])
    
    def get_best_action(self, state: np.ndarray) -> Dict:
        """
        Retourne la recommandation RL pour l'inference.
        Lecture seule (pas de mise à jour).
        """
        q_values = self.get_q_values(state)
        best_idx = int(np.argmax(q_values))
        
        return {
            'recommended_action': ACTIONS[best_idx],
            'q_values': {a: float(q_values[i]) for i, a in enumerate(ACTIONS)},
            'q_best': float(q_values[best_idx]),
            'q_no_bet': float(q_values[3]),
            'recommend_bet': ACTIONS[best_idx] != 'NO_BET' and q_values[best_idx] > q_values[3],
        }
    
    def update(self, state: np.ndarray, action: str, reward: float,
               next_state: np.ndarray, done: bool = True):
        """
        Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        Appelé depuis Thread A après chaque résultat.
        """
        key = self._hash_state(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(4)
        
        action_idx = ACTION_TO_IDX.get(action, 3)
        q_current = self.q_table[key][action_idx]
        
        if done:
            q_target = reward
        else:
            next_q = self.get_q_values(next_state)
            q_target = reward + self.gamma * np.max(next_q)
        
        # Q-learning update
        self.q_table[key][action_idx] += self.alpha * (q_target - q_current)
        
        # Decay epsilon
        self.t += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def compute_reward(self, action: str, profit_loss: float,
                       outcome: str, correct: bool) -> float:
        """
        Compute reward from action taken and result.
        BET_* correct → profit_loss (positive)
        BET_* wrong → -profit_loss (loss as negative reward)
        NO_BET correct → small neutral (0)
        NO_BET when should have bet → -0.1
        """
        if action == 'NO_BET':
            if not correct:  # Would have been wrong bet
                return 0.15  # Good decision
            else:
                return -0.05  # Missed a winning bet
        else:
            return float(profit_loss)
    
    def build_state_vector(self, cache: Dict) -> np.ndarray:
        """
        Convert cache dict to compact state vector (10 features).
        Lecture depuis cache RAM: ~0.001s.
        """
        fourier = cache.get('fourier', {})
        cycle = cache.get('cycle', {})
        streak = cache.get('streak', {})
        runs = cache.get('runs_test', {})
        autocorr = cache.get('autocorr', {}).get('lag1', {})
        model_scores = cache.get('model_scores', {})
        bankroll_data = cache.get('bankroll', {})
        cp = cache.get('changepoint', {})
        
        state = np.array([
            cycle.get('N', {}).get('overdue_score', 0),
            cycle.get('V', {}).get('overdue_score', 0),
            cycle.get('D', {}).get('overdue_score', 0),
            autocorr.get('autocorr', 0),
            1.0 if fourier.get('cycle_detected') else 0.0,
            float(fourier.get('phase', 0) or 0) / max(float(fourier.get('cycle_length', 1) or 1), 1),
            runs.get('p_value', 0.5),
            model_scores.get('V', 0.33),
            model_scores.get('N', 0.33),
            1.0 if cp.get('changepoint_detected') else 0.0,
        ], dtype=float)
        
        return np.clip(state, -1, 1)
    
    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            't': self.t,
            'q_table_size': len(self.q_table),
        }
    
    def save(self, path: str):
        """Save Q-table to JSON."""
        data = {
            'q_table': {k: v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            't': self.t,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"RL agent saved: {len(self.q_table)} states")
    
    def load(self, path: str):
        """Load Q-table from JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.q_table = {k: np.array(v) for k, v in data['q_table'].items()}
            self.epsilon = data.get('epsilon', self.epsilon_min)
            self.t = data.get('t', 0)
            logger.info(f"RL agent loaded: {len(self.q_table)} states")
        except Exception as e:
            logger.warning(f"Could not load RL agent: {e}")


# ─────────────────────────────────────────────────────────────────────
# 3B — UCB Contextual Bandit pour sélection modèle
# ─────────────────────────────────────────────────────────────────────
class UCBBandit:
    """
    Upper Confidence Bound bandit pour pondération dynamique des modèles.
    UCB[model] = mean_reward + sqrt(2 × log(t) / n[model])
    Contextes: (heure_tranche, régime, type_match_dominant)
    """
    
    MODELS = ['lstm', 'xgb', 'lgb', 'rf', 'cat', 'mlp', 'gbm', 'mc', 'poisson', 'elo', 'markov']
    
    def __init__(self, c: float = math.sqrt(2)):
        self.c = c  # Exploration coefficient
        self.counts: Dict[str, Dict[str, int]] = {}      # context → model → n
        self.rewards: Dict[str, Dict[str, float]] = {}   # context → model → sum_reward
        self.t = 0
    
    def _get_context(self, heure: int, regime: str, dominant_type: str) -> str:
        tranche = 'matin' if 6 <= heure < 12 else \
                  'apres_midi' if 12 <= heure < 18 else \
                  'soir' if 18 <= heure < 22 else 'nuit'
        return f"{tranche}|{regime}|{dominant_type}"
    
    def get_ucb_weights(self, heure: int = 12, regime: str = 'STABLE',
                         dominant_type: str = 'V') -> Dict[str, float]:
        """
        Retourne les poids UCB pour chaque modèle dans ce contexte.
        Lecture: ~0.001s.
        """
        ctx = self._get_context(heure, regime, dominant_type)
        
        if ctx not in self.counts or not self.counts[ctx]:
            # Equal weights si pas de données contextuelles
            return {m: 1.0 / len(self.MODELS) for m in self.MODELS}
        
        cnt = self.counts[ctx]
        rew = self.rewards[ctx]
        t_ctx = sum(cnt.values())
        
        ucb_values = {}
        for model in self.MODELS:
            n = cnt.get(model, 0)
            r = rew.get(model, 0)
            if n == 0:
                ucb_values[model] = float('inf')  # Force exploration
            else:
                mean_reward = r / n
                confidence = self.c * math.sqrt(math.log(max(t_ctx, 1)) / n)
                ucb_values[model] = mean_reward + confidence
        
        # Softmax over UCB values (finite only)
        finite_vals = {m: v for m, v in ucb_values.items() if v != float('inf')}
        inf_models = [m for m, v in ucb_values.items() if v == float('inf')]
        
        if not finite_vals:
            return {m: 1.0 / len(self.MODELS) for m in self.MODELS}
        
        max_val = max(finite_vals.values())
        exp_vals = {m: math.exp(v - max_val) for m, v in finite_vals.items()}
        
        # Inf models get bonus weight
        for m in inf_models:
            exp_vals[m] = max(exp_vals.values()) * 1.5
        
        total = sum(exp_vals.values())
        weights = {m: v / total for m, v in exp_vals.items()}
        
        # Boost best model by 1.40 (spec)
        best_model = max(weights, key=weights.get)
        weights[best_model] = min(weights[best_model] * 1.40, 0.50)
        
        # Renormalize
        total = sum(weights.values())
        return {m: v / total for m, v in weights.items()}
    
    def update(self, model: str, reward: float,
               heure: int = 12, regime: str = 'STABLE', dominant_type: str = 'V'):
        """Update bandit après résultat. Thread A: ~0.02s."""
        ctx = self._get_context(heure, regime, dominant_type)
        
        if ctx not in self.counts:
            self.counts[ctx] = {m: 0 for m in self.MODELS}
            self.rewards[ctx] = {m: 0.0 for m in self.MODELS}
        
        if model in self.counts[ctx]:
            self.counts[ctx][model] += 1
            self.rewards[ctx][model] += reward
        
        self.t += 1
    
    def get_best_model(self, heure: int = 12, regime: str = 'STABLE',
                        dominant_type: str = 'V') -> str:
        """Retourne le meilleur modèle dans ce contexte."""
        weights = self.get_ucb_weights(heure, regime, dominant_type)
        return max(weights, key=weights.get)
    
    def get_model_stats(self) -> Dict:
        """Summary des performances par modèle."""
        aggregated: Dict[str, Dict] = {}
        for ctx, cnt in self.counts.items():
            for model, n in cnt.items():
                if model not in aggregated:
                    aggregated[model] = {'total_n': 0, 'total_reward': 0.0}
                aggregated[model]['total_n'] += n
                aggregated[model]['total_reward'] += self.rewards.get(ctx, {}).get(model, 0)
        
        result = {}
        for model, data in aggregated.items():
            n = data['total_n']
            result[model] = {
                'n': n,
                'mean_reward': data['total_reward'] / max(n, 1),
                'ucb_global': data['total_reward'] / max(n, 1) + self.c * math.sqrt(math.log(max(self.t, 1)) / max(n, 1)),
            }
        return result
    
    def save(self, path: str):
        """Persist bandit state."""
        data = {
            'counts': self.counts,
            'rewards': self.rewards,
            't': self.t,
            'c': self.c,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load bandit state."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.counts = data['counts']
            self.rewards = data['rewards']
            self.t = data.get('t', 0)
            logger.info(f"UCB Bandit loaded: {len(self.counts)} contexts")
        except Exception as e:
            logger.warning(f"Could not load UCB Bandit: {e}")


# ─────────────────────────────────────────────────────────────────────
# 3C — Anti-Martingale Adaptatif
# ─────────────────────────────────────────────────────────────────────
class AntiMartingaleManager:
    """
    Augmente la mise après wins consécutifs si signal fort.
    Reset immédiat après toute perte.
    """
    
    def __init__(self, max_multiplier: float = 1.25):
        self.max_multiplier = max_multiplier
        self.wins_streak = 0
        self.current_multiplier = 1.0
    
    def get_multiplier(self, signal_strength: float) -> float:
        """Retourne le multiplicateur courant."""
        if self.wins_streak >= 3 and signal_strength > 0.75:
            self.current_multiplier = min(
                self.max_multiplier,
                1.0 + self.wins_streak * 0.05
            )
        else:
            self.current_multiplier = 1.0
        return self.current_multiplier
    
    def on_win(self):
        """Appelé après un pari gagnant."""
        self.wins_streak += 1
    
    def on_loss(self):
        """Appelé après un pari perdu → reset immédiat."""
        self.wins_streak = 0
        self.current_multiplier = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'wins_streak': self.wins_streak,
            'current_multiplier': self.current_multiplier,
            'max_multiplier': self.max_multiplier,
        }
