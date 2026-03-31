"""
Conformal Prediction Intervals + ECE Drift Detection.
Module 4A + 4B + 4C + 4D de la spécification élite.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger
import math


# ─────────────────────────────────────────────────────────────────────
# 4A — Conformal Prediction Intervals
# ─────────────────────────────────────────────────────────────────────
class ConformalPredictor:
    """
    Calibration sur 100 matchs récents.
    scores_calib = [abs(pred - observed) for each]
    quantile_90 = percentile(scores_calib, 90)
    """
    
    def __init__(self, window: int = 100, alpha: float = 0.10):
        self.window = window          # Taille fenêtre calibration
        self.alpha = alpha            # 1-alpha = coverage cible (90%)
        self.calibration_scores: deque = deque(maxlen=window)
        self.quantile_90: float = 0.20  # Default
        self._is_calibrated = False
    
    def add_calibration_sample(self, predicted_prob: float, observed_result: bool):
        """
        Ajoute un échantillon calibration.
        predicted_prob: probabilité prédite pour l'outcome.
        observed_result: True si outcome s'est réalisé.
        """
        conformity_score = abs(predicted_prob - (1.0 if observed_result else 0.0))
        self.calibration_scores.append(conformity_score)
        
        # Recalculate quantile
        if len(self.calibration_scores) >= 20:
            self.quantile_90 = float(np.percentile(list(self.calibration_scores), 90))
            self._is_calibrated = True
    
    def compute_interval(self, prob: float) -> Tuple[float, float, float]:
        """
        Returns (lower, upper, width).
        lower = max(0, prob - quantile_90)
        upper = min(1, prob + quantile_90)
        """
        lower = max(0.0, prob - self.quantile_90)
        upper = min(1.0, prob + self.quantile_90)
        width = upper - lower
        return lower, upper, width
    
    def get_confidence_multiplier(self, width: float) -> float:
        """
        width < 0.15: mult 1.20 (très fiable)
        0.15-0.22: 1.00
        0.22-0.28: 0.80
        0.28-0.35: 0.60
        > 0.35: 0.00 (NO BET)
        """
        if width < 0.15:
            return 1.20
        elif width < 0.22:
            return 1.00
        elif width < 0.28:
            return 0.80
        elif width <= 0.35:
            return 0.60
        else:
            return 0.00  # NO BET
    
    def should_bet(self, width: float) -> bool:
        return width <= 0.35
    
    def get_signal_level(self, width: float) -> str:
        if width < 0.15:
            return 'TRÈS FIABLE'
        elif width < 0.22:
            return 'FIABLE'
        elif width < 0.28:
            return 'MODÉRÉ'
        elif width <= 0.35:
            return 'INCERTAIN'
        else:
            return 'NO BET'
    
    def get_stats(self) -> Dict:
        return {
            'is_calibrated': self._is_calibrated,
            'n_samples': len(self.calibration_scores),
            'quantile_90': self.quantile_90,
            'window': self.window,
        }


# ─────────────────────────────────────────────────────────────────────
# 4B — Ensemble Diversity Score
# ─────────────────────────────────────────────────────────────────────
def compute_diversity_score(model_predictions: Dict[str, Dict]) -> float:
    """
    diversity = 1 - mean(pairwise_correlations(predictions))
    model_predictions = {'xgb': {'V': 0.5, 'N': 0.3, 'D': 0.2}, ...}
    """
    if not model_predictions or len(model_predictions) < 2:
        return 0.50  # Neutral
    
    # Convert to vectors
    outcomes = ['V', 'N', 'D']
    vectors = []
    for model, probs in model_predictions.items():
        vec = [probs.get(o, 0.33) for o in outcomes]
        vectors.append(vec)
    
    # Pairwise correlations
    correlations = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = np.array(vectors[i]), np.array(vectors[j])
            if v1.std() > 0 and v2.std() > 0:
                corr = float(np.corrcoef(v1, v2)[0, 1])
                correlations.append(corr)
    
    if not correlations:
        return 0.50
    
    diversity = 1.0 - float(np.mean(correlations))
    return float(np.clip(diversity, 0, 1))


def get_diversity_signal(diversity: float) -> Tuple[str, float]:
    """
    Returns (level, bonus/penalty).
    < 0.30: modèles trop corrélés → pénalité
    > 0.60: bonne diversité → bonus confiance
    """
    if diversity < 0.30:
        return 'FAIBLE', -0.10
    elif diversity > 0.60:
        return 'BON', 0.05
    else:
        return 'NORMAL', 0.0


# ─────────────────────────────────────────────────────────────────────
# 4C — ECE (Expected Calibration Error) + Isotonic Drift Detection
# ─────────────────────────────────────────────────────────────────────
class CalibrationDriftDetector:
    """
    ECE = mean(abs(confidence - accuracy)) sur 20 derniers matchs.
    ECE > 0.07 → recalibration en background
    ECE > 0.12 → pause 5 matchs + analyse
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        self.calibration_data: deque = deque(maxlen=window)
        self.ece = 0.0
        self.recalibration_count = 0
    
    def add_sample(self, confidence: float, was_correct: bool):
        """Ajoute un échantillon après résultat connu."""
        self.calibration_data.append({
            'confidence': confidence,
            'correct': 1.0 if was_correct else 0.0,
        })
        self._compute_ece()
    
    def _compute_ece(self):
        """Calcule l'ECE courant."""
        if len(self.calibration_data) < 5:
            return
        
        data = list(self.calibration_data)
        # Bin par confidence
        bins = {}
        for sample in data:
            bucket = round(sample['confidence'] * 10) / 10
            if bucket not in bins:
                bins[bucket] = {'sum_conf': 0.0, 'sum_correct': 0.0, 'n': 0}
            bins[bucket]['sum_conf'] += sample['confidence']
            bins[bucket]['sum_correct'] += sample['correct']
            bins[bucket]['n'] += 1
        
        ece_parts = []
        total = len(data)
        for bucket_data in bins.values():
            n = bucket_data['n']
            if n == 0:
                continue
            avg_conf = bucket_data['sum_conf'] / n
            avg_acc = bucket_data['sum_correct'] / n
            ece_parts.append((n / total) * abs(avg_conf - avg_acc))
        
        self.ece = float(sum(ece_parts))
    
    def get_ece_status(self) -> Dict:
        """Retourne le statut ECE + recommandations."""
        status = 'OK'
        action = None
        
        if self.ece > 0.12:
            status = 'DRIFT CRITIQUE'
            action = 'PAUSE 5 MATCHS + RECALIBRATION'
        elif self.ece > 0.07:
            status = 'DRIFT'
            action = 'RECALIBRATION BACKGROUND'
        
        return {
            'ece': self.ece,
            'status': status,
            'action': action,
            'n_samples': len(self.calibration_data),
            'recalibration_count': self.recalibration_count,
        }
    
    def needs_recalibration(self) -> bool:
        return self.ece > 0.07
    
    def needs_pause(self) -> bool:
        return self.ece > 0.12


# ─────────────────────────────────────────────────────────────────────
# 4D — Variance-based Confidence
# ─────────────────────────────────────────────────────────────────────
def compute_variance_confidence(pred_scores_last_20: List[float]) -> Dict:
    """
    pred_variance = var([final_scores last 20])
    variance_factor = 1 / (1 + 5 × pred_variance)
    """
    if len(pred_scores_last_20) < 3:
        return {'variance': 0.0, 'factor': 1.0}
    
    pred_variance = float(np.var(pred_scores_last_20))
    factor = 1.0 / (1.0 + 5 * pred_variance)
    
    return {
        'variance': pred_variance,
        'factor': float(np.clip(factor, 0.20, 1.0)),
        'level': 'STABLE' if pred_variance < 0.02 else 'VOLATILE' if pred_variance < 0.05 else 'INSTABLE',
    }


# ─────────────────────────────────────────────────────────────────────
# Entropy calculation
# ─────────────────────────────────────────────────────────────────────
def compute_entropy(probs: Dict[str, float]) -> float:
    """
    entropy_norm = -sum(p × log(p)) / log(3)
    """
    entropy = -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)
    entropy_norm = entropy / math.log(3)
    return float(np.clip(entropy_norm, 0, 1))


def get_entropy_signal(entropy_norm: float) -> str:
    if entropy_norm < 0.60:
        return 'CLAIR'
    elif entropy_norm < 0.80:
        return 'MODÉRÉ'
    elif entropy_norm < 0.95:
        return 'INCERTAIN'
    else:
        return 'TRÈS INCERTAIN'


# ─────────────────────────────────────────────────────────────────────
# Module Agreement Score
# ─────────────────────────────────────────────────────────────────────
def compute_model_agreement(model_predictions: Dict[str, Dict]) -> Tuple[float, Dict]:
    """
    Calcule le ratio de modèles en accord sur le résultat prédit.
    Returns (agreement_ratio, vote_counts).
    """
    votes = {'V': 0, 'N': 0, 'D': 0}
    
    for model, probs in model_predictions.items():
        best = max(probs, key=probs.get)
        votes[best] = votes.get(best, 0) + 1
    
    total = sum(votes.values())
    if total == 0:
        return 0.0, votes
    
    dominant = max(votes, key=votes.get)
    agreement = votes[dominant] / total
    
    return float(agreement), votes
