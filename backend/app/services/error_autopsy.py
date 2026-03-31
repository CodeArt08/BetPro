"""
Error Autopsy System — Module 5 de la spécification élite.
Classification, correction immédiate, Recovery Mode, Meta-patterns.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict, deque
import json


# ─────────────────────────────────────────────────────────────────────
# 5A — 10 Types d'erreurs
# ─────────────────────────────────────────────────────────────────────
ERROR_TYPES = {
    'WRONG_DIRECTION':         1,   # Mauvaise direction, signal engine ignoré
    'OVERCONFIDENT':           2,   # Confidence trop haute, perdu
    'WRONG_VALUE':             3,   # EV mal calculé
    'PATTERN_MISREAD':         4,   # Pattern engine mal interprété
    'WRONG_REGIME':            5,   # Régime mal détecté
    'STREAK_UNDERESTIMATED':   6,   # Streak continuation sous-estimée
    'CHANGEPOINT_MISSED':      7,   # Changepoint non détecté
    'CONFORMAL_IGNORED':       8,   # Conformal interval trop large ignoré
    'MOVEMENT_IGNORED':        9,   # Odds movement ignoré ou mal lu
    'TIMING_MISSED':           10,  # Décision trop tardive
}

ERROR_TYPE_NAMES = {v: k for k, v in ERROR_TYPES.items()}

# Normal thresholds (non-recovery)
NORMAL_THRESHOLDS = {
    'ev_minimum':          0.10,
    'confidence_min':      0.72,
    'model_agreement_min': 3/5,
    'entropy_max':         0.95,
    'conformal_max':       0.35,
}

# Recovery thresholds (post-error)
RECOVERY_THRESHOLDS = {
    'ev_minimum':          0.15,
    'confidence_min':      0.80,
    'model_agreement_min': 4/5,
    'entropy_max':         0.88,
    'conformal_max':       0.28,
}


class ErrorAutopsySystem:
    """
    Système complet de classification et correction immédiate des erreurs.
    """
    
    def __init__(self):
        # Correction weights (multiplicative)
        self.weights = {
            'confidence_threshold':   0.72,
            'conformal_max_width':    0.35,
            'entropy_weight':         1.0,
            'ml_weight':              1.0,
            'engine_signal_weight':   1.0,
            'draw_cycle_weight':      1.0,
            'fourier_weight':         1.0,
            'fourier_amplitude_threshold': 0.30,
            'odds_movement_weight':   1.0,
            'bocpd_sensitivity':      0.10,
            'post_changepoint_discount': 0.25,
            'max_streak_multiplier':  1.0,
        }
        
        # Error history
        self.error_log: List[Dict] = []
        self.recent_errors: deque = deque(maxlen=50)
        
        # Recovery state
        self.recovery_mode = False
        self.recovery_wins_needed = 2
        self.recovery_wins_count = 0
        self.consecutive_errors = 0
        
        # Meta-pattern tracking
        self.errors_by_hour: Dict[int, List[bool]] = defaultdict(list)    # hour → [correct/wrong]
        self.errors_by_cote_bracket: Dict[str, List[bool]] = defaultdict(list)
        self.errors_after_win: List[bool] = []
        
        # Max streaks for M3
        self.max_streaks_hist: Dict[str, int] = {'V': 5, 'N': 4, 'D': 4}
    
    # ─────────────────────────────────────────────────────────────────
    # 5A — Classification immédiate (< 30s post-résultat)
    # ─────────────────────────────────────────────────────────────────
    def classify_error(self, prediction_context: Dict) -> str:
        """
        Classifie le type d'erreur selon le contexte de prédiction.
        prediction_context = {
          'predicted': 'V/N/D', 'actual': 'V/N/D',
          'confidence': float, 'conformal_width': float,
          'entropy': float, 'draw_overdue': float,
          'fourier_signal': str, 'odds_movement': str,
          'changepoint_last_10': bool, 'inference_time': float,
          'regime': str, 'streak_type': str, 'streak_len': int,
        }
        """
        predicted = prediction_context.get('predicted')
        actual = prediction_context.get('actual')
        confidence = prediction_context.get('confidence', 0.5)
        conformal_width = prediction_context.get('conformal_width', 0.35)
        draw_overdue = prediction_context.get('draw_overdue', 0)
        fourier_signal = prediction_context.get('fourier_signal')
        odds_movement = prediction_context.get('odds_movement')
        changepoint = prediction_context.get('changepoint_last_10', False)
        inference_time = prediction_context.get('inference_time', 2.0)
        regime = prediction_context.get('regime', 'STABLE')
        streak_type = prediction_context.get('streak_type')
        streak_len = prediction_context.get('streak_len', 0)
        
        # T10 — Timing
        if inference_time > 15:
            return 'TIMING_MISSED'
        
        # T7 — Changepoint
        if changepoint and actual != predicted:
            return 'CHANGEPOINT_MISSED'
        
        # T5 — Wrong regime
        if regime == 'CHAOTIC':
            return 'WRONG_REGIME'
        
        # T8 — Conformal ignored
        if conformal_width > 0.35 and actual != predicted:
            return 'CONFORMAL_IGNORED'
        
        # T9 — Odds movement ignored
        if odds_movement and odds_movement != predicted and actual == odds_movement:
            return 'MOVEMENT_IGNORED'
        
        # T4 — Fourier missed
        if fourier_signal and fourier_signal != predicted and actual == fourier_signal:
            return 'PATTERN_MISREAD'
        
        # T6 — Streak continuation
        if streak_type == actual and streak_len >= 3 and predicted != actual:
            return 'STREAK_UNDERESTIMATED'
        
        # T2 — Overconfident
        if confidence > 0.85 and actual != predicted:
            return 'OVERCONFIDENT'
        
        # T3 — Wrong draw overdue
        if draw_overdue > 0.35 and actual == 'N' and predicted != 'N':
            return 'WRONG_DIRECTION'
        
        # Default
        return 'WRONG_DIRECTION'
    
    # ─────────────────────────────────────────────────────────────────
    # 5B — Error Autopsy complet
    # ─────────────────────────────────────────────────────────────────
    def run_autopsy(self, prediction_context: Dict, match_id: int,
                    inference_log: Dict = None) -> Dict:
        """
        Exécute l'autopsy complète. < 0.2s total.
        """
        error_type = self.classify_error(prediction_context)
        lesson = self._extract_lesson(error_type, prediction_context)
        
        autopsy = {
            'match_id': match_id,
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': error_type,
            'severity': self._get_severity(error_type),
            'predicted': prediction_context.get('predicted'),
            'actual': prediction_context.get('actual'),
            'features_responsible': self._identify_responsible_features(error_type, prediction_context),
            'entropy_at_time': prediction_context.get('entropy', 0),
            'conformal_width': prediction_context.get('conformal_width', 0),
            'draw_overdue_ignored': (
                prediction_context.get('draw_overdue', 0) > 0.35
                and prediction_context.get('predicted') != 'N'
            ),
            'changepoint_active': prediction_context.get('changepoint_last_10', False),
            'fourier_cycle_ignored': (
                prediction_context.get('fourier_signal') != prediction_context.get('predicted')
            ),
            'movement_ignored': (
                prediction_context.get('odds_movement') is not None
                and prediction_context.get('odds_movement') != prediction_context.get('predicted')
            ),
            'runs_pvalue': prediction_context.get('runs_pvalue', 1.0),
            'inference_time': prediction_context.get('inference_time', 2.0),
            'lesson': lesson,
            'correction': self._compute_correction(error_type, prediction_context),
        }
        
        self.error_log.append(autopsy)
        self.recent_errors.appendleft(autopsy)
        
        logger.warning(f"Error autopsy [{error_type}]: {lesson}")
        return autopsy
    
    # ─────────────────────────────────────────────────────────────────
    # 5C — Correction immédiate des poids
    # ─────────────────────────────────────────────────────────────────
    def apply_corrections(self, error_type: str, prediction_context: Dict):
        """
        Applique les corrections de poids immédiatement.
        Appelé dans Thread A dès l'autopsy.
        """
        if error_type == 'OVERCONFIDENT':
            self.weights['confidence_threshold'] = min(
                self.weights['confidence_threshold'] + 0.03, 0.92)
            self.weights['conformal_max_width'] = max(
                self.weights['conformal_max_width'] - 0.02, 0.15)
            self.weights['entropy_weight'] *= 1.15
        
        elif error_type == 'WRONG_DIRECTION':
            self.weights['ml_weight'] *= 0.88
            self.weights['engine_signal_weight'] *= 1.12
            if prediction_context.get('draw_overdue', 0) > 0.35:
                self.weights['draw_cycle_weight'] *= 1.20
        
        elif error_type == 'CHANGEPOINT_MISSED':
            self.weights['bocpd_sensitivity'] += 0.10
            self.weights['post_changepoint_discount'] += 0.05
        
        elif error_type == 'PATTERN_MISREAD':
            self.weights['fourier_weight'] *= 1.15
            self.weights['fourier_amplitude_threshold'] *= 0.90
        
        elif error_type == 'MOVEMENT_IGNORED':
            self.weights['odds_movement_weight'] *= 1.20
        
        elif error_type == 'STREAK_UNDERESTIMATED':
            for t in ['V', 'N', 'D']:
                self.max_streaks_hist[t] = int(self.max_streaks_hist.get(t, 5) * 1.25)
        
        # Clamp all weights
        for k in self.weights:
            if isinstance(self.weights[k], float):
                self.weights[k] = float(np.clip(self.weights[k], 0.01, 5.0))
        
        logger.info(f"Applied correction for [{error_type}]")
    
    # ─────────────────────────────────────────────────────────────────
    # 5D — Recovery Mode post-erreur
    # ─────────────────────────────────────────────────────────────────
    def on_result(self, correct: bool, heure: int = 12,
                  cote_bracket: str = '1.5-2.0', was_win_before: bool = False):
        """
        Appelé après chaque match (correct=True si prédiction correcte).
        Met à jour les états Recovery + méta-patterns.
        """
        if correct:
            self.consecutive_errors = 0
            if self.recovery_mode:
                self.recovery_wins_count += 1
                if self.recovery_wins_count >= self.recovery_wins_needed:
                    self.recovery_mode = False
                    self.recovery_wins_count = 0
                    logger.info("Recovery mode DÉSACTIVÉ — 2 paris gagnants consécutifs")
        else:
            self.consecutive_errors += 1
            self.recovery_mode = True
            self.recovery_wins_count = 0
            logger.warning(f"Recovery mode ACTIVÉ (erreur #{self.consecutive_errors})")
        
        # Meta-patterns
        self.errors_by_hour[heure].append(correct)
        self.errors_by_cote_bracket[cote_bracket].append(correct)
        if was_win_before:
            self.errors_after_win.append(correct)
    
    def get_thresholds(self) -> Dict:
        """Retourne les seuils actifs (normaux ou recovery)."""
        if self.recovery_mode:
            return RECOVERY_THRESHOLDS.copy()
        return NORMAL_THRESHOLDS.copy()
    
    def get_active_lessons(self) -> List[str]:
        """Retourne les leçons actives (corrections appliquées)."""
        lessons = []
        
        if self.weights.get('confidence_threshold', 0.72) > 0.74:
            lessons.append(f"LECON: confidence seuil → {self.weights['confidence_threshold']:.2f}")
        if self.weights.get('draw_cycle_weight', 1.0) > 1.15:
            lessons.append("LECON: cycle draw ignoré → draw_cycle_weight boosté")
        if self.weights.get('fourier_weight', 1.0) > 1.10:
            lessons.append("LECON: cycle Fourier ignoré → fourier_weight boosté")
        if self.weights.get('odds_movement_weight', 1.0) > 1.15:
            lessons.append("LECON: odds movement ignoré → movement_weight boosté")
        if self.weights.get('bocpd_sensitivity', 0.10) > 0.15:
            lessons.append("LECON: changepoint manqué → BOCPD sensibilité augmentée")
        if self.recovery_mode:
            lessons.append(f"RECOVERY: critères renforcés actifs (besoin {self.recovery_wins_needed - self.recovery_wins_count} wins)")
        
        return lessons
    
    # ─────────────────────────────────────────────────────────────────
    # 5E — Meta-pattern erreurs
    # ─────────────────────────────────────────────────────────────────
    def check_meta_patterns(self, heure: int = 12, cote_bracket: str = '1.5-2.0') -> Dict:
        """
        Vérifie si on doit stopper paris selon taux d'erreur.
        Returns: {stop_this_hour, stop_this_bracket, pause_required}
        """
        alerts = {}
        
        # Error rate par heure
        hour_results = self.errors_by_hour.get(heure, [])
        if len(hour_results) >= 5:
            error_rate = 1 - (sum(hour_results) / len(hour_results))
            if error_rate > 0.65:
                alerts['stop_this_hour'] = {
                    'heure': heure,
                    'error_rate': error_rate,
                    'message': f"STOP paris heure {heure}h — erreur {error_rate:.0%}"
                }
        
        # Error rate par tranche de cotes
        bracket_results = self.errors_by_cote_bracket.get(cote_bracket, [])
        if len(bracket_results) >= 5:
            error_rate_b = 1 - (sum(bracket_results) / len(bracket_results))
            if error_rate_b > 0.60:
                alerts['stop_this_bracket'] = {
                    'bracket': cote_bracket,
                    'error_rate': error_rate_b,
                    'message': f"STOP tranche cote {cote_bracket} — erreur {error_rate_b:.0%}"
                }
        
        # Erreurs consécutives
        if self.consecutive_errors >= 3:
            alerts['pause_required'] = {
                'consecutive': self.consecutive_errors,
                'message': f"PAUSE OBLIGATOIRE — {self.consecutive_errors} erreurs consécutives"
            }
        
        # Error rate après win
        if len(self.errors_after_win) >= 5:
            er_win = 1 - sum(self.errors_after_win[-10:]) / min(len(self.errors_after_win), 10)
            if er_win > 0.65:
                alerts['caution_after_win'] = {
                    'rate': er_win,
                    'message': "ATTENTION: taux d'erreur élevé après win"
                }
        
        return alerts
    
    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────
    def _extract_lesson(self, error_type: str, ctx: Dict) -> str:
        lessons = {
            'WRONG_DIRECTION':       "Signaux engine non pris en compte — renforcer cycle/distribution",
            'OVERCONFIDENT':         "Confiance excessivement haute — élever seuil confidence + Conformal",
            'WRONG_VALUE':           "EV mal estimé — recalibrer Shin + calibration tracker",
            'PATTERN_MISREAD':       "Pattern Fourier/Symbolique mal interprété — augmenter poids Fourier",
            'WRONG_REGIME':          "Pari en régime CHAOTIC interdit — vérifier régime AVANT",
            'STREAK_UNDERESTIMATED': "Streak trop long ignoré — augmenter max_streak_ref",
            'CHANGEPOINT_MISSED':    "Changepoint non détecté — augmenter sensibilité BOCPD",
            'CONFORMAL_IGNORED':     "Intervalle conformal > 0.35 ignoré — NO BET toujours",
            'MOVEMENT_IGNORED':      "Mouvement cotes ignoré — augmenter poids odds_movement",
            'TIMING_MISSED':         f"Inference trop lente ({ctx.get('inference_time', 0):.1f}s) — optimiser goulot",
        }
        return lessons.get(error_type, "Erreur non classifiée")
    
    def _get_severity(self, error_type: str) -> str:
        high = {'CONFORMAL_IGNORED', 'WRONG_REGIME', 'CHANGEPOINT_MISSED', 'TIMING_MISSED'}
        return 'HIGH' if error_type in high else 'MEDIUM'
    
    def _identify_responsible_features(self, error_type: str, ctx: Dict) -> List[str]:
        mapping = {
            'WRONG_DIRECTION':       ['distribution_deviation', 'cycle_overdue_score'],
            'OVERCONFIDENT':         ['confidence', 'entropy', 'conformal_width'],
            'PATTERN_MISREAD':       ['fourier_cycle_signal', 'symbolic_pattern_lift'],
            'STREAK_UNDERESTIMATED': ['streak_correction'],
            'CHANGEPOINT_MISSED':    ['bocpd_changepoint'],
            'MOVEMENT_IGNORED':      ['odds_movement_signal'],
            'TIMING_MISSED':         ['inference_time'],
        }
        return mapping.get(error_type, ['unknown'])
    
    def _compute_correction(self, error_type: str, ctx: Dict) -> Dict:
        """Calcule la correction à appliquer et retourne un résumé."""
        corrections = {
            'error_type': error_type,
            'immediate_action': ERROR_TYPE_NAMES.get(ERROR_TYPES.get(error_type, 0), error_type),
        }
        
        if error_type == 'WRONG_DIRECTION' and ctx.get('actual') == 'N':
            corrections['next_bet_bias'] = 'Chercher DRAW fort (draw_overdue)'
        elif error_type == 'WRONG_DIRECTION':
            corrections['next_bet_bias'] = f"Chercher {ctx.get('actual')} fort"
        
        return corrections
    
    def to_dict(self) -> Dict:
        """Status résumé pour API."""
        return {
            'recovery_mode': self.recovery_mode,
            'recovery_wins_needed': self.recovery_wins_needed,
            'recovery_wins_count': self.recovery_wins_count,
            'consecutive_errors': self.consecutive_errors,
            'active_lessons': self.get_active_lessons(),
            'weight_ml': self.weights.get('ml_weight', 1.0),
            'weight_engine': self.weights.get('engine_signal_weight', 1.0),
            'weight_draw_cycle': self.weights.get('draw_cycle_weight', 1.0),
            'weight_fourier': self.weights.get('fourier_weight', 1.0),
            'weight_movement': self.weights.get('odds_movement_weight', 1.0),
            'thresholds': self.get_thresholds(),
        }
    
    def save(self, path: str):
        """Persist error system state."""
        data = {
            'weights': self.weights,
            'recovery_mode': self.recovery_mode,
            'recovery_wins_count': self.recovery_wins_count,
            'consecutive_errors': self.consecutive_errors,
            'max_streaks_hist': self.max_streaks_hist,
            'error_log': self.error_log[-100:],  # Keep last 100
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, path: str):
        """Load error system state."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.weights.update(data.get('weights', {}))
            self.recovery_mode = data.get('recovery_mode', False)
            self.recovery_wins_count = data.get('recovery_wins_count', 0)
            self.consecutive_errors = data.get('consecutive_errors', 0)
            self.max_streaks_hist = data.get('max_streaks_hist', {'V': 5, 'N': 4, 'D': 4})
            self.error_log = data.get('error_log', [])
            logger.info("Error autopsy system loaded")
        except Exception as e:
            logger.warning(f"Could not load error system: {e}")
