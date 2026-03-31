"""
Bankroll V2 — Module 7 de la spécification élite.
Kelly + Variance-based staking + Drawdown curve + Anti-Martingale.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from collections import deque
from pathlib import Path
import json
import math


# ─────────────────────────────────────────────────────────────────────
# Constantes de la spécification
# ─────────────────────────────────────────────────────────────────────
DRAWDOWN_MULTIPLIERS = {
    (0.00, 0.05): 1.00,
    (0.05, 0.08): 0.85,
    (0.08, 0.12): 0.70,
    (0.12, 0.16): 0.55,
    (0.16, 0.20): 0.40,
    (0.20, 1.00): 0.00,  # STOP
}

CONFORMAL_MULTIPLIERS = {
    (0.00, 0.15): 1.20,
    (0.15, 0.22): 1.00,
    (0.22, 0.28): 0.80,
    (0.28, 0.35): 0.60,
    (0.35, 1.00): 0.00,  # NO BET
}

REGIME_MULTIPLIERS = {
    'STABLE':   1.00,
    'VOLATILE': 0.65,
    'CHAOTIC':  0.00,  # NO BET
}

MIN_STAKE_PCT = 0.005
MAX_STAKE_PCT = 0.025


class BankrollManagerV2:
    """
    Bankroll ultra-précise avec:
    - Kelly fractionnel (×0.20)
    - Variance-based sizing
    - Drawdown curve progressive
    - Conformal multiplier
    - Regime multiplier  
    - Anti-Martingale (×max 1.25)
    - Changepoint discount (×0.75)
    """
    
    def __init__(self, initial_bankroll: float = 100_000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        
        # History
        self.bankroll_history: List[Dict] = []
        self.stakes_history: deque = deque(maxlen=100)  # Increased for better chart
        self.pred_variance_history: deque = deque(maxlen=20)
        
        # Season tracking
        self.current_season_id: Optional[int] = None
        self.season_start_bankroll = initial_bankroll
        self.matchday_history: Dict[int, Dict] = {}  # matchday -> stats
        
        # Metrics
        self.total_profit = 0.0
        self.total_stake = 0.0
        self.wins = 0
        self.losses = 0
        
        # Anti-Martingale state
        self.wins_streak = 0
        self.anti_martingale_mult = 1.0
        
        # Season specific totals
        self.season_total_stake = 0.0
        self.season_total_profit = 0.0
        
        # Drawdown state
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0  # Track actual max drawdown
        self.is_stopped = False
        
        # Load persisted state
        self._load_state()
    
    def _load_state(self):
        """Load bankroll state from disk."""
        try:
            state_dir = Path('data/engine_state')
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / 'bankroll_state.json'
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.bankroll = data.get('bankroll', self.initial_bankroll)
                self.peak_bankroll = data.get('peak_bankroll', self.bankroll)
                self.total_profit = data.get('total_profit', 0.0)
                self.total_stake = data.get('total_stake', 0.0)
                self.wins = data.get('wins', 0)
                self.losses = data.get('losses', 0)
                self.wins_streak = data.get('wins_streak', 0)
                self.current_drawdown = data.get('current_drawdown', 0.0)
                self.max_drawdown = data.get('max_drawdown', 0.0)
                self.current_season_id = data.get('current_season_id')
                self.season_start_bankroll = data.get('season_start_bankroll', self.initial_bankroll)
                self.matchday_history = data.get('matchday_history', {})
                if data.get('stakes_history'):
                    self.stakes_history.extend(data['stakes_history'])
                self.season_total_stake = data.get('season_total_stake', 0.0)
                self.season_total_profit = data.get('season_total_profit', 0.0)
                logger.info(f"Bankroll loaded: {self.bankroll:.0f} Ar, Season: {self.current_season_id}")
        except Exception as e:
            logger.warning(f"Could not load bankroll state: {e}")
    
    def _save_state(self):
        """Save bankroll state to disk."""
        try:
            state_dir = Path('data/engine_state')
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / 'bankroll_state.json'
            
            data = {
                'bankroll': self.bankroll,
                'initial_bankroll': self.initial_bankroll,
                'peak_bankroll': self.peak_bankroll,
                'total_profit': self.total_profit,
                'total_stake': self.total_stake,
                'wins': self.wins,
                'losses': self.losses,
                'wins_streak': self.wins_streak,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'current_season_id': self.current_season_id,
                'season_start_bankroll': self.season_start_bankroll,
                'matchday_history': self.matchday_history,
                'stakes_history': list(self.stakes_history)[-50:],  # Keep last 50
                'season_total_stake': self.season_total_stake,
                'season_total_profit': self.season_total_profit,
            }
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save bankroll state: {e}")
    
    def start_new_season(self, season_id: int):
        """Reset bankroll for a new season."""
        self.current_season_id = season_id
        self.season_start_bankroll = self.bankroll
        self.matchday_history = {}
        self.season_total_stake = 0.0
        self.season_total_profit = 0.0
        self._save_state()
        logger.info(f"New season {season_id} started with bankroll: {self.bankroll:.0f} Ar")
    
    def record_matchday_result(self, matchday: int, profit_loss: float, won: bool, stake: float, odds: float):
        """Record result for a specific matchday."""
        if matchday not in self.matchday_history:
            self.matchday_history[matchday] = {
                'profit_loss': 0.0,
                'wins': 0,
                'losses': 0,
                'total_stake': 0.0,
                'bankroll_after': self.bankroll
            }
        
        self.matchday_history[matchday]['profit_loss'] += profit_loss
        self.matchday_history[matchday]['total_stake'] += stake
        if won:
            self.matchday_history[matchday]['wins'] += 1
        else:
            self.matchday_history[matchday]['losses'] += 1
        self.matchday_history[matchday]['bankroll_after'] = self.bankroll
        
        self._save_state()
    
    # ─────────────────────────────────────────────────────────────────
    # KELLY + VARIANCE STAKING
    # ─────────────────────────────────────────────────────────────────
    def compute_kelly(self, prob: float, odds: float) -> float:
        """
        kelly_raw = (P × odds - 1) / (odds - 1)
        kelly_safe = kelly_raw × 0.20
        """
        if odds <= 1.0 or prob <= 0:
            return 0.0
        kelly_raw = (prob * odds - 1) / (odds - 1)
        kelly_safe = max(0.0, kelly_raw) * 0.20
        return float(kelly_safe)
    
    def compute_variance_factor(self) -> float:
        """
        variance_factor = 1 / (1 + 5 × pred_variance)
        pred_variance: variance des scores finaux des 20 derniers matchs.
        """
        if len(self.pred_variance_history) < 3:
            return 1.0
        pred_variance = float(np.var(list(self.pred_variance_history)))
        factor = 1.0 / (1.0 + 5 * pred_variance)
        return float(np.clip(factor, 0.20, 1.0))
    
    def update_pred_variance(self, final_score: float):
        """Appelé en Thread A après chaque résultat."""
        self.pred_variance_history.append(final_score)
    
    def compute_drawdown(self) -> float:
        """Calcule le drawdown courant vs peak."""
        if self.peak_bankroll == 0:
            return 0.0
        self.current_drawdown = max(0, (self.peak_bankroll - self.bankroll) / self.peak_bankroll)
        return self.current_drawdown
    
    def get_drawdown_multiplier(self) -> float:
        """Retourne le multiplicateur selon la courbe drawdown."""
        dd = self.compute_drawdown()
        for (low, high), mult in DRAWDOWN_MULTIPLIERS.items():
            if low <= dd < high:
                return mult
        return 0.0
    
    def get_conformal_multiplier(self, conformal_width: float) -> float:
        """Retourne le multiplicateur selon la largeur conformal."""
        for (low, high), mult in CONFORMAL_MULTIPLIERS.items():
            if low <= conformal_width < high:
                return mult
        return 0.0
    
    def get_regime_multiplier(self, regime: str) -> float:
        """Retourne le multiplicateur selon le régime."""
        return REGIME_MULTIPLIERS.get(regime, 0.65)
    
    def get_anti_martingale_multiplier(self, signal_strength: float) -> float:
        """
        Anti-martingale: max ×1.25, reset après perte.
        """
        if self.wins_streak >= 3 and signal_strength > 0.75:
            self.anti_martingale_mult = min(1.25, 1.0 + self.wins_streak * 0.05)
        else:
            self.anti_martingale_mult = 1.0
        return self.anti_martingale_mult
    
    # ─────────────────────────────────────────────────────────────────
    # STAKE FINAL COMPOSITE
    # ─────────────────────────────────────────────────────────────────
    def compute_stake(self,
                      prob: float,
                      odds: float,
                      conformal_width: float,
                      regime: str = 'STABLE',
                      signal_strength: float = 0.5,
                      changepoint_discount: float = 1.0,
                      correction_factor: float = 1.0) -> Dict:
        """
        Calcule la mise finale composite selon toutes les règles.
        Returns dict avec stake et tous les multiplicateurs.
        """
        if self.is_stopped:
            return {
                'stake': 0,
                'reason': 'STOP COMPLET (drawdown > 20%)',
                'allowed': False,
            }
        
        # Check NO BET conditions
        if conformal_width > 0.35:
            return {'stake': 0, 'reason': 'Conformal width > 0.35 → NO BET', 'allowed': False}
        if regime == 'CHAOTIC':
            return {'stake': 0, 'reason': 'Régime CHAOTIC → NO BET', 'allowed': False}
        
        drawdown_mult = self.get_drawdown_multiplier()
        if drawdown_mult == 0.0:
            self.is_stopped = True
            return {'stake': 0, 'reason': 'Drawdown > 20% → STOP COMPLET', 'allowed': False}
        
        # Calculer Kelly
        kelly_safe = self.compute_kelly(prob, odds)
        
        # Variance
        variance_factor = self.compute_variance_factor()
        
        # Conformal
        conformal_mult = self.get_conformal_multiplier(conformal_width)
        if conformal_mult == 0.0:
            return {'stake': 0, 'reason': 'Conformal → NO BET', 'allowed': False}
        
        # Regime
        regime_mult = self.get_regime_multiplier(regime)
        if regime_mult == 0.0:
            return {'stake': 0, 'reason': f'Régime {regime} → NO BET', 'allowed': False}
        
        # Anti-Martingale
        am_mult = self.get_anti_martingale_multiplier(signal_strength)
        
        # Stake composite
        stake_raw = (
            kelly_safe
            * self.bankroll
            * variance_factor
            * drawdown_mult
            * conformal_mult
            * regime_mult
            * am_mult
            * changepoint_discount
            * correction_factor
        )
        
        # Clip to [0.5%, 2.5%] of bankroll
        min_stake = self.bankroll * MIN_STAKE_PCT
        max_stake = self.bankroll * MAX_STAKE_PCT
        stake_final = float(np.clip(stake_raw, min_stake, max_stake))
        
        # Round to nearest 100 Ar
        stake_final = round(stake_final / 100) * 100
        stake_final = max(stake_final, 500)  # Minimum 500 Ar
        
        return {
            'stake': stake_final,
            'allowed': True,
            'kelly_safe': kelly_safe,
            'kelly_raw_pct': kelly_safe * 100,
            'variance_factor': variance_factor,
            'drawdown_mult': drawdown_mult,
            'conformal_mult': conformal_mult,
            'regime_mult': regime_mult,
            'anti_martingale_mult': am_mult,
            'changepoint_discount': changepoint_discount,
            'correction_factor': correction_factor,
            'stake_pct': stake_final / max(self.bankroll, 1),
            'potential_return': stake_final * odds,
            'potential_profit': stake_final * (odds - 1),
        }
    
    # ─────────────────────────────────────────────────────────────────
    # SETTLE + UPDATE
    # ─────────────────────────────────────────────────────────────────
    def settle_bet(self, stake: float, odds: float, won: bool, matchday: int = None, season_id: int = None):
        """Règle un pari et met à jour la bankroll."""
        self.total_stake += stake
        
        # Check for new season
        if season_id and season_id != self.current_season_id:
            self.start_new_season(season_id)
        
        if won:
            profit = stake * (odds - 1)
            self.bankroll += profit
            self.total_profit += profit
            self.season_total_profit += profit
            self.wins += 1
            self.wins_streak += 1
            self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
            logger.info(f"Pari GAGNÉ: +{profit:.0f} Ar | Bankroll: {self.bankroll:.0f} Ar")
        else:
            self.bankroll -= stake
            self.total_profit -= stake
            self.season_total_profit -= stake
            self.losses += 1
            self.wins_streak = 0
            self.anti_martingale_mult = 1.0
            logger.info(f"Pari PERDU: -{stake:.0f} Ar | Bankroll: {self.bankroll:.0f} Ar")
        
        self.season_total_stake += stake
        
        # Check drawdown
        dd = self.compute_drawdown()
        self.max_drawdown = max(self.max_drawdown, dd)
        
        if dd > 0.20:
            self.is_stopped = True
            logger.critical(f"STOP COMPLET — drawdown {dd:.1%} > 20%")
        
        # Record in history
        self.stakes_history.append({
            'stake': stake, 'won': won, 'odds': odds,
            'bankroll_after': self.bankroll,
            'drawdown': dd,
            'matchday': matchday,
            'season_id': season_id,
        })
        
        # Record by matchday if provided
        if matchday is not None:
            profit_loss = stake * (odds - 1) if won else -stake
            self.record_matchday_result(matchday, profit_loss, won, stake, odds)
        else:
            self._save_state()
    
    # ─────────────────────────────────────────────────────────────────
    # VALIDATION CRITÈRES (Module 7)
    # ─────────────────────────────────────────────────────────────────
    def validate_bet(self, ev_adjusted: float, confidence: float,
                     model_agreement: float, entropy_norm: float,
                     conformal_width: float, diversity_score: float,
                     runs_pvalue: float, engine_anomaly: float,
                     regime: str, odds: float, drawdown: float,
                     consecutive_errors: int, error_rate_hour: float,
                     inference_time: float,
                     recovery_mode: bool = False) -> Tuple[bool, List[str]]:
        """
        Valide TOUS les critères du Module 7 (14 critères).
        Retourne (ok, raisons_rejet).
        """
        thresholds = RECOVERY_THRESHOLDS if recovery_mode else {
            'ev_minimum': 0.10,
            'confidence_min': 0.72,
            'model_agreement_min': 3/5,
            'entropy_max': 0.95,
            'conformal_max': 0.35,
        }
        
        reasons = []
        
        if ev_adjusted < thresholds['ev_minimum']:
            reasons.append(f"EV {ev_adjusted:.2f} < {thresholds['ev_minimum']:.2f}")
        if confidence < thresholds['confidence_min']:
            reasons.append(f"Confidence {confidence:.2f} < {thresholds['confidence_min']:.2f}")
        if model_agreement < thresholds['model_agreement_min']:
            reasons.append(f"Agreement {model_agreement:.2f} < {thresholds['model_agreement_min']:.2f}")
        if entropy_norm > thresholds['entropy_max']:
            reasons.append(f"Entropy {entropy_norm:.2f} > {thresholds['entropy_max']:.2f}")
        if conformal_width > thresholds['conformal_max']:
            reasons.append(f"Conformal {conformal_width:.2f} > {thresholds['conformal_max']:.2f}")
        if diversity_score < 0.25:
            reasons.append(f"Diversity {diversity_score:.2f} < 0.25")
        if runs_pvalue > 0.20:
            reasons.append(f"Runs p={runs_pvalue:.2f} > 0.20")
        if engine_anomaly > 0.65:
            reasons.append(f"Engine anomaly {engine_anomaly:.2f} > 0.65")
        if regime == 'CHAOTIC':
            reasons.append("Régime CHAOTIC")
        if odds < 1.70:
            reasons.append(f"Cote {odds:.2f} < 1.70")
        if drawdown >= 0.20:
            reasons.append(f"Drawdown {drawdown:.1%} ≥ 20%")
        if consecutive_errors >= 3:
            reasons.append(f"{consecutive_errors} erreurs consécutives")
        if error_rate_hour > 0.65:
            reasons.append(f"Error rate heure {error_rate_hour:.1%} > 65%")
        if inference_time > 15:
            reasons.append(f"Inference trop lente {inference_time:.1f}s")
        
        return len(reasons) == 0, reasons
    
    # ─────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        """Retourne les stats complètes de la bankroll."""
        roi = self.total_profit / max(self.total_stake, 1)
        win_rate = self.wins / max(self.wins + self.losses, 1)
        
        # Build chart data from matchday history
        chart_data = []
        running_bankroll = self.season_start_bankroll
        for md in sorted(self.matchday_history.keys()):
            running_bankroll = self.matchday_history[md]['bankroll_after']
            chart_data.append({
                'matchday': md,
                'bankroll': running_bankroll,
                'profit_loss': self.matchday_history[md]['profit_loss'],
                'wins': self.matchday_history[md]['wins'],
                'losses': self.matchday_history[md]['losses'],
            })
        
        return {
            'bankroll': self.bankroll,
            'initial_bankroll': self.initial_bankroll,
            'peak_bankroll': self.peak_bankroll,
            'total_profit': self.total_profit,
            'total_stake': self.total_stake,
            'season_profit': self.season_total_profit,
            'season_stake': self.season_total_stake,
            'roi': roi,
            'roi_pct': roi * 100,
            'season_roi_pct': (self.season_total_profit / max(self.season_total_stake, 1)) * 100,
            'win_rate': win_rate,
            'wins': self.wins,
            'losses': self.losses,
            'current_drawdown': self.current_drawdown,
            'drawdown_pct': self.current_drawdown * 100,
            'max_drawdown': self.max_drawdown,
            'wins_streak': self.wins_streak,
            'anti_martingale_mult': self.anti_martingale_mult,
            'is_stopped': self.is_stopped,
            'variance_factor': self.compute_variance_factor(),
            'history': list(self.stakes_history),
            'chart_data': chart_data,
            'season_id': self.current_season_id,
            'season_start_bankroll': self.season_start_bankroll,
            'matchday_history': self.matchday_history,
        }
    
    def save(self, path: str):
        """Save bankroll state."""
        data = {
            'bankroll': self.bankroll,
            'initial_bankroll': self.initial_bankroll,
            'peak_bankroll': self.peak_bankroll,
            'total_profit': self.total_profit,
            'total_stake': self.total_stake,
            'wins': self.wins,
            'losses': self.losses,
            'wins_streak': self.wins_streak,
            'is_stopped': self.is_stopped,
            'current_drawdown': self.current_drawdown,
            'pred_variance_history': list(self.pred_variance_history),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load bankroll state."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.bankroll = data.get('bankroll', self.initial_bankroll)
            self.peak_bankroll = data.get('peak_bankroll', self.bankroll)
            self.total_profit = data.get('total_profit', 0.0)
            self.total_stake = data.get('total_stake', 0.0)
            self.wins = data.get('wins', 0)
            self.losses = data.get('losses', 0)
            self.wins_streak = data.get('wins_streak', 0)
            self.is_stopped = data.get('is_stopped', False)
            self.current_drawdown = data.get('current_drawdown', 0.0)
            self.pred_variance_history.extend(data.get('pred_variance_history', []))
            logger.info(f"Bankroll loaded: {self.bankroll:.0f} Ar")
        except Exception as e:
            logger.warning(f"Could not load bankroll: {e}")


def get_regime(kl_divergence: float, runs_pvalue: float,
               changepoint_recent: bool, min_matches: int = 50) -> str:
    """
    Détermine le régime actuel: STABLE / VOLATILE / CHAOTIC.
    """
    if min_matches < 50:
        return 'VOLATILE'
    if changepoint_recent:
        return 'CHAOTIC'
    if kl_divergence > 0.15 or runs_pvalue < 0.01:
        return 'CHAOTIC'
    if kl_divergence > 0.05 or runs_pvalue < 0.10:
        return 'VOLATILE'
    return 'STABLE'
