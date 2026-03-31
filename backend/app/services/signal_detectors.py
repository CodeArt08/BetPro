"""
Signal Detectors — M1 to M15 (Engine statistique côté background).
Tous précalculés en Thread B, lecture < 1ms en cache RAM.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger
import math


# ─────────────────────────────────────────────────────────────
# M16 — DRAW DETECTION SPECIALIST (NOUVEAU)
# ─────────────────────────────────────────────────────────────
def compute_draw_signals(results: List[str], odds_implied: Dict[str, float]) -> Dict:
    """
    Détecteur spécialisé pour les matchs nuls.
    Combine plusieurs indicateurs pour identifier les situations de nul probables.
    """
    if not results:
        return {'draw_signal_strength': 0.0, 'draw_factors': {}}
    
    signals = {}
    
    # 1. Draw rate analysis (fenêtres glissantes)
    draw_rates = {}
    for window in [5, 10, 15, 20]:
        window_results = results[-window:] if len(results) >= window else results
        n = len(window_results)
        if n > 0:
            draw_rates[f'rate_{window}'] = window_results.count('N') / n
    
    # 2. Low-scoring pattern detection
    # Les nuls sont plus fréquents après des séries de faibles scores
    recent_draws = 0
    low_score_indicators = 0
    for i, result in enumerate(reversed(results[-15:])):
        if result == 'N':
            recent_draws += 1
            # Si les 2-3 derniers matchs avaient peu de buts, probabilité de nul augmente
            if i < 5:  # Très récent
                low_score_indicators += 1
    
    # 3. Odds-based draw opportunity
    draw_odds = odds_implied.get('N', 0.28)
    draw_value = draw_odds - 0.28  # Écart vs baseline
    
    # 4. Alternating pattern (V-D-V-D ou D-V-D-V)
    alternating_score = 0
    if len(results) >= 4:
        last_4 = results[-4:]
        patterns = [
            ['V', 'N', 'V', 'N'],
            ['N', 'V', 'N', 'V'],
            ['V', 'D', 'V'],  # V-D-V pattern
            ['D', 'V', 'D']   # D-V-D pattern
        ]
        for pattern in patterns:
            if last_4 == pattern:
                alternating_score += 0.3
                break
    
    # 5. Compression des cotes (V et D proches, N plus élevé)
    home_odds = odds_implied.get('V', 0.45)
    away_odds = odds_implied.get('D', 0.27)
    compression_score = 0
    if abs(home_odds - away_odds) < 0.15 and draw_odds > 0.30:
        compression_score = 0.25
    
    # 5. Calculate final draw signal strength (REDUCED thresholds)
    draw_strength = (
        min(draw_rates.get('rate_10', 0) * 1.5, 0.25) +  # REDUCED from 2.0 to 1.5
        min(recent_draws * 0.08, 0.15) +              # REDUCED from 0.1 to 0.08
        min(low_score_indicators * 0.03, 0.10) +      # REDUCED from 0.05 to 0.03
        min(max(draw_value, 0) * 0.3, 0.20) +        # REDUCED from 0.5 to 0.3
        alternating_score +                              # KEPT at 0.3
        compression_score                               # KEPT at 0.25
    )
    
    # Normalize to 0-1 range
    draw_strength = min(draw_strength, 1.0)
    
    signals = {
        'draw_signal_strength': draw_strength,
        'draw_factors': {
            'recent_draw_rate_10': draw_rates.get('rate_10', 0),
            'recent_draws_count': recent_draws,
            'draw_odds_value': draw_value,
            'alternating_pattern': alternating_score > 0,
            'compression_detected': compression_score > 0,
            'low_score_indicators': low_score_indicators
        },
        'draw_recommendation': 'HIGH' if draw_strength > 0.75 else 'MEDIUM' if draw_strength > 0.4 else 'LOW'  # RAISED threshold from 0.6 to 0.75
    }
    
    return signals


# ─────────────────────────────────────────────────────────────
# M17b — AWAY WIN DETECTION (NOUVEAU)
# ─────────────────────────────────────────────────────
def compute_away_win_signals(results: List[str], odds_implied: Dict[str, float]) -> Dict:
    """
    Détecteur spécialisé pour les victoires extérieur (D).
    Combine plusieurs indicateurs pour identifier les situations où l'extérieur est favorisé.
    """
    if not results:
        return {'away_signal_strength': 0.0, 'away_factors': {}}
    
    signals = {}
    
    # 1. Away win rate analysis (fenêtres glissantes)
    away_rates = {}
    for window in [5, 10, 15, 20]:
        window_results = results[-window:] if len(results) >= window else results
        n = len(window_results)
        if n > 0:
            away_rates[f'rate_{window}'] = window_results.count('D') / n
    
    # 2. Recent away wins momentum
    recent_away_wins = 0
    for i, result in enumerate(reversed(results[-10:])):
        if result == 'D':
            recent_away_wins += 1
    
    # 3. Odds-based away opportunity
    away_odds = odds_implied.get('D', 0.27)
    home_odds = odds_implied.get('V', 0.45)
    away_value = away_odds - 0.27  # Écart vs baseline
    
    # 4. Underdog pattern (away team favored by odds)
    underdog_score = 0
    if away_odds > home_odds:
        # L'extérieur est outsider mais a une probabilité décemment élevée
        underdog_score = min((away_odds - home_odds) * 0.5, 0.30)
    
    # 5. Sequence pattern: D après V-V ou V-N (correction)
    sequence_boost = 0
    if len(results) >= 3:
        last_3 = results[-3:]
        # Patterns favorisant D
        d_patterns = [
            ['V', 'V', 'V'],  # Trop de V → correction possible
            ['V', 'N', 'V'],  # Alternance sans D
            ['N', 'V', 'N'],  # Nuls sans D
        ]
        for pattern in d_patterns:
            if last_3 == pattern:
                sequence_boost = 0.20
                break
    
    # 6. Calculate final away signal strength
    away_strength = (
        min(away_rates.get('rate_10', 0) * 2.0, 0.25) +
        min(recent_away_wins * 0.10, 0.20) +
        min(max(away_value, 0) * 0.4, 0.25) +
        underdog_score +
        sequence_boost
    )
    
    # Normalize to 0-1 range
    away_strength = min(away_strength, 1.0)
    
    signals = {
        'away_signal_strength': away_strength,
        'away_factors': {
            'recent_away_rate_10': away_rates.get('rate_10', 0),
            'recent_away_wins_count': recent_away_wins,
            'away_odds_value': away_value,
            'underdog_situation': underdog_score > 0,
            'sequence_pattern': sequence_boost > 0
        },
        'away_recommendation': 'HIGH' if away_strength > 0.65 else 'MEDIUM' if away_strength > 0.35 else 'LOW'
    }
    
    return signals


# ─────────────────────────────────────────────────────────────
# M17 — GOAL EXPECTATION DETECTOR (NOUVEAU)
# ─────────────────────────────────────────────────────
def compute_goal_expectation_signals(results: List[str], odds_implied: Dict[str, float]) -> Dict:
    """
    Détecte les matchs à faible attente de buts (propices aux nuls).
    """
    if not results:
        return {'low_goal_expectation': 0.0}
    
    # Analyse des cotes pour détecter les matchs à faible buts attendus
    home_odds = odds_implied.get('V', 0.45)
    draw_odds = odds_implied.get('N', 0.28)
    away_odds = odds_implied.get('D', 0.27)
    
    # Plus les cotes sont serrées, plus la probabilité de nul augmente
    odds_spread = max(home_odds, away_odds) - min(home_odds, away_odds)
    draw_favorability = draw_odds / (home_odds + away_odds)
    
    # Score de faible attente de buts
    low_goal_score = 0
    if odds_spread < 0.20:  # Cotes très serrées
        low_goal_score += 0.3
    if draw_favorability > 0.35:  # Nul relativement favorisé
        low_goal_score += 0.3
    if draw_odds > 0.32:  # Cote du nul élevée
        low_goal_score += 0.2
    
    low_goal_score = min(low_goal_score, 1.0)
    
    return {
        'low_goal_expectation': low_goal_score,
        'odds_spread': odds_spread,
        'draw_favorability': draw_favorability,
        'draw_likely': low_goal_score > 0.5
    }


# ─────────────────────────────────────────────────────────────
# M1 — Distribution Analysis (fenêtres glissantes 10/20/50/100)
# ─────────────────────────────────────────────────────────────
def compute_distribution_signals(results: List[str], odds_implied: Dict[str, float]) -> Dict:
    """
    Compare distribution observée vs théorique (cotes).
    results = derniers N résultats ['V','N','D',...]
    odds_implied = {'V': implied_prob, 'N': ..., 'D': ...}
    """
    signals = {}
    for window in [10, 20, 50, 100]:
        window_results = results[-window:] if len(results) >= window else results
        n = len(window_results)
        if n == 0:
            signals[f'dist_{window}'] = {'V': 0, 'N': 0, 'D': 0, 'signal': None}
            continue
        
        observed = {t: window_results.count(t) / n for t in ['V', 'N', 'D']}
        deviation = {t: observed.get(t, 0) - odds_implied.get(t, 0.33) for t in ['V', 'N', 'D']}
        
        # KL divergence
        kl = sum(
            observed.get(t, 1e-9) * math.log(observed.get(t, 1e-9) / max(odds_implied.get(t, 0.33), 1e-9))
            for t in ['V', 'N', 'D']
        )
        
        threshold = 0.08
        signal = {t: dev for t, dev in deviation.items() if abs(dev) > threshold}
        signals[f'dist_{window}'] = {
            'observed': observed,
            'deviation': deviation,
            'kl': kl,
            'signal': signal if signal else None
        }
    return signals


# ─────────────────────────────────────────────────────────────────────
# M2 — Cycle Detection: overdue / saturated
# ─────────────────────────────────────────────────────────────────────
# Probabilités de base (BASELINE) ajustées pour réduire le biais 'V'
BASELINE = {'V': 0.40, 'N': 0.30, 'D': 0.30}

def compute_cycle_signals(results: List[str]) -> Dict:
    """Détecte si un type de résultat est overdue ou saturé."""
    last_10 = results[-10:] if len(results) >= 10 else results
    n10 = len(last_10)
    
    signals = {}
    for t in ['V', 'N', 'D']:
        rate_10 = last_10.count(t) / n10 if n10 > 0 else BASELINE[t]
        baseline = BASELINE[t]
        overdue_score = max(0, (baseline - rate_10) / baseline)
        saturated = rate_10 > baseline * 1.6
        
        signals[t] = {
            'rate_10': rate_10,
            'baseline': baseline,
            'overdue_score': overdue_score,
            'overdue': overdue_score > 0.40,
            'very_overdue': overdue_score > 0.60,
            'saturated': saturated,
        }
    return signals


# ─────────────────────────────────────────────────────────────────────
# M3 — Streak Pattern Detection
# ─────────────────────────────────────────────────────────────────────
def compute_streak_signals(results: List[str], max_streaks_hist: Dict[str, int]) -> Dict:
    """Détecte la longueur de streak courante et probabilité de correction."""
    if not results:
        return {}
    
    signals = {}
    for t in ['V', 'N', 'D']:
        streak = 0
        for r in reversed(results):
            if r == t:
                streak += 1
            else:
                break
        max_hist = max_streaks_hist.get(t, max(streak, 1))
        correction_prob = streak / max(max_hist, 1)
        signals[t] = {
            'current_streak': streak,
            'max_historical': max_hist,
            'correction_prob': correction_prob,
            'correction_imminent': correction_prob > 0.80,
        }
        # Update max historical streak
        if streak > max_hist:
            max_streaks_hist[t] = streak
    return signals


# ─────────────────────────────────────────────────────────────────────
# M4 — Autocorrélation Séquences (lag 1–5)
# ─────────────────────────────────────────────────────────────────────
def compute_autocorrelation(results: List[str], max_lag: int = 5) -> Dict:
    """
    Encode H=2, D=1, A=0 et calcule l'autocorrélation rolling 50 derniers.
    """
    encode = {'V': 2, 'N': 1, 'D': 0}
    series = [encode.get(r, 1) for r in results[-50:]]
    
    signals = {}
    for lag in range(1, max_lag + 1):
        if len(series) <= lag:
            signals[f'lag{lag}'] = {'autocorr': 0.0, 'pattern': 'unknown'}
            continue
        s1 = np.array(series[:-lag], dtype=float)
        s2 = np.array(series[lag:], dtype=float)
        if s1.std() == 0 or s2.std() == 0:
            ac = 0.0
        else:
            ac = float(np.corrcoef(s1, s2)[0, 1])
        
        pattern = 'random'
        if abs(ac) > 0.15:
            pattern = 'continuation' if ac > 0 else 'alternance'
        signals[f'lag{lag}'] = {'autocorr': ac, 'pattern': pattern}
    return signals


# ─────────────────────────────────────────────────────────────────────
# M5 — Fourier Transform (FFT cycles)
# ─────────────────────────────────────────────────────────────────────
def compute_fourier_signals(results: List[str], min_matches: int = 30) -> Dict:
    """
    FFT sur derniers 100 matchs. Returns dominant cycle + phase courante.
    """
    encode = {'V': 2, 'N': 1, 'D': 0}
    series = np.array([encode.get(r, 1) for r in results[-100:]], dtype=float)
    
    if len(series) < min_matches:
        return {'cycle_detected': False, 'cycle_length': None, 'phase': None, 'dominant_type': None}
    
    # Detrend
    series = series - series.mean()
    
    fft_result = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series))
    amplitudes = np.abs(fft_result)
    
    # Exclude DC component
    amplitudes[0] = 0
    
    dominant_idx = np.argmax(amplitudes)
    dominant_freq = freqs[dominant_idx]
    dominant_amplitude = float(amplitudes[dominant_idx])
    
    # Threshold: amplitude > mean + 2*std suggests real cycle
    amp_threshold = amplitudes[1:].mean() + 2 * amplitudes[1:].std()
    cycle_detected = bool(dominant_amplitude > amp_threshold and dominant_freq > 0)
    
    cycle_length = int(round(1 / dominant_freq)) if (cycle_detected and dominant_freq > 0) else None
    
    # Phase: position dans cycle
    phase = None
    if cycle_length and cycle_length > 1:
        phase = len(series) % cycle_length
    
    # Dominant outcome type at current phase
    dominant_type = None
    if cycle_length and cycle_length > 0 and len(results) >= cycle_length:
        # Look at results at same phase historically
        phase_results = [results[-(i * cycle_length + (cycle_length - phase))]
                         for i in range(1, 6)
                         if i * cycle_length + (cycle_length - phase) <= len(results)]
        if phase_results:
            counts = {t: phase_results.count(t) for t in ['V', 'N', 'D']}
            dominant_type = max(counts, key=counts.get)
    
    return {
        'cycle_detected': cycle_detected,
        'cycle_length': cycle_length,
        'phase': phase,
        'dominant_freq': float(dominant_freq),
        'dominant_amplitude': dominant_amplitude,
        'dominant_type': dominant_type,
    }


# ─────────────────────────────────────────────────────────────────────
# M6 — BOCPD Changepoint Detection (simplified online Bayesian)
# ─────────────────────────────────────────────────────────────────────
class BOCPDDetector:
    """Online Bayesian Online Changepoint Detection."""
    
    def __init__(self, hazard: float = 0.01, alpha: float = 1.0, beta: float = 1.0):
        self.hazard = hazard  # Prior probability of changepoint per timestep
        self.alpha = alpha
        self.beta = beta
        self.R = np.array([1.0])  # Run length probabilities
        self.changepoints: List[Dict] = []
        self.t = 0
    
    def update(self, x: float) -> Dict:
        """Update with new observation. Returns changepoint signal."""
        self.t += 1
        H = self.hazard
        
        # Predictive probabilities under Student-t with params
        alphas = self.alpha + np.arange(len(self.R)) / 2
        betas = self.beta + np.arange(len(self.R)) * (x - 0.5) ** 2 / 2
        
        # Predictive pdf (simplified scalar)
        pred = np.ones_like(self.R) * (1 / 3.0)  # Uniform for categorical simplification
        
        # Update
        R_new = np.zeros(len(self.R) + 1)
        R_new[0] = np.sum(self.R * H)           # changepoint (run length resets to 0)
        R_new[1:] = self.R * (1 - H) * pred
        
        # Normalize
        total = R_new.sum()
        if total > 0:
            R_new /= total
        
        self.R = R_new
        
        # Detect changepoint: if R[0] (run=0) > threshold
        cp_prob = float(self.R[0])
        changepoint_detected = cp_prob > 0.15
        
        if changepoint_detected:
            self.changepoints.append({'t': self.t, 'prob': cp_prob})
        
        return {
            'changepoint_detected': changepoint_detected,
            'changepoint_prob': cp_prob,
            'most_likely_run_length': int(np.argmax(self.R)),
        }
    
    def recent_changepoint(self, within_n: int = 30) -> Optional[Dict]:
        """Return the most recent changepoint within N steps, or None."""
        for cp in reversed(self.changepoints):
            if self.t - cp['t'] <= within_n:
                return cp
        return None


# ─────────────────────────────────────────────────────────────────────
# M7 — Runs Test de Wald-Wolfowitz
# ─────────────────────────────────────────────────────────────────────
def compute_runs_test(results: List[str]) -> Dict:
    """
    Test de Wald-Wolfowitz sur rolling 50 matchs.
    Retourne z_score, p_value, et interprétation.
    """
    window = results[-50:] if len(results) >= 50 else results
    n = len(window)
    if n < 10:
        return {'z_score': 0.0, 'p_value': 1.0, 'random': True}
    
    # Encode: non-draw vs draw (binaire simple)
    binary = [1 if r != 'N' else 0 for r in window]
    
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            runs += 1
    
    n1 = binary.count(1)
    n2 = binary.count(0)
    
    if n1 == 0 or n2 == 0:
        return {'z_score': 0.0, 'p_value': 1.0, 'random': True}
    
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    sigma_sq = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    sigma = math.sqrt(max(sigma_sq, 1e-9))
    
    z = (runs - mu) / sigma
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    
    return {
        'z_score': float(z),
        'p_value': p_value,
        'runs': runs,
        'expected_runs': float(mu),
        'random': p_value > 0.20,
        'exploitable': p_value < 0.05,
        'reduce_stakes': p_value > 0.30,
    }


# ─────────────────────────────────────────────────────────────────────
# M8 — Symbolic Sequence Mining (simplified PrefixSpan-like)
# ─────────────────────────────────────────────────────────────────────
def compute_symbolic_patterns(results: List[str],
                               min_length: int = 3, max_length: int = 5,
                               min_lift: float = 1.30) -> Dict:
    """
    Cherche patterns fréquents de longueur 3-5.
    Lift = observed_freq / expected_freq.
    """
    n = len(results)
    if n < 20:
        return {'top_patterns': [], 'exploitable': []}
    
    # Expected freq for uniform random
    base_prob = {'V': BASELINE['V'], 'N': BASELINE['N'], 'D': BASELINE['D']}
    
    pattern_stats = []
    
    for length in range(min_length, max_length + 1):
        counts: Dict[Tuple, int] = defaultdict(int)
        next_after: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for i in range(n - length):
            pat = tuple(results[i:i + length])
            next_r = results[i + length]
            counts[pat] += 1
            next_after[pat][next_r] += 1
        
        total_patterns = sum(counts.values())
        
        for pat, cnt in counts.items():
            if cnt < 3:
                continue
            observed_freq = cnt / max(total_patterns, 1)
            expected_freq = math.prod(base_prob.get(s, 0.33) for s in pat)
            lift = observed_freq / max(expected_freq, 1e-9)
            
            # Next result prediction
            next_counts = next_after[pat]
            total_next = sum(next_counts.values())
            if total_next > 0:
                next_probs = {t: next_counts.get(t, 0) / total_next for t in ['V', 'N', 'D']}
                next_type = max(next_probs, key=next_probs.get)
                next_conf = next_probs[next_type]
            else:
                next_probs = {'V': 0.33, 'N': 0.33, 'D': 0.33}
                next_type = 'N'
                next_conf = 0.33
            
            pattern_stats.append({
                'pattern': ''.join(pat),
                'length': length,
                'count': cnt,
                'observed_freq': observed_freq,
                'expected_freq': expected_freq,
                'lift': lift,
                'next_type': next_type,
                'next_conf': next_conf,
                'next_probs': next_probs,
                'exploitable': lift >= min_lift,
            })
    
    # Sort by lift
    pattern_stats.sort(key=lambda x: -x['lift'])
    exploitable = [p for p in pattern_stats if p['exploitable']]
    
    return {
        'top_patterns': pattern_stats[:10],
        'exploitable': exploitable[:5],
        'best_pattern': pattern_stats[0] if pattern_stats else None,
    }


def get_current_pattern(results: List[str], max_length: int = 5) -> str:
    """Renvoie le pattern courant (derniers N résultats)."""
    return ''.join(results[-max_length:]) if len(results) >= max_length else ''.join(results)


# ─────────────────────────────────────────────────────────────────────
# M9 — Line Position Bias
# ─────────────────────────────────────────────────────────────────────
def compute_line_bias(results_by_line: Dict[int, List[str]]) -> Dict[int, Dict]:
    """
    Pour chaque position de ligne, calcule le biais vs global.
    results_by_line = {1: ['V','N',...], 2: [...], ...}
    """
    # Global rates
    all_results = [r for rs in results_by_line.values() for r in rs]
    n_global = len(all_results)
    global_rates = {t: all_results.count(t) / max(n_global, 1) for t in ['V', 'N', 'D']}
    
    line_biases = {}
    for pos, results in results_by_line.items():
        n = len(results)
        if n < 10:
            line_biases[pos] = {'enough_data': False, 'bias': {}}
            continue
        
        rates = {t: results.count(t) / n for t in ['V', 'N', 'D']}
        bias = {t: (rates[t] - global_rates[t]) / max(global_rates[t], 1e-9) for t in ['V', 'N', 'D']}
        strong_bias = {t: b for t, b in bias.items() if abs(b) > 0.25}
        
        line_biases[pos] = {
            'enough_data': n >= 200,
            'reliable': n >= 100,
            'n': n,
            'rates': rates,
            'global_rates': global_rates,
            'bias': bias,
            'strong_bias': strong_bias if strong_bias else None,
        }
    return line_biases


# ─────────────────────────────────────────────────────────────────────
# M10 — Score Distribution Fingerprint
# ─────────────────────────────────────────────────────────────────────
def compute_score_distribution(score_counts: Dict[str, int],
                                lambda_h: float = 1.35,
                                lambda_a: float = 1.10) -> Dict:
    """
    Compare distribution des scores observés vs Poisson.
    score_counts = {'1-0': 45, '0-0': 23, ...}
    """
    from scipy.stats import poisson
    
    total = sum(score_counts.values())
    if total == 0:
        return {'kl_divergence': 0.0, 'biased_scores': []}
    
    biased_scores = []
    kl = 0.0
    
    for score_str, cnt in score_counts.items():
        try:
            h_str, a_str = score_str.split('-')
            h, a = int(h_str), int(a_str)
        except:
            continue
        
        observed = cnt / total
        expected = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
        
        if expected > 0:
            kl += observed * math.log(observed / expected + 1e-9)
        
        bias = (observed - expected) / max(expected, 1e-9)
        if abs(bias) > 0.5 and cnt >= 5:
            biased_scores.append({
                'score': score_str,
                'observed': observed,
                'expected': expected,
                'bias': bias,
            })
    
    biased_scores.sort(key=lambda x: -abs(x['bias']))
    
    return {
        'kl_divergence': float(kl),
        'kl_level': 'FORT' if kl > 0.1 else 'MOYEN' if kl > 0.05 else 'FAIBLE',
        'biased_scores': biased_scores[:10],
        'total_matches': total,
    }


# ─────────────────────────────────────────────────────────────────────
# M11 — Odds Calibration Tracker
# ─────────────────────────────────────────────────────────────────────
def compute_calibration_edge(calibration_data: List[Dict]) -> Dict:
    """
    calibration_data = [{'implied': 0.5, 'result': 'V', 'outcome': 'V'}, ...]
    Compute real win rate vs implied, by bracket.
    """
    brackets = {}
    
    for item in calibration_data:
        implied = item.get('implied', 0.5)
        bracket_key = round(implied * 10) / 10  # Round to nearest 10%
        if bracket_key not in brackets:
            brackets[bracket_key] = {'implied': bracket_key, 'total': 0, 'wins': 0}
        brackets[bracket_key]['total'] += 1
        if item['result'] == item['outcome']:
            brackets[bracket_key]['wins'] += 1
    
    result = {}
    for key, data in brackets.items():
        if data['total'] >= 30:
            real_rate = data['wins'] / data['total']
            edge = real_rate - data['implied']
            result[key] = {
                'implied': data['implied'],
                'real_rate': real_rate,
                'value_edge': edge,
                'structural_value': abs(edge) > 0.05,
                'sample_size': data['total'],
            }
    
    return result


# ─────────────────────────────────────────────────────────────────────
# M12 — Implied Probability Decomposition (Shin)
# ─────────────────────────────────────────────────────────────────────
def compute_shin_probabilities(odds_h: float, odds_d: float, odds_a: float) -> Dict:
    """
    3 méthodes: Additive, Power, Shin.
    """
    if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
        return {'additive': {}, 'power': {}, 'shin': {}, 'divergence_score': 0}
    
    raw = {'V': 1 / odds_h, 'N': 1 / odds_d, 'D': 1 / odds_a}
    bookmaker_margin = sum(raw.values()) - 1
    
    # Additive normalization
    total_raw = sum(raw.values())
    additive = {t: v / total_raw for t, v in raw.items()}
    
    # Power normalization (find k such that sum(p^k) = 1)
    from scipy.optimize import brentq
    def power_eq(k):
        return sum(v ** k for v in raw.values()) - 1.0
    try:
        k_opt = brentq(power_eq, 0.1, 5.0, xtol=1e-6)
        power = {t: v ** k_opt for t, v in raw.items()}
    except:
        power = additive
    
    # Shin method: find z such that probabilities sum to 1
    #   p_i = sqrt(z^2 + 4*(1-z)*q_i^2/Q) / (2*(1-z)/Q) - z/(2*(1-z))
    #   where q_i = implied, Q = sum(q_i)
    from scipy.optimize import brentq
    q = list(raw.values())
    Q = sum(q)
    
    def shin_eq(z):
        if z >= 1 or z <= 0:
            return float('inf')
        s = sum(math.sqrt(z**2 + 4*(1-z)*qi**2/Q) for qi in q)
        return s - (2 - z)
    
    try:
        z_opt = brentq(shin_eq, 0.001, 0.999, xtol=1e-6)
        shin_probs = {}
        for t, qi in zip(['V', 'N', 'D'], q):
            shin_probs[t] = (math.sqrt(z_opt**2 + 4*(1-z_opt)*qi**2/Q) - z_opt) / (2*(1-z_opt))
        # Normalize
        shin_total = sum(shin_probs.values())
        shin = {t: v / shin_total for t, v in shin_probs.items()}
    except:
        shin = additive
    
    # Divergence score (std across 3 methods)
    divergence = float(np.std([
        additive.get('V', 0.33), power.get('V', 0.33), shin.get('V', 0.33)
    ]))
    
    return {
        'additive': additive,
        'power': power,
        'shin': shin,
        'bookmaker_margin': bookmaker_margin,
        'divergence_score': divergence,
        'anomaly': divergence > 0.04,
        'raw_implied': raw,
    }


# ─────────────────────────────────────────────────────────────────────
# M14 — Time-of-Day Bias
# ─────────────────────────────────────────────────────────────────────
def compute_time_bias(results_by_hour: Dict[int, List[str]]) -> Dict[str, Dict]:
    """
    Taux H/D/A par tranche horaire.
    results_by_hour = {hour: [résultats]}
    """
    # Tranches
    tranches = {
        'matin': list(range(6, 12)),
        'apres_midi': list(range(12, 18)),
        'soir': list(range(18, 22)),
        'nuit': list(range(22, 24)) + list(range(0, 6)),
    }
    
    # Global
    all_r = [r for rs in results_by_hour.values() for r in rs]
    n_g = max(len(all_r), 1)
    global_rates = {t: all_r.count(t) / n_g for t in ['V', 'N', 'D']}
    
    biases = {}
    for tranche, hours in tranches.items():
        tranche_results = [r for h, rs in results_by_hour.items() if h in hours for r in rs]
        n = len(tranche_results)
        if n < 10:
            biases[tranche] = {'n': n, 'bias': {}, 'signal': None}
            continue
        rates = {t: tranche_results.count(t) / n for t in ['V', 'N', 'D']}
        bias = {t: rates[t] - global_rates[t] for t in ['V', 'N', 'D']}
        signal = {t: b for t, b in bias.items() if abs(b) > 0.06}
        biases[tranche] = {
            'n': n,
            'rates': rates,
            'bias': bias,
            'signal': signal if signal else None,
        }
    return biases


# ─────────────────────────────────────────────────────────────────────
# M15 — Cross-Match Correlation
# ─────────────────────────────────────────────────────────────────────
def compute_cross_match_correlation(results_by_line: Dict[int, List[str]]) -> Dict:
    """
    Corrélation entre résultats de différentes lignes d'un même jour.
    """
    encode = {'V': 2, 'N': 1, 'D': 0}
    
    line_ids = sorted(results_by_line.keys())
    if len(line_ids) < 2:
        return {'correlation': 0.0, 'dependent': False}
    
    min_len = min(len(results_by_line[l]) for l in line_ids)
    if min_len < 5:
        return {'correlation': 0.0, 'dependent': False}
    
    # Use last min_len results for each line
    vectors = [
        np.array([encode.get(r, 1) for r in results_by_line[l][-min_len:]], dtype=float)
        for l in line_ids
    ]
    
    corr_values = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if vectors[i].std() > 0 and vectors[j].std() > 0:
                c = float(np.corrcoef(vectors[i], vectors[j])[0, 1])
                corr_values.append(c)
    
    mean_corr = float(np.mean(corr_values)) if corr_values else 0.0
    return {
        'correlation': mean_corr,
        'dependent': abs(mean_corr) > 0.10,
        'n_pairs': len(corr_values),
    }


# ─────────────────────────────────────────────────────────────────────
# Engine Signal Composite (Engine Score = somme pondérée tous signaux)
# ─────────────────────────────────────────────────────────────────────
def compute_engine_score(cache: Dict, target_type: str) -> float:
    """
    Calcule le engine_score final pour un type (V/N/D) depuis le cache RAM.
    Temps: ~0.02s (tout lecture cache).
    """
    # Weights from spec (Module 6) - UPDATED to reduce global signal dominance
    w = {
        'distribution_deviation': 0.08,  # Reduced from 0.10
        'cycle_overdue_score':    0.10,  # Reduced from 0.14
        'streak_correction':      0.08,  
        'autocorr_signal':        0.06,  
        'fourier_cycle_signal':   0.07,  # Reduced from 0.08
        'symbolic_pattern_lift':  0.06,  
        'line_bias':              0.08,  # INCREASED from 0.05
        'time_bias':              0.06,  # INCREASED from 0.04
        'cross_match_corr':       0.04,  # INCREASED from 0.03
        'kl_divergence_signal':   0.06,  
        'score_dist_bias':        0.05,  
        'odds_calib_edge':        0.06,  # INCREASED from 0.05
        'draw_signal_strength':   0.10,  # Reduced from 0.12 to balance with away
        'goal_expectation':       0.08,
        'away_signal_strength':   0.10,  # NOUVEAU - pour D (victoire extérieur)
    }
    
    score = 0.0
    
    # 1. Distribution deviation
    dist = cache.get('dist_50', {}).get('deviation', {})
    score += dist.get(target_type, 0) * w['distribution_deviation']
    
    # 2. Cycle overdue score
    cycle = cache.get('cycle', {}).get(target_type, {})
    score += cycle.get('overdue_score', 0) * w['cycle_overdue_score']
    
    # 3. Streak correction (negative for same type)
    streak = cache.get('streak', {}).get(target_type, {})
    # If correction prob high, it signals AGAINST the streak type → positive for others
    cp = streak.get('correction_prob', 0)
    # For the streak type: negative signal; for others: slightly positive
    if streak.get('current_streak', 0) > 2:
        score -= cp * 0.5 * w['streak_correction']
    
    # 4. Autocorr signal
    ac = cache.get('autocorr', {}).get('lag1', {}).get('autocorr', 0)
    # Continuation: positive for same type as last result
    last_result = cache.get('last_result')
    if last_result == target_type:
        score += max(0, ac) * w['autocorr_signal']
    else:
        score += max(0, -ac) * w['autocorr_signal']
    
    # 5. Fourier cycle signal
    fourier = cache.get('fourier', {})
    if fourier.get('dominant_type') == target_type and fourier.get('cycle_detected'):
        score += 0.5 * w['fourier_cycle_signal']
    
    # 6. Symbolic pattern lift
    sym = cache.get('symbolic', {}).get('best_pattern')
    if sym and sym.get('next_type') == target_type:
        lift_bonus = min((sym.get('lift', 1.0) - 1.0) / 2.0, 1.0)
        score += lift_bonus * w['symbolic_pattern_lift']
    
    # 7. Line bias
    line_pos = cache.get('line_position', 1)
    line_biases = cache.get('line_bias', {})
    lb = line_biases.get(line_pos, {}).get('bias', {})
    score += lb.get(target_type, 0) * w['line_bias']
    
    # 8. Time bias
    tb = cache.get('time_bias_current', {}).get('bias', {})
    score += tb.get(target_type, 0) * w['time_bias']
    
    # 9. Cross match correlation (shared signal)
    cross = cache.get('cross_match_corr', {})
    corr = cross.get('correlation', 0)
    if cross.get('dependent') and target_type == cache.get('previous_line_result'):
        score += corr * w['cross_match_corr']
    
    # 10. KL divergence signal
    kl = cache.get('dist_50', {}).get('kl', 0)
    signal_types = cache.get('dist_50', {}).get('signal', {})
    if signal_types and target_type in signal_types:
        score += min(kl, 1.0) * w['kl_divergence_signal']
    
    # 11. Score distribution bias (generic — affects O/U mostly)
    score_dist = cache.get('score_distribution', {})
    score += score_dist.get('kl_divergence', 0) * 0.1 * w['score_dist_bias']
    
    # 12. Odds calibration edge
    odds_bracket = cache.get('odds_bracket_edge', {})
    edge = odds_bracket.get(target_type, {}).get('value_edge', 0)
    score += edge * w['odds_calib_edge']
    
    # 13. Draw Detection Specialist (NOUVEAU)
    draw_signals = cache.get('draw_detection', {})
    draw_strength = draw_signals.get('draw_signal_strength', 0)
    if target_type == 'N':  # Pour les nuls
        score += draw_strength * w['draw_signal_strength']
    # Pour V et D: pas de pénalité, seulement pas de bonus
    
    # 14. Goal Expectation Detector (NOUVEAU)
    goal_signals = cache.get('goal_expectation', {})
    low_goal_exp = goal_signals.get('low_goal_expectation', 0)
    if target_type == 'N' and goal_signals.get('draw_likely', False):
        score += low_goal_exp * w['goal_expectation']
    
    # 15. Away Win Detector (NOUVEAU)
    away_signals = cache.get('away_detection', {})
    away_strength = away_signals.get('away_signal_strength', 0)
    if target_type == 'D':  # Pour les victoires extérieur
        score += away_strength * w.get('away_signal_strength', 0.10)
    
    return float(np.clip(score, -1.0, 1.0))
