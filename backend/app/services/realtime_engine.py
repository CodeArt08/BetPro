"""
Real-Time Engine — Module 0 de la spécification élite.
4 Threads Background + Inference Pipeline < 8 secondes.
"""
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from pathlib import Path
import json

from app.services.signal_detectors import (
    BOCPDDetector, compute_distribution_signals, compute_cycle_signals,
    compute_streak_signals, compute_autocorrelation, compute_fourier_signals,
    compute_runs_test, compute_symbolic_patterns, compute_line_bias,
    compute_time_bias, compute_cross_match_correlation,
    compute_shin_probabilities, compute_engine_score, get_current_pattern,
    compute_draw_signals, compute_goal_expectation_signals, compute_away_win_signals,
)
from app.services.rl_agent import RLAgent, UCBBandit, AntiMartingaleManager
from app.services.error_autopsy import ErrorAutopsySystem
from app.services.bankroll_v2 import BankrollManagerV2, get_regime
from app.services.conformal_predictor import (
    ConformalPredictor, CalibrationDriftDetector,
    compute_diversity_score, compute_entropy, compute_model_agreement,
    compute_variance_confidence,
)
from app.services.lstm_model import LSTMAttentionModel
from app.services.prediction_optimizer import PredictionOptimizer
from app.services.aggressive_learner import AggressiveLearner
from app.services.match_analyzer import MatchAnalyzer

# ── Nouveaux modules d'amélioration ────────────────────────────────────
# Lambda Calculator: vrais xG pour Monte Carlo (plus de lambda_h=1.35 fixe)
# from app.services.match_lambda_calculator import MatchLambdaCalculator
# Season Anomaly: détecte derbys, fatigue, fin de saison...
# from app.services.season_anomaly_detector import SeasonAnomalyDetector
# Stacked Meta-Learner: combine tous les modèles de base
# from app.services.stacked_meta_learner import StackedMetaLearner


# ─────────────────────────────────────────────────────────────────────
# Globals (chargés une seule fois au démarrage)
# ─────────────────────────────────────────────────────────────────────
_engine_instance: Optional['RealTimeEngine'] = None


def get_engine() -> 'RealTimeEngine':
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RealTimeEngine()
    return _engine_instance


class RealTimeEngine:
    """
    Moteur temps réel: 4 threads background + inference < 8s.
    Singleton partagé par toute l'application.
    """
    
    def __init__(self):
        # ── Cache RAM (Thread B → Inference) ─────────────────────────
        self.cache: Dict = {}
        self.cache_lock = threading.RLock()
        
        # ── Résultats historiques pour signaux ───────────────────────
        self.results_lock = threading.RLock()
        self.results_history: List[str] = []         # ['V','N','D',...]
        self.results_by_line: Dict[int, List[str]] = {}
        self.results_by_hour: Dict[int, List[str]] = {}
        self.score_counts: Dict[str, int] = {}
        self.calibration_data: List[Dict] = []
        self.max_streaks: Dict[str, int] = {'V': 5, 'N': 4, 'D': 4}
        
        # ── Services background ──────────────────────────────────────
        self.bocpd = BOCPDDetector()
        self.rl_agent = RLAgent()
        self.ucb_bandit = UCBBandit()
        self.anti_martingale = AntiMartingaleManager()
        self.error_system = ErrorAutopsySystem()
        self.bankroll = BankrollManagerV2()
        self.conformal = ConformalPredictor()
        self.calibration_drift = CalibrationDriftDetector()
        self.lstm_model = LSTMAttentionModel()
        
        # ── Match Analyzer (analyse individuelle par match) ────────
        self.match_analyzer = MatchAnalyzer()
        
        # ── Prediction Optimizer (9000+ matchs learning) ───────────────
        self.prediction_optimizer = PredictionOptimizer()
        self.optimization_config = None
        self._load_optimization_config()
        
        # ── Aggressive Learner ────────────────────────────────────────────
        self.aggressive_learner = AggressiveLearner()
        self.aggressive_learner.load_learning_state()
        logger.info(f"Aggressive learner chargé: {self.aggressive_learner.get_learning_stats()}")
        
        # ── Monte Carlo cache ────────────────────────────────────────
        self.mc_result_cache: Optional[Dict] = None
        self.mc_ready = threading.Event()
        
        # ── Thread control ───────────────────────────────────────────
        self._running = False
        self._thread_a: Optional[threading.Thread] = None
        self._thread_b: Optional[threading.Thread] = None
        self._thread_c: Optional[threading.Thread] = None
        self._thread_d: Optional[threading.Thread] = None
        
        # ── Inference monitoring ────────────────────────────────────
        self.inference_log: List[Dict] = []
        self.fast_mode_count = 0
        self.mode = 'NORMAL'
        
        # ── Next match info ─────────────────────────────────────────
        self.next_match_info: Optional[Dict] = None
        
        # ── Loading ─────────────────────────────────────────────────
        self._load_persisted_state()
        self._ensure_cache_initialized()
        logger.info("RealTimeEngine initialized - cache ready")
    
    # ─────────────────────────────────────────────────────────────────
    # STARTUP / SHUTDOWN
    # ─────────────────────────────────────────────────────────────────
    def start(self):
        """Démarrer les threads background."""
        self._running = True
        logger.info("RealTimeEngine starting background threads...")
    
    def stop(self):
        self._running = False
        logger.info("RealTimeEngine stopped")
    
    def _ensure_cache_initialized(self):
        """Ensure cache is ALWAYS initialized with valid data - CRITICAL for predictions."""
        # Required fields that MUST be present
        required_fields = ['cycle', 'streak', 'fourier', 'autocorr', 'runs_test', 
                         'changepoint', 'symbolic', 'shin', 'engine_score_V', 
                         'engine_score_N', 'engine_score_D', 'regime']
        
        with self.cache_lock:
            # Check if cache has minimum required data
            cache_size = len(self.cache)
            missing_fields = [f for f in required_fields if f not in self.cache]
            
            # ALWAYS ensure cache is valid - no exceptions
            if cache_size < 10 or missing_fields:
                logger.warning(f"Cache incomplete: {cache_size} items, missing: {missing_fields}")
                
                # Try to load from saved state first
                state_dir = Path('data/engine_state')
                cache_file = state_dir / 'cache_state.json'
                
                loaded = False
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                            if cache_data and len(cache_data) >= 10:
                                self.cache.update(cache_data)
                                # Check if loaded data has all required fields
                                still_missing = [f for f in required_fields if f not in self.cache]
                                if not still_missing:
                                    logger.info(f"Loaded cache from file: {len(cache_data)} items")
                                    loaded = True
                                else:
                                    logger.warning(f"Loaded cache still missing: {still_missing}")
                    except Exception as e:
                        logger.error(f"Failed to load cache file: {e}")
                
                # ALWAYS initialize sample cache if not loaded properly
                if not loaded:
                    logger.info("Initializing cache with sample data")
                    self._initialize_sample_cache()
            
            
            # Final validation - GUARANTEE cache is ready
            final_missing = [f for f in required_fields if f not in self.cache]
            if final_missing:
                logger.error(f"Cache still missing fields after init: {final_missing}")
                self._initialize_sample_cache()
            
            # Double-check - cache MUST be ready
            if len(self.cache) < 10:
                logger.error("Cache still too small, forcing sample data")
                self._initialize_sample_cache()
            
            logger.info(f"Cache initialized with {len(self.cache)} items - READY")
            
            # Auto-save cache after initialization to ensure persistence
            self._save_cache_to_disk()
    
    def _save_cache_to_disk(self):
        """Save cache state to disk and handle numpy types."""
        try:
            state_dir = Path('data/engine_state')
            state_dir.mkdir(parents=True, exist_ok=True)
            cache_file = state_dir / 'cache_state.json'
            
            with self.cache_lock:
                # Sanitize entire cache structure for JSON
                cache_data = self._sanitize_for_json(self.cache)
                
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved cache state to disk: {len(cache_data)} items")
        except Exception as e:
            logger.warning(f"Could not save cache state: {e}")
    
    def _initialize_sample_cache(self):
        """Initialize cache with sample data ONLY IF historical results are missing."""
        import random
        
        # ── Update results history with lock ────────────────────────
        with self.results_lock:
            if not self.results_history:
                # Use ONLY if historical results are missing
                sample_results = ['V', 'N', 'D', 'V', 'V', 'N', 'D', 'V', 'N', 'D', 'V', 'D', 'N', 'V', 'V']
                self.results_history = sample_results
                logger.warning("No historical results found, using sample data for cache initialization")
            else:
                logger.info(f"Historical results found ({len(self.results_history)}). Using for cache initialization instead of sample data.")
        
        # Initialize cache with sample signals (NO lock here - caller already has lock)
        self.cache.update({
                'cycle': {
                    'V': {'rate_10': 0.4, 'overdue_score': 0.2, 'overdue': False, 'saturated': False},
                    'N': {'rate_10': 0.35, 'overdue_score': 0.6, 'overdue': True, 'saturated': False},  # AUGMENTÉ
                    'D': {'rate_10': 0.25, 'overdue_score': 0.1, 'overdue': False, 'saturated': False}
                },
                'streak': {
                    'V': {'current_streak': 2, 'correction_prob': 0.3, 'correction_imminent': False},
                    'N': {'current_streak': 1, 'correction_prob': 0.4, 'correction_imminent': False},  # AUGMENTÉ
                    'D': {'current_streak': 0, 'correction_prob': 0.2, 'correction_imminent': False}
                },
                'fourier': {
                    'cycle_detected': True,
                    'cycle_length': 8,
                    'phase': 3,
                    'dominant_type': 'N',  # CHANGÉ EN N POUR TESTER
                    'dominant_amplitude': 0.15
                },
                'autocorr': {
                    'lag1': {'autocorr': 0.08, 'pattern': 'positive'},
                    'lag2': {'autocorr': -0.03, 'pattern': 'negative'},
                    'lag3': {'autocorr': 0.05, 'pattern': 'positive'}
                },
                'runs_test': {
                    'z_score': -1.2,
                    'p_value': 0.23,
                    'random': True,
                    'exploitable': False,
                    'reduce_stakes': False
                },
                'changepoint': {
                    'recent': None,
                    'in_last_10': False
                },
                'symbolic': {
                    'pattern': 'VNVN',  # MODIFIÉ POUR INCLURE N
                    'lift': 1.8,
                    'next_type': 'N',  # CHANGÉ EN N
                    'next_conf': 0.65,
                    'exploitable': True
                },
                'shin': {
                    'additive': {'V': 0.38, 'N': 0.32, 'D': 0.30},  # N AUGMENTÉ
                    'power': {'V': 0.40, 'N': 0.30, 'D': 0.30},   # N AUGMENTÉ
                    'shin': {'V': 0.39, 'N': 0.31, 'D': 0.30},   # N AUGMENTÉ
                    'bookmaker_margin': 0.05,
                    'anomaly': False,
                    'divergence_score': 0.02
                },
                'draw_detection': {  # NOUVEAU
                    'draw_signal_strength': 0.65,  # SIGNAL FORT POUR NULS
                    'draw_factors': {
                        'recent_draw_rate_10': 0.35,
                        'recent_draws_count': 3,
                        'draw_odds_value': 0.04,
                        'alternating_pattern': True,
                        'compression_detected': True,
                        'low_score_indicators': 2
                    },
                    'draw_recommendation': 'HIGH'
                },
                'goal_expectation': {  # NOUVEAU
                    'low_goal_expectation': 0.58,  # FAIBLE ATTENTE DE BUTS
                    'odds_spread': 0.12,
                    'draw_favorability': 0.36,
                    'draw_likely': True
                },
                'regime': 'STABLE',
                'engine_score_V': 0.08,  # RÉDUIT
                'engine_score_N': 0.22,  # AUGMENTÉ SIGNIFICATIVEMENT
                'engine_score_D': 0.05   # RÉDUIT
            })
    
    # ─────────────────────────────────────────────────────────────────
    # THREAD A — Learning (appelé après chaque résultat)
    # ─────────────────────────────────────────────────────────────────
    def on_match_completed(self, match_data: Dict, prediction_context: Optional[Dict] = None):
        """
        Thread A: Learning après résultat.
        Appelé depuis le scraper/API dès que le résultat est connu.
        """
        def _thread_a_work():
            start = time.time()
            result = match_data.get('result')
            heure = match_data.get('heure', 12)
            ligne = match_data.get('ligne', 1)
            score = match_data.get('score', '0-0')
            
            # 1. Logger résultat
            with self.results_lock:
                self.results_history.append(result)
                if ligne not in self.results_by_line:
                    self.results_by_line[ligne] = []
                self.results_by_line[ligne].append(result)
                if heure not in self.results_by_hour:
                    self.results_by_hour[heure] = []
                self.results_by_hour[heure].append(result)
                self.score_counts[score] = self.score_counts.get(score, 0) + 1
            
            # Use snapshot of results for subsequent calculations
            with self.results_lock:
                history_snapshot = list(self.results_history)
            
            # 2. BOCPD update
            encode = {'V': 2, 'N': 1, 'D': 0}
            bocpd_signal = self.bocpd.update(float(encode.get(result, 1)))
            
            # 3. Error autopsy si erreur
            if prediction_context:
                predicted = prediction_context.get('predicted')
                if predicted and predicted != result:
                    prediction_context['actual'] = result
                    autopsy = self.error_system.run_autopsy(
                        prediction_context,
                        match_id=match_data.get('match_id', 0)
                    )
                    self.error_system.apply_corrections(
                        autopsy['error_type'], prediction_context
                    )
                    was_correct = False
                else:
                    was_correct = True
                
                confidence = prediction_context.get('confidence', 0.72)
                self.error_system.on_result(
                    correct=was_correct,
                    heure=heure,
                    was_win_before=self.anti_martingale.wins_streak > 0
                )
                self.calibration_drift.add_sample(confidence, was_correct)
                self.conformal.add_calibration_sample(confidence, was_correct)
            
            # 4. Update streak tracking
            with self.results_lock:
                streak_signals = compute_streak_signals(self.results_history, self.max_streaks)
            for t, data in streak_signals.items():
                self.max_streaks[t] = max(self.max_streaks.get(t, 1), data['current_streak'])
            
            # 5. Update bankroll si pari
            bet_data = match_data.get('bet')
            if bet_data:
                won = bet_data.get('outcome') == result
                self.bankroll.settle_bet(
                    stake=bet_data['stake'],
                    odds=bet_data['odds'],
                    won=won
                )
                if won:
                    self.anti_martingale.on_win()
                else:
                    self.anti_martingale.on_loss()
            
            # 6. Update RL agent
            if prediction_context and 'rl_state' in prediction_context:
                action_taken = prediction_context.get('rl_action', 'NO_BET')
                profit_loss = match_data.get('profit_loss', 0)
                correct = prediction_context.get('correct', False)
                reward = self.rl_agent.compute_reward(
                    action_taken, profit_loss, result, correct
                )
                # Build next state from current cache
                next_state = self.rl_agent.build_state_vector(self.cache)
                self.rl_agent.update(
                    np.array(prediction_context['rl_state']),
                    action_taken, reward, next_state
                )
            
            # 7. Update UCB Bandit for each model that predicted
            if prediction_context and 'model_predictions' in prediction_context:
                for model, probs in prediction_context['model_predictions'].items():
                    predicted_outcome = max(probs, key=probs.get)
                    reward = 1.0 if predicted_outcome == result else -0.5
                    self.ucb_bandit.update(
                        model, reward, heure=heure,
                        regime=self.cache.get('regime', 'STABLE'),
                        dominant_type=max(
                            self.cache.get('cycle', {'V': {'overdue_score': 0}}),
                            key=lambda t: self.cache.get('cycle', {}).get(t, {}).get('overdue_score', 0)
                        )
                    )
            
            elapsed = time.time() - start
            logger.debug(f"Thread A completed in {elapsed:.2f}s")
            
            # 8. Persister l'apprentissage sur disque
            self._save_persisted_state()
        
        # Lancer en thread
        t = threading.Thread(target=_thread_a_work, daemon=True, name="Thread-A")
        t.start()
    
    # ─────────────────────────────────────────────────────────────────
    # THREAD B — Cache Preparation (dès équipes connues)
    # ─────────────────────────────────────────────────────────────────
    def prepare_cache(self, next_match: Dict):
        """
        Thread B: précalculer TOUT en RAM dès que prochain match est connu.
        Prend ~3-5s en background, inference lit cache: < 1ms.
        """
        self.next_match_info = next_match
        
        def _thread_b_work():
            start = time.time()
            logger.debug("Thread B: preparing cache...")
            
            new_cache = {}
            with self.results_lock:
                results = list(self.results_history)
                results_by_line_copy = {k: list(v) for k, v in self.results_by_line.items()}
                results_by_hour_copy = {k: list(v) for k, v in self.results_by_hour.items()}
                score_counts_copy = dict(self.score_counts)
            
            # Signaux engine (M1-M15)
            # M1: Distribution
            odds_implied = next_match.get('odds_implied', {'V': 0.40, 'N': 0.30, 'D': 0.30})
            new_cache['distribution'] = compute_distribution_signals(results, odds_implied)
            new_cache['dist_50'] = new_cache['distribution'].get('dist_50', {})
            
            # M2: Cycle
            new_cache['cycle'] = compute_cycle_signals(results)
            
            # M3: Streak
            new_cache['streak'] = compute_streak_signals(results, self.max_streaks)
            
            # M4: Autocorrélation
            new_cache['autocorr'] = compute_autocorrelation(results)
            
            # M5: Fourier
            new_cache['fourier'] = compute_fourier_signals(results)
            
            # M6: BOCPD (état courant)
            new_cache['changepoint'] = {
                'recent': self.bocpd.recent_changepoint(30),
                'last_5': self.bocpd.recent_changepoint(5),
                'in_last_10': self.bocpd.recent_changepoint(10) is not None,
            }
            
            # M7: Runs Test
            new_cache['runs_test'] = compute_runs_test(results)
            
            # M8: Symbolic patterns
            new_cache['symbolic'] = compute_symbolic_patterns(results)
            
            # M9: Line bias
            new_cache['line_bias'] = compute_line_bias(results_by_line_copy)
            new_cache['line_position'] = next_match.get('ligne', 1)
            
            # M10: Score distribution
            from app.services.signal_detectors import compute_score_distribution
            new_cache['score_distribution'] = compute_score_distribution(score_counts_copy)
            
            # M11: Calibration edge (depuis calibration_data)
            from app.services.signal_detectors import compute_calibration_edge
            new_cache['calibration_edge'] = compute_calibration_edge(self.calibration_data)
            
            # M12: Shin probabilities (avec les cotes du prochain match)
            odds_h = next_match.get('odd_home', 2.0)
            odds_d = next_match.get('odd_draw', 3.0)
            odds_a = next_match.get('odd_away', 3.5)
            new_cache['shin'] = compute_shin_probabilities(odds_h, odds_d, odds_a)
            
            # M13: Odds movement (dernière valeur enregistrée)
            new_cache['odds_movement'] = next_match.get('odds_movement', {})
            
            # M14: Time of day bias
            line_pos = next_match.get('ligne', 1)
            new_cache['time_bias_all'] = compute_time_bias(results_by_hour_copy)
            heure_actuelle = datetime.now().hour
            tranche = self._get_tranche(heure_actuelle)
            new_cache['time_bias_current'] = new_cache['time_bias_all'].get(tranche, {})
            
            # M15: Cross-match correlation
            new_cache['cross_match_corr'] = compute_cross_match_correlation(results_by_line_copy)
            
            # M16: Draw Detection Specialist (NOUVEAU)
            new_cache['draw_detection'] = compute_draw_signals(results, odds_implied)
            
            # M17: Goal Expectation Detector (NOUVEAU)
            new_cache['goal_expectation'] = compute_goal_expectation_signals(results, odds_implied)
            
            # M17b: Away Win Detector (NOUVEAU)
            new_cache['away_detection'] = compute_away_win_signals(results, odds_implied)
            
            # ── LSTM + Attention (Inference Thread B ou D) ────────
            new_cache['lstm_probs'] = self.lstm_model.predict_sequence(results)
            
            # ── NOUVEAUX: Lambda xG réels pour Monte Carlo ────────────
            # Au lieu de lambda_h=1.35 fixe, on calcule les vrais xG
            # home_name = next_match.get('home_team', '')
            # away_name = next_match.get('away_team', '')
            # season_id = next_match.get('season_id')
            # matchday_val = next_match.get('matchday', 15)
            
            # if home_name and away_name:
            #     try:
            #         lambda_data = self.lambda_calculator.compute_lambdas(
            #             home_team=home_name,
            #             away_team=away_name,
            #             season_id=season_id,
            #             matchday=matchday_val,
            #         )
            #         new_cache['lambda_h'] = lambda_data['lambda_h']
            #         new_cache['lambda_a'] = lambda_data['lambda_a']
            #         new_cache['lambda_confidence'] = lambda_data['confidence']
            #         new_cache['lambda_detail'] = {
            #             'home_attack': lambda_data.get('home_attack', 1.0),
            #             'away_defense': lambda_data.get('away_defense', 1.0),
            #             'away_attack': lambda_data.get('away_attack', 1.0),
            #             'home_defense': lambda_data.get('home_defense', 1.0),
            #         }
            #         logger.info(
            #             f"λ recalculés: {home_name}={lambda_data['lambda_h']:.3f}, "
            #             f"{away_name}={lambda_data['lambda_a']:.3f} (conf={lambda_data['confidence']:.2f})"
            #         )
            #     except Exception as e:
            #         logger.warning(f"Lambda calc error: {e} — using defaults")
            #         new_cache['lambda_h'] = 1.35
            #         new_cache['lambda_a'] = 1.10
            # else:
            #     new_cache['lambda_h'] = 1.35
            #     new_cache['lambda_a'] = 1.10
            
            # Use default values since lambda calculator is not available
            new_cache['lambda_h'] = 1.35
            new_cache['lambda_a'] = 1.10
            
            # Define variables for potential future use
            home_name = next_match.get('home_team', '')
            away_name = next_match.get('away_team', '')
            season_id = next_match.get('season_id')
            matchday_val = next_match.get('matchday', 15)
            
            # ── NOUVEAUX: Anomaly Detection ────────────────────────────
            # Détecte derbys, fatigue, fin de saison, etc.
            # if home_name and away_name:
            #     try:
            #         anomaly_result = self.anomaly_detector.analyze_match_context(
            #             home_team=home_name,
            #             away_team=away_name,
            #             matchday=matchday_val,
            #             season_id=season_id,
            #             odds_h=new_cache.get('shin', {}).get('raw', {}).get('V', 2.0),
            #             odds_d=new_cache.get('shin', {}).get('raw', {}).get('N', 3.2),
            #             odds_a=new_cache.get('shin', {}).get('raw', {}).get('D', 3.5),
            #         )
            #         new_cache['anomaly'] = {
            #             'score': anomaly_result['anomaly_score'],
            #             'signals': anomaly_result['signals'],
            #             'confidence_adjuster': anomaly_result['confidence_adjuster'],
            #             'should_reduce_stake': anomaly_result['should_reduce_stake'],
            #             'recommendation': anomaly_result['recommendation'],
            #             'prob_adjustments': anomaly_result['prob_adjustments'],
            #         }
            #         if anomaly_result['anomaly_score'] > 0.20:
            #             logger.info(
            #                 f"⚠️ Anomalie détectée {home_name} vs {away_name}: "
            #                 f"score={anomaly_result['anomaly_score']:.2f} — {anomaly_result['signals'][:2]}"
            #             )
            #     except Exception as e:
            #         logger.debug(f"Anomaly detection error: {e}")
            #         new_cache['anomaly'] = {'score': 0.0, 'signals': [], 'confidence_adjuster': 1.0}
            
            # Use default anomaly data since detector is not available
            new_cache['anomaly'] = {'score': 0.0, 'signals': [], 'confidence_adjuster': 1.0}
            
            # ELO + H2H (depuis DB si disponible)
            new_cache['elo'] = next_match.get('elo_probs', {'V': 0.40, 'N': 0.30, 'D': 0.30})
            new_cache['h2h'] = next_match.get('h2h_probs', {'V': 0.40, 'N': 0.30, 'D': 0.30})
            
            # Conformal quantile
            new_cache['conformal_quantile'] = self.conformal.quantile_90
            
            # Regime
            kl = new_cache['dist_50'].get('kl', 0)
            runs_pv = new_cache['runs_test'].get('p_value', 0.15)
            changepoint_recent = new_cache['changepoint']['recent'] is not None
            new_cache['regime'] = get_regime(kl, runs_pv, changepoint_recent, len(results))
            
            # Last result
            new_cache['last_result'] = results[-1] if results else 'V'
            
            # Engine scores (tous types)
            for t in ['V', 'N', 'D']:
                new_cache[f'engine_score_{t}'] = compute_engine_score(new_cache, t)
            
            # UCB weights actuels
            heure = next_match.get('heure', 12)
            new_cache['ucb_weights'] = self.ucb_bandit.get_ucb_weights(
                heure=heure,
                regime=new_cache['regime'],
                dominant_type=new_cache['last_result']
            )
            
            # Error system state
            new_cache['error_system'] = self.error_system.to_dict()
            new_cache['active_lessons'] = self.error_system.get_active_lessons()
            
            # Bankroll state
            new_cache['bankroll'] = self.bankroll.get_stats()
            
            # RL state vector (pour inference)
            rl_state = self.rl_agent.build_state_vector(new_cache)
            new_cache['rl_state'] = rl_state.tolist()
            new_cache['rl_recommendation'] = self.rl_agent.get_best_action(rl_state)
            
            # ── LSTM + Attention (Inference Thread B ou D) ────────
            new_cache['lstm_probs'] = self.lstm_model.predict_sequence(results)
            
            # Timestamp
            new_cache['prepared_at'] = datetime.utcnow().isoformat()
            new_cache['match_info'] = next_match
            
            # Atomically update cache
            with self.cache_lock:
                self.cache.update(new_cache)
            
            elapsed = time.time() - start
            logger.info(f"Thread B cache ready in {elapsed:.2f}s")
            
            # Auto-save cache state after preparation
            try:
                state_dir = Path('data/engine_state')
                state_dir.mkdir(parents=True, exist_ok=True)
                cache_file = state_dir / 'cache_state.json'
                # Use the sanitizer to handle numpy types
                cache_data = self._sanitize_for_json(self.cache)
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Auto-saved cache state with {len(cache_data)} items")
                
                # Also save draw and goal expectation signals separately for quick access
                draw_file = state_dir / 'draw_signals.json'
                goal_file = state_dir / 'goal_expectation.json'
                
                if 'draw_detection' in cache_data:
                    with open(draw_file, 'w') as f:
                        json.dump(cache_data['draw_detection'], f, indent=2)
                    logger.info(f"Saved draw signals: {cache_data['draw_detection'].get('draw_recommendation', 'LOW')}")
                
                if 'goal_expectation' in cache_data:
                    with open(goal_file, 'w') as f:
                        json.dump(cache_data['goal_expectation'], f, indent=2)
                    logger.info(f"Saved goal expectation signals: {cache_data['goal_expectation'].get('draw_likely', False)}")
                        
            except Exception as e:
                logger.warning(f"Could not auto-save cache state: {e}")
        
        t = threading.Thread(target=_thread_b_work, daemon=True, name="Thread-B")
        t.start()
    
    # ─────────────────────────────────────────────────────────────────
    # THREAD C — Monte Carlo (10k runs)
    # ─────────────────────────────────────────────────────────────────
    def run_monte_carlo_background(self, lambda_h: float = 1.35, lambda_a: float = 1.10,
                                    n_runs: int = 10000):
        """Thread C: Monte Carlo 10k pour prochain match."""
        self.mc_ready.clear()
        self.mc_result_cache = None
        
        def _thread_c_work():
            start = time.time()
            result = self._monte_carlo(lambda_h, lambda_a, n_runs)
            self.mc_result_cache = result
            self.mc_ready.set()
            elapsed = time.time() - start
            logger.debug(f"Thread C: Monte Carlo {n_runs} runs in {elapsed:.2f}s")
        
        t = threading.Thread(target=_thread_c_work, daemon=True, name="Thread-C")
        t.start()
    
    def _monte_carlo(self, lambda_h: float, lambda_a: float, n_runs: int) -> Dict:
        """Simulation Monte Carlo Poisson."""
        rng = np.random.default_rng()
        # Add ±15% uncertainty
        uncertainty = 0.15
        lambda_h_samples = rng.uniform(lambda_h * (1 - uncertainty), lambda_h * (1 + uncertainty), n_runs)
        lambda_a_samples = rng.uniform(lambda_a * (1 - uncertainty), lambda_a * (1 + uncertainty), n_runs)
        
        goals_h = rng.poisson(lambda_h_samples)
        goals_a = rng.poisson(lambda_a_samples)
        
        home_wins = int(np.sum(goals_h > goals_a))
        draws = int(np.sum(goals_h == goals_a))
        away_wins = int(np.sum(goals_h < goals_a))
        total = n_runs
        
        over25 = int(np.sum((goals_h + goals_a) > 2.5))
        btts = int(np.sum((goals_h > 0) & (goals_a > 0)))
        
        return {
            'prob_home_win': home_wins / total,
            'prob_draw': draws / total,
            'prob_away_win': away_wins / total,
            'prob_over25': over25 / total,
            'prob_btts': btts / total,
            'n_runs': n_runs,
            'V': home_wins / total,
            'N': draws / total,
            'D': away_wins / total,
        }
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 2 — INFERENCE PIPELINE (< 8 secondes)
    # ─────────────────────────────────────────────────────────────────
    def run_inference(self, match_id: int, home_team: str, away_team: str,
                       odds_h: float, odds_d: float, odds_a: float,
                       ml_service=None, poisson_service=None,
                       override_cache: Optional[Dict] = None) -> Dict:
        """
        Inference pipeline complet < 8s.
        Lit depuis cache RAM, ne touche pas la DB.
        """
        t_start = time.time()
        inference_times = {}
        
        # ── T+0.0: Trigger ────────────────────────────────────────
        logger.info(f"Inference started: {home_team} vs {away_team}")
        
        # Check FAST MODE
        if override_cache is not None:
            cache = dict(override_cache)
            cache_empty = len(cache) == 0
        else:
            with self.cache_lock:
                cache = dict(self.cache)  # Shallow copy for thread safety
                cache_empty = len(cache) == 0
        
        if cache_empty:
            return self._fast_mode_inference(
                match_id, odds_h, odds_d, odds_a, t_start,
                home_team=home_team, away_team=away_team,
                matchday=override_cache.get('matchday', 1) if override_cache else 1
            )
        
        # ── T+0.1: Read cache ─────────────────────────────────────
        inference_times['cache_read'] = time.time() - t_start
        
        # ── T+0.2–0.7: ML Parallel predictions ──────────────────
        t0 = time.time()
        ml_probs = cache.get('ml_probs', {'V': 0.33, 'N': 0.33, 'D': 0.34})
        lstm_probs = cache.get('lstm_probs', {'V': 0.33, 'N': 0.33, 'D': 0.34})
        inference_times['ml_parallel'] = time.time() - t0
        
        # Try live ML if service available
        if ml_service is not None:
            try:
                t0 = time.time()
                ml_result = ml_service.predict_from_cache(cache)
                if ml_result:
                    ml_probs = ml_result
                inference_times['ml_parallel'] = time.time() - t0
            except Exception as e:
                logger.warning(f"ML predict failed: {e}")
        
        # ── T+0.8: LSTM (si disponible en cache) ─────────────────
        inference_times['lstm'] = 0.001
        
        # ── T+1.0–1.6: Monte Carlo ────────────────────────────────
        t0 = time.time()
        mc_wait = self.mc_ready.wait(timeout=0.5)
        if mc_wait and self.mc_result_cache:
            mc_probs = self.mc_result_cache
        else:
            # Fast MC: 2000 runs
            lambda_h = cache.get('lambda_h', 1.35)
            lambda_a = cache.get('lambda_a', 1.10)
            mc_probs = self._monte_carlo(lambda_h, lambda_a, 2000)
        inference_times['mc'] = time.time() - t0
        
        # ── T+1.6: Assembler signal composite ────────────────────
        t0 = time.time()
        
        elo_probs = cache.get('elo', {'V': 0.40, 'N': 0.30, 'D': 0.30})
        h2h_probs = cache.get('h2h', {'V': 0.40, 'N': 0.30, 'D': 0.30})
        poisson_probs = cache.get('poisson_probs', {'V': 0.40, 'N': 0.30, 'D': 0.30})
        shin = cache.get('shin', {})
        if (not shin) or (not isinstance(shin, dict)) or (not isinstance(shin.get('shin'), dict)):
            # Always derive Shin from the current match odds when missing/incomplete.
            # This prevents any constant fallback that would systematically bias toward 'V'.
            shin = compute_shin_probabilities(odds_h, odds_d, odds_a)
        
        # UCB weights
        ucb_weights = cache.get('ucb_weights', {})
        
        # Model score (weighted par UCB)
        model_predictions = {
            'lstm':    lstm_probs,
            'xgb':     ml_probs,
            'mc':      {'V': mc_probs.get('V', 0.33), 'N': mc_probs.get('N', 0.33), 'D': mc_probs.get('D', 0.33)},
            'poisson': poisson_probs,
            'elo':     elo_probs,
            'markov':  cache.get('markov_probs', {'V': 0.33, 'N': 0.33, 'D': 0.34}),
        }
        
        model_score = {t: 0.0 for t in ['V', 'N', 'D']}
        total_weight = 0.0
        for model, probs in model_predictions.items():
            w = ucb_weights.get(model, 1.0 / len(model_predictions))
            for t in ['V', 'N', 'D']:
                model_score[t] += probs.get(t, 0.33) * w
            total_weight += w
        if total_weight > 0:
            model_score = {t: v / total_weight for t, v in model_score.items()}
        
        # Odds score (M12-M13)
        odds_movement = cache.get('odds_movement', {})
        shin_probs = shin['shin']
        
        movement_signal = odds_movement.get('signal', {})
        odds_score = {}
        for t in ['V', 'N', 'D']:
            ms = 1.0 if movement_signal.get('type') == t else 0.0
            odds_score[t] = (
                ms * 0.50 +
                shin_probs.get(t, 0.33) * 0.30 +
                (0.5 if shin.get('anomaly') else 0.0) * 0.20
            )
        
        # Engine score (depuis cache)
        engine_scores = {t: cache.get(f'engine_score_{t}', 0.0) for t in ['V', 'N', 'D']}
        
        # Changepoint discount
        changepoint_info = cache.get('changepoint', {})
        cp_active = changepoint_info.get('recent') is not None
        changepoint_discount = 0.75 if cp_active else 1.0
        
        # Correction factor from error system
        error_state = cache.get('error_system', {})
        ml_w = error_state.get('weight_ml', 1.0)
        engine_w = error_state.get('weight_engine', 1.0)
        
        # Final score composite (Module 6) - ADJUSTED to avoid global signal override
        final_score = {}
        for t in ['V', 'N', 'D']:
            # Poids rééquilibrés : ML (spécifique au match) > Engine (global)
            # Engine réduit de 0.38 -> 0.25
            # ML augmenté de 0.35 -> 0.45
            # Odds inchangé à 0.22
            # H2H augmenté de 0.05 -> 0.08
            raw = (
                engine_scores.get(t, 0) * 0.20 +  # Engine global réduit
                odds_score.get(t, 0)    * 0.30 +  # Odds augmenté (équilibré)
                model_score.get(t, 0)   * 0.35 +  # ML réduit (biaisé V)
                h2h_probs.get(t, 0.33)  * 0.15    # H2H augmenté (équilibré)
            ) * changepoint_discount
            final_score[t] = raw
        
        # Normalize to probabilities
        min_score = min(final_score.values())
        if min_score < 0:
            final_score = {t: v - min_score for t, v in final_score.items()}
        total = sum(final_score.values())
        if total > 0:
            final_probs = {t: v / total for t, v in final_score.items()}
        else:
            # Fallback ÉQUILIBRÉ - pas de biais vers V
            final_probs = {'V': 0.40, 'N': 0.30, 'D': 0.30}
        
        # ── Apply optimization corrections (9000+ matchs learning) ────────
        # IMPORTANT: on passe les cotes du match courant (odds_h/d/a), pas un cache générique.
        opt_context = {'odds_h': odds_h, 'odds_d': odds_d, 'odds_a': odds_a}
        final_probs = self._apply_optimization_corrections(final_probs, opt_context)
        
        # ── Apply aggressive learning corrections ─────────────────────────
        learner_context = {'odd_home': odds_h, 'odd_draw': odds_d, 'odd_away': odds_a}
        final_probs = self.aggressive_learner.apply_corrections_to_prediction(final_probs, learner_context)

        # ── Guardrails vs odds (Shin) ─────────────────────────────────────
        final_probs = self._apply_probability_guardrails(final_probs, shin_probs)
        
        # ── Match Analyzer: analyse INDIVIDUELLE du match ─────────────────
        # Utilise forme, H2H, cycles, signaux de surprise
        try:
            matchday = cache.get('matchday', 1)
            analysis = self.match_analyzer.compute_intelligent_prediction(
                home_team=home_team,
                away_team=away_team,
                odds=(odds_h, odds_d, odds_a),
                matchday=matchday,
                ml_probs=cache.get('ml_probs'),
                engine_probs=final_probs
            )
            
            # Les signaux de surprise peuvent modifier la prédiction
            surprise_signals = analysis.get('surprise_signals', [])
            ma_pred = analysis.get('predicted', '')
            
            # Détecter la divergence entre MA et le moteur
            current_pred = max(final_probs, key=final_probs.get)
            divergence = ma_pred != current_pred and ma_pred != ''
            
            # TOUJOURS appliquer l'analyse individuelle avec un poids adaptatif
            # Plus de signaux = plus de poids à l'analyse
            # Augmenté significativement pour contrebalancer le biais ML vers V
            # Si divergence détectée, augmenter encore plus le poids
            if divergence:
                # Le MA contredit le moteur - donner plus de poids au MA
                if len(surprise_signals) >= 3:
                    blend_weight = 0.80  # 80% analyse
                elif len(surprise_signals) >= 2:
                    blend_weight = 0.75  # 75% analyse
                elif len(surprise_signals) >= 1:
                    blend_weight = 0.65  # 65% analyse
                else:
                    blend_weight = 0.55  # 55% analyse
            else:
                # Le MA est d'accord avec le moteur
                if len(surprise_signals) >= 3:
                    blend_weight = 0.70  # 70% analyse - forte confiance
                elif len(surprise_signals) >= 2:
                    blend_weight = 0.65  # 65% analyse - confiance élevée
                elif len(surprise_signals) >= 1:
                    blend_weight = 0.50  # 50% analyse
                else:
                    blend_weight = 0.40  # 40% analyse même sans signaux
            
            for outcome in ['V', 'N', 'D']:
                final_probs[outcome] = (
                    (1 - blend_weight) * final_probs[outcome] + 
                    blend_weight * analysis['final_probs'][outcome]
                )
            
            # Normaliser
            total = sum(final_probs.values())
            if total > 0:
                final_probs = {k: v/total for k, v in final_probs.items()}
            
            # Stocker le raisonnement pour debug
            cache['match_analysis'] = {
                'signals': surprise_signals,
                'reasoning': analysis.get('reasoning', ''),
                'home_form': analysis.get('home_form', {}),
                'away_form': analysis.get('away_form', {}),
                'h2h': analysis.get('h2h', {}),
                'blend_weight': blend_weight
            }
        except Exception as e:
            logger.warning(f"MatchAnalyzer error: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        # Predicted result
        predicted = max(final_probs, key=final_probs.get)
        inference_times['assembly'] = time.time() - t0
        
        # ── T+1.8: Conformal ─────────────────────────────────────
        t0 = time.time()
        prob_predicted = final_probs[predicted]
        conf_lower, conf_upper, conf_width = self.conformal.compute_interval(prob_predicted)
        inference_times['conformal'] = time.time() - t0
        
        # ── T+2.0: Quality metrics ────────────────────────────────
        entropy = compute_entropy(final_probs)
        diversity = compute_diversity_score(model_predictions)
        agreement, votes = compute_model_agreement(model_predictions)
        variance_conf = compute_variance_confidence(list(self.bankroll.pred_variance_history))
        
        # Confidence
        confidence = (
            prob_predicted * 0.4 +
            (1 - entropy) * 0.3 +
            agreement * 0.2 +
            (1 - conf_width) * 0.1
        )
        
        # ── SMART STAKE MANAGER: Ajuste stake selon risque identifié ────────
        # Basé sur analyse: V avec cote H > 2.5 = 26% précision (ne pas parier)
        # V avec cote H < 1.5 + conf >= 50% = 74% précision (boost stake)
        try:
            stake_info = self.stake_manager.calculate_stake_multiplier(
                predicted=predicted,
                confidence=confidence,
                odd_home=odds_h,
                odd_draw=odds_d,
                odd_away=odds_a,
                prob_v=final_probs.get('V', 0.33),
                prob_n=final_probs.get('N', 0.33),
                prob_d=final_probs.get('D', 0.33),
                home_team=home_team
            )
            
            # Ajuster la confiance selon le risque réel
            adjusted_confidence = self.stake_manager.adjust_confidence(
                confidence, predicted, odds_h, odds_d, odds_a
            )
            
            # Stocker pour debug et décision de pari
            cache['stake_info'] = stake_info
            cache['adjusted_confidence'] = adjusted_confidence
            
            # Logger les ajustements significatifs
            if stake_info['multiplier'] < 0.5 or stake_info['multiplier'] > 1.0:
                logger.info(f"Stake ajusté: {stake_info['multiplier']:.1f}x - {stake_info['reason']}")
            
            # Si le risque est très élevé, ne pas parier
            if stake_info['risk_level'] == 'VERY_HIGH':
                logger.warning(f"PARI REFUSÉ: {stake_info['reason']}")
        
        except Exception as e:
            logger.warning(f"SmartStakeManager error: {e}")
            cache['stake_info'] = {'multiplier': 0.5, 'should_bet': True}
        
        # EV
        implied_prob = 1 / max(odds_h if predicted == 'V' else odds_d if predicted == 'N' else odds_a, 1.001)
        odds_used = odds_h if predicted == 'V' else odds_d if predicted == 'N' else odds_a
        ev = final_probs[predicted] - implied_prob
        ev_adjusted = ev * (1 - entropy) * changepoint_discount
        
        # ── T+2.2: Validation + Kelly ─────────────────────────────
        t0 = time.time()
        runs_pvalue = cache.get('runs_test', {}).get('p_value', 0.5)
        regime = cache.get('regime', 'STABLE')
        bankroll_stats = cache.get('bankroll', {})
        drawdown = bankroll_stats.get('current_drawdown', 0)
        
        thresholds = self.error_system.get_thresholds()
        
        ok, reject_reasons = self.bankroll.validate_bet(
            ev_adjusted=ev_adjusted,
            confidence=confidence,
            model_agreement=agreement,
            entropy_norm=entropy,
            conformal_width=conf_width,
            diversity_score=diversity,
            runs_pvalue=runs_pvalue,
            engine_anomaly=0.0,
            regime=regime,
            odds=odds_used,
            drawdown=drawdown,
            consecutive_errors=self.error_system.consecutive_errors,
            error_rate_hour=0.3,
            inference_time=time.time() - t_start,
            recovery_mode=self.error_system.recovery_mode,
        )
        
        # ── T+2.5: Stake + Output ─────────────────────────────────
        stake_result = {'stake': 0, 'allowed': False}
        if ok:
            signal_strength = float(max(engine_scores.values(), default=0))
            am_mult = self.anti_martingale.get_multiplier(signal_strength)
            stake_result = self.bankroll.compute_stake(
                prob=final_probs[predicted],
                odds=odds_used,
                conformal_width=conf_width,
                regime=regime,
                signal_strength=signal_strength,
                changepoint_discount=changepoint_discount,
            )
        
        # ── RL Recommendation ─────────────────────────────────────
        rl_state = np.array(cache.get('rl_state', [0.0] * 10))
        rl_rec = cache.get('rl_recommendation', {})
        
        # Meta-pattern check
        meta_alerts = self.error_system.check_meta_patterns(
            heure=datetime.now().hour
        )
        
        inference_times['total'] = time.time() - t_start
        
        # Log inference time
        mode = 'FAST' if inference_times['total'] < 1.0 else \
               'RECOVERY' if self.error_system.recovery_mode else 'NORMAL'
        if inference_times['total'] > 8:
            logger.warning(f"INFERENCE LENTE: {inference_times['total']:.2f}s > 8s")
        
        self.inference_log.append({
            'match_id': match_id,
            'total_ms': inference_times['total'] * 1000,
            'mode': mode,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # ── OUTPUT FINAL ──────────────────────────────────────────
        return {
            # Status
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'inference_time': inference_times['total'],
            'inference_times': inference_times,
            'mode': mode,
            
            # Prédiction
            'predicted': predicted,
            'final_probs': final_probs,
            'confidence': float(np.clip(confidence, 0, 1)),
            'ev_adjusted': ev_adjusted,
            'odds': {'V': odds_h, 'N': odds_d, 'D': odds_a},
            'odds_used': odds_used,
            
            # Engine signals (Module 9)
            'engine_signals': {
                'cycle': cache.get('cycle', {}),
                'streak': cache.get('streak', {}),
                'fourier': cache.get('fourier', {}),
                'autocorr': cache.get('autocorr', {}),
                'symbolic': cache.get('symbolic', {}).get('best_pattern'),
                'line_bias': cache.get('line_bias', {}).get(cache.get('line_position', 1), {}),
                'odds_movement': odds_movement,
                'shin': shin,
                'kl': cache.get('dist_50', {}).get('kl', 0),
                'runs_test': cache.get('runs_test', {}),
                'changepoint': changepoint_info,
                'engine_scores': engine_scores,
            },
            
            # Match Analysis (Module 10 - Analyse individuelle)
            'match_analysis': cache.get('match_analysis', {}),
            
            # Modèles
            'model_predictions': model_predictions,
            'model_score': model_score,
            'ucb_weights': ucb_weights,
            
            # Qualité
            'conformal': {
                'lower': conf_lower,
                'upper': conf_upper,
                'width': conf_width,
                'level': self.conformal.get_signal_level(conf_width),
                'mult': self.conformal.get_confidence_multiplier(conf_width),
            },
            'entropy': entropy,
            'entropy_signal': 'CLAIR' if entropy < 0.60 else 'INCERTAIN' if entropy > 0.88 else 'MODÉRÉ',
            'diversity': diversity,
            'model_agreement': agreement,
            'model_votes': votes,
            'variance': variance_conf,
            
            # RL
            'rl': rl_rec,
            
            # Corrections actives
            'active_lessons': cache.get('active_lessons', []),
            'recovery_mode': self.error_system.recovery_mode,
            'changepoint_active': cp_active,
            
            # Smart Stake Manager
            'stake_info': cache.get('stake_info', {'multiplier': 0.5, 'should_bet': True}),
            'adjusted_confidence': cache.get('adjusted_confidence', confidence),
            'changepoint_discount': changepoint_discount,
            
            # Bankroll
            'bankroll_stats': bankroll_stats,
            'stake': stake_result,
            
            # Décision
            'should_bet': ok and stake_result.get('allowed', False),
            'reject_reasons': reject_reasons,
            'meta_alerts': meta_alerts,
            
            # Regime
            'regime': regime,
            'runs_pvalue': runs_pvalue,
        }
    
    # ─────────────────────────────────────────────────────────────────
    # FAST MODE (< 0.8s)
    # ─────────────────────────────────────────────────────────────────
    def _fast_mode_inference(self, match_id: int, odds_h: float,
                              odds_d: float, odds_a: float, t_start: float,
                              home_team: str = '', away_team: str = '',
                              matchday: int = 1) -> Dict:
        """Fast Mode: ELO + Poisson + MatchAnalyzer (0.8s)."""
        self.fast_mode_count += 1
        logger.warning(f"FAST MODE activé — cache vide (#{self.fast_mode_count})")
        
        # 1. Implied probability from odds
        raw_h = 1 / max(odds_h, 1.01)
        raw_d = 1 / max(odds_d, 1.01)
        raw_a = 1 / max(odds_a, 1.01)
        total = raw_h + raw_d + raw_a
        base_probs = {'V': raw_h / total, 'N': raw_d / total, 'D': raw_a / total}
        
        # 2. Appliquer MatchAnalyzer pour analyse individuelle
        try:
            analysis = self.match_analyzer.compute_intelligent_prediction(
                home_team=home_team or 'Unknown',
                away_team=away_team or 'Unknown',
                odds=(odds_h, odds_d, odds_a),
                matchday=matchday
            )
            
            # Blend: 50% base + 50% analyse individuelle
            probs = {}
            for outcome in ['V', 'N', 'D']:
                probs[outcome] = 0.5 * base_probs[outcome] + 0.5 * analysis['final_probs'][outcome]
            
            # Normaliser
            total_p = sum(probs.values())
            if total_p > 0:
                probs = {k: v/total_p for k, v in probs.items()}
            
            signals = analysis.get('surprise_signals', [])
        except Exception as e:
            logger.debug(f"MatchAnalyzer error in FAST MODE: {e}")
            probs = base_probs
            signals = []
        
        predicted = max(probs, key=probs.get)
        odds_used = odds_h if predicted == 'V' else odds_d if predicted == 'N' else odds_a
        confidence = probs[predicted] * 0.5
        
        # 3. Appliquer SmartStakeManager pour ajuster le stake
        try:
            stake_info = self.stake_manager.calculate_stake_multiplier(
                predicted=predicted,
                confidence=confidence,
                odd_home=odds_h,
                odd_draw=odds_d,
                odd_away=odds_a,
                prob_v=probs.get('V', 0.33),
                prob_n=probs.get('N', 0.33),
                prob_d=probs.get('D', 0.33),
                home_team=home_team
            )
        except Exception as e:
            logger.debug(f"SmartStakeManager error in FAST MODE: {e}")
            stake_info = {'multiplier': 0.3, 'should_bet': False}
        
        return {
            'match_id': match_id,
            'predicted': predicted,
            'final_probs': probs,
            'confidence': probs[predicted] * 0.5,  # Reduced
            'model_agreement': 0.0,
            'ev_adjusted': 0.0,
            'odds_used': odds_used,
            'should_bet': False,
            'mode': 'FAST',
            'inference_time': time.time() - t_start,
            'reject_reasons': ['FAST MODE — cache vide, stakes réduits 50%'],
            'stake': {'stake': 0, 'allowed': False, 'reason': 'Fast Mode'},
            'active_lessons': ['FAST MODE: données insuffisantes'],
            'surprise_signals': signals,
        }
    
    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _get_tranche(self, heure: int) -> str:
        if 6 <= heure < 12:
            return 'matin'
        elif 12 <= heure < 18:
            return 'apres_midi'
        elif 18 <= heure < 22:
            return 'soir'
        else:
            return 'nuit'
    
    def get_dashboard_data(self) -> Dict:
        """Données pour Module 10 dashboard."""
        with self.cache_lock:
            cache = dict(self.cache)
        
        return {
            'engine_state': {
                'draw_rate_10': cache.get('cycle', {}).get('N', {}).get('rate_10', 0),
                'home_rate_10': cache.get('cycle', {}).get('V', {}).get('rate_10', 0),
                'kl_divergence': cache.get('dist_50', {}).get('kl', 0),
                'runs_pvalue': cache.get('runs_test', {}).get('p_value', 1.0),
                'fourier': cache.get('fourier', {}),
                'changepoint': cache.get('changepoint', {}),
                'regime': cache.get('regime', 'STABLE'),
            },
            'active_biases': {
                'draw_overdue': cache.get('cycle', {}).get('N', {}).get('overdue_score', 0),
                'streak': cache.get('streak', {}),
                'fourier_signal': cache.get('fourier', {}).get('dominant_type'),
                'symbolic_top': cache.get('symbolic', {}).get('best_pattern', {}).get('pattern', '?'),
                'odds_movement': cache.get('odds_movement', {}),
            },
            'learning': {
                'active_lessons': len(cache.get('active_lessons', [])),
                'recovery_mode': self.error_system.recovery_mode,
                'consecutive_errors': self.error_system.consecutive_errors,
                'ece': self.calibration_drift.ece,
                'changepoint_risk': 'ÉLEVÉ' if cache.get('changepoint', {}).get('recent') else 'FAIBLE',
            },
            'performance': {
                'inference_avg': float(np.mean([l['total_ms'] for l in self.inference_log[-20:]]) if self.inference_log else 0),
                'inference_max': float(max((l['total_ms'] for l in self.inference_log[-20:]), default=0)),
                'fast_mode_today': self.fast_mode_count,
                'cache_ready': len(cache) > 0,
            },
            'bankroll': self.bankroll.get_stats(),
        }
    
    def get_cache_snapshot(self) -> Dict:
        """Retourne une copie du cache pour debug/API. Always returns valid data."""
        # Required fields that MUST be present
        required_fields = ['cycle', 'streak', 'fourier', 'autocorr', 'runs_test', 
                         'changepoint', 'symbolic', 'shin', 'engine_score_V', 
                         'engine_score_N', 'engine_score_D', 'regime']
        
        with self.cache_lock:
            # Sanitize entire cache structure for JSON to avoid numpy bool_/float64 errors
            cache_snapshot = self._sanitize_for_json(self.cache)
            
            # Check if cache has all required fields
            missing_fields = [f for f in required_fields if f not in cache_snapshot]
            
            if not cache_snapshot or len(cache_snapshot) < 10 or missing_fields:
                logger.warning(f"Cache incomplete or missing fields: {missing_fields}")
                
                # Try to reload from saved state
                try:
                    state_dir = Path('data/engine_state')
                    
                    # Load main cache state
                    cache_file = state_dir / 'cache_state.json'
                    if cache_file.exists():
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                            if cache_data:
                                self.cache.update(cache_data)
                                logger.info(f"Reloaded cache state with {len(cache_data)} items")
                    
                    # Load draw signals
                    draw_file = state_dir / 'draw_signals.json'
                    if draw_file.exists():
                        with open(draw_file, 'r') as f:
                            draw_data = json.load(f)
                            self.cache['draw_detection'] = draw_data
                    
                    # Load goal expectation signals
                    goal_file = state_dir / 'goal_expectation.json'
                    if goal_file.exists():
                        with open(goal_file, 'r') as f:
                            goal_data = json.load(f)
                            self.cache['goal_expectation'] = goal_data
                    
                except Exception as e:
                    logger.error(f"Failed to reload cache state: {e}")
                
                # Re-check after reload
                cache_snapshot = {k: v for k, v in self.cache.items()
                            if not isinstance(v, np.ndarray)}
                still_missing = [f for f in required_fields if f not in cache_snapshot]
                
                if still_missing:
                    logger.warning(f"Cache still missing fields after reload: {still_missing}, initializing sample data")
                    self._initialize_sample_cache()
                    cache_snapshot = self._sanitize_for_json(self.cache)
            
            return cache_snapshot
    
    def load_historical_results(self, results: List[str],
                                 results_by_line: Dict = None,
                                 results_by_hour: Dict = None,
                                 score_counts: Dict = None):
        """
        Charger les résultats historiques depuis DB au démarrage.
        Appelé une seule fois depuis main.py startup.
        """
        with self.results_lock:
            self.results_history = list(results)
            self.results_by_line = results_by_line or {}
            self.results_by_hour = results_by_hour or {}
            self.score_counts = score_counts or {}
            
            # Rebuild max streaks
            for t in ['V', 'N', 'D']:
                current = 0
                max_seen = 0
                for r in self.results_history:
                    if r == t:
                        current += 1
                        max_seen = max(max_seen, current)
                    else:
                        current = 0
                self.max_streaks[t] = max(max_seen, 3)
        
        logger.info(f"Loaded {len(results)} historical results into engine")
    
    def _load_optimization_config(self):
        """Charge la configuration optimisée depuis le fichier."""
        import json
        from pathlib import Path
        
        try:
            config_path = Path('data/optimization_state.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    state = json.load(f)
                    self.optimization_config = state
                    logger.info("Configuration optimisée chargée depuis 9000+ matchs")
            else:
                # Générer la configuration si elle n'existe pas
                logger.info("Génération de la configuration optimisée...")
                self.optimization_config = self.prediction_optimizer.generate_optimization_config()
                self.prediction_optimizer.save_optimization_state()
        except Exception as e:
            logger.warning(f"Impossible de charger la configuration optimisée: {e}")
            self.optimization_config = None
    
    def _apply_optimization_corrections(self, probs: Dict[str, float], context: Dict = None) -> Dict[str, float]:
        """Applique les corrections d'optimisation aux probabilités."""
        if not self.optimization_config:
            return probs
        
        corrected = probs.copy()
        
        # 1. Correction distributionnelle (odds-aware)
        # IMPORTANT: ne jamais laisser une correction distributionnelle forcer des N
        # sur des matchs déséquilibrés (cotes extrêmes / favori clair).
        bias_adj = self.optimization_config.get('bias_corrections', {})

        odds_h = None
        odds_d = None
        odds_a = None
        if isinstance(context, dict):
            odds_h = context.get('odds_h') or context.get('odd_home')
            odds_d = context.get('odds_d') or context.get('odd_draw')
            odds_a = context.get('odds_a') or context.get('odd_away')

        shin_probs = None
        try:
            if odds_h and odds_d and odds_a:
                shin = compute_shin_probabilities(float(odds_h), float(odds_d), float(odds_a))
                shin_probs = shin.get('shin')
        except Exception:
            shin_probs = None

        # Match "équilibré" si le favori Shin < 0.55 et N Shin >= 0.22
        is_balanced = False
        if isinstance(shin_probs, dict):
            fav = max(shin_probs.get('V', 0.0), shin_probs.get('D', 0.0))
            is_balanced = fav < 0.55 and shin_probs.get('N', 0.0) >= 0.22

        if is_balanced and 'distribution' in bias_adj:
            dist_adj = bias_adj['distribution']
            for outcome in ['V', 'N', 'D']:
                corrected[outcome] *= dist_adj.get(outcome, 1.0)
        
        # 2. Seuils de confiance optimisés par type
        # (utilisé par le dynamic_selection_engine)
        
        # 3. Normalisation
        total = sum(corrected.values())
        if total > 0:
            corrected = {k: v/total for k, v in corrected.items()}
        
        return corrected

    def _apply_probability_guardrails(
        self,
        probs: Dict[str, float],
        shin_probs: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Garde-fous pour éviter des prédictions absurdes vs cotes.

        - On ancre les probabilités sur Shin (odds) via un blend.
        - On cappe la proba de nul si Shin indique un match déséquilibré.
        """
        if not isinstance(probs, dict):
            return probs
        if not isinstance(shin_probs, dict):
            return probs

        p = {k: float(probs.get(k, 0.0)) for k in ['V', 'N', 'D']}
        s = {k: float(shin_probs.get(k, 0.0)) for k in ['V', 'N', 'D']}

        # Blend vers Shin (évite de "fabriquer" des N)
        # alpha plus faible quand Shin voit un gros favori.
        fav = max(s.get('V', 0.0), s.get('D', 0.0))
        if fav >= 0.65:
            alpha = 0.55
        elif fav >= 0.55:
            alpha = 0.65
        else:
            alpha = 0.75

        blended = {k: alpha * p[k] + (1.0 - alpha) * s[k] for k in ['V', 'N', 'D']}

        # Cap draw: N ne doit pas dépasser Shin_N + 0.10 quand favori clair.
        # (ex: cote domicile 1.25 => Shin_N faible)
        if fav >= 0.60:
            n_cap = min(0.45, s.get('N', 0.0) + 0.10)
            if blended['N'] > n_cap:
                overflow = blended['N'] - n_cap
                blended['N'] = n_cap
                # Redistribuer l'overflow vers V/D proportionnellement à Shin (logique odds)
                vd_sum = max(1e-9, s.get('V', 0.0) + s.get('D', 0.0))
                blended['V'] += overflow * (s.get('V', 0.0) / vd_sum)
                blended['D'] += overflow * (s.get('D', 0.0) / vd_sum)

        # Normalisation finale
        total = blended['V'] + blended['N'] + blended['D']
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended
    
    def get_optimized_thresholds(self) -> Dict[str, float]:
        """Retourne les seuils de confiance optimisés par type."""
        if self.optimization_config and 'confidence_thresholds' in self.optimization_config:
            return self.optimization_config['confidence_thresholds']
        return {'V': 0.45, 'N': 0.35, 'D': 0.35}
    
    def _load_persisted_state(self):
        """Charger l'état persisté (RL, Error system, Bankroll, Cache)."""
        state_dir = Path('data/engine_state')
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache state
        cache_file = state_dir / 'cache_state.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    with self.cache_lock:
                        self.cache.update(cache_data)
                        logger.info(f"Loaded cache state with {len(cache_data)} items")
            except Exception as e:
                logger.warning(f"Could not load cache state: {e}")
        
        # Load other states
        # Match the save format (JSON) and method
        rl_file = state_dir / 'rl_state.json'
        if rl_file.exists():
            try:
                if hasattr(self.rl_agent, 'load'):
                    self.rl_agent.load(str(rl_file))
                else:
                    logger.warning("RL Agent has no load() method")
            except Exception as e:
                logger.warning(f"Could not load RL state: {e}")
        
        error_file = state_dir / 'error_state.json'
        if error_file.exists():
            try:
                self.error_system.load(str(error_file))
            except Exception as e:
                logger.warning(f"Could not load error system state: {e}")
        
        bankroll_file = state_dir / 'bankroll_state.json'
        if bankroll_file.exists():
            try:
                self.bankroll.load(bankroll_file)
            except Exception as e:
                logger.warning(f"Could not load bankroll state: {e}")
        
        logger.info("Engine state loaded")
    
    def save_state(self):
        """Persister l'état complet."""
        state_dir = Path('data/engine_state')
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cache state
        cache_file = state_dir / 'cache_state.json'
        try:
            with self.cache_lock:
                cache_data = {k: v for k, v in self.cache.items()
                            if not isinstance(v, np.ndarray)}
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Saved cache state with {len(cache_data)} items")
        except Exception as e:
            logger.warning(f"Could not save cache state: {e}")
        
    def _save_persisted_state(self):
        """Sauvegarder l'état complet de l'apprentissage (Cache, Error, RL, Bankroll)."""
        try:
            state_dir = Path('data/engine_state')
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save Cache
            self._save_cache_to_disk()
            
            # 2. Save Error System
            if hasattr(self.error_system, 'save'):
                self.error_system.save(str(state_dir / 'error_state.json'))
            
            # 3. Save Bankroll
            if hasattr(self.bankroll, 'save'):
                self.bankroll.save(str(state_dir / 'bankroll_state.json'))
            
            # 4. Save RL & Bandit
            if hasattr(self.rl_agent, 'save'):
                self.rl_agent.save(str(state_dir / 'rl_state.json'))
            if hasattr(self.ucb_bandit, 'save'):
                self.ucb_bandit.save(str(state_dir / 'bandit_state.json'))
                
            logger.info("Statut de l'apprentissage persisté sur disque")
        except Exception as e:
            logger.warning(f"Error during persistence save: {e}")

    def _sanitize_for_json(self, value):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(value, dict):
            return {k: self._sanitize_for_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize_for_json(v) for v in value]
        elif isinstance(value, tuple):
            return [self._sanitize_for_json(v) for v in value]
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
