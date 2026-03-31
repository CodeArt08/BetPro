"""Adaptive Ensemble System with dynamic weighting and meta-model."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import joblib
from sqlalchemy.orm import Session

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

from app.core.config import settings
from app.models import MatchFeatures, MethodPerformance, Prediction


class AdaptiveEnsemble:
    """
    Combines all prediction methods with dynamic weights:
    - ML Ensemble
    - Monte Carlo
    - Bivariate Poisson
    - ELO
    - Head-to-Head
    - Real-Time Engine (M1-M15 signals)
    
    Weights are updated based on recent performance.
    """
    
    METHOD_NAMES = ['ml_ensemble', 'monte_carlo', 'poisson', 'elo', 'h2h', 'realtime']
    
    def __init__(self):
        # When ML is not trained, other methods have more weight
        # Dynamic method weights (adjusted by performance)
        self.method_weights = {
            'ml_ensemble': 0.30,      # Reduced slightly to make room for H2H
            'monte_carlo': 0.20,
            'poisson': 0.15,
            'elo': 0.12,              # Reduced slightly
            'h2h': 0.18,              # Increased from 0.08 to 0.18 (more data = more weight)
            'realtime': 0.05
        }
        self.calibrators = {}
        self.meta_model = None
        self.is_calibrated = False
    
    def get_dynamic_weights(self, db: Session) -> Dict[str, float]:
        """
        Get dynamic weights based on stored method performance.
        Now uses adaptive weights manager for performance-based adjustments.
        """
        # Try to load adaptive weights first
        try:
            from app.services.adaptive_weights_manager import AdaptiveWeightsManager
            
            manager = AdaptiveWeightsManager()
            if manager.load_weights():
                logger.info("✅ Using adaptive weights from performance analysis")
                updated_weights = manager.get_current_weights()
                self.method_weights = updated_weights
                return updated_weights
        except Exception as e:
            logger.warning(f"❌ Could not load adaptive weights: {e}")
        
        # Fallback to base weights
        logger.info("⚠️ Using base weights (adaptive weights not available)")
        weights = {}
        for method in self.METHOD_NAMES:
            weights[method] = 1.0
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        self.method_weights = weights
        return weights
    
    def combine_predictions(self, ml_probs: Dict, mc_probs: Dict, poisson_probs: Dict,
                            elo_probs: Dict, h2h_probs: Dict, 
                            rt_probs: Optional[Dict] = None,
                            context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Combine predictions using odds-implied probabilities as BASE.
        Bookmaker odds are the most reliable predictor - models only adjust slightly.
        """
        # Get weights - FORCER l'utilisation des poids adaptatifs
        if db:
            # Forcer l'utilisation des poids adaptatifs
            try:
                from app.services.adaptive_weights_manager import AdaptiveWeightsManager
                manager = AdaptiveWeightsManager()
                if manager.load_weights():
                    weights = manager.get_current_weights()
                    logger.info("✅ Using adaptive weights from performance analysis")
                else:
                    logger.warning("⚠️ Using base weights (adaptive weights not available)")
                    weights = self.method_weights
            except Exception as e:
                logger.warning(f"❌ Error loading adaptive weights: {e}")
                weights = self.method_weights
        else:
            weights = self.method_weights
        
        # Use odds-implied probabilities as REFERENCE (not absolute base)
        # Bookmakers are ~70% accurate but heavily favor home team
        if odds and all(odds.get(k, 0) > 0 for k in ['home', 'draw', 'away']):
            implied_V = 1 / odds['home']
            implied_N = 1 / odds['draw']
            implied_D = 1 / odds['away']
            
            # Normalize to remove bookmaker margin
            total_implied = implied_V + implied_N + implied_D
            odds_ref = {
                'V': implied_V / total_implied,
                'N': implied_N / total_implied,
                'D': implied_D / total_implied
            }
            
            # RADICAL CORRECTION: Ignore odds completely, use balanced distribution
            # User requested: 38% V / 32% N / 30% D - force this distribution
            balanced_distribution_base = {'V': 0.38, 'N': 0.32, 'D': 0.30}
            odds_base = {
                'V': odds_ref['V'] * 0.0 + balanced_distribution_base['V'] * 1.0,  # 0% odds, 100% balanced
                'N': odds_ref['N'] * 0.0 + balanced_distribution_base['N'] * 1.0,  # 0% odds, 100% balanced
                'D': odds_ref['D'] * 0.0 + balanced_distribution_base['D'] * 1.0  # 0% odds, 100% balanced
            }
        else:
            # Fallback if no odds available
            odds_base = {'V': 0.38, 'N': 0.30, 'D': 0.32}
        
        
        # Model adjustments (small deviations from odds base)
        # Each model suggests a small adjustment
        model_adjustments = {'V': 0.0, 'N': 0.0, 'D': 0.0}
        
        # Monte Carlo adjustment
        mc_adj = {
            'V': mc_probs.get('prob_home_win', 0.33) - 0.33,
            'N': mc_probs.get('prob_draw', 0.33) - 0.33,
            'D': mc_probs.get('prob_away_win', 0.33) - 0.33
        }
        for t in ['V', 'N', 'D']:
            model_adjustments[t] += mc_adj[t] * weights['monte_carlo'] * 0.5
        
        # Poisson adjustment
        poisson_adj = {
            'V': poisson_probs.get('prob_home_win', 0.33) - 0.33,
            'N': poisson_probs.get('prob_draw', 0.33) - 0.33,
            'D': poisson_probs.get('prob_away_win', 0.33) - 0.33
        }
        for t in ['V', 'N', 'D']:
            model_adjustments[t] += poisson_adj[t] * weights['poisson'] * 0.5
        
        # ELO adjustment
        elo_adj = {
            'V': elo_probs.get('V', 0.33) - 0.33,
            'N': elo_probs.get('N', 0.33) - 0.33,
            'D': elo_probs.get('D', 0.33) - 0.33
        }
        for t in ['V', 'N', 'D']:
            model_adjustments[t] += elo_adj[t] * weights['elo'] * 0.5
        
        # H2H adjustment
        h2h_adj = {
            'V': h2h_probs.get('V', 0.33) - 0.33,
            'N': h2h_probs.get('N', 0.33) - 0.33,
            'D': h2h_probs.get('D', 0.33) - 0.33
        }
        for t in ['V', 'N', 'D']:
            model_adjustments[t] += h2h_adj[t] * weights['h2h'] * 0.5
        
        # Real-time engine signals
        if rt_probs:
            draw_signal = rt_probs.get('draw_signal_strength', 0)
            if draw_signal > 0.4:
                model_adjustments['N'] += 0.05  # Boost draw slightly
        
        # RADICAL CORRECTION: Force balanced distribution regardless of odds
        # Use team strength to adjust slightly from balanced base
        v_prob = odds_base.get('V', 0.38)
        d_prob = odds_base.get('D', 0.30)
        
        # Strong adjustments based on team strength differences
        if abs(v_prob - d_prob) > 0.05:  # Teams are not equal (lower threshold)
            if v_prob > d_prob:  # Home stronger
                model_adjustments['V'] += 0.10  # Strong boost to stronger team
                model_adjustments['D'] -= 0.10
                model_adjustments['N'] -= 0.05  # Reduce draw when clear favorite
            else:  # Away stronger
                model_adjustments['D'] += 0.10
                model_adjustments['V'] -= 0.10
                model_adjustments['N'] -= 0.05  # Reduce draw when clear favorite
        
        # Add draw boost for close matches
        if abs(v_prob - d_prob) < 0.05:  # Very close teams
            model_adjustments['N'] += 0.15  # Strong draw boost for balanced matches
        
        # Apply adjustments to odds base (very aggressive corrections allowed)
        combined = {}
        for t in ['V', 'N', 'D']:
            adjustment = max(-0.20, min(0.20, model_adjustments[t]))  # ±20% cap (was ±15%)
            combined[t] = odds_base[t] + adjustment
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        
        # CORRECTED CAPS: Based on real statistics
        combined['V'] = max(0.35, min(0.60, combined['V']))  # V = 48% in reality
        combined['N'] = max(0.20, min(0.35, combined['N']))  # N = 28% in reality  
        combined['D'] = max(0.15, min(0.35, combined['D']))  # D = 24% in reality  # D: 20-40% (base: 30%)
        
        # Final normalization
        total = sum(combined.values())
        combined = {k: v / total for k, v in combined.items()}
        
        # REALISTIC DISTRIBUTION: Based on 9000+ actual match results
        # Real football statistics: V=48%, N=28%, D=24%
        real_distribution = {'V': 0.48, 'N': 0.28, 'D': 0.24}
        
        # Blend 90% calculated + 10% realistic distribution for balance
        for key in combined:
            combined[key] = combined[key] * 0.9 + real_distribution[key] * 0.1
        
        # Final normalization to ensure 100%
        total = sum(combined.values())
        combined = {k: v / total for k, v in combined.items()}
        
        # APPLY AGGRESSIVE LEARNER CORRECTIONS
        try:
            from app.services.aggressive_learner import AggressiveLearner, run_aggressive_learning
            
            # First, update learning with latest data
            learner = AggressiveLearner()
            learner.load_learning_state()
            
            # Update weights from all history (automatic learning)
            corrections = learner.update_weights_from_history()
            learner.save_learning_state()
            
            # Then apply corrections to current prediction
            combined = learner.apply_corrections_to_prediction(combined, context)
            
            print(f"✅ AggressiveLearner updated: {len(corrections)} corrections applied")
        except Exception as e:
            print(f"Warning: Could not apply AggressiveLearner corrections: {e}")
        
        # APPLY CONTEXTUAL PATTERNS
        try:
            from app.services.contextual_patterns_learner import ContextualPatternsLearner
            
            pattern_learner = ContextualPatternsLearner()
            pattern_learner.load_patterns()
            
            if pattern_learner.learned_patterns:
                # Apply patterns si disponible
                if context and 'match' in context:
                    combined = pattern_learner.apply_patterns_to_prediction(
                        context['match'], 
                        db, 
                        combined
                    )
                    print("✅ Contextual patterns applied")
                else:
                    # Si pas de contexte match, essayer quand même
                    # Créer un contexte minimal
                    minimal_context = {'match': context} if context else {}
                    try:
                        combined = pattern_learner.apply_patterns_to_prediction(
                            minimal_context.get('match'), 
                            db, 
                            combined
                        )
                        print("✅ Contextual patterns applied (minimal context)")
                    except:
                        print("⚠️ Could not apply patterns (no valid context)")
        except Exception as e:
            print(f"Warning: Could not apply contextual patterns: {e}")
        
        return combined
    
    def compute_h2h_probabilities(self, h2h_dominance: float, 
                                  h2h_home_win_rate: float,
                                  h2h_away_win_rate: float,
                                  h2h_strict_home_win_rate: float = 0.0,
                                  h2h_strict_away_win_rate: float = 0.0,
                                  h2h_strict_total_matches: int = 0) -> Dict:
        """
        Convert H2H features to probability predictions.
        Now uses strict home/away separation for better accuracy.
        """
        # If we have strict H2H data, use it with higher weight (increased from 0.7)
        if h2h_strict_total_matches and h2h_strict_total_matches >= 2:
            # Weight based on number of H2H matches (more matches = more reliable)
            # Increased maximum weight from 0.7 to 0.85 since we now use ALL H2H matches
            weight = min(0.85, 0.4 + h2h_strict_total_matches * 0.03)
            
            # Use strict rates: home team at home vs away team
            strict_home = h2h_strict_home_win_rate if h2h_strict_home_win_rate else 0.33
            strict_away = h2h_strict_away_win_rate if h2h_strict_away_win_rate else 0.33
            
            # Combine: home team's home performance + away team's home performance (inverted)
            prob_V = strict_home * weight + (1 - strict_away) * (1 - weight) * 0.5
            prob_D = (1 - strict_home) * weight + strict_away * (1 - weight) * 0.5
            prob_N = 0.25 * (1 - abs(prob_V - prob_D))  # Draw more likely when teams are even
        else:
            # Fallback to basic H2H features - BALANCED defaults
            base_home = 0.38
            base_draw = 0.32
            base_away = 0.30
            
            # Adjust based on dominance score (-1 to 1)
            dominance_adjustment = h2h_dominance * 0.15
            
            # Adjust based on win rates
            home_adjustment = (h2h_home_win_rate - 0.5) * 0.1
            away_adjustment = (h2h_away_win_rate - 0.5) * 0.1
            
            prob_V = base_home + dominance_adjustment + home_adjustment - away_adjustment
            prob_D = base_away - dominance_adjustment - home_adjustment + away_adjustment
            prob_N = base_draw - abs(dominance_adjustment) * 0.3
        
        # Normalize
        total = prob_V + prob_N + prob_D
        return {
            'V': max(0.05, min(0.85, prob_V / total)),
            'N': max(0.05, min(0.40, prob_N / total)),
            'D': max(0.05, min(0.85, prob_D / total))
        }
    
    def calibrate_probabilities(self, method_name: str, probs: Dict, 
                                features: MatchFeatures = None) -> Dict:
        """
        Calibrate probabilities using Platt scaling or isotonic regression.
        """
        if method_name not in self.calibrators:
            return probs
        
        calibrator = self.calibrators[method_name]
        
        # Apply calibration
        calibrated = {}
        for outcome in ['V', 'N', 'D']:
            raw_prob = probs.get(outcome, 0.33)
            try:
                if isinstance(calibrator, IsotonicRegression):
                    calibrated[outcome] = float(calibrator.predict([raw_prob])[0])
                else:
                    calibrated[outcome] = float(calibrator.predict_proba([[raw_prob]])[0][1])
            except:
                calibrated[outcome] = raw_prob
        
        # Normalize
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in calibrated.items()}
        
        return calibrated
    
    def train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train meta-model that combines all method outputs.
        Input: predictions from each method
        Output: final optimized prediction
        """
        logger.info(f"Training meta-model on {len(X)} samples...")
        
        # Use logistic regression as meta-learner
        self.meta_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            C=0.1, # Regularization
            random_state=42
        )
        
        self.meta_model.fit(X, y)
        
        logger.info("Meta-model trained successfully")
    
    def predict_with_meta_model(self, ml_probs: Dict, mc_probs: Dict,
                                poisson_probs: Dict, elo_probs: Dict,
                                h2h_probs: Dict, confidence: float) -> Dict:
        """
        Use meta-model for final prediction.
        """
        if self.meta_model is None:
            # Fall back to weighted combination
            return self.combine_predictions(ml_probs, mc_probs, poisson_probs, elo_probs, h2h_probs)
        
        # Prepare features for meta-model
        features = np.array([[
            ml_probs.get('V', 0.33), ml_probs.get('N', 0.33), ml_probs.get('D', 0.33),
            mc_probs.get('prob_home_win', 0.33), mc_probs.get('prob_draw', 0.33), mc_probs.get('prob_away_win', 0.33),
            poisson_probs.get('prob_home_win', 0.33), poisson_probs.get('prob_draw', 0.33), poisson_probs.get('prob_away_win', 0.33),
            elo_probs.get('V', 0.33), elo_probs.get('N', 0.33), elo_probs.get('D', 0.33),
            h2h_probs.get('V', 0.33), h2h_probs.get('N', 0.33), h2h_probs.get('D', 0.33),
            confidence
        ]])
        
        try:
            meta_probs = self.meta_model.predict_proba(features)[0]
            classes = self.meta_model.classes_
            
            return {
                'V': float(meta_probs[list(classes).index('V')] if 'V' in classes else 0.33),
                'N': float(meta_probs[list(classes).index('N')] if 'N' in classes else 0.33),
                'D': float(meta_probs[list(classes).index('D')] if 'D' in classes else 0.33)
            }
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}")
            return self.combine_predictions(ml_probs, mc_probs, poisson_probs, elo_probs, h2h_probs)
    
    def save(self, path: Path = None):
        """Save ensemble state."""
        if path is None:
            path = settings.MODELS_DIR
        
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.method_weights, path / "adaptive_weights.joblib")
        joblib.dump(self.calibrators, path / "calibrators.joblib")
        if self.meta_model:
            joblib.dump(self.meta_model, path / "meta_model.joblib")
        
        logger.info("Adaptive ensemble saved")
    
    def load(self, path: Path = None):
        """Load ensemble state."""
        if path is None:
            path = settings.MODELS_DIR
        
        try:
            self.method_weights = joblib.load(path / "adaptive_weights.joblib")
            self.calibrators = joblib.load(path / "calibrators.joblib")
            
            meta_path = path / "meta_model.joblib"
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
            
            logger.info("Adaptive ensemble loaded")
        except Exception as e:
            logger.warning(f"Could not load adaptive ensemble: {e}")


class PredictionFilter:
    """
    Prediction filtering for match predictions.
    BALANCED thresholds to maximize predictions while maintaining quality.
    Now uses adaptive filters that adjust based on recent performance.
    """
    
    def __init__(self, min_odds: float = 1.50, min_confidence: float = 0.40,
                 min_gap: float = 0.05, min_value: float = 0.05):
        self.min_odds = min_odds  # Accept lower odds favorites
        self.min_confidence = min_confidence  # 40% minimum - BALANCED
        self.min_gap = min_gap  # Smaller gap required
        self.min_value = min_value  # 5% minimum value edge
        
        # Try to load adaptive filters
        self._load_adaptive_filters()
    
    def _load_adaptive_filters(self):
        """Load adaptive filter thresholds from file."""
        try:
            from app.services.adaptive_filters import AdaptiveFilters
            filters = AdaptiveFilters()
            if filters.load_filters():
                # Apply adaptive filters if available
                self.min_confidence = filters.current_filters.get('min_confidence', self.min_confidence)
                self.min_value = filters.current_filters.get('min_value', self.min_value)
                logger.info("✅ Using adaptive filters")
        except Exception as e:
            logger.warning(f"Could not load adaptive filters: {e}")
    
    def should_predict(self, probs: Dict, odds: Dict[str, float] = None) -> Tuple[bool, str]:
        """
        Determine if prediction meets BALANCED quality criteria.
        Allow more predictions through while filtering obvious bad ones.
        Now reloads adaptive filters for latest adjustments.
        """
        # Reload adaptive filters to get latest values
        self._load_adaptive_filters()
        
        max_prob = max(probs.values())
        
        # BALANCED: Require only 40% confidence (above random 33%)
        if max_prob < self.min_confidence:
            return False, f"Confiance {max_prob:.1%} insuffisante (min: {self.min_confidence:.0%})"
        
        # Check gap between top two probabilities - relaxed
        sorted_probs = sorted(probs.values(), reverse=True)
        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
            if gap < self.min_gap:
                return False, f"Écart insuffisant ({gap:.1%} < {self.min_gap:.0%})"
        
        # Accept prediction
        return True, "Prédiction approuvée"
    
    def find_value_bets(self, probs: Dict, odds: Dict[str, float],
                        min_confidence: float = 0.45) -> List[Dict]:
        """
        Find value bets meeting BALANCED criteria.
        More permissive to capture more opportunities.
        """
        value_bets = []
        
        outcome_map = {'V': 'home', 'N': 'draw', 'D': 'away'}
        outcome_names = {'V': 'Victoire Domicile', 'N': 'Match Nul', 'D': 'Victoire Extérieur'}
        
        for outcome, prob in probs.items():
            odd_key = outcome_map[outcome]
            odd = odds.get(odd_key, 0)
            
            if odd <= 0:
                continue
            
            implied_prob = 1 / odd
            value = prob - implied_prob
            value_percent = value / implied_prob if implied_prob > 0 else 0
            
            # BALANCED criteria: moderate confidence + some value
            # Value >= 5% edge OR confidence >= 45%
            if (value >= self.min_value or value_percent >= 0.08) and prob >= min_confidence and odd >= self.min_odds:
                value_bets.append({
                    'outcome': outcome,
                    'outcome_name': outcome_names[outcome],
                    'model_probability': prob,
                    'implied_probability': implied_prob,
                    'value': value,
                    'value_percent': value_percent,
                    'odds': odd,
                    'confidence': prob,
                    'is_value_bet': True
                })
        
        # Sort by value percentage (relative value)
        value_bets.sort(key=lambda x: x['value_percent'], reverse=True)
        
        return value_bets
    
    def evaluate_prediction_quality(self, probs: Dict, model_agreement: float = 0.0) -> Dict:
        """
        Evaluate overall prediction quality.
        """
        max_prob = max(probs.values())
        sorted_probs = sorted(probs.values(), reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
        
        # Entropy (lower = more certain)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs.values())
        max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
        normalized_entropy = entropy / max_entropy
        
        # Quality score (0-1)
        quality = (
            max_prob * 0.4 +
            (1 - normalized_entropy) * 0.3 +
            model_agreement * 0.2 +
            min(gap * 5, 1.0) * 0.1
        )
        
        return {
            'quality_score': quality,
            'confidence': max_prob,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'gap': gap,
            'model_agreement': model_agreement,
            'quality_level': 'high' if quality >= 0.7 else 'medium' if quality >= 0.5 else 'low'
        }


def update_method_performance(db: Session, method_name: str, 
                              predicted_probs: Dict, actual_result: str):
    """
    Update performance tracking for a prediction method.
    """
    perf = db.query(MethodPerformance).filter(
        MethodPerformance.method_name == method_name
    ).first()
    
    if not perf:
        perf = MethodPerformance(method_name=method_name)
        db.add(perf)
    
    perf.update_prediction(predicted_probs, actual_result)
    perf.calculate_dynamic_weight()
    
    db.commit()
    
    return perf
