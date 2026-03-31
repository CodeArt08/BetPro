"""Enhanced Continuous Learning with per-method performance tracking."""
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy.orm import Session
from datetime import datetime
import numpy as np
import pandas as pd

from app.core.config import settings
from app.models import Match, Prediction, ModelMetrics, MatchFeatures, MethodPerformance
from app.services.ml_ensemble import MachineLearningEnsemble
from app.services.team_strength import TeamStrengthEngine
from app.services.sequence_analysis import SequencePatternAnalyzer
from app.services.adaptive_ensemble import AdaptiveEnsemble, update_method_performance
from app.services.advanced_features import AdvancedFeatureEngine


class EnhancedContinuousLearning:
    """
    Enhanced continuous learning with:
    - Per-method performance tracking
    - Automatic retraining every 50 matches
    - Dynamic weight updates
    - Full historical data training
    """
    
    # Retraining configuration
    RETRAIN_INTERVAL = 50  # Retrain after every 50 new verified predictions
    MIN_TRAINING_SAMPLES = 100
    
    def __init__(self):
        self.ml_ensemble = MachineLearningEnsemble()
        self.team_strength = TeamStrengthEngine()
        self.sequence_analyzer = SequencePatternAnalyzer()
        self.adaptive_ensemble = AdaptiveEnsemble()
        self.advanced_features = AdvancedFeatureEngine()
        
        self.predictions_since_retrain = 0
        self.last_retrain_time: Optional[datetime] = None
        self.matches_processed = 0
    
    def on_match_completed(self, match: Match, db: Session, 
                          prediction: Prediction = None):
        """
        Called when a match is completed to update all models and track performance.
        """
        logger.info(f"Processing completed match {match.id}")
        
        # 1. Update team strength ratings
        self.team_strength.update_all_ratings(match, db)
        
        # 2. Update sequence patterns
        self._update_sequences(match, db)
        
        # 3. Get prediction if not provided
        if not prediction:
            prediction = db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).first()
        
        if prediction:
            # 4. Verify prediction
            prediction.verify(match.result)
            
            # 5. Update per-method performance
            self._update_all_method_performance(prediction, match.result, db)
            
            # 6. Update model metrics
            self._update_model_metrics(prediction, match.result, db)
        
        # 7. Check if retraining needed
        self.predictions_since_retrain += 1
        self.matches_processed += 1
        
        if self.predictions_since_retrain >= self.RETRAIN_INTERVAL:
            self._retrain_models(db)
        
        db.commit()
    
    def _update_sequences(self, match: Match, db: Session):
        """Update sequence patterns for both teams."""
        home_result = match.result
        inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
        away_result = inverted.get(match.result, 'N')
        
        self.sequence_analyzer.update_sequence(match.home_team_id, home_result)
        self.sequence_analyzer.update_sequence(match.away_team_id, away_result)
    
    def _update_all_method_performance(self, prediction: Prediction, 
                                       actual_result: str, db: Session):
        """Update performance for each prediction method."""
        
        # ML Ensemble
        if prediction.model_outputs:
            ml_probs = prediction.model_outputs.get('ensemble', 
                      {'V': prediction.prob_home_win, 'N': prediction.prob_draw, 'D': prediction.prob_away_win})
            update_method_performance(db, 'ml_ensemble', ml_probs, actual_result)
        
        # Monte Carlo
        if prediction.monte_carlo_results:
            mc_probs = {
                'V': prediction.monte_carlo_results.get('prob_home_win', 0.33),
                'N': prediction.monte_carlo_results.get('prob_draw', 0.33),
                'D': prediction.monte_carlo_results.get('prob_away_win', 0.33)
            }
            update_method_performance(db, 'monte_carlo', mc_probs, actual_result)
        
        # Extract method outputs from extra data if available
        extra = prediction.__dict__.get('extra_data', {}) or {}
        
        # Poisson
        if 'poisson_probs' in extra:
            update_method_performance(db, 'poisson', extra['poisson_probs'], actual_result)
        
        # ELO
        if 'elo_probs' in extra:
            update_method_performance(db, 'elo', extra['elo_probs'], actual_result)
        
        # H2H
        if 'h2h_probs' in extra:
            update_method_performance(db, 'h2h', extra['h2h_probs'], actual_result)
    
    def _update_model_metrics(self, prediction: Prediction, actual_result: str, db: Session):
        """Update metrics for each ML model."""
        if not prediction.model_outputs:
            return
        
        for model_name, output in prediction.model_outputs.items():
            if model_name == 'ensemble':
                continue
            
            metrics = db.query(ModelMetrics).filter(
                ModelMetrics.model_name == model_name
            ).first()
            
            if not metrics:
                metrics = ModelMetrics(
                    model_name=model_name,
                    model_version='1.0'
                )
                db.add(metrics)
            
            predicted = max(output, key=output.get)
            metrics.update_prediction(predicted, actual_result, output)
            metrics.calculate_ensemble_weight()
    
    def _retrain_models(self, db: Session):
        """Retrain ML models with ALL historical data."""
        logger.info("Starting full data model retraining...")
        
        # Get ALL completed matches with features (no limit)
        query = db.query(Match).filter(Match.is_completed == True)
        matches = query.all()
        
        total_matches = len(matches)
        logger.info(f"Training on {total_matches} completed matches")
        
        if total_matches < self.MIN_TRAINING_SAMPLES:
            logger.warning(f"Not enough data for retraining ({total_matches} < {self.MIN_TRAINING_SAMPLES})")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for match in matches:
            features = db.query(MatchFeatures).filter(
                MatchFeatures.match_id == match.id
            ).first()
            
            if features:
                X.append(features.to_dict())
                y.append(match.result)
        
        if len(X) < self.MIN_TRAINING_SAMPLES:
            logger.warning(f"Not enough features for retraining ({len(X)} < {self.MIN_TRAINING_SAMPLES})")
            return
        
        df_X = pd.DataFrame(X)
        df_y = pd.Series(y)
        
        logger.info(f"Training ensemble on {len(df_X)} samples with {len(df_X.columns)} features")
        
        # Train ensemble
        self.ml_ensemble.train(df_X, df_y, db)
        
        # Save models
        self.ml_ensemble.save_models()
        
        # Update weights from metrics
        self.ml_ensemble.update_weights_from_metrics(db)
        
        # Rebuild sequence patterns
        self.sequence_analyzer.analyze_patterns(db)
        
        # Train meta-model if we have enough verified predictions
        self._train_meta_model(db)
        
        # Update adaptive weights based on recent performance
        from app.services.adaptive_weights_manager import update_adaptive_weights
        new_weights = update_adaptive_weights(db)
        logger.info(f"Updated adaptive weights: {new_weights}")
        
        # Update adaptive filters based on recent performance
        from app.services.adaptive_filters import update_adaptive_filters
        new_filters = update_adaptive_filters(db)
        logger.info(f"Updated adaptive filters: {new_filters}")
        
        # Learn contextual patterns from historical data
        from app.services.contextual_patterns_learner import learn_contextual_patterns
        patterns = learn_contextual_patterns(db)
        logger.info(f"Learned contextual patterns: {len(patterns)} pattern types")
        
        # Save adaptive ensemble
        self.adaptive_ensemble.save()
        
        self.predictions_since_retrain = 0
        self.last_retrain_time = datetime.utcnow()
        
        logger.info(f"Model retraining completed on {total_matches} matches")
    
    def _train_meta_model(self, db: Session):
        """Train the meta-model using verified predictions."""
        # Get verified predictions with method outputs
        predictions = db.query(Prediction).filter(
            Prediction.actual_result != None
        ).limit(500).all()
        
        if len(predictions) < 50:
            logger.info("Not enough verified predictions for meta-model training")
            return
        
        X = []
        y = []
        
        for pred in predictions:
            # Build feature vector from method outputs
            features = []
            
            # ML probabilities
            ml_out = pred.model_outputs or {}
            ensemble = ml_out.get('ensemble', {})
            features.extend([
                ensemble.get('V', pred.prob_home_win),
                ensemble.get('N', pred.prob_draw),
                ensemble.get('D', pred.prob_away_win)
            ])
            
            # Monte Carlo probabilities
            mc = pred.monte_carlo_results or {}
            features.extend([
                mc.get('prob_home_win', 0.33),
                mc.get('prob_draw', 0.33),
                mc.get('prob_away_win', 0.33)
            ])
            
            # Use stored probabilities as proxy for other methods
            features.extend([
                pred.prob_home_win,
                pred.prob_draw,
                pred.prob_away_win,
                pred.confidence
            ])
            
            X.append(features)
            y.append(pred.actual_result)
        
        if len(X) >= 50:
            X = np.array(X)
            y = np.array(y)
            
            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            self.adaptive_ensemble.train_meta_model(X, y_encoded)
            logger.info(f"Meta-model trained on {len(X)} verified predictions")
    
    def load_models(self):
        """Load saved models on startup."""
        try:
            self.ml_ensemble.load_models()
            self.adaptive_ensemble.load()
            logger.info("Loaded saved ML models and adaptive ensemble")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def ensure_models_trained(self, db: Session, min_samples: int = 50):
        """Ensure ML models are trained with ALL available data."""
        if self.ml_ensemble.is_trained:
            feature_mismatch = self._check_feature_mismatch(db)
            if not feature_mismatch:
                return
            logger.warning("Feature mismatch detected. Retraining...")
        
        logger.info("Training models from historical data...")
        
        # Get ALL completed matches WITH features
        matches = (
            db.query(Match)
            .join(MatchFeatures, MatchFeatures.match_id == Match.id)
            .filter(Match.is_completed == True)
            .all()
        )
        
        total_matches = len(matches)
        logger.info(f"Found {total_matches} completed matches with features")
        
        if total_matches < min_samples:
            logger.warning(f"Not enough samples ({total_matches} < {min_samples})")
            return
        
        X = []
        y = []
        
        for match in matches:
            features = db.query(MatchFeatures).filter(
                MatchFeatures.match_id == match.id
            ).first()
            if features:
                X.append(features.to_dict())
                y.append(match.result)
        
        df_X = pd.DataFrame(X)
        df_y = pd.Series(y)
        
        logger.info(f"Training on {len(df_X)} samples")
        
        self.ml_ensemble.train(df_X, df_y, db)
        self.ml_ensemble.save_models()
        self.ml_ensemble.update_weights_from_metrics(db)
        
        logger.info(f"Initial training completed on {total_matches} matches")
    
    def _check_feature_mismatch(self, db: Session) -> bool:
        """Check for feature schema mismatch."""
        sample_features = (
            db.query(MatchFeatures)
            .join(Match, Match.id == MatchFeatures.match_id)
            .first()
        )
        
        if not sample_features:
            return False
        
        try:
            feature_dict = sample_features.to_dict()
            current_count = len(feature_dict)
            
            scaler_count = self.ml_ensemble.scaler.n_features_in_ if hasattr(
                self.ml_ensemble.scaler, 'n_features_in_') else None
            
            if scaler_count and scaler_count != current_count:
                logger.warning(f"Feature mismatch: {scaler_count} vs {current_count}")
                self.ml_ensemble.is_trained = False
                return True
            
            self.ml_ensemble.predict(sample_features)
            return False
        except Exception as e:
            logger.warning(f"Feature mismatch check failed: {e}")
            return True
    
    def get_learning_status(self, db: Session) -> Dict:
        """Get comprehensive learning status."""
        # Overall predictions
        total_verified = db.query(Prediction).filter(
            Prediction.actual_result != None
        ).count()
        
        correct = db.query(Prediction).filter(
            Prediction.actual_result != None,
            Prediction.is_correct == True
        ).count()
        
        # Method performance
        method_perfs = db.query(MethodPerformance).all()
        method_stats = {}
        
        for mp in method_perfs:
            method_stats[mp.method_name] = {
                'accuracy': mp.accuracy,
                'recent_accuracy': mp.recent_accuracy,
                'dynamic_weight': mp.dynamic_weight,
                'total_predictions': mp.total_predictions,
                'avg_brier_score': mp.avg_brier_score
            }
        
        # Model metrics
        model_metrics = db.query(ModelMetrics).all()
        
        return {
            'total_predictions_verified': total_verified,
            'correct_predictions': correct,
            'overall_accuracy': correct / total_verified if total_verified > 0 else 0,
            'predictions_since_retrain': self.predictions_since_retrain,
            'retrain_threshold': self.RETRAIN_INTERVAL,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'matches_processed': self.matches_processed,
            'method_performance': method_stats,
            'model_metrics': [{
                'model_name': m.model_name,
                'accuracy': m.accuracy,
                'ensemble_weight': m.ensemble_weight,
                'recent_accuracy': m.recent_accuracy
            } for m in model_metrics],
            'adaptive_weights': self.adaptive_ensemble.method_weights
        }
