"""Continuous Learning Engine."""
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy.orm import Session
from datetime import datetime
import numpy as np

from app.core.config import settings
from app.models import Match, Prediction, ModelMetrics, MatchFeatures
from app.services.ml_ensemble import MachineLearningEnsemble
from app.services.team_strength import TeamStrengthEngine
from app.services.sequence_analysis import SequencePatternAnalyzer


class ContinuousLearningEngine:
    """
    Manages continuous learning and model updates.
    """
    
    def __init__(self):
        self.ml_ensemble = MachineLearningEnsemble()
        self.team_strength = TeamStrengthEngine()
        self.sequence_analyzer = SequencePatternAnalyzer()
        
        self.predictions_since_retrain = 0
        self.last_retrain_time: Optional[datetime] = None
    
    def on_match_completed(self, match: Match, db: Session):
        """
        Called when a match is completed to update all models.
        """
        logger.info(f"Processing completed match {match.id}")
        
        # 1. Update team strength ratings
        self.team_strength.update_all_ratings(match, db)
        
        # 2. Update sequence patterns
        self._update_sequences(match, db)
        
        # 3. Verify predictions
        self._verify_predictions(match, db)
        
        # 4. Update model metrics
        self._update_model_metrics(match, db)
        
        # 5. Check if retraining needed
        self.predictions_since_retrain += 1
        if self.predictions_since_retrain >= settings.MODEL_RETRAIN_INTERVAL:
            self._retrain_models(db)
    
    def _update_sequences(self, match: Match, db: Session):
        """Update sequence patterns for both teams."""
        # Update home team sequence
        home_result = match.result  # V, N, or D from home perspective
        
        # Update away team sequence (inverted)
        inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
        away_result = inverted.get(match.result, 'N')
        
        self.sequence_analyzer.update_sequence(match.home_team_id, home_result)
        self.sequence_analyzer.update_sequence(match.away_team_id, away_result)
    
    def _verify_predictions(self, match: Match, db: Session):
        """Verify and update predictions for the match."""
        prediction = db.query(Prediction).filter(Prediction.match_id == match.id).first()
        
        if prediction:
            prediction.verify(match.result)
            db.commit()
            
            logger.info(f"Prediction verified: predicted={prediction.predicted_result}, "
                       f"actual={match.result}, correct={prediction.is_correct}")
    
    def _update_model_metrics(self, match: Match, db: Session):
        """Update metrics for each model."""
        prediction = db.query(Prediction).filter(Prediction.match_id == match.id).first()
        
        if not prediction or not prediction.model_outputs:
            return
        
        for model_name, output in prediction.model_outputs.items():
            metrics = db.query(ModelMetrics).filter(
                ModelMetrics.model_name == model_name
            ).first()
            
            if not metrics:
                metrics = ModelMetrics(
                    model_name=model_name,
                    model_version='1.0'
                )
                db.add(metrics)
            
            # Get predicted outcome for this model
            predicted = max(output, key=output.get)
            
            metrics.update_prediction(predicted, match.result, output)
            
            # Update ensemble weight
            metrics.calculate_ensemble_weight()
            
            db.commit()
    
    def _retrain_models(self, db: Session):
        """Retrain ML models with all historical data."""
        logger.info("Starting model retraining...")
        
        # Get all completed matches with features
        matches = db.query(Match).filter(Match.is_completed == True).all()
        
        if len(matches) < 50:
            logger.warning("Not enough data for retraining")
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
        
        if len(X) < 50:
            logger.warning("Not enough features for retraining")
            return
        
        import pandas as pd
        df_X = pd.DataFrame(X)
        df_y = pd.Series(y)
        
        # Train ensemble
        self.ml_ensemble.train(df_X, df_y, db)
        
        # Save models
        self.ml_ensemble.save_models()
        
        # Update weights from metrics
        self.ml_ensemble.update_weights_from_metrics(db)
        
        # Rebuild sequence patterns
        self.sequence_analyzer.analyze_patterns(db)
        
        self.predictions_since_retrain = 0
        self.last_retrain_time = datetime.utcnow()
        
        logger.info("Model retraining completed")
    
    def load_models(self):
        """Load saved models on startup."""
        try:
            self.ml_ensemble.load_models()
            logger.info("Loaded saved ML models")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")

    def ensure_models_trained(self, db: Session, min_samples: int = 50):
        """Ensure ML models are trained.

        If no saved models could be loaded (ml_ensemble.is_trained=False), train once
        from historical completed matches that already have features.
        Also detects feature mismatch between saved models and current feature schema.
        """
        if self.ml_ensemble.is_trained:
            # Check for feature mismatch by doing a test prediction
            feature_mismatch = self._check_feature_mismatch(db)
            if not feature_mismatch:
                return
            logger.warning("Feature mismatch detected between saved models and current schema. Retraining...")

        logger.warning("ML ensemble not trained. Attempting initial training from historical data...")

        # Use only completed matches WITH features to avoid expensive feature recomputation at startup.
        matches = (
            db.query(Match)
            .join(MatchFeatures, MatchFeatures.match_id == Match.id)
            .filter(Match.is_completed == True)
            .all()
        )

        if len(matches) < min_samples:
            logger.warning(f"Not enough historical samples to train ({len(matches)} < {min_samples}).")
            return

        X = []
        y = []
        for match in matches:
            features = db.query(MatchFeatures).filter(MatchFeatures.match_id == match.id).first()
            if not features:
                continue
            X.append(features.to_dict())
            y.append(match.result)

        if len(X) < min_samples:
            logger.warning(f"Not enough feature rows to train ({len(X)} < {min_samples}).")
            return

        import pandas as pd

        df_X = pd.DataFrame(X)
        df_y = pd.Series(y)

        self.ml_ensemble.train(df_X, df_y, db)
        self.ml_ensemble.save_models()
        self.ml_ensemble.update_weights_from_metrics(db)
        logger.info("Initial ML training completed and models saved.")

    def _check_feature_mismatch(self, db: Session) -> bool:
        """Check if there's a feature mismatch between loaded models and current schema.

        Returns True if mismatch detected (models need retraining), False otherwise.
        """
        # Get a sample match with features to test prediction
        sample_features = (
            db.query(MatchFeatures)
            .join(Match, Match.id == MatchFeatures.match_id)
            .first()
        )

        if not sample_features:
            logger.warning("No sample features available to check for mismatch")
            return False

        try:
            # Try to prepare features and run through scaler
            import numpy as np
            feature_dict = sample_features.to_dict()
            current_feature_count = len(feature_dict)

            # Check if scaler's feature count matches
            scaler_feature_count = self.ml_ensemble.scaler.n_features_in_ if hasattr(self.ml_ensemble.scaler, 'n_features_in_') else None

            if scaler_feature_count is not None and scaler_feature_count != current_feature_count:
                logger.warning(f"Feature count mismatch: scaler expects {scaler_feature_count}, current schema has {current_feature_count}")
                self.ml_ensemble.is_trained = False
                return True

            # Try actual prediction to catch any other issues
            self.ml_ensemble.predict(sample_features)
            return False

        except ValueError as e:
            if "features" in str(e).lower():
                logger.warning(f"Feature mismatch detected: {e}")
                self.ml_ensemble.is_trained = False
                return True
            raise
        except Exception as e:
            logger.warning(f"Error during feature mismatch check: {e}")
            return False
    
    def get_learning_status(self, db: Session) -> Dict:
        """Get current learning status."""
        # Get overall prediction accuracy
        total_predictions = db.query(Prediction).filter(
            Prediction.actual_result != None
        ).count()
        
        correct_predictions = db.query(Prediction).filter(
            Prediction.actual_result != None,
            Prediction.is_correct == True
        ).count()
        
        # Get model metrics
        model_metrics = db.query(ModelMetrics).all()
        
        return {
            'total_predictions_verified': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'predictions_since_retrain': self.predictions_since_retrain,
            'retrain_threshold': settings.MODEL_RETRAIN_INTERVAL,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'model_metrics': [{
                'model_name': m.model_name,
                'accuracy': m.accuracy,
                'ensemble_weight': m.ensemble_weight,
                'total_predictions': m.total_predictions
            } for m in model_metrics]
        }
    
    def calculate_prediction_confidence(self, prediction: Prediction, db: Session) -> float:
        """
        Calculate confidence score combining multiple factors.
        """
        # Factor 1: Probability strength
        max_prob = max(prediction.prob_home_win, prediction.prob_draw, prediction.prob_away_win)
        
        # Factor 2: Model agreement
        agreement = self.ml_ensemble.calculate_model_agreement(prediction.model_outputs)
        
        # Factor 3: Historical accuracy (from model metrics)
        model_metrics = db.query(ModelMetrics).all()
        avg_accuracy = np.mean([m.accuracy for m in model_metrics]) if model_metrics else 0.5
        
        # Combined confidence
        confidence = (
            max_prob * 0.4 +
            agreement * 0.3 +
            avg_accuracy * 0.3
        )
        
        return min(confidence, 1.0)
    
    def export_learning_report(self, db: Session, output_path: str):
        """Export learning report to file."""
        import json
        
        status = self.get_learning_status(db)
        
        # Get recent predictions
        recent_predictions = db.query(Prediction).filter(
            Prediction.actual_result != None
        ).order_by(Prediction.verified_at.desc()).limit(100).all()
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'learning_status': status,
            'recent_predictions': [{
                'match_id': p.match_id,
                'predicted': p.predicted_result,
                'actual': p.actual_result,
                'correct': p.is_correct,
                'confidence': p.confidence,
                'brier_score': p.prediction_error
            } for p in recent_predictions]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Learning report exported to {output_path}")
