"""Method Performance Tracking for adaptive ensemble weighting."""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class MethodPerformance(Base):
    """Tracks performance for each prediction method (ML, MC, Poisson, ELO, H2H)."""
    __tablename__ = "method_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Method identification
    method_name = Column(String(50), nullable=False, unique=True, index=True)
    # Values: 'ml_ensemble', 'monte_carlo', 'poisson', 'elo', 'h2h'
    
    # Overall metrics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    
    # Recent performance (last 100 predictions)
    recent_accuracy = Column(Float, default=0.0)
    recent_predictions_list = Column(JSON, default=list)
    
    # Log loss and Brier score
    avg_log_loss = Column(Float, default=0.0)
    avg_brier_score = Column(Float, default=0.0)
    
    # Dynamic weight (updated after each prediction)
    dynamic_weight = Column(Float, default=1.0)
    
    # Per-outcome accuracy
    home_win_accuracy = Column(Float, default=0.0)
    draw_accuracy = Column(Float, default=0.0)
    away_win_accuracy = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<MethodPerformance(method={self.method_name}, accuracy={self.accuracy:.3f}, weight={self.dynamic_weight:.3f})>"
    
    def update_prediction(self, predicted_probs: dict, actual_result: str):
        """Update metrics after a prediction is verified."""
        import numpy as np
        
        self.total_predictions += 1
        
        # Get predicted outcome
        predicted = max(predicted_probs, key=predicted_probs.get)
        
        if predicted == actual_result:
            self.correct_predictions += 1
        
        self.accuracy = self.correct_predictions / self.total_predictions
        
        # Calculate Brier score
        actual_onehot = {'V': 0.0, 'N': 0.0, 'D': 0.0}
        actual_onehot[actual_result] = 1.0
        brier = sum((predicted_probs.get(k, 0.33) - v) ** 2 for k, v in actual_onehot.items()) / 3
        
        # Update average Brier score
        if self.avg_brier_score == 0:
            self.avg_brier_score = brier
        else:
            self.avg_brier_score = (self.avg_brier_score * (self.total_predictions - 1) + brier) / self.total_predictions
        
        # Calculate log loss
        prob_actual = predicted_probs.get(actual_result, 0.33)
        prob_actual = max(prob_actual, 1e-10)  # Avoid log(0)
        log_loss = -np.log(prob_actual)
        
        if self.avg_log_loss == 0:
            self.avg_log_loss = log_loss
        else:
            self.avg_log_loss = (self.avg_log_loss * (self.total_predictions - 1) + log_loss) / self.total_predictions
        
        # Update recent predictions
        recent = self.recent_predictions_list or []
        recent.append({
            'predicted': predicted,
            'actual': actual_result,
            'correct': predicted == actual_result,
            'brier': brier
        })
        self.recent_predictions_list = recent[-100:]  # Keep last 100
        
        # Calculate recent accuracy
        if self.recent_predictions_list:
            self.recent_accuracy = sum(1 for r in self.recent_predictions_list if r['correct']) / len(self.recent_predictions_list)
        
        # Update per-outcome accuracy
        self._update_outcome_accuracy(predicted, actual_result)
    
    def _update_outcome_accuracy(self, predicted: str, actual: str):
        """Update per-outcome accuracy tracking."""
        # Track accuracy when predicting each outcome
        recent = self.recent_predictions_list or []
        
        for outcome in ['V', 'N', 'D']:
            outcome_preds = [r for r in recent if r['predicted'] == outcome]
            if outcome_preds:
                correct = sum(1 for r in outcome_preds if r['correct'])
                if outcome == 'V':
                    self.home_win_accuracy = correct / len(outcome_preds)
                elif outcome == 'N':
                    self.draw_accuracy = correct / len(outcome_preds)
                elif outcome == 'D':
                    self.away_win_accuracy = correct / len(outcome_preds)
    
    def calculate_dynamic_weight(self) -> float:
        """Calculate dynamic weight based on performance metrics."""
        if self.total_predictions < 10:
            return 1.0  # Default weight for new methods
        
        # Weight components:
        # 1. Overall accuracy (40%)
        # 2. Recent accuracy (30%) - more important for adaptation
        # 3. Calibration (30%) - lower Brier = better
        
        accuracy_score = self.accuracy
        recent_score = self.recent_accuracy
        calibration_score = 1 - min(self.avg_brier_score, 1.0)
        
        # Combined score
        raw_weight = (
            accuracy_score * 0.40 +
            recent_score * 0.30 +
            calibration_score * 0.30
        )
        
        # Ensure minimum weight of 0.1 (don't completely disable)
        self.dynamic_weight = max(0.1, raw_weight)
        
        return self.dynamic_weight
