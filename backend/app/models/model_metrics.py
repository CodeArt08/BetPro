"""Model metrics tracking for continuous learning."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class ModelMetrics(Base):
    """Tracks performance metrics for each model."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_name = Column(String(100), nullable=False, index=True)  # e.g., "xgboost", "logistic_regression"
    model_version = Column(String(50), nullable=False)
    
    # Overall accuracy metrics
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    
    # Per-class metrics
    home_win_predictions = Column(Integer, default=0)
    home_win_correct = Column(Integer, default=0)
    home_win_accuracy = Column(Float, default=0.0)
    
    draw_predictions = Column(Integer, default=0)
    draw_correct = Column(Integer, default=0)
    draw_accuracy = Column(Float, default=0.0)
    
    away_win_predictions = Column(Integer, default=0)
    away_win_correct = Column(Integer, default=0)
    away_win_accuracy = Column(Float, default=0.0)
    
    # Probability calibration
    avg_brier_score = Column(Float, default=0.0)
    brier_scores_count = Column(Integer, default=0)
    
    # Log loss
    avg_log_loss = Column(Float, default=0.0)
    log_loss_count = Column(Integer, default=0)
    
    # Betting performance (if used for betting)
    bets_made = Column(Integer, default=0)
    bets_won = Column(Integer, default=0)
    betting_accuracy = Column(Float, default=0.0)
    total_profit = Column(Float, default=0.0)
    roi = Column(Float, default=0.0)
    
    # Ensemble weight
    ensemble_weight = Column(Float, default=1.0)
    
    # Recent performance (last 50 predictions)
    recent_accuracy = Column(Float, default=0.0)
    recent_predictions = Column(JSON, default=list)  # List of recent results
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_retrained_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}, accuracy={self.accuracy:.3f})>"
    
    def update_prediction(self, predicted: str, actual: str, probabilities: dict):
        """Update metrics after a prediction is verified."""
        self.total_predictions += 1
        
        # Update overall accuracy
        if predicted == actual:
            self.correct_predictions += 1
        self.accuracy = self.correct_predictions / self.total_predictions
        
        # Update per-class metrics
        if predicted == 'V':
            self.home_win_predictions += 1
            if actual == 'V':
                self.home_win_correct += 1
            self.home_win_accuracy = self.home_win_correct / self.home_win_predictions if self.home_win_predictions > 0 else 0.0
        elif predicted == 'N':
            self.draw_predictions += 1
            if actual == 'N':
                self.draw_correct += 1
            self.draw_accuracy = self.draw_correct / self.draw_predictions if self.draw_predictions > 0 else 0.0
        elif predicted == 'D':
            self.away_win_predictions += 1
            if actual == 'D':
                self.away_win_correct += 1
            self.away_win_accuracy = self.away_win_correct / self.away_win_predictions if self.away_win_predictions > 0 else 0.0
        
        # Update Brier score
        actual_probs = {'V': 0.0, 'N': 0.0, 'D': 0.0}
        actual_probs[actual] = 1.0
        brier = sum((probabilities.get(k, 0) - v) ** 2 for k, v in actual_probs.items()) / 3
        
        self.avg_brier_score = (self.avg_brier_score * self.brier_scores_count + brier) / (self.brier_scores_count + 1)
        self.brier_scores_count += 1
        
        # Update recent performance
        recent = self.recent_predictions or []
        recent.append({'predicted': predicted, 'actual': actual, 'correct': predicted == actual})
        self.recent_predictions = recent[-50:]  # Keep last 50
        
        # Calculate recent accuracy
        if self.recent_predictions:
            self.recent_accuracy = sum(1 for r in self.recent_predictions if r['correct']) / len(self.recent_predictions)
    
    def calculate_ensemble_weight(self):
        """Calculate weight for ensemble based on performance."""
        # Weight based on accuracy and calibration
        # Higher accuracy and lower brier score = higher weight
        if self.total_predictions >= 10:
            accuracy_factor = self.accuracy
            calibration_factor = 1 - min(self.avg_brier_score, 1.0)
            recent_factor = self.recent_accuracy
            
            # Combined weight
            self.ensemble_weight = (accuracy_factor * 0.4 + calibration_factor * 0.3 + recent_factor * 0.3)
        else:
            self.ensemble_weight = 1.0  # Default weight for new models
