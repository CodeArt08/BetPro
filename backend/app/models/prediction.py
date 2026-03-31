"""Prediction model for storing match predictions."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Prediction(Base):
    """Represents a match prediction from the ensemble."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Match reference
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True, index=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False, index=True)
    
    # Model probabilities
    prob_home_win = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away_win = Column(Float, nullable=False)
    
    # Individual model outputs (stored as JSON)
    model_outputs = Column(JSON, default=dict)
    # Example: {"logistic_regression": {"V": 0.45, "N": 0.28, "D": 0.27}, ...}
    
    # Ensemble weights used
    ensemble_weights = Column(JSON, default=dict)
    
    # Predicted outcome
    predicted_result = Column(String(1), nullable=False)  # V, N, or D
    predicted_result_name = Column(String(50), nullable=False)  # "Home Win", "Draw", "Away Win"
    
    # Confidence metrics
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    model_agreement = Column(Float, nullable=False)  # How many models agree
    probability_strength = Column(Float, nullable=False)  # Max probability
    
    # Value analysis
    value_home = Column(Float, nullable=True)  # Model prob - Market prob
    value_draw = Column(Float, nullable=True)
    value_away = Column(Float, nullable=True)
    
    # Best value bet
    best_value_outcome = Column(String(1), nullable=True)
    best_value_amount = Column(Float, nullable=True)
    
    # Monte Carlo simulation results
    monte_carlo_results = Column(JSON, nullable=True)
    # Example: {"home_wins": 4500, "draws": 2500, "away_wins": 3000, "avg_home_goals": 1.45, ...}
    
    # Actual result (filled after match)
    actual_result = Column(String(1), nullable=True)
    is_correct = Column(Boolean, nullable=True)
    
    # Prediction error for learning
    prediction_error = Column(Float, nullable=True)  # Brier score
    
    # Selection fields for dynamic betting
    is_selected_for_bet = Column(Boolean, default=False, index=True)
    selection_rank = Column(Integer, nullable=True)  # 1-N ranking within matchday
    selection_reason = Column(String(50), nullable=True)  # STRONG_DRAW, HIGH_VALUE, MODEL_CONSENSUS, SEQUENCE_PATTERN
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    match = relationship("Match", back_populates="prediction")
    season = relationship("Season", back_populates="predictions")
    bets = relationship("Bet", back_populates="prediction", cascade="all, delete-orphan", passive_deletes=False)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, match_id={self.match_id}, predicted={self.predicted_result})>"
    
    def verify(self, actual_result: str):
        """Verify prediction against actual result."""
        self.actual_result = actual_result
        self.is_correct = (self.predicted_result == actual_result)
        self.verified_at = func.now()
        
        # Calculate Brier score
        probs = [self.prob_home_win, self.prob_draw, self.prob_away_win]
        actuals = [1.0 if actual_result == 'V' else 0.0,
                   1.0 if actual_result == 'N' else 0.0,
                   1.0 if actual_result == 'D' else 0.0]
        self.prediction_error = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / 3
    
    def calculate_value(self, odd_home: float, odd_draw: float, odd_away: float):
        """Calculate value against bookmaker odds."""
        from loguru import logger
        
        logger.info(f"Calculating value for match: odds={odd_home}/{odd_draw}/{odd_away}")
        
        if odd_home and odd_home > 0:
            implied_home = 1 / odd_home
            self.value_home = self.prob_home_win - implied_home
            logger.info(f"  value_home = {self.prob_home_win:.3f} - {implied_home:.3f} = {self.value_home:.3f}")
        
        if odd_draw and odd_draw > 0:
            implied_draw = 1 / odd_draw
            self.value_draw = self.prob_draw - implied_draw
            logger.info(f"  value_draw = {self.prob_draw:.3f} - {implied_draw:.3f} = {self.value_draw:.3f}")
        
        if odd_away and odd_away > 0:
            implied_away = 1 / odd_away
            self.value_away = self.prob_away_win - implied_away
            logger.info(f"  value_away = {self.prob_away_win:.3f} - {implied_away:.3f} = {self.value_away:.3f}")
        
        # Find best value
        values = {
            'V': self.value_home or 0,
            'N': self.value_draw or 0,
            'D': self.value_away or 0
        }
        best_outcome = max(values, key=values.get)
        self.best_value_outcome = best_outcome
        self.best_value_amount = values[best_outcome]
        
        logger.info(f"  Best value: {best_outcome} = {self.best_value_amount:.3f}")
    
    @property
    def predicted_outcome_description(self) -> str:
        """Human-readable prediction."""
        descriptions = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}
        return descriptions.get(self.predicted_result, 'Unknown')
