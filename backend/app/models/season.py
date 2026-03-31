"""Season model for tracking league seasons."""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Season(Base):
    """Represents a league season (38 matchdays)."""
    __tablename__ = "seasons"
    
    id = Column(Integer, primary_key=True, index=True)
    season_number = Column(Integer, nullable=False)
    start_date = Column(DateTime(timezone=True), server_default=func.now())
    end_date = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    is_completed = Column(Boolean, default=False)
    
    # Statistics
    total_matches = Column(Integer, default=0)
    total_home_wins = Column(Integer, default=0)
    total_draws = Column(Integer, default=0)
    total_away_wins = Column(Integer, default=0)
    total_goals = Column(Integer, default=0)
    avg_goals_per_match = Column(Float, default=0.0)
    
    # Betting stats
    total_bets = Column(Integer, default=0)
    winning_bets = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    roi = Column(Float, default=0.0)
    
    # Relationships - NO CASCADE DELETE to preserve historical data
    matches = relationship("Match", back_populates="season")
    predictions = relationship("Prediction", back_populates="season")
    bets = relationship("Bet", back_populates="season")
    
    def __repr__(self):
        return f"<Season(id={self.id}, number={self.season_number}, active={self.is_active})>"
    
    def close_season(self):
        """Mark season as completed."""
        self.is_active = False
        self.is_completed = True
        self.end_date = func.now()
        
    def update_statistics(self):
        """Update season statistics from matches."""
        if self.matches:
            self.total_matches = len(self.matches)
            self.total_home_wins = sum(1 for m in self.matches if m.result == 'V')
            self.total_draws = sum(1 for m in self.matches if m.result == 'N')
            self.total_away_wins = sum(1 for m in self.matches if m.result == 'D')
            self.total_goals = sum((m.score_home or 0) + (m.score_away or 0) for m in self.matches)
            self.avg_goals_per_match = self.total_goals / self.total_matches if self.total_matches > 0 else 0.0
