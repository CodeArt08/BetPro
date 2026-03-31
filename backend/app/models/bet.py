"""Bet model for tracking betting history and profit."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class BetStatus(enum.Enum):
    """Bet status enumeration."""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"


class BetOutcome(enum.Enum):
    """What was bet on."""
    HOME_WIN = "V"
    DRAW = "N"
    AWAY_WIN = "D"


class Bet(Base):
    """Represents a placed bet."""
    __tablename__ = "bets"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # References
    match_id = Column(Integer, ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    season_id = Column(Integer, ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Bet details
    bet_outcome = Column(String(1), nullable=False)  # V, N, or D
    bet_outcome_name = Column(String(50), nullable=False)
    odds = Column(Float, nullable=False)
    
    # Stake and returns
    stake = Column(Float, nullable=False)
    potential_return = Column(Float, nullable=False)
    actual_return = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    
    # Bankroll tracking
    bankroll_before = Column(Float, nullable=False)
    bankroll_after = Column(Float, nullable=True)
    
    # Kelly criterion
    kelly_fraction_used = Column(Float, nullable=False)
    kelly_full = Column(Float, nullable=False)  # Full Kelly fraction
    
    # Value
    value_edge = Column(Float, nullable=False)  # Model prob - implied prob
    
    # Confidence
    confidence = Column(Float, nullable=False)
    
    # Status
    status = Column(String(20), default="pending")
    is_settled = Column(Boolean, default=False)
    
    # Timestamps
    placed_at = Column(DateTime(timezone=True), server_default=func.now())
    settled_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    match = relationship("Match", back_populates="bets")
    prediction = relationship("Prediction", back_populates="bets")
    season = relationship("Season", back_populates="bets")
    
    def __repr__(self):
        return f"<Bet(id={self.id}, match_id={self.match_id}, outcome={self.bet_outcome}, stake={self.stake})>"
    
    def settle(self, actual_result: str):
        """Settle the bet based on actual result."""
        self.status = "won" if self.bet_outcome == actual_result else "lost"
        self.is_settled = True
        self.settled_at = func.now()
        
        if self.status == "won":
            self.actual_return = self.stake * self.odds
            self.profit_loss = self.actual_return - self.stake
        else:
            self.actual_return = 0.0
            self.profit_loss = -self.stake
        
        self.bankroll_after = self.bankroll_before + self.profit_loss
    
    @property
    def roi(self) -> float:
        """Return on investment for this bet."""
        if self.stake > 0:
            return (self.profit_loss or 0) / self.stake
        return 0.0
