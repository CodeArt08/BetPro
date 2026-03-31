"""Match model for storing match results and data."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class MatchResult(enum.Enum):
    """Match result enumeration."""
    HOME_WIN = "V"
    DRAW = "N"
    AWAY_WIN = "D"


class Match(Base):
    """Represents a single match."""
    __tablename__ = "matches"
    __table_args__ = (
        # Index composite pour requêtes fréquentes
        Index('ix_matches_season_upcoming', 'season_id', 'is_upcoming'),
        Index('ix_matches_season_completed', 'season_id', 'is_completed'),
        Index('ix_matches_season_matchday', 'season_id', 'matchday'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Season and Matchday
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False, index=True)
    matchday = Column(Integer, nullable=False, index=True)  # 1-38
    line_position = Column(Integer, nullable=False)  # 1-10 (position in matchday)
    
    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    home_team_name = Column(String(100), nullable=False)
    away_team_name = Column(String(100), nullable=False)
    
    # Score and Result
    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    result = Column(String(1), nullable=True)  # V (home win), N (draw), D (away win)
    
    # Odds (from bookmaker)
    odd_home = Column(Float, nullable=True)
    odd_draw = Column(Float, nullable=True)
    odd_away = Column(Float, nullable=True)
    
    # Implied probabilities (normalized)
    implied_prob_home = Column(Float, nullable=True)
    implied_prob_draw = Column(Float, nullable=True)
    implied_prob_away = Column(Float, nullable=True)
    
    # Bookmaker margin
    bookmaker_margin = Column(Float, nullable=True)
    
    # Timestamps
    match_time = Column(DateTime(timezone=True), nullable=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    result_recorded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_completed = Column(Boolean, default=False)
    is_upcoming = Column(Boolean, default=True)
    has_odds = Column(Boolean, default=False)
    
    # Calculated fields
    total_goals = Column(Integer, nullable=True)
    goal_difference = Column(Integer, nullable=True)
    both_teams_scored = Column(Boolean, nullable=True)
    over_2_5 = Column(Boolean, nullable=True)
    
    # Relationships
    season = relationship("Season", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_match_refs")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_match_refs")
    prediction = relationship("Prediction", back_populates="match", uselist=False)
    features = relationship("MatchFeatures", back_populates="match", uselist=False)
    bets = relationship("Bet", back_populates="match")
    
    def __repr__(self):
        return f"<Match(id={self.id}, {self.home_team_name} vs {self.away_team_name}, md={self.matchday})>"
    
    def set_result(self, score_home: int, score_away: int):
        """Set match result."""
        self.score_home = score_home
        self.score_away = score_away
        self.total_goals = score_home + score_away
        self.goal_difference = score_home - score_away
        self.both_teams_scored = score_home > 0 and score_away > 0
        self.over_2_5 = self.total_goals > 2.5
        
        # Determine result
        if score_home > score_away:
            self.result = 'V'
        elif score_home < score_away:
            self.result = 'D'
        else:
            self.result = 'N'
        
        self.is_completed = True
        self.is_upcoming = False
        self.result_recorded_at = func.now()
    
    def set_odds(self, odd_home: float, odd_draw: float, odd_away: float):
        """Set and normalize odds."""
        self.odd_home = odd_home
        self.odd_draw = odd_draw
        self.odd_away = odd_away
        self.has_odds = True
        
        # Calculate implied probabilities
        raw_home = 1 / odd_home if odd_home > 0 else 0
        raw_draw = 1 / odd_draw if odd_draw > 0 else 0
        raw_away = 1 / odd_away if odd_away > 0 else 0
        
        total = raw_home + raw_draw + raw_away
        self.bookmaker_margin = total - 1
        
        # Normalize
        if total > 0:
            self.implied_prob_home = raw_home / total
            self.implied_prob_draw = raw_draw / total
            self.implied_prob_away = raw_away / total
    
    @property
    def result_description(self) -> str:
        """Get human-readable result."""
        if self.result == 'V':
            return f"{self.home_team_name} Win"
        elif self.result == 'D':
            return f"{self.away_team_name} Win"
        elif self.result == 'N':
            return "Draw"
        return "Not played"
