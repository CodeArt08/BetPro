"""Team model with strength ratings and statistics."""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Team(Base):
    """Represents a team in the Spanish League."""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    
    # League Standing
    league_position = Column(Integer, default=0)  # Current position in league table
    league_points = Column(Integer, default=0)  # Current points
    league_played = Column(Integer, default=0)  # Matches played in current season
    
    # ELO Rating
    elo_rating = Column(Float, default=1500.0)
    elo_home = Column(Float, default=1500.0)
    elo_away = Column(Float, default=1500.0)
    
    # Bayesian Rating
    bayesian_rating = Column(Float, default=0.0)
    bayesian_variance = Column(Float, default=1.0)
    
    # Attack/Defense Strength (Poisson-based)
    attack_strength = Column(Float, default=1.0)
    defense_strength = Column(Float, default=1.0)
    attack_strength_home = Column(Float, default=1.0)
    defense_strength_home = Column(Float, default=1.0)
    attack_strength_away = Column(Float, default=1.0)
    defense_strength_away = Column(Float, default=1.0)
    
    # Overall Statistics
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_scored = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    
    # Home Statistics
    home_matches = Column(Integer, default=0)
    home_wins = Column(Integer, default=0)
    home_draws = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    home_goals_scored = Column(Integer, default=0)
    home_goals_conceded = Column(Integer, default=0)
    
    # Away Statistics
    away_matches = Column(Integer, default=0)
    away_wins = Column(Integer, default=0)
    away_draws = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)
    away_goals_scored = Column(Integer, default=0)
    away_goals_conceded = Column(Integer, default=0)
    
    # Form and Momentum
    current_form = Column(String(10), default="")  # Last 5 results: "VVDND"
    winning_streak = Column(Integer, default=0)
    losing_streak = Column(Integer, default=0)
    draw_streak = Column(Integer, default=0)
    unbeaten_streak = Column(Integer, default=0)
    
    # Expected Goals
    avg_xg_for = Column(Float, default=0.0)
    avg_xg_against = Column(Float, default=0.0)
    
    # Metadata
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Additional stats stored as JSON
    line_position_stats = Column(JSON, default=dict)  # Stats per line position
    
    # Relationships
    home_match_refs = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_match_refs = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    
    def __repr__(self):
        return f"<Team(id={self.id}, name={self.name}, elo={self.elo_rating})>"
    
    @property
    def goal_difference(self) -> int:
        return self.goals_scored - self.goals_conceded
    
    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played > 0 else 0.0
    
    @property
    def home_advantage(self) -> float:
        """Calculate home advantage factor."""
        if self.home_matches > 0 and self.away_matches > 0:
            home_win_rate = self.home_wins / self.home_matches
            away_win_rate = self.away_wins / self.away_matches
            return home_win_rate - away_win_rate
        return 0.0
    
    def update_form(self, result: str):
        """Update current form with new result."""
        form = self.current_form + result
        self.current_form = form[-5:]  # Keep last 5
        
        # Update streaks
        if result == 'V':
            self.winning_streak += 1
            self.losing_streak = 0
            self.draw_streak = 0
            self.unbeaten_streak += 1
        elif result == 'D':
            self.winning_streak = 0
            self.losing_streak += 1
            self.draw_streak = 0
            self.unbeaten_streak = 0
        else:  # Draw
            self.winning_streak = 0
            self.losing_streak = 0
            self.draw_streak += 1
            self.unbeaten_streak += 1
