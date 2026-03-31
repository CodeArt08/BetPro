"""Match features model for storing engineered features."""
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class MatchFeatures(Base):
    """Stores all engineered features for a match."""
    __tablename__ = "features"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, unique=True, index=True)
    
    # Team form features (last 5 matches)
    home_form_points = Column(Float, default=0.0)  # Points from last 5
    away_form_points = Column(Float, default=0.0)
    home_form_goals_scored = Column(Float, default=0.0)
    away_form_goals_scored = Column(Float, default=0.0)
    home_form_goals_conceded = Column(Float, default=0.0)
    away_form_goals_conceded = Column(Float, default=0.0)
    home_form_wins = Column(Integer, default=0)
    away_form_wins = Column(Integer, default=0)
    home_form_draws = Column(Integer, default=0)
    away_form_draws = Column(Integer, default=0)
    home_form_losses = Column(Integer, default=0)
    away_form_losses = Column(Integer, default=0)
    
    # Home/Away specific form
    home_home_form_points = Column(Float, default=0.0)  # Home team at home
    away_away_form_points = Column(Float, default=0.0)  # Away team away
    home_home_form_goals = Column(Float, default=0.0)
    away_away_form_goals = Column(Float, default=0.0)
    home_home_form_conceded = Column(Float, default=0.0)
    away_away_form_conceded = Column(Float, default=0.0)
    
    # Goal difference trends
    home_gd_trend = Column(Float, default=0.0)  # Trend over last 5
    away_gd_trend = Column(Float, default=0.0)
    
    # Scoring frequency
    home_scoring_freq = Column(Float, default=0.0)  # Goals per match
    away_scoring_freq = Column(Float, default=0.0)
    home_conceding_freq = Column(Float, default=0.0)
    away_conceding_freq = Column(Float, default=0.0)
    
    # Head-to-head features (basic)
    h2h_home_wins = Column(Integer, default=0)
    h2h_away_wins = Column(Integer, default=0)
    h2h_draws = Column(Integer, default=0)
    h2h_avg_goals = Column(Float, default=0.0)
    h2h_home_avg_goals = Column(Float, default=0.0)
    h2h_away_avg_goals = Column(Float, default=0.0)
    h2h_draw_rate = Column(Float, default=0.0)
    
    # Advanced H2H features (weighted by recency)
    h2h_home_win_rate = Column(Float, default=0.0)  # Win rate when home team is at home
    h2h_away_win_rate = Column(Float, default=0.0)  # Win rate when away team is at home (inverted)
    h2h_weighted_home_goals = Column(Float, default=0.0)  # Goals weighted by recency
    h2h_weighted_away_goals = Column(Float, default=0.0)
    h2h_weighted_home_conceded = Column(Float, default=0.0)
    h2h_weighted_away_conceded = Column(Float, default=0.0)
    h2h_dominance_score = Column(Float, default=0.0)  # -1 to 1, positive = home team dominant
    h2h_last_5_results = Column(JSON, default=list)  # Last 5 H2H results as list
    h2h_recency_weighted_accuracy = Column(Float, default=0.0)  # How predictable this matchup is
    
    # STRICT H2H features (home team AT HOME vs away team)
    h2h_strict_home_wins = Column(Integer, default=0)  # Wins when home team played at home
    h2h_strict_home_losses = Column(Integer, default=0)  # Losses when home team played at home
    h2h_strict_home_draws = Column(Integer, default=0)  # Draws when home team played at home
    h2h_strict_home_goals_scored = Column(Float, default=0.0)  # Goals scored by home team when at home
    h2h_strict_home_goals_conceded = Column(Float, default=0.0)  # Goals conceded by home team when at home
    h2h_strict_away_wins = Column(Integer, default=0)  # Wins when away team played at home (vs home team)
    h2h_strict_away_losses = Column(Integer, default=0)  # Losses when away team played at home
    h2h_strict_away_draws = Column(Integer, default=0)  # Draws when away team played at home
    h2h_strict_away_goals_scored = Column(Float, default=0.0)  # Goals scored by away team when at home
    h2h_strict_away_goals_conceded = Column(Float, default=0.0)  # Goals conceded by away team when at home
    h2h_strict_home_win_rate = Column(Float, default=0.0)  # Win rate in direct confrontations at home
    h2h_strict_away_win_rate = Column(Float, default=0.0)  # Win rate in direct confrontations when away team is home
    h2h_strict_total_matches = Column(Integer, default=0)  # Total direct confrontations
    
    # Strength features
    home_elo = Column(Float, default=1500.0)
    away_elo = Column(Float, default=1500.0)
    elo_diff = Column(Float, default=0.0)
    home_elo_home = Column(Float, default=1500.0)
    away_elo_away = Column(Float, default=1500.0)
    
    # League Position (classement)
    home_league_position = Column(Integer, default=0)  # Current league position
    away_league_position = Column(Integer, default=0)
    league_position_diff = Column(Integer, default=0)  # home - away (negative = home team higher)
    home_league_points = Column(Integer, default=0)
    away_league_points = Column(Integer, default=0)
    points_diff = Column(Integer, default=0)
    
    home_bayesian_rating = Column(Float, default=0.0)
    away_bayesian_rating = Column(Float, default=0.0)
    bayesian_diff = Column(Float, default=0.0)
    
    home_attack_strength = Column(Float, default=1.0)
    away_attack_strength = Column(Float, default=1.0)
    home_defense_strength = Column(Float, default=1.0)
    away_defense_strength = Column(Float, default=1.0)
    
    # Momentum features
    home_winning_streak = Column(Integer, default=0)
    away_winning_streak = Column(Integer, default=0)
    home_losing_streak = Column(Integer, default=0)
    away_losing_streak = Column(Integer, default=0)
    home_unbeaten_streak = Column(Integer, default=0)
    away_unbeaten_streak = Column(Integer, default=0)
    
    # Home advantage
    home_advantage = Column(Float, default=0.0)
    away_disadvantage = Column(Float, default=0.0)
    
    # Line position features
    line_position = Column(Integer, default=1)
    line_home_win_rate = Column(Float, default=0.0)
    line_draw_rate = Column(Float, default=0.0)
    line_away_win_rate = Column(Float, default=0.0)
    line_avg_goals = Column(Float, default=0.0)
    
    # Sequence features (Markov)
    home_sequence_prob_V = Column(Float, default=0.0)
    home_sequence_prob_N = Column(Float, default=0.0)
    home_sequence_prob_D = Column(Float, default=0.0)
    away_sequence_prob_V = Column(Float, default=0.0)
    away_sequence_prob_N = Column(Float, default=0.0)
    away_sequence_prob_D = Column(Float, default=0.0)
    
    # Line sequence features (Markov for line position)
    line_sequence_prob_V = Column(Float, default=0.0)
    line_sequence_prob_N = Column(Float, default=0.0)
    line_sequence_prob_D = Column(Float, default=0.0)
    line_recent_form = Column(Float, default=0.0)  # Encoded recent form (e.g., VVVN = 0.8)
    
    # Expected goals (from Poisson model)
    home_xg = Column(Float, default=0.0)
    away_xg = Column(Float, default=0.0)
    
    # Overall team stats
    home_win_rate = Column(Float, default=0.0)
    away_win_rate = Column(Float, default=0.0)
    home_draw_rate = Column(Float, default=0.0)
    away_draw_rate = Column(Float, default=0.0)
    home_avg_goals_scored = Column(Float, default=0.0)
    away_avg_goals_scored = Column(Float, default=0.0)
    home_avg_goals_conceded = Column(Float, default=0.0)
    away_avg_goals_conceded = Column(Float, default=0.0)
    
    # Matchday context
    matchday = Column(Integer, default=1)
    is_early_matchday = Column(Integer, default=0)  # 1 if md <= 5
    is_late_matchday = Column(Integer, default=0)  # 1 if md >= 34
    
    # Interaction features (derived from existing features)
    form_ratio = Column(Float, default=0.0)
    elo_ratio = Column(Float, default=0.0)
    attack_defense_ratio = Column(Float, default=0.0)
    defense_attack_ratio = Column(Float, default=0.0)
    goal_ratio = Column(Float, default=0.0)
    
    # Strength indicators
    home_strength_index = Column(Float, default=0.0)
    away_strength_index = Column(Float, default=0.0)
    strength_diff = Column(Float, default=0.0)
    
    # Form momentum indicators
    home_form_momentum = Column(Float, default=0.0)
    away_form_momentum = Column(Float, default=0.0)
    form_diff = Column(Float, default=0.0)
    
    # H2H weighted features
    h2h_recency_weight = Column(Float, default=0.0)
    h2h_predictability = Column(Float, default=0.0)
    
    # Additional features as JSON
    extra_features = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="features")
    
    def __repr__(self):
        return f"<MatchFeatures(id={self.id}, match_id={self.match_id})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for ML models."""
        return {
            'home_form_points': self.home_form_points,
            'away_form_points': self.away_form_points,
            'home_form_goals_scored': self.home_form_goals_scored,
            'away_form_goals_scored': self.away_form_goals_scored,
            'home_form_goals_conceded': self.home_form_goals_conceded,
            'away_form_goals_conceded': self.away_form_goals_conceded,
            'home_form_wins': self.home_form_wins,
            'away_form_wins': self.away_form_wins,
            'home_form_draws': self.home_form_draws,
            'away_form_draws': self.away_form_draws,
            'home_form_losses': self.home_form_losses,
            'away_form_losses': self.away_form_losses,
            'home_home_form_points': self.home_home_form_points,
            'away_away_form_points': self.away_away_form_points,
            'home_home_form_goals': self.home_home_form_goals,
            'away_away_form_goals': self.away_away_form_goals,
            'home_home_form_conceded': self.home_home_form_conceded,
            'away_away_form_conceded': self.away_away_form_conceded,
            'home_gd_trend': self.home_gd_trend,
            'away_gd_trend': self.away_gd_trend,
            'home_scoring_freq': self.home_scoring_freq,
            'away_scoring_freq': self.away_scoring_freq,
            'home_conceding_freq': self.home_conceding_freq,
            'away_conceding_freq': self.away_conceding_freq,
            'h2h_home_wins': self.h2h_home_wins,
            'h2h_away_wins': self.h2h_away_wins,
            'h2h_draws': self.h2h_draws,
            'h2h_avg_goals': self.h2h_avg_goals,
            'h2h_home_avg_goals': self.h2h_home_avg_goals,
            'h2h_away_avg_goals': self.h2h_away_avg_goals,
            'h2h_draw_rate': self.h2h_draw_rate,
            'h2h_home_win_rate': self.h2h_home_win_rate,
            'h2h_away_win_rate': self.h2h_away_win_rate,
            'h2h_weighted_home_goals': self.h2h_weighted_home_goals,
            'h2h_weighted_away_goals': self.h2h_weighted_away_goals,
            'h2h_weighted_home_conceded': self.h2h_weighted_home_conceded,
            'h2h_weighted_away_conceded': self.h2h_weighted_away_conceded,
            'h2h_dominance_score': self.h2h_dominance_score,
            'h2h_recency_weighted_accuracy': self.h2h_recency_weighted_accuracy,
            # Strict H2H features
            'h2h_strict_home_wins': self.h2h_strict_home_wins,
            'h2h_strict_home_losses': self.h2h_strict_home_losses,
            'h2h_strict_home_draws': self.h2h_strict_home_draws,
            'h2h_strict_home_goals_scored': self.h2h_strict_home_goals_scored,
            'h2h_strict_home_goals_conceded': self.h2h_strict_home_goals_conceded,
            'h2h_strict_away_wins': self.h2h_strict_away_wins,
            'h2h_strict_away_losses': self.h2h_strict_away_losses,
            'h2h_strict_away_draws': self.h2h_strict_away_draws,
            'h2h_strict_away_goals_scored': self.h2h_strict_away_goals_scored,
            'h2h_strict_away_goals_conceded': self.h2h_strict_away_goals_conceded,
            'h2h_strict_home_win_rate': self.h2h_strict_home_win_rate,
            'h2h_strict_away_win_rate': self.h2h_strict_away_win_rate,
            'h2h_strict_total_matches': self.h2h_strict_total_matches,
            'home_elo': self.home_elo,
            'away_elo': self.away_elo,
            'elo_diff': self.elo_diff,
            'home_elo_home': self.home_elo_home,
            'away_elo_away': self.away_elo_away,
            # League position
            'home_league_position': self.home_league_position,
            'away_league_position': self.away_league_position,
            'league_position_diff': self.league_position_diff,
            'home_league_points': self.home_league_points,
            'away_league_points': self.away_league_points,
            'points_diff': self.points_diff,
            'home_bayesian_rating': self.home_bayesian_rating,
            'away_bayesian_rating': self.away_bayesian_rating,
            'bayesian_diff': self.bayesian_diff,
            'home_attack_strength': self.home_attack_strength,
            'away_attack_strength': self.away_attack_strength,
            'home_defense_strength': self.home_defense_strength,
            'away_defense_strength': self.away_defense_strength,
            'home_winning_streak': self.home_winning_streak,
            'away_winning_streak': self.away_winning_streak,
            'home_losing_streak': self.home_losing_streak,
            'away_losing_streak': self.away_losing_streak,
            'home_unbeaten_streak': self.home_unbeaten_streak,
            'away_unbeaten_streak': self.away_unbeaten_streak,
            'home_advantage': self.home_advantage,
            'away_disadvantage': self.away_disadvantage,
            'line_position': self.line_position,
            'line_home_win_rate': self.line_home_win_rate,
            'line_draw_rate': self.line_draw_rate,
            'line_away_win_rate': self.line_away_win_rate,
            'line_avg_goals': self.line_avg_goals,
            'home_sequence_prob_V': self.home_sequence_prob_V,
            'home_sequence_prob_N': self.home_sequence_prob_N,
            'home_sequence_prob_D': self.home_sequence_prob_D,
            'away_sequence_prob_V': self.away_sequence_prob_V,
            'away_sequence_prob_N': self.away_sequence_prob_N,
            'away_sequence_prob_D': self.away_sequence_prob_D,
            'line_sequence_prob_V': self.line_sequence_prob_V,
            'line_sequence_prob_N': self.line_sequence_prob_N,
            'line_sequence_prob_D': self.line_sequence_prob_D,
            'line_recent_form': self.line_recent_form,
            'home_xg': self.home_xg,
            'away_xg': self.away_xg,
            'home_win_rate': self.home_win_rate,
            'away_win_rate': self.away_win_rate,
            'home_draw_rate': self.home_draw_rate,
            'away_draw_rate': self.away_draw_rate,
            'home_avg_goals_scored': self.home_avg_goals_scored,
            'away_avg_goals_scored': self.away_avg_goals_scored,
            'home_avg_goals_conceded': self.home_avg_goals_conceded,
            'away_avg_goals_conceded': self.away_avg_goals_conceded,
            'matchday': self.matchday,
            'is_early_matchday': self.is_early_matchday,
            'is_late_matchday': self.is_late_matchday,
            # Interaction features
            'form_ratio': self.form_ratio,
            'elo_ratio': self.elo_ratio,
            'attack_defense_ratio': self.attack_defense_ratio,
            'defense_attack_ratio': self.defense_attack_ratio,
            'goal_ratio': self.goal_ratio,
            # Strength indicators
            'home_strength_index': self.home_strength_index,
            'away_strength_index': self.away_strength_index,
            'strength_diff': self.strength_diff,
            # Form momentum
            'home_form_momentum': self.home_form_momentum,
            'away_form_momentum': self.away_form_momentum,
            'form_diff': self.form_diff,
            # H2H weighted
            'h2h_recency_weight': self.h2h_recency_weight,
            'h2h_predictability': self.h2h_predictability,
        }
