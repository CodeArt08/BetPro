"""Team Strength Engine with ELO, Bayesian, and Poisson models."""
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from loguru import logger
from sqlalchemy.orm import Session

from app.models import Team, Match


class TeamStrengthEngine:
    """
    Manages team strength ratings using multiple models.
    """
    
    # ELO constants
    ELO_K = 32  # ELO K-factor
    ELO_HOME_ADVANTAGE_BASE = 100  # Base home advantage in ELO points
    ELO_HOME_ADVANTAGE_MIN = 0  # Minimum home advantage (weak home team)
    ELO_HOME_ADVANTAGE_MAX = 200  # Maximum home advantage (strong home team)
    ELO_DRAW_BIAS = 0.22
    ELO_DIFF_SCALE = 400
    
    # Bayesian prior
    BAYESIAN_PRIOR_MEAN = 0.0
    BAYESIAN_PRIOR_VARIANCE = 1.0
    
    def __init__(self):
        # Balanced league averages - home advantage is captured adaptively
        self.league_avg_goals_home = 1.35
        self.league_avg_goals_away = 1.20
    
    def update_elo(self, match: Match, db: Session):
        """
        Update ELO ratings after a match.
        """
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        if not home_team or not away_team:
            return
        
        # Expected scores
        home_expected = self._expected_elo(home_team.elo_rating, away_team.elo_rating + self.ELO_HOME_ADVANTAGE)
        away_expected = 1 - home_expected
        
        # Actual scores
        if match.result == 'V':
            home_actual, away_actual = 1.0, 0.0
        elif match.result == 'N':
            home_actual, away_actual = 0.5, 0.5
        else:
            home_actual, away_actual = 0.0, 1.0
        
        # Update ratings
        home_team.elo_rating += self.ELO_K * (home_actual - home_expected)
        away_team.elo_rating += self.ELO_K * (away_actual - away_expected)
        
        # Update home/away specific ELO
        home_team.elo_home += self.ELO_K * 0.5 * (home_actual - home_expected)
        away_team.elo_away += self.ELO_K * 0.5 * (away_actual - away_expected)
        
        db.commit()
        logger.debug(f"ELO updated: {home_team.name}={home_team.elo_rating:.1f}, {away_team.name}={away_team.elo_rating:.1f}")
    
    def _expected_elo(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_bayesian(self, match: Match, db: Session):
        """
        Update Bayesian ratings after a match.
        Uses a simple Bayesian update on goal difference.
        """
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        if not home_team or not away_team:
            return
        
        # Goal difference from perspective of each team
        gd_home = (match.score_home or 0) - (match.score_away or 0)
        gd_away = -gd_home
        
        # Update home team
        prior_var = home_team.bayesian_variance
        prior_mean = home_team.bayesian_rating
        
        # Observation variance (assume high uncertainty)
        obs_var = 2.0
        
        # Posterior
        posterior_var = 1 / (1/prior_var + 1/obs_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + gd_home/obs_var)
        
        home_team.bayesian_rating = posterior_mean
        home_team.bayesian_variance = posterior_var + 0.1  # Add small variance to prevent collapse
        
        # Update away team
        prior_var = away_team.bayesian_variance
        prior_mean = away_team.bayesian_rating
        
        posterior_var = 1 / (1/prior_var + 1/obs_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + gd_away/obs_var)
        
        away_team.bayesian_rating = posterior_mean
        away_team.bayesian_variance = posterior_var + 0.1
        
        db.commit()
    
    def update_poisson_strength(self, match: Match, db: Session):
        """
        Update attack/defense strength based on Poisson goal model.
        """
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        if not home_team or not away_team:
            return
        
        # Update team statistics
        self._update_team_stats(home_team, is_home=True, 
                               goals_scored=match.score_home or 0,
                               goals_conceded=match.score_away or 0,
                               result=match.result)
        
        self._update_team_stats(away_team, is_home=False,
                               goals_scored=match.score_away or 0,
                               goals_conceded=match.score_home or 0,
                               result=match.result)
        
        # Calculate attack/defense strength
        self._calculate_strength(home_team, db)
        self._calculate_strength(away_team, db)
        
        db.commit()
    
    def _update_team_stats(self, team: Team, is_home: bool, 
                          goals_scored: int, goals_conceded: int, result: str):
        """Update team statistics after a match."""
        team.matches_played += 1
        team.goals_scored += goals_scored
        team.goals_conceded += goals_conceded
        
        if result == 'V' and is_home:
            team.wins += 1
            team.home_wins += 1
            team.home_matches += 1
            team.home_goals_scored += goals_scored
            team.home_goals_conceded += goals_conceded
            team.update_form('V')
        elif result == 'D' and not is_home:
            team.wins += 1
            team.away_wins += 1
            team.away_matches += 1
            team.away_goals_scored += goals_scored
            team.away_goals_conceded += goals_conceded
            team.update_form('V')
        elif result == 'N':
            team.draws += 1
            if is_home:
                team.home_draws += 1
                team.home_matches += 1
                team.home_goals_scored += goals_scored
                team.home_goals_conceded += goals_conceded
            else:
                team.away_draws += 1
                team.away_matches += 1
                team.away_goals_scored += goals_scored
                team.away_goals_conceded += goals_conceded
            team.update_form('N')
        else:
            team.losses += 1
            if is_home:
                team.home_losses += 1
                team.home_matches += 1
                team.home_goals_scored += goals_scored
                team.home_goals_conceded += goals_conceded
            else:
                team.away_losses += 1
                team.away_matches += 1
                team.away_goals_scored += goals_scored
                team.away_goals_conceded += goals_conceded
            team.update_form('D')
    
    def _calculate_strength(self, team: Team, db: Session):
        """Calculate attack and defense strength."""
        # Overall attack strength = goals scored / league average
        if team.matches_played > 0:
            team.attack_strength = (team.goals_scored / team.matches_played) / self.league_avg_goals_home
            team.defense_strength = (team.goals_conceded / team.matches_played) / self.league_avg_goals_home
        
        # Home attack/defense
        if team.home_matches > 0:
            team.attack_strength_home = (team.home_goals_scored / team.home_matches) / self.league_avg_goals_home
            team.defense_strength_home = (team.home_goals_conceded / team.home_matches) / self.league_avg_goals_away
        
        # Away attack/defense
        if team.away_matches > 0:
            team.attack_strength_away = (team.away_goals_scored / team.away_matches) / self.league_avg_goals_away
            team.defense_strength_away = (team.away_goals_conceded / team.away_matches) / self.league_avg_goals_home
    
    def predict_poisson_goals(self, home_team: Team, away_team: Team) -> Tuple[float, float]:
        """
        Predict expected goals using Poisson model.
        Returns (expected_home_goals, expected_away_goals).
        """
        # Home expected goals
        home_attack = home_team.attack_strength_home if home_team.attack_strength_home > 0 else home_team.attack_strength
        away_defense = away_team.defense_strength_away if away_team.defense_strength_away > 0 else away_team.defense_strength
        expected_home = home_attack * away_defense * self.league_avg_goals_home
        
        # Away expected goals
        away_attack = away_team.attack_strength_away if away_team.attack_strength_away > 0 else away_team.attack_strength
        home_defense = home_team.defense_strength_home if home_team.defense_strength_home > 0 else home_team.defense_strength
        expected_away = away_attack * home_defense * self.league_avg_goals_away
        
        return expected_home, expected_away
    
    def _calculate_adaptive_home_advantage(self, home_team: Team, away_team: Team) -> float:
        """
        Calculate adaptive home advantage based on actual home/away performance.
        Strong home team vs weak away team = high advantage
        Weak home team vs strong away team = low/negative advantage
        """
        # Calculate home team's actual home performance
        home_win_rate = 0.5  # Default
        if home_team.home_matches and home_team.home_matches > 0:
            home_win_rate = (home_team.home_wins or 0) / home_team.home_matches
        
        # Calculate away team's actual away performance
        away_loss_rate = 0.5  # Default (loss rate when playing away)
        if away_team.away_matches and away_team.away_matches > 0:
            away_loss_rate = (away_team.away_losses or 0) / away_team.away_matches
        
        # Combined factor: how much does home team benefit from being at home
        # against this specific away team's away weakness
        home_strength_factor = (home_win_rate - 0.5) * 2  # -1 to 1
        away_weakness_factor = (away_loss_rate - 0.5) * 2  # -1 to 1
        
        # Combined: ranges from -1 (home disadvantage) to +1 (strong home advantage)
        combined_factor = (home_strength_factor + away_weakness_factor) / 2
        
        # Map to ELO points: 0 to 200, centered at 100 for neutral
        adaptive_advantage = self.ELO_HOME_ADVANTAGE_BASE + combined_factor * 100
        adaptive_advantage = max(self.ELO_HOME_ADVANTAGE_MIN, 
                                 min(self.ELO_HOME_ADVANTAGE_MAX, adaptive_advantage))
        
        logger.debug(f"Adaptive home advantage: {adaptive_advantage:.1f} (home_wr={home_win_rate:.2f}, away_lr={away_loss_rate:.2f})")
        return adaptive_advantage
    
    def predict_elo_probabilities(self, home_team: Team, away_team: Team) -> Dict[str, float]:
        """
        Predict match outcome probabilities using ELO.
        """
        home_elo = home_team.elo_home if home_team.elo_home != 1500 else home_team.elo_rating
        away_elo = away_team.elo_away if away_team.elo_away != 1500 else away_team.elo_rating

        # Calculate adaptive home advantage
        home_advantage = self._calculate_adaptive_home_advantage(home_team, away_team)
        
        # Convert ELO difference to a smooth win/lose odds ratio.
        # This avoids the hard saturation that can happen when upstream ratings drift.
        elo_diff = (home_elo + home_advantage) - away_elo
        x = elo_diff / float(self.ELO_DIFF_SCALE)

        # Bradley-Terry style odds for home vs away.
        # p_home_base in (0,1)
        p_home_base = 1.0 / (1.0 + 10.0 ** (-x))
        p_away_base = 1.0 - p_home_base

        # Draw propensity is highest when teams are close, lower when mismatch is large.
        # Keep within a realistic band.
        draw_propensity = self.ELO_DRAW_BIAS * (1.0 - min(1.0, abs(elo_diff) / 600.0))
        draw_propensity = max(0.05, min(0.30, draw_propensity))

        # Allocate probability mass: remove draw mass from win/lose proportionally.
        non_draw = 1.0 - draw_propensity
        home_win_prob = non_draw * p_home_base
        away_win_prob = non_draw * p_away_base
        draw_prob = draw_propensity

        total = home_win_prob + draw_prob + away_win_prob
        return {
            'V': home_win_prob / total,
            'N': draw_prob / total,
            'D': away_win_prob / total,
        }
    
    def update_all_ratings(self, match: Match, db: Session):
        """Update all rating systems after a match."""
        self.update_elo(match, db)
        self.update_bayesian(match, db)
        self.update_poisson_strength(match, db)
        logger.info(f"Updated all ratings for match {match.id}")
    
    def initialize_team_ratings(self, team: Team):
        """Initialize ratings for a new team."""
        team.elo_rating = 1500.0
        team.elo_home = 1500.0
        team.elo_away = 1500.0
        team.bayesian_rating = self.BAYESIAN_PRIOR_MEAN
        team.bayesian_variance = self.BAYESIAN_PRIOR_VARIANCE
        team.attack_strength = 1.0
        team.defense_strength = 1.0
        team.attack_strength_home = 1.0
        team.defense_strength_home = 1.0
        team.attack_strength_away = 1.0
        team.defense_strength_away = 1.0
