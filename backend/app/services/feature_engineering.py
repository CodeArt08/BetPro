"""Feature Engineering Pipeline for match prediction."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db_context
from app.models import Match, Team, MatchFeatures, Season


class FeatureEngineeringPipeline:
    """
    Computes predictive features for each match.
    """
    
    def __init__(self):
        self.form_matches = 5  # Number of matches for form calculation
    
    def compute_features(self, match: Match, db: Session) -> MatchFeatures:
        """
        Compute all features for a match.
        """
        features = MatchFeatures(match_id=match.id)
        
        # Get teams
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        if not home_team or not away_team:
            return features
        
        # Form features
        self._compute_form_features(features, home_team, away_team, match.season_id, db)
        
        # Head-to-head features
        self._compute_h2h_features(features, match, db)
        
        # Strength features
        self._compute_strength_features(features, home_team, away_team)
        
        # Momentum features
        self._compute_momentum_features(features, home_team, away_team)
        
        # Line position features
        self._compute_line_position_features(features, match, db)
        
        # Sequence features (from Markov analysis)
        self._compute_sequence_features(features, home_team, away_team)
        
        # Line sequence features (NEW - from line position Markov analysis)
        self._compute_line_sequence_features(features, match, db)
        
        # Expected goals
        self._compute_xg_features(features, home_team, away_team)
        
        # Matchday context
        features.matchday = match.matchday
        features.is_early_matchday = 1 if match.matchday <= 5 else 0
        features.is_late_matchday = 1 if match.matchday >= 34 else 0
        
        # Compute interaction features
        self._compute_interaction_features(features, home_team, away_team)
        
        # Overall team stats
        self._compute_overall_stats(features, home_team, away_team)
        
        return features
    
    def _compute_interaction_features(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute interaction and derived features for better predictions."""
        # Form ratio (home advantage in form)
        total_form = features.home_form_points + features.away_form_points
        if total_form > 0:
            features.form_ratio = features.home_form_points / total_form
        else:
            features.form_ratio = 0.5
        
        # ELO ratio
        total_elo = home_team.elo_rating + away_team.elo_rating
        if total_elo > 0:
            features.elo_ratio = home_team.elo_rating / total_elo
        else:
            features.elo_ratio = 0.5
        
        # Attack/Defense ratios
        if away_team.defense_strength > 0:
            features.attack_defense_ratio = home_team.attack_strength / away_team.defense_strength
        else:
            features.attack_defense_ratio = home_team.attack_strength
        
        if away_team.attack_strength > 0:
            features.defense_attack_ratio = home_team.defense_strength / away_team.attack_strength
        else:
            features.defense_attack_ratio = home_team.defense_strength
        
        # Goal ratio
        total_xg = features.home_xg + features.away_xg
        if total_xg > 0:
            features.goal_ratio = features.home_xg / total_xg
        else:
            features.goal_ratio = 0.5
        
        # Strength indices (combined metrics)
        features.home_strength_index = (
            home_team.elo_rating / 2000 +
            home_team.attack_strength +
            home_team.defense_strength +
            features.home_form_points / 15 +
            home_team.win_rate
        ) / 5
        
        features.away_strength_index = (
            away_team.elo_rating / 2000 +
            away_team.attack_strength +
            away_team.defense_strength +
            features.away_form_points / 15 +
            away_team.win_rate
        ) / 5
        
        features.strength_diff = features.home_strength_index - features.away_strength_index
        
        # Form momentum (weighted recent form)
        features.home_form_momentum = (
            features.home_gd_trend * 0.3 +
            home_team.winning_streak * 0.1 -
            home_team.losing_streak * 0.1 +
            features.home_form_points / 15 * 0.5
        )
        
        features.away_form_momentum = (
            features.away_gd_trend * 0.3 +
            away_team.winning_streak * 0.1 -
            away_team.losing_streak * 0.1 +
            features.away_form_points / 15 * 0.5
        )
        
        features.form_diff = features.home_form_points - features.away_form_points
        
        # H2H weighted features
        h2h_count = features.h2h_strict_total_matches or 0
        features.h2h_recency_weight = min(h2h_count / 10, 1.0)
        
        # H2H predictability
        if h2h_count > 0:
            draw_rate = features.h2h_draw_rate or 0.33
            dominance = abs(features.h2h_dominance_score or 0)
            features.h2h_predictability = (1 - draw_rate) * 0.5 + dominance * 0.5
        else:
            features.h2h_predictability = 0.0
    
    def _compute_form_features(self, features: MatchFeatures, home_team: Team, 
                               away_team: Team, season_id: int, db: Session):
        """Compute form-based features."""
        # Home team overall form
        home_matches = self._get_recent_matches(home_team.id, season_id, db, limit=self.form_matches)
        features.home_form_points = self._calculate_points(home_matches, home_team.id)
        features.home_form_goals_scored = self._calculate_goals_scored(home_matches, home_team.id)
        features.home_form_goals_conceded = self._calculate_goals_conceded(home_matches, home_team.id)
        features.home_form_wins, features.home_form_draws, features.home_form_losses = \
            self._calculate_wdl(home_matches, home_team.id)
        
        # Away team overall form
        away_matches = self._get_recent_matches(away_team.id, season_id, db, limit=self.form_matches)
        features.away_form_points = self._calculate_points(away_matches, away_team.id)
        features.away_form_goals_scored = self._calculate_goals_scored(away_matches, away_team.id)
        features.away_form_goals_conceded = self._calculate_goals_conceded(away_matches, away_team.id)
        features.away_form_wins, features.away_form_draws, features.away_form_losses = \
            self._calculate_wdl(away_matches, away_team.id)
        
        # Home team at home form
        home_home_matches = self._get_recent_home_matches(home_team.id, season_id, db, limit=self.form_matches)
        features.home_home_form_points = self._calculate_points(home_home_matches, home_team.id)
        features.home_home_form_goals = self._calculate_goals_scored(home_home_matches, home_team.id)
        features.home_home_form_conceded = self._calculate_goals_conceded(home_home_matches, home_team.id)
        
        # Away team away form
        away_away_matches = self._get_recent_away_matches(away_team.id, season_id, db, limit=self.form_matches)
        features.away_away_form_points = self._calculate_points(away_away_matches, away_team.id)
        features.away_away_form_goals = self._calculate_goals_scored(away_away_matches, away_team.id)
        features.away_away_form_conceded = self._calculate_goals_conceded(away_away_matches, away_team.id)
        
        # Goal difference trends
        features.home_gd_trend = self._calculate_gd_trend(home_matches, home_team.id)
        features.away_gd_trend = self._calculate_gd_trend(away_matches, away_team.id)
        
        # Scoring frequency
        features.home_scoring_freq = home_team.goals_scored / home_team.matches_played if home_team.matches_played > 0 else 0
        features.away_scoring_freq = away_team.goals_scored / away_team.matches_played if away_team.matches_played > 0 else 0
        features.home_conceding_freq = home_team.goals_conceded / home_team.matches_played if home_team.matches_played > 0 else 0
        features.away_conceding_freq = away_team.goals_conceded / away_team.matches_played if away_team.matches_played > 0 else 0
    
    def _compute_h2h_features(self, features: MatchFeatures, match: Match, db: Session):
        """Compute head-to-head features with strict home/away separation."""
        h2h_matches = db.query(Match).filter(
            ((Match.home_team_id == match.home_team_id) & (Match.away_team_id == match.away_team_id)) |
            ((Match.home_team_id == match.away_team_id) & (Match.away_team_id == match.home_team_id)),
            Match.is_completed == True
        ).order_by(Match.matchday.desc()).all()  # Most recent first
        
        if h2h_matches:
            features.h2h_home_wins = sum(1 for m in h2h_matches if 
                (m.home_team_id == match.home_team_id and m.result == 'V') or
                (m.away_team_id == match.home_team_id and m.result == 'D'))
            features.h2h_away_wins = sum(1 for m in h2h_matches if 
                (m.home_team_id == match.away_team_id and m.result == 'V') or
                (m.away_team_id == match.away_team_id and m.result == 'D'))
            features.h2h_draws = sum(1 for m in h2h_matches if m.result == 'N')
            
            total_matches = len(h2h_matches)
            features.h2h_avg_goals = sum(m.total_goals or 0 for m in h2h_matches) / total_matches
            features.h2h_draw_rate = features.h2h_draws / total_matches
            
            # Goals by home team in H2H
            home_goals = sum(
                m.score_home if m.home_team_id == match.home_team_id else m.score_away
                for m in h2h_matches if m.score_home is not None
            )
            features.h2h_home_avg_goals = home_goals / total_matches
            
            away_goals = sum(
                m.score_away if m.home_team_id == match.home_team_id else m.score_home
                for m in h2h_matches if m.score_away is not None
            )
            features.h2h_away_avg_goals = away_goals / total_matches
            
            # === STRICT H2H: Home team AT HOME vs away team ===
            strict_home_matches = [m for m in h2h_matches if m.home_team_id == match.home_team_id]
            strict_away_matches = [m for m in h2h_matches if m.away_team_id == match.home_team_id]
            
            features.h2h_strict_total_matches = total_matches
            
            # When home team played at home
            if strict_home_matches:
                features.h2h_strict_home_wins = sum(1 for m in strict_home_matches if m.result == 'V')
                features.h2h_strict_home_losses = sum(1 for m in strict_home_matches if m.result == 'D')
                features.h2h_strict_home_draws = sum(1 for m in strict_home_matches if m.result == 'N')
                features.h2h_strict_home_goals_scored = sum(m.score_home or 0 for m in strict_home_matches) / len(strict_home_matches)
                features.h2h_strict_home_goals_conceded = sum(m.score_away or 0 for m in strict_home_matches) / len(strict_home_matches)
                features.h2h_strict_home_win_rate = features.h2h_strict_home_wins / len(strict_home_matches)
            
            # When away team played at home (vs home team)
            if strict_away_matches:
                features.h2h_strict_away_wins = sum(1 for m in strict_away_matches if m.result == 'D')  # Away team won
                features.h2h_strict_away_losses = sum(1 for m in strict_away_matches if m.result == 'V')  # Away team lost
                features.h2h_strict_away_draws = sum(1 for m in strict_away_matches if m.result == 'N')
                features.h2h_strict_away_goals_scored = sum(m.score_away or 0 for m in strict_away_matches) / len(strict_away_matches)
                features.h2h_strict_away_goals_conceded = sum(m.score_home or 0 for m in strict_away_matches) / len(strict_away_matches)
                features.h2h_strict_away_win_rate = features.h2h_strict_away_wins / len(strict_away_matches)
            
            # Last 5 H2H results (from home team perspective)
            last_5 = []
            for m in h2h_matches[:5]:
                if m.home_team_id == match.home_team_id:
                    last_5.append(m.result)  # V, N, D as is
                else:
                    # Invert result when home team was away
                    inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
                    last_5.append(inverted.get(m.result, 'N'))
            features.h2h_last_5_results = last_5
            
            # Dominance score based on strict home/away
            home_strength = features.h2h_strict_home_win_rate if features.h2h_strict_home_win_rate else 0.33
            away_weakness = 1 - (features.h2h_strict_away_win_rate if features.h2h_strict_away_win_rate else 0.33)
            features.h2h_dominance_score = (home_strength + away_weakness) / 2 - 0.5  # -0.5 to 0.5
    
    def _compute_strength_features(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute team strength features including league position."""
        features.home_elo = home_team.elo_rating
        features.away_elo = away_team.elo_rating
        features.elo_diff = home_team.elo_rating - away_team.elo_rating
        features.home_elo_home = home_team.elo_home
        features.away_elo_away = away_team.elo_away
        
        # League position (classement)
        features.home_league_position = home_team.league_position or 0
        features.away_league_position = away_team.league_position or 0
        features.league_position_diff = (home_team.league_position or 0) - (away_team.league_position or 0)
        features.home_league_points = home_team.league_points or 0
        features.away_league_points = away_team.league_points or 0
        features.points_diff = (home_team.league_points or 0) - (away_team.league_points or 0)
        
        features.home_bayesian_rating = home_team.bayesian_rating
        features.away_bayesian_rating = away_team.bayesian_rating
        features.bayesian_diff = home_team.bayesian_rating - away_team.bayesian_rating
        
        features.home_attack_strength = home_team.attack_strength
        features.away_attack_strength = away_team.attack_strength
        features.home_defense_strength = home_team.defense_strength
        features.away_defense_strength = away_team.defense_strength
    
    def _compute_momentum_features(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute momentum and streak features."""
        features.home_winning_streak = home_team.winning_streak
        features.away_winning_streak = away_team.winning_streak
        features.home_losing_streak = home_team.losing_streak
        features.away_losing_streak = away_team.losing_streak
        features.home_unbeaten_streak = home_team.unbeaten_streak
        features.away_unbeaten_streak = away_team.unbeaten_streak
        
        features.home_advantage = home_team.home_advantage
        features.away_disadvantage = -away_team.home_advantage
    
    def _compute_line_position_features(self, features: MatchFeatures, match: Match, db: Session):
        """Compute features based on line position in matchday."""
        features.line_position = match.line_position
        
        # Get historical stats for this line position
        line_matches = db.query(Match).filter(
            Match.line_position == match.line_position,
            Match.is_completed == True
        ).limit(100).all()
        
        if line_matches:
            total = len(line_matches)
            features.line_home_win_rate = sum(1 for m in line_matches if m.result == 'V') / total
            features.line_draw_rate = sum(1 for m in line_matches if m.result == 'N') / total
            features.line_away_win_rate = sum(1 for m in line_matches if m.result == 'D') / total
            features.line_avg_goals = sum(m.total_goals or 0 for m in line_matches) / total
    
    def _compute_sequence_features(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute sequence pattern features (placeholder - updated by sequence analyzer)."""
        # These will be populated by the SequencePatternAnalyzer
        features.home_sequence_prob_V = 0.33
        features.home_sequence_prob_N = 0.33
        features.home_sequence_prob_D = 0.33
        features.away_sequence_prob_V = 0.33
        features.away_sequence_prob_N = 0.33
        features.away_sequence_prob_D = 0.33
        
        # Line sequence features - will be populated by SequencePatternAnalyzer
        features.line_sequence_prob_V = 0.33
        features.line_sequence_prob_N = 0.33
        features.line_sequence_prob_D = 0.33
        features.line_recent_form = 0.0
    
    def _compute_line_sequence_features(self, features: MatchFeatures, match: Match, db: Session):
        """Compute line sequence features using Markov analysis on line position history."""
        from app.services.sequence_analysis import SequencePatternAnalyzer
        
        # Create analyzer and load line sequences
        analyzer = SequencePatternAnalyzer()
        analyzer.load_line_sequences(db)
        
        line_position = match.line_position or 1
        
        # Get line form analysis
        line_form = analyzer.get_line_form_analysis(line_position)
        recent_form = line_form.get('recent_form', '')
        
        # Get sequence probability for next result
        probs = analyzer.get_line_sequence_probability(line_position, recent_form)
        
        features.line_sequence_prob_V = probs.get('V', 0.33)
        features.line_sequence_prob_N = probs.get('N', 0.33)
        features.line_sequence_prob_D = probs.get('D', 0.33)
        
        # Encode recent form as a float (V=1, N=0.5, D=0)
        form_value = 0.0
        if recent_form:
            form_map = {'V': 1.0, 'N': 0.5, 'D': 0.0}
            form_values = [form_map.get(c, 0.5) for c in recent_form]
            form_value = sum(form_values) / len(form_values) if form_values else 0.5
        features.line_recent_form = form_value
    
    def _compute_xg_features(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute expected goals features."""
        # Based on attack/defense strength
        league_avg_goals = 1.35  # Typical for Spanish football
        
        features.home_xg = home_team.attack_strength * away_team.defense_strength * league_avg_goals
        features.away_xg = away_team.attack_strength * home_team.defense_strength * league_avg_goals * 0.8  # Away factor
    
    def _compute_overall_stats(self, features: MatchFeatures, home_team: Team, away_team: Team):
        """Compute overall team statistics."""
        features.home_win_rate = home_team.win_rate
        features.away_win_rate = away_team.win_rate
        features.home_draw_rate = home_team.draws / home_team.matches_played if home_team.matches_played > 0 else 0
        features.away_draw_rate = away_team.draws / away_team.matches_played if away_team.matches_played > 0 else 0
        features.home_avg_goals_scored = home_team.goals_scored / home_team.matches_played if home_team.matches_played > 0 else 0
        features.away_avg_goals_scored = away_team.goals_scored / away_team.matches_played if away_team.matches_played > 0 else 0
        features.home_avg_goals_conceded = home_team.goals_conceded / home_team.matches_played if home_team.matches_played > 0 else 0
        features.away_avg_goals_conceded = away_team.goals_conceded / away_team.matches_played if away_team.matches_played > 0 else 0
    
    def _get_recent_matches(self, team_id: int, season_id: int, db: Session, limit: int) -> List[Match]:
        """Get recent matches for a team."""
        return db.query(Match).filter(
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
            Match.season_id == season_id,
            Match.is_completed == True
        ).order_by(Match.matchday.desc()).limit(limit).all()
    
    def _get_recent_home_matches(self, team_id: int, season_id: int, db: Session, limit: int) -> List[Match]:
        """Get recent home matches for a team."""
        return db.query(Match).filter(
            Match.home_team_id == team_id,
            Match.season_id == season_id,
            Match.is_completed == True
        ).order_by(Match.matchday.desc()).limit(limit).all()
    
    def _get_recent_away_matches(self, team_id: int, season_id: int, db: Session, limit: int) -> List[Match]:
        """Get recent away matches for a team."""
        return db.query(Match).filter(
            Match.away_team_id == team_id,
            Match.season_id == season_id,
            Match.is_completed == True
        ).order_by(Match.matchday.desc()).limit(limit).all()
    
    def _calculate_points(self, matches: List[Match], team_id: int) -> float:
        """Calculate points from matches."""
        points = 0
        for m in matches:
            if m.home_team_id == team_id:
                if m.result == 'V':
                    points += 3
                elif m.result == 'N':
                    points += 1
            else:
                if m.result == 'D':
                    points += 3
                elif m.result == 'N':
                    points += 1
        return points
    
    def _calculate_goals_scored(self, matches: List[Match], team_id: int) -> float:
        """Calculate goals scored in matches."""
        goals = 0
        for m in matches:
            if m.home_team_id == team_id:
                goals += m.score_home or 0
            else:
                goals += m.score_away or 0
        return goals
    
    def _calculate_goals_conceded(self, matches: List[Match], team_id: int) -> float:
        """Calculate goals conceded in matches."""
        goals = 0
        for m in matches:
            if m.home_team_id == team_id:
                goals += m.score_away or 0
            else:
                goals += m.score_home or 0
        return goals
    
    def _calculate_wdl(self, matches: List[Match], team_id: int) -> tuple:
        """Calculate wins, draws, losses."""
        wins = draws = losses = 0
        for m in matches:
            if m.home_team_id == team_id:
                if m.result == 'V':
                    wins += 1
                elif m.result == 'N':
                    draws += 1
                else:
                    losses += 1
            else:
                if m.result == 'D':
                    wins += 1
                elif m.result == 'N':
                    draws += 1
                else:
                    losses += 1
        return wins, draws, losses
    
    def _calculate_gd_trend(self, matches: List[Match], team_id: int) -> float:
        """Calculate goal difference trend."""
        if len(matches) < 2:
            return 0.0
        
        gds = []
        for m in matches:
            if m.home_team_id == team_id:
                gds.append((m.score_home or 0) - (m.score_away or 0))
            else:
                gds.append((m.score_away or 0) - (m.score_home or 0))
        
        # Linear trend
        x = np.arange(len(gds))
        if len(gds) > 1:
            slope, _ = np.polyfit(x, gds, 1)
            return slope
        return 0.0
    
    def prepare_training_data(self, db: Session) -> tuple:
        """
        Prepare training data for ML models.
        Returns X (features) and y (targets).
        """
        matches = db.query(Match).filter(Match.is_completed == True).all()
        
        X = []
        y = []
        
        for match in matches:
            features = db.query(MatchFeatures).filter(MatchFeatures.match_id == match.id).first()
            if features:
                X.append(features.to_dict())
                y.append(match.result)
        
        df_X = pd.DataFrame(X)
        df_y = pd.Series(y)
        
        return df_X, df_y
