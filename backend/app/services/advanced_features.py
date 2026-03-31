"""Advanced Feature Engineering for ultra-precise predictions."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime

from app.models import Match, Team, MatchFeatures


class AdvancedFeatureEngine:
    """
    Computes advanced predictive features including:
    - Weighted H2H with recency
    - Momentum and streaks
    - Attack vs Defense differentials
    - Efficiency metrics
    """
    
    # Recency weights for H2H (more recent = higher weight)
    RECENCY_WEIGHTS = {
        1: 1.0,   # Last match
        2: 0.85,
        3: 0.70,
        4: 0.55,
        5: 0.40,
        6: 0.30,
        7: 0.25,
        8: 0.20,
        9: 0.15,
        10: 0.10
    }
    
    def __init__(self):
        self.form_window = 5
    
    def compute_advanced_h2h(self, match: Match, db: Session) -> Dict:
        """
        Compute advanced head-to-head features with recency weighting.
        Separates home/away performance properly.
        """
        home_team_id = match.home_team_id
        away_team_id = match.away_team_id
        
        # Get all H2H matches ordered by date (most recent first)
        h2h_matches = db.query(Match).filter(
            ((Match.home_team_id == home_team_id) & (Match.away_team_id == away_team_id)) |
            ((Match.home_team_id == away_team_id) & (Match.away_team_id == home_team_id)),
            Match.is_completed == True
        ).order_by(desc(Match.matchday)).limit(10).all()
        
        if not h2h_matches:
            return self._default_h2h_features()
        
        # Reverse to get chronological order (oldest first)
        h2h_matches = list(reversed(h2h_matches))
        
        # Calculate weighted metrics
        total_weight = 0.0
        weighted_home_goals = 0.0
        weighted_away_goals = 0.0
        weighted_home_conceded = 0.0
        weighted_away_conceded = 0.0
        
        home_wins_at_home = 0
        home_matches_at_home = 0
        away_wins_at_home = 0  # Away team playing at home
        away_matches_at_home = 0
        
        last_5_results = []
        dominance_points = 0.0
        
        for idx, h2h_match in enumerate(h2h_matches):
            weight = self.RECENCY_WEIGHTS.get(idx + 1, 0.1)
            total_weight += weight
            
            # Determine which team is home in this H2H match
            home_is_original_home = (h2h_match.home_team_id == home_team_id)
            
            # Goals from perspective of current home team
            if home_is_original_home:
                # Current home team played at home in this H2H
                home_goals = h2h_match.score_home or 0
                away_goals = h2h_match.score_away or 0
                
                home_matches_at_home += 1
                if h2h_match.result == 'V':
                    home_wins_at_home += 1
                    dominance_points += weight
                elif h2h_match.result == 'D':
                    dominance_points -= weight * 0.5
            else:
                # Current home team played away in this H2H
                home_goals = h2h_match.score_away or 0
                away_goals = h2h_match.score_home or 0
                
                away_matches_at_home += 1
                if h2h_match.result == 'D':
                    away_wins_at_home += 1
                    dominance_points -= weight
                elif h2h_match.result == 'N':
                    dominance_points += weight * 0.5
            
            weighted_home_goals += home_goals * weight
            weighted_away_goals += away_goals * weight
            weighted_home_conceded += away_goals * weight
            weighted_away_conceded += home_goals * weight
            
            # Track last 5 results from home team perspective
            if len(last_5_results) < 5:
                if home_is_original_home:
                    last_5_results.append(h2h_match.result)
                else:
                    # Invert result
                    inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
                    last_5_results.append(inverted.get(h2h_match.result, 'N'))
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_home_goals /= total_weight
            weighted_away_goals /= total_weight
            weighted_home_conceded /= total_weight
            weighted_away_conceded /= total_weight
            dominance_score = dominance_points / total_weight
        else:
            dominance_score = 0.0
        
        # Calculate win rates
        h2h_home_win_rate = home_wins_at_home / home_matches_at_home if home_matches_at_home > 0 else 0.5
        h2h_away_win_rate = away_wins_at_home / away_matches_at_home if away_matches_at_home > 0 else 0.5
        
        # Calculate predictability (how consistent are results)
        if len(last_5_results) >= 3:
            result_counts = {'V': 0, 'N': 0, 'D': 0}
            for r in last_5_results:
                result_counts[r] = result_counts.get(r, 0) + 1
            max_count = max(result_counts.values())
            recency_weighted_accuracy = max_count / len(last_5_results)
        else:
            recency_weighted_accuracy = 0.5
        
        return {
            'h2h_home_win_rate': h2h_home_win_rate,
            'h2h_away_win_rate': h2h_away_win_rate,
            'h2h_weighted_home_goals': weighted_home_goals,
            'h2h_weighted_away_goals': weighted_away_goals,
            'h2h_weighted_home_conceded': weighted_home_conceded,
            'h2h_weighted_away_conceded': weighted_away_conceded,
            'h2h_dominance_score': dominance_score,  # -1 to 1
            'h2h_last_5_results': last_5_results,
            'h2h_recency_weighted_accuracy': recency_weighted_accuracy
        }
    
    def compute_weighted_form(self, team_id: int, season_id: int, db: Session, 
                              is_home: bool = True, window: int = 5) -> Dict:
        """
        Compute weighted form with exponential decay.
        More recent matches have higher weight.
        """
        if is_home:
            matches = db.query(Match).filter(
                Match.home_team_id == team_id,
                Match.season_id == season_id,
                Match.is_completed == True
            ).order_by(desc(Match.matchday)).limit(window).all()
        else:
            matches = db.query(Match).filter(
                Match.away_team_id == team_id,
                Match.season_id == season_id,
                Match.is_completed == True
            ).order_by(desc(Match.matchday)).limit(window).all()
        
        if not matches:
            return self._default_form_features()
        
        # Reverse to chronological order
        matches = list(reversed(matches))
        
        # Exponential decay weights
        decay_factor = 0.8
        weights = [decay_factor ** (window - i - 1) for i in range(len(matches))]
        total_weight = sum(weights)
        
        weighted_points = 0.0
        weighted_goals_scored = 0.0
        weighted_goals_conceded = 0.0
        
        results = []
        
        for i, m in enumerate(matches):
            weight = weights[i]
            
            if m.home_team_id == team_id:
                # Team played at home
                goals_scored = m.score_home or 0
                goals_conceded = m.score_away or 0
                if m.result == 'V':
                    points = 3
                    results.append('W')
                elif m.result == 'N':
                    points = 1
                    results.append('D')
                else:
                    points = 0
                    results.append('L')
            else:
                # Team played away
                goals_scored = m.score_away or 0
                goals_conceded = m.score_home or 0
                if m.result == 'D':
                    points = 3
                    results.append('W')
                elif m.result == 'N':
                    points = 1
                    results.append('D')
                else:
                    points = 0
                    results.append('L')
            
            weighted_points += points * weight
            weighted_goals_scored += goals_scored * weight
            weighted_goals_conceded += goals_conceded * weight
        
        # Normalize
        weighted_points /= total_weight
        weighted_goals_scored /= total_weight
        weighted_goals_conceded /= total_weight
        
        # Calculate form rating (0-1 scale)
        max_points = 3 * len(matches)
        form_rating = weighted_points / max_points if max_points > 0 else 0.5
        
        # Calculate momentum (trend in last 3 vs previous)
        if len(results) >= 3:
            recent_points = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in results[-3:])
            older_points = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in results[:-3]) if len(results) > 3 else recent_points
            momentum = (recent_points - older_points) / 9.0  # Normalized
        else:
            momentum = 0.0
        
        return {
            'weighted_form_points': weighted_points,
            'weighted_goals_scored': weighted_goals_scored,
            'weighted_goals_conceded': weighted_goals_conceded,
            'form_rating': form_rating,
            'momentum': momentum,
            'recent_results': results
        }
    
    def compute_attack_defense_differential(self, home_team: Team, away_team: Team) -> Dict:
        """
        Compute attack vs defense differentials.
        How does home attack compare to away defense and vice versa.
        """
        # Home attack vs Away defense
        home_attack = home_team.attack_strength_home or home_team.attack_strength or 1.0
        away_defense = away_team.defense_strength_away or away_team.defense_strength or 1.0
        home_attack_advantage = home_attack - away_defense
        
        # Away attack vs Home defense
        away_attack = away_team.attack_strength_away or away_team.attack_strength or 1.0
        home_defense = home_team.defense_strength_home or home_team.defense_strength or 1.0
        away_attack_advantage = away_attack - home_defense
        
        # Overall differential
        overall_diff = home_attack_advantage - away_attack_advantage
        
        # Efficiency metrics (goals per opportunity)
        home_efficiency = home_team.goals_scored / max(home_team.matches_played, 1)
        away_efficiency = away_team.goals_scored / max(away_team.matches_played, 1)
        
        home_defensive_efficiency = home_team.goals_conceded / max(home_team.matches_played, 1)
        away_defensive_efficiency = away_team.goals_conceded / max(away_team.matches_played, 1)
        
        return {
            'home_attack_advantage': home_attack_advantage,
            'away_attack_advantage': away_attack_advantage,
            'overall_attack_defense_diff': overall_diff,
            'home_offensive_efficiency': home_efficiency,
            'away_offensive_efficiency': away_efficiency,
            'home_defensive_efficiency': home_defensive_efficiency,
            'away_defensive_efficiency': away_defensive_efficiency
        }
    
    def compute_momentum_features(self, team: Team) -> Dict:
        """
        Compute intelligent momentum features.
        """
        # Streaks
        winning_streak = team.winning_streak or 0
        losing_streak = team.losing_streak or 0
        unbeaten_streak = team.unbeaten_streak or 0
        
        # Streak strength (exponential bonus for longer streaks)
        winning_momentum = min(winning_streak ** 1.5 / 10.0, 1.0)
        losing_momentum = -min(losing_streak ** 1.5 / 10.0, 1.0)
        
        # Combined momentum
        if winning_streak > 0:
            momentum = winning_momentum
        elif losing_streak > 0:
            momentum = losing_momentum
        else:
            momentum = 0.0
        
        # Form string analysis (use empty if not available)
        form_string = getattr(team, 'form', '') or ""
        form_points = 0
        for char in form_string:
            if char == 'W':
                form_points += 3
            elif char == 'D':
                form_points += 1
        
        form_points_per_game = form_points / len(form_string) if form_string else 1.0
        
        return {
            'momentum': momentum,
            'winning_streak': winning_streak,
            'losing_streak': losing_streak,
            'unbeaten_streak': unbeaten_streak,
            'form_points_per_game': form_points_per_game
        }
    
    def compute_markov_probabilities(self, team_id: int, season_id: int, db: Session) -> Dict:
        """
        Compute Markov chain probabilities for next result based on sequence.
        """
        # Get recent results
        matches = db.query(Match).filter(
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
            Match.season_id == season_id,
            Match.is_completed == True
        ).order_by(desc(Match.matchday)).limit(10).all()
        
        if len(matches) < 3:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        # Build sequence from team perspective
        sequence = []
        for m in reversed(matches):
            if m.home_team_id == team_id:
                sequence.append(m.result)
            else:
                inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
                sequence.append(inverted.get(m.result, 'N'))
        
        # Build transition matrix
        transitions = {'V': {'V': 0, 'N': 0, 'D': 0}, 
                       'N': {'V': 0, 'N': 0, 'D': 0}, 
                       'D': {'V': 0, 'N': 0, 'D': 0}}
        
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_result = sequence[i + 1]
            transitions[current][next_result] += 1
        
        # Get last result
        last_result = sequence[-1] if sequence else 'N'
        
        # Calculate probabilities
        total = sum(transitions[last_result].values())
        if total > 0:
            probs = {
                'V': transitions[last_result]['V'] / total,
                'N': transitions[last_result]['N'] / total,
                'D': transitions[last_result]['D'] / total
            }
        else:
            probs = {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        return probs
    
    def update_features(self, features: MatchFeatures, match: Match, db: Session) -> MatchFeatures:
        """
        Update features with advanced computations.
        """
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        if not home_team or not away_team:
            return features
        
        # Advanced H2H
        h2h_features = self.compute_advanced_h2h(match, db)
        features.h2h_home_win_rate = h2h_features['h2h_home_win_rate']
        features.h2h_away_win_rate = h2h_features['h2h_away_win_rate']
        features.h2h_weighted_home_goals = h2h_features['h2h_weighted_home_goals']
        features.h2h_weighted_away_goals = h2h_features['h2h_weighted_away_goals']
        features.h2h_weighted_home_conceded = h2h_features['h2h_weighted_home_conceded']
        features.h2h_weighted_away_conceded = h2h_features['h2h_weighted_away_conceded']
        features.h2h_dominance_score = h2h_features['h2h_dominance_score']
        features.h2h_last_5_results = h2h_features['h2h_last_5_results']
        features.h2h_recency_weighted_accuracy = h2h_features['h2h_recency_weighted_accuracy']
        
        # Store advanced features in extra_features
        extra = features.extra_features or {}
        
        # Weighted form
        home_form = self.compute_weighted_form(match.home_team_id, match.season_id, db, is_home=True)
        away_form = self.compute_weighted_form(match.away_team_id, match.season_id, db, is_home=False)
        
        extra['home_weighted_form_rating'] = home_form['form_rating']
        extra['home_momentum'] = home_form['momentum']
        extra['away_weighted_form_rating'] = away_form['form_rating']
        extra['away_momentum'] = away_form['momentum']
        
        # Attack vs Defense differential
        ad_diff = self.compute_attack_defense_differential(home_team, away_team)
        extra['home_attack_advantage'] = ad_diff['home_attack_advantage']
        extra['away_attack_advantage'] = ad_diff['away_attack_advantage']
        extra['overall_attack_defense_diff'] = ad_diff['overall_attack_defense_diff']
        extra['home_offensive_efficiency'] = ad_diff['home_offensive_efficiency']
        extra['away_offensive_efficiency'] = ad_diff['away_offensive_efficiency']
        
        # Momentum features
        home_momentum = self.compute_momentum_features(home_team)
        away_momentum = self.compute_momentum_features(away_team)
        extra['home_momentum_strength'] = home_momentum['momentum']
        extra['away_momentum_strength'] = away_momentum['momentum']
        
        # Markov probabilities
        home_markov = self.compute_markov_probabilities(match.home_team_id, match.season_id, db)
        away_markov = self.compute_markov_probabilities(match.away_team_id, match.season_id, db)
        extra['home_markov_V'] = home_markov['V']
        extra['home_markov_N'] = home_markov['N']
        extra['home_markov_D'] = home_markov['D']
        extra['away_markov_V'] = away_markov['V']
        extra['away_markov_N'] = away_markov['N']
        extra['away_markov_D'] = away_markov['D']
        
        features.extra_features = extra
        
        return features
    
    def _default_h2h_features(self) -> Dict:
        """Return default H2H features when no history exists."""
        return {
            'h2h_home_win_rate': 0.5,
            'h2h_away_win_rate': 0.5,
            'h2h_weighted_home_goals': 1.35,
            'h2h_weighted_away_goals': 1.1,
            'h2h_weighted_home_conceded': 1.1,
            'h2h_weighted_away_conceded': 1.35,
            'h2h_dominance_score': 0.0,
            'h2h_last_5_results': [],
            'h2h_recency_weighted_accuracy': 0.5
        }
    
    def _default_form_features(self) -> Dict:
        """Return default form features when no history exists."""
        return {
            'weighted_form_points': 4.5,
            'weighted_goals_scored': 1.35,
            'weighted_goals_conceded': 1.1,
            'form_rating': 0.5,
            'momentum': 0.0,
            'recent_results': []
        }
