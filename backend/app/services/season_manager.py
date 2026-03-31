"""Season Lifecycle Manager."""
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db_context
from app.models import Season, Match, Team, Bet


class SeasonManager:
    """
    Manages season lifecycle and transitions.
    """
    
    MATCHDAYS_PER_SEASON = 38
    MATCHES_PER_MATCHDAY = 10
    
    def __init__(self):
        self.current_season: Optional[Season] = None
    
    def get_active_season(self, db: Session) -> Season:
        """Get or create the active season."""
        season = db.query(Season).filter(Season.is_active == True).first()
        
        if not season:
            season = self._create_new_season(db)
        
        self.current_season = season
        return season

    def create_new_season(self, db: Session) -> Season:
        """Public wrapper for creating a new season."""
        return self._create_new_season(db)
    
    def _create_new_season(self, db: Session) -> Season:
        """Create a new season."""
        # Get next season number
        last_season = db.query(Season).order_by(Season.season_number.desc()).first()
        next_number = (last_season.season_number + 1) if last_season else 1
        
        season = Season(
            season_number=next_number,
            is_active=True,
            is_completed=False
        )
        
        db.add(season)
        db.commit()
        db.refresh(season)
        
        logger.info(f"Created new season: {season.season_number}")
        
        return season
    
    def check_season_completion(self, db: Session) -> bool:
        """Check if current season is complete."""
        season = self.get_active_season(db)
        
        completed_matches = db.query(Match).filter(
            Match.season_id == season.id,
            Match.is_completed == True
        ).count()
        
        expected_matches = self.MATCHDAYS_PER_SEASON * self.MATCHES_PER_MATCHDAY
        
        return completed_matches >= expected_matches
    
    def close_season(self, db: Session) -> Season:
        """Close the current season and prepare for next."""
        season = self.get_active_season(db)
        
        # Update final statistics
        self._update_season_statistics(season, db)
        
        # Close season
        season.close_season()
        db.commit()
        
        logger.info(f"Closed season {season.season_number}")
        
        # Create new season
        new_season = self._create_new_season(db)
        
        return new_season
    
    def _update_season_statistics(self, season: Season, db: Session):
        """Update season statistics before closing."""
        from app.models import Prediction
        
        matches = db.query(Match).filter(Match.season_id == season.id).all()
        
        season.total_matches = len(matches)
        season.total_home_wins = sum(1 for m in matches if m.result == 'V')
        season.total_draws = sum(1 for m in matches if m.result == 'N')
        season.total_away_wins = sum(1 for m in matches if m.result == 'D')
        season.total_goals = sum((m.score_home or 0) + (m.score_away or 0) for m in matches)
        season.avg_goals_per_match = season.total_goals / season.total_matches if season.total_matches > 0 else 0
        
        # Betting statistics - try bets first, then predictions
        bets = db.query(Bet).filter(Bet.season_id == season.id, Bet.is_settled == True).all()
        
        if bets:
            season.total_bets = len(bets)
            season.winning_bets = sum(1 for b in bets if b.status == 'won')
            season.total_profit = sum(b.profit_loss or 0 for b in bets)
            total_stake = sum(b.stake for b in bets)
            season.roi = season.total_profit / total_stake if total_stake > 0 else 0
            logger.info(f"Season {season.season_number} stats from bets: {season.total_bets} bets, {season.winning_bets} wins, {season.total_profit} Ar profit")
        else:
            # Calculate from verified predictions (simulated betting)
            # CRITICAL: Only use SELECTED predictions for consistency with strategy
            # MIN_CONFIDENCE filter: only predictions with confidence >= 50%
            # MIN_ODDS filter: only predictions with odds >= 1.80
            from app.models import Prediction
            MIN_CONFIDENCE = 0.5
            MIN_ODDS = 1.80
            verified_preds = db.query(Prediction).join(Match).filter(
                Prediction.season_id == season.id,
                Prediction.actual_result != None,
                Prediction.is_selected_for_bet == True,  # Only use selected ones
                Prediction.confidence >= MIN_CONFIDENCE,
                # Odds filter based on predicted result
                ((Prediction.predicted_result == 'V') & (Match.odd_home >= MIN_ODDS)) |
                ((Prediction.predicted_result == 'N') & (Match.odd_draw >= MIN_ODDS)) |
                ((Prediction.predicted_result == 'D') & (Match.odd_away >= MIN_ODDS))
            ).all()
            
            if verified_preds:
                STAKE = 1000
                winning_bets = 0
                total_profit = 0.0
                
                for pred in verified_preds:
                    if pred.is_correct:
                        winning_bets += 1
                        match = pred.match
                        # Logic for determining odds
                        if pred.predicted_result == 'V':
                            odds = match.odd_home if match and match.odd_home else 2.0
                        elif pred.predicted_result == 'N':
                            odds = match.odd_draw if match and match.odd_draw else 3.5
                        else:
                            odds = match.odd_away if match and match.odd_away else 3.0
                        profit = (STAKE * odds) - STAKE
                        total_profit += profit
                    else:
                        total_profit -= STAKE
                
                season.total_bets = len(verified_preds)
                season.winning_bets = winning_bets
                season.total_profit = total_profit
                total_stake = len(verified_preds) * STAKE
                season.roi = total_profit / total_stake if total_stake > 0 else 0
                logger.info(f"Season {season.season_number} stats from selected predictions: {season.total_bets} preds, {season.winning_bets} correct, {season.total_profit} Ar profit")
            else:
                season.total_bets = 0
                season.winning_bets = 0
                season.total_profit = 0.0
                season.roi = 0.0
                logger.warning(f"Season {season.season_number} has no bets or verified selected predictions - stats set to 0")
    
    def get_season_progress(self, db: Session) -> Dict:
        """Get current season progress."""
        season = self.get_active_season(db)
        
        completed_matches = db.query(Match).filter(
            Match.season_id == season.id,
            Match.is_completed == True
        ).count()
        
        completed_matchdays = db.query(Match.matchday).filter(
            Match.season_id == season.id,
            Match.is_completed == True
        ).distinct().count()
        
        total_matches = self.MATCHDAYS_PER_SEASON * self.MATCHES_PER_MATCHDAY
        
        # Cap progress at 100%
        progress_percent = (completed_matches / total_matches * 100) if total_matches > 0 else 0
        progress_percent = min(progress_percent, 100.0)  # Never exceed 100%
        
        return {
            'season_number': season.season_number,
            'season_id': season.id,
            'completed_matchdays': min(completed_matchdays, self.MATCHDAYS_PER_SEASON),
            'total_matchdays': self.MATCHDAYS_PER_SEASON,
            'completed_matches': min(completed_matches, total_matches),
            'total_matches': total_matches,
            'progress_percent': progress_percent,
            'is_complete': completed_matches >= total_matches,
            'start_date': season.start_date,
            'days_elapsed': (datetime.utcnow() - season.start_date).days if season.start_date else 0
        }
    
    def get_standings(self, season_id: int, db: Session) -> List[Dict]:
        """Calculate league standings for a season - max 20 teams."""
        matches = db.query(Match).filter(
            Match.season_id == season_id,
            Match.is_completed == True
        ).all()
        
        # Get unique team names from matches only (not from teams table)
        team_names = set()
        for match in matches:
            if match.home_team_name:
                team_names.add(match.home_team_name)
            if match.away_team_name:
                team_names.add(match.away_team_name)
        
        # Initialize standings from actual matches
        standings = {name: {
            'team_id': None,
            'team_name': name,
            'played': 0,
            'won': 0,
            'drawn': 0,
            'lost': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0,
            'points': 0
        } for name in team_names}
        
        # Get team IDs
        teams = db.query(Team).all()
        team_map = {t.name: t.id for t in teams}
        for name in standings:
            if name in team_map:
                standings[name]['team_id'] = team_map[name]
        
        # Calculate from matches
        for match in matches:
            home = match.home_team_name
            away = match.away_team_name
            
            if home in standings:
                standings[home]['played'] += 1
                standings[home]['goals_for'] += match.score_home or 0
                standings[home]['goals_against'] += match.score_away or 0
                
                if match.result == 'V':
                    standings[home]['won'] += 1
                    standings[home]['points'] += 3
                elif match.result == 'N':
                    standings[home]['drawn'] += 1
                    standings[home]['points'] += 1
                else:
                    standings[home]['lost'] += 1
            
            if away in standings:
                standings[away]['played'] += 1
                standings[away]['goals_for'] += match.score_away or 0
                standings[away]['goals_against'] += match.score_home or 0
                
                if match.result == 'D':
                    standings[away]['won'] += 1
                    standings[away]['points'] += 3
                elif match.result == 'N':
                    standings[away]['drawn'] += 1
                    standings[away]['points'] += 1
                else:
                    standings[away]['lost'] += 1
        
        # Calculate goal difference
        for team in standings:
            standings[team]['goal_difference'] = (
                standings[team]['goals_for'] - standings[team]['goals_against']
            )
        
        # Sort by points, then GD, then GF
        sorted_standings = sorted(
            standings.values(),
            key=lambda x: (x['points'], x['goal_difference'], x['goals_for']),
            reverse=True
        )
        
        # Limit to 20 teams max
        sorted_standings = sorted_standings[:20]
        
        # Add position
        for i, team in enumerate(sorted_standings):
            team['position'] = i + 1
        
        return sorted_standings
    
    def get_historical_seasons(self, db: Session) -> List[Dict]:
        """Get all historical seasons summary."""
        seasons = db.query(Season).order_by(Season.season_number).all()
        
        return [{
            'season_id': s.id,
            'season_number': s.season_number,
            'is_active': s.is_active,
            'is_completed': s.is_completed,
            'start_date': s.start_date,
            'end_date': s.end_date,
            'total_matches': s.total_matches,
            'total_home_wins': s.total_home_wins,
            'total_draws': s.total_draws,
            'total_away_wins': s.total_away_wins,
            'total_goals': s.total_goals,
            'avg_goals_per_match': s.avg_goals_per_match,
            'total_bets': s.total_bets,
            'winning_bets': s.winning_bets,
            'total_profit': s.total_profit,
            'roi': s.roi
        } for s in seasons]
    
    def reset_teams_for_new_season(self, db: Session):
        """Reset team statistics for new season (keep ratings)."""
        teams = db.query(Team).all()
        
        for team in teams:
            # Keep ratings (ELO, Bayesian, strength)
            # Reset season-specific stats
            team.matches_played = 0
            team.wins = 0
            team.draws = 0
            team.losses = 0
            team.goals_scored = 0
            team.goals_conceded = 0
            team.home_matches = 0
            team.home_wins = 0
            team.home_draws = 0
            team.home_losses = 0
            team.home_goals_scored = 0
            team.home_goals_conceded = 0
            team.away_matches = 0
            team.away_wins = 0
            team.away_draws = 0
            team.away_losses = 0
            team.away_goals_scored = 0
            team.away_goals_conceded = 0
            team.current_form = ""
            team.winning_streak = 0
            team.losing_streak = 0
            team.draw_streak = 0
            team.unbeaten_streak = 0
        
        db.commit()
        logger.info("Reset team statistics for new season")
