"""Dashboard API routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from typing import Dict, List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models import Match, Prediction, Bet, Season, Team
from app.services.season_manager import SeasonManager
from app.services.elite_selector import get_elite_selector


router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview(db: Session = Depends(get_db)):
    """Get main dashboard overview."""
    season_manager = SeasonManager()
    
    # Current season info
    season = season_manager.get_active_season(db)
    progress = season_manager.get_season_progress(db)
    
    # Recent results (last completed matchday, up to 10 matches)
    last_completed = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_completed == True
    ).order_by(Match.matchday.desc()).first()
    last_completed_matchday = last_completed.matchday if last_completed else None

    recent_matches_query = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_completed == True
    )
    if last_completed_matchday is not None:
        recent_matches_query = recent_matches_query.filter(Match.matchday == last_completed_matchday)

    recent_matches = recent_matches_query.order_by(Match.line_position.asc()).limit(10).all()
    
    # Upcoming matches (limit to 10)
    upcoming_matches = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True,
        Match.has_odds == True
    ).order_by(Match.matchday.asc(), Match.line_position.asc()).limit(10).all()
    
    # MIN_CONFIDENCE filter: only predictions with confidence >= 50%
    # MIN_ODDS filter: only predictions with odds >= 1.80
    MIN_CONFIDENCE = 0.5
    MIN_ODDS = 1.80
    
    # Ensure all upcoming matches have predictions (trigger background if not)
    # Check for upcoming matches with odds but no prediction in CURRENT season
    has_unpredicted_matches = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True,
        Match.has_odds == True,
        ~Match.id.in_(db.query(Prediction.match_id).filter(Prediction.season_id == season.id))
    ).count() > 0
    
    if has_unpredicted_matches:
        import asyncio
        from app.api.scrape import generate_predictions_for_matches
        # Trigger background generation of predictions (non-blocking)
        asyncio.create_task(generate_predictions_for_matches())
        print(f"Dashboard triggered background generation for Season {season.season_number}")
    
    # Elite selector status — 5 predictions per season system
    elite_selector = get_elite_selector()
    elite_selector.ensure_season(season.id)
    
    # Evaluate upcoming predictions if not already done this matchday
    upcoming_preds_with_matches = db.query(Prediction, Match).join(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True,
        Match.has_odds == True
    ).all()
    
    if upcoming_preds_with_matches and elite_selector.get_slots_remaining() > 0:
        # Run elite evaluation to update candidates_rejected counter
        elite_selector.evaluate_matchday(upcoming_preds_with_matches, season.id)
    
    elite_status = elite_selector.get_status()
    
    # Show ALL predictions for upcoming matches (for information)
    all_upcoming_predictions = db.query(Prediction).options(
        joinedload(Prediction.match)
    ).join(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True
    ).order_by(Match.matchday.asc(), Match.line_position.asc()).limit(10).all()
    
    # Elite predictions only (selected for betting)
    predicted_upcoming = db.query(Prediction).options(
        joinedload(Prediction.match)
    ).join(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True,
        Prediction.is_selected_for_bet == True
    ).order_by(Prediction.selection_rank.asc(), Match.matchday.asc(), Match.line_position.asc()).limit(5).all()
    
    # Prediction stats (confidence >= 50%, odds >= 1.80, selected for betting only)
    total_predictions = db.query(Prediction).join(Match).filter(
        Prediction.season_id == season.id,
        Prediction.actual_result != None,
        Prediction.is_selected_for_bet == True,  # Only use selected predictions for consistency
        Prediction.confidence >= MIN_CONFIDENCE,
        # Odds filter based on predicted result
        ((Prediction.predicted_result == 'V') & (Match.odd_home >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'N') & (Match.odd_draw >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'D') & (Match.odd_away >= MIN_ODDS))
    ).count()
    
    correct_predictions = db.query(Prediction).join(Match).filter(
        Prediction.season_id == season.id,
        Prediction.actual_result != None,
        Prediction.is_correct == True,
        Prediction.is_selected_for_bet == True,  # Only use selected predictions for consistency
        Prediction.confidence >= MIN_CONFIDENCE,
        # Odds filter based on predicted result
        ((Prediction.predicted_result == 'V') & (Match.odd_home >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'N') & (Match.odd_draw >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'D') & (Match.odd_away >= MIN_ODDS))
    ).count()
    
    # Betting stats
    total_bets = db.query(Bet).filter(
        Bet.season_id == season.id,
        Bet.is_settled == True
    ).count()
    winning_bets = db.query(Bet).filter(
        Bet.season_id == season.id,
        Bet.is_settled == True,
        Bet.status == 'won'
    ).count()
    
    total_profit = sum(b.profit_loss or 0 for b in db.query(Bet).filter(
        Bet.season_id == season.id,
        Bet.is_settled == True
    ).all())
    
    # Bankroll - calculate from DB bets
    FIXED_STAKE_ARIARY = 1000
    settled_bets = db.query(Bet).filter(
        Bet.season_id == season.id,
        Bet.is_settled == True
    ).all()
    
    # Calculate total profit from settled bets
    total_profit_from_bets = sum(b.profit_loss or 0 for b in settled_bets)
    total_bets_count = len(settled_bets)
    
    # Starting bankroll (initial)
    initial_bankroll = 1000.0  # Default starting bankroll
    current_bankroll = initial_bankroll + total_profit_from_bets
    
    # Calculate ROI based on total stake
    total_stake = total_bets_count * FIXED_STAKE_ARIARY
    roi = total_profit_from_bets / total_stake if total_stake > 0 else 0
    
    bankroll_stats = {
        'initial_bankroll': initial_bankroll,
        'current_bankroll': current_bankroll,
        'profit': total_profit_from_bets,
        'roi': roi,
        'total_bets': total_bets_count,
        'total_settled': total_bets_count
    }
    
    return {
        "season": {
            "number": season.season_number,
            "progress": progress
        },
        "last_completed_matchday": last_completed_matchday,
        "predictions": {
            "total": total_predictions,
            "correct": correct_predictions,
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0
        },
        "betting": {
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "win_rate": winning_bets / total_bets if total_bets > 0 else 0,
            "total_profit": total_profit
        },
        "bankroll": bankroll_stats,
        "elite_status": {
            "max_predictions": elite_status['max_predictions'],
            "predictions_used": elite_status['predictions_used'],
            "slots_remaining": elite_status['slots_remaining'],
            "elite_predictions": elite_status['elite_predictions'],
            "candidates_rejected": elite_status['candidates_rejected'],
            "correct_count": elite_status['correct_count'],
            "verified_count": elite_status['verified_count'],
            "accuracy": elite_status['accuracy'],
            "total_profit": elite_status['total_profit'],
        },
        "recent_results": [{
            "id": m.id,
            "home_team": m.home_team_name,
            "away_team": m.away_team_name,
            "score": f"{m.score_home}-{m.score_away}",
            "result": m.result
        } for m in recent_matches],
        "upcoming_matches": [{
            "id": m.id,
            "matchday": m.matchday,
            "home_team": m.home_team_name,
            "away_team": m.away_team_name,
            "odds": {
                "home": m.odd_home,
                "draw": m.odd_draw,
                "away": m.odd_away
            }
        } for m in upcoming_matches],
        "all_predictions": [{
            "id": pred.match.id,
            "matchday": pred.match.matchday,
            "home_team": pred.match.home_team_name,
            "away_team": pred.match.away_team_name,
            "predicted_result": pred.predicted_result,
            "predicted_result_name": pred.predicted_result_name,
            "confidence": pred.confidence,
            "model_agreement": pred.model_agreement,
            "is_elite": pred.is_selected_for_bet,
            "selection_reason": pred.selection_reason,
            "odds": {
                "home": pred.match.odd_home,
                "draw": pred.match.odd_draw,
                "away": pred.match.odd_away
            }
        } for pred in all_upcoming_predictions],
        "predicted_matches": [{
            "id": pred.match.id,
            "matchday": pred.match.matchday,
            "home_team": pred.match.home_team_name,
            "away_team": pred.match.away_team_name,
            "predicted_result": pred.predicted_result,
            "predicted_result_name": pred.predicted_result_name,
            "confidence": pred.confidence,
            "model_agreement": pred.model_agreement,
            "selection_rank": pred.selection_rank,
            "selection_reason": pred.selection_reason,
            "odds": {
                "home": pred.match.odd_home,
                "draw": pred.match.odd_draw,
                "away": pred.match.odd_away
            }
        } for pred in predicted_upcoming]
    }


@router.get("/standings")
async def get_league_standings(db: Session = Depends(get_db)):
    """Get current league standings."""
    season_manager = SeasonManager()
    season = season_manager.get_active_season(db)
    
    standings = season_manager.get_standings(season.id, db)
    
    return {"standings": standings}


@router.get("/statistics")
async def get_league_statistics(db: Session = Depends(get_db)):
    """Get league statistics."""
    season_manager = SeasonManager()
    season = season_manager.get_active_season(db)
    
    matches = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_completed == True
    ).all()
    
    if not matches:
        return {"message": "No matches completed yet"}
    
    total_matches = len(matches)
    total_goals = sum((m.score_home or 0) + (m.score_away or 0) for m in matches)
    
    home_wins = sum(1 for m in matches if m.result == 'V')
    draws = sum(1 for m in matches if m.result == 'N')
    away_wins = sum(1 for m in matches if m.result == 'D')
    
    both_teams_scored = sum(1 for m in matches if m.both_teams_scored)
    over_2_5 = sum(1 for m in matches if m.over_2_5)
    
    # Goals distribution
    goals_per_match = [m.total_goals for m in matches]
    
    return {
        "total_matches": total_matches,
        "total_goals": total_goals,
        "avg_goals_per_match": total_goals / total_matches,
        "home_win_rate": home_wins / total_matches,
        "draw_rate": draws / total_matches,
        "away_win_rate": away_wins / total_matches,
        "both_teams_scored_rate": both_teams_scored / total_matches,
        "over_2_5_rate": over_2_5 / total_matches,
        "goals_distribution": {
            "0-1": sum(1 for g in goals_per_match if g <= 1),
            "2-3": sum(1 for g in goals_per_match if 2 <= g <= 3),
            "4-5": sum(1 for g in goals_per_match if 4 <= g <= 5),
            "6+": sum(1 for g in goals_per_match if g >= 6)
        }
    }


@router.get("/line-position-analysis")
async def get_line_position_analysis(db: Session = Depends(get_db)):
    """Analyze results by line position."""
    matches = db.query(Match).filter(Match.is_completed == True).all()
    
    line_stats = {}
    
    for pos in range(1, 11):
        line_matches = [m for m in matches if m.line_position == pos]
        
        if line_matches:
            total = len(line_matches)
            home_wins = sum(1 for m in line_matches if m.result == 'V')
            draws = sum(1 for m in line_matches if m.result == 'N')
            away_wins = sum(1 for m in line_matches if m.result == 'D')
            avg_goals = sum(m.total_goals or 0 for m in line_matches) / total
            
            line_stats[pos] = {
                "total_matches": total,
                "home_win_rate": home_wins / total,
                "draw_rate": draws / total,
                "away_win_rate": away_wins / total,
                "avg_goals": avg_goals
            }
    
    return {"line_position_stats": line_stats}


@router.get("/team-form/{team_name}")
async def get_team_form(team_name: str, db: Session = Depends(get_db)):
    """Get team form analysis."""
    team = db.query(Team).filter(Team.name == team_name).first()
    
    if not team:
        return {"error": "Team not found"}
    
    # Get recent matches
    recent_matches = db.query(Match).filter(
        (Match.home_team_name == team_name) | (Match.away_team_name == team_name),
        Match.is_completed == True
    ).order_by(Match.matchday.desc()).limit(10).all()
    
    matches_data = []
    for m in recent_matches:
        is_home = m.home_team_name == team_name
        team_goals = m.score_home if is_home else m.score_away
        opp_goals = m.score_away if is_home else m.score_home
        
        # Result from team perspective
        if is_home:
            result = m.result
        else:
            inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
            result = inverted.get(m.result, 'N')
        
        matches_data.append({
            "matchday": m.matchday,
            "opponent": m.away_team_name if is_home else m.home_team_name,
            "home_away": "home" if is_home else "away",
            "score": f"{team_goals}-{opp_goals}",
            "result": result
        })
    
    return {
        "team": team_name,
        "elo_rating": team.elo_rating,
        "attack_strength": team.attack_strength,
        "defense_strength": team.defense_strength,
        "current_form": team.current_form,
        "winning_streak": team.winning_streak,
        "losing_streak": team.losing_streak,
        "matches_played": team.matches_played,
        "wins": team.wins,
        "draws": team.draws,
        "losses": team.losses,
        "goals_scored": team.goals_scored,
        "goals_conceded": team.goals_conceded,
        "recent_matches": matches_data
    }


@router.get("/learning-status")
async def get_learning_status(db: Session = Depends(get_db)):
    """Get continuous learning status."""
    from app.services.continuous_learning import ContinuousLearningEngine
    
    engine = ContinuousLearningEngine()
    status = engine.get_learning_status(db)
    
    return status


@router.get("/historical-seasons")
async def get_historical_seasons(db: Session = Depends(get_db)):
    """Get historical seasons data."""
    season_manager = SeasonManager()
    seasons = season_manager.get_historical_seasons(db)
    
    return {"seasons": seasons}


@router.get("/elite")
async def get_elite_status(db: Session = Depends(get_db)):
    """Get Elite 5-predictions-per-season system status."""
    season_manager = SeasonManager()
    season = season_manager.get_active_season(db)
    
    elite_selector = get_elite_selector()
    elite_selector.ensure_season(season.id)
    status = elite_selector.get_status()
    
    return {
        "season_number": season.season_number,
        "system": "ELITE_5",
        "description": "5 prédictions ultra-sélectives par saison (cote > 2.0, consensus élevé)",
        **status
    }


@router.post("/elite/reset")
async def reset_elite_for_season(db: Session = Depends(get_db)):
    """Reset elite selector for current season (admin only)."""
    season_manager = SeasonManager()
    season = season_manager.get_active_season(db)
    
    elite_selector = get_elite_selector()
    elite_selector.reset_for_season(season.id)
    
    return {
        "message": f"Elite selector reset for season {season.season_number}",
        "status": elite_selector.get_status()
    }
