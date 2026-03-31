"""Predictions API routes."""

from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models import Match, Prediction, Season, Team
from app.services.conservative_predictor import ConservativePredictor, QualityScorer


router = APIRouter()


class MatchOut(BaseModel):
    id: int
    season_id: int
    matchday: int
    line_position: int
    home_team_name: str
    away_team_name: str
    odd_home: Optional[float] = None
    odd_draw: Optional[float] = None
    odd_away: Optional[float] = None
    has_odds: bool
    is_upcoming: bool
    is_completed: bool
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    result: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    id: int
    match_id: int
    season_id: int
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_result: str
    predicted_result_name: str
    confidence: float
    actual_result: Optional[str] = None
    is_correct: Optional[bool] = None
    verified_at: Optional[Any] = None
    # Selection fields
    is_selected_for_bet: bool = False
    selection_rank: Optional[int] = None
    selection_reason: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionWithMatchOut(BaseModel):
    match: MatchOut
    prediction: Optional[PredictionOut] = None


class PredictionAccuracyResponse(BaseModel):
    total_verified: int
    correct: int
    accuracy: float


def _get_active_season_id(db: Session) -> Optional[int]:
    season = db.query(Season).filter(Season.is_active == True).first()
    return season.id if season else None


@router.get("/upcoming", response_model=List[PredictionWithMatchOut])
async def get_upcoming_predictions(
    limit: int = 10,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Return upcoming matches that have predictions generated.
    
    Auto-selects ALL predictions if not already selected.
    """
    if season_id is None:
        season_id = _get_active_season_id(db)

    # Auto-select ALL predictions for upcoming matches
    unselected = db.query(Prediction).join(Match).filter(
        Match.is_upcoming == True,
        Prediction.is_selected_for_bet == False
    )
    if season_id is not None:
        unselected = unselected.filter(Prediction.season_id == season_id)
    
    unselected_preds = unselected.all()
    if unselected_preds:
        rank_start = db.query(Prediction).filter(
            Prediction.is_selected_for_bet == True
        ).count() + 1
        
        for idx, pred in enumerate(unselected_preds):
            pred.is_selected_for_bet = True
            pred.selection_rank = rank_start + idx
            pred.selection_reason = "AUTO_SELECTED"
        
        db.commit()
        print(f"Auto-selected {len(unselected_preds)} predictions")

    # Query predictions for upcoming matches that are SELECTED for betting
    # Remove strict filtering so all predictions are shown, but keep them marked correctly
    q = db.query(Prediction).join(Match).filter(
        Match.is_upcoming == True,
        Prediction.is_selected_for_bet == True
    )
    if season_id is not None:
        q = q.filter(Prediction.season_id == season_id)

    predictions = (
        q.order_by(Prediction.selection_rank.asc(), Match.matchday.asc(), Match.line_position.asc())
        .limit(limit)
        .all()
    )

    result: List[PredictionWithMatchOut] = []
    for pred in predictions:
        result.append({"match": pred.match, "prediction": pred})

    return result


@router.get("/", response_model=List[PredictionWithMatchOut])
async def list_predictions(
    season_id: Optional[int] = None,
    is_verified: Optional[bool] = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """List predictions (optionally filtered), joined with match info."""
    if season_id is None:
        season_id = _get_active_season_id(db)

    q = db.query(Prediction)
    if season_id is not None:
        q = q.filter(Prediction.season_id == season_id)

    if is_verified is True:
        q = q.filter(Prediction.actual_result != None)  # noqa: E711
    elif is_verified is False:
        q = q.filter(Prediction.actual_result == None)  # noqa: E711

    preds = q.order_by(Prediction.created_at.desc()).limit(limit).all()

    out: List[PredictionWithMatchOut] = []
    for p in preds:
        m = db.query(Match).filter(Match.id == p.match_id).first()
        out.append({"match": m, "prediction": p})

    return out


@router.get("/accuracy/stats", response_model=PredictionAccuracyResponse)
async def get_prediction_accuracy_stats(
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Accuracy over verified predictions."""
    if season_id is None:
        season_id = _get_active_season_id(db)

    # MIN_CONFIDENCE filter: only predictions with confidence >= 50%
    # MIN_ODDS filter: only predictions with odds >= 1.80
    MIN_CONFIDENCE = 0.5
    MIN_ODDS = 1.80
    
    q = db.query(Prediction).join(Match).filter(
        Prediction.actual_result != None,  # noqa: E711
        Prediction.confidence >= MIN_CONFIDENCE,
        # Odds filter based on predicted result
        ((Prediction.predicted_result == 'V') & (Match.odd_home >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'N') & (Match.odd_draw >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'D') & (Match.odd_away >= MIN_ODDS))
    )
    if season_id is not None:
        q = q.filter(Prediction.season_id == season_id)

    total = q.count()
    correct = q.filter(Prediction.is_correct == True).count()  # noqa: E712

    return PredictionAccuracyResponse(
        total_verified=total,
        correct=correct,
        accuracy=(correct / total) if total > 0 else 0.0,
    )


class MatchdayComparisonResult(BaseModel):
    """Result comparison for a single matchday."""
    matchday: int
    predictions: List[Dict]
    total_stake: int
    total_profit: float
    correct_count: int
    total_predictions: int
    accuracy: float


class ComparisonDataResponse(BaseModel):
    """Full comparison data for multiple matchdays."""
    predictions: List[Dict]
    by_matchday: List[MatchdayComparisonResult]
    total_stake: int
    total_profit: float
    correct_count: int
    total_predictions: int
    accuracy: float
    timestamp: str


@router.get("/comparison/{matchday}", response_model=MatchdayComparisonResult)
async def get_matchday_comparison(
    matchday: int,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get prediction results for a specific matchday.
    
    Returns comparison data for verified predictions of the ACTIVE season by default.
    """
    STAKE = 1000  # Fixed stake per prediction
    result_name = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}

    # Default to active season if not specified
    if season_id is None:
        season_id = _get_active_season_id(db)

    # Get HIGH CONFIDENCE verified predictions for this matchday (PRÉDICTIONS ASSURÉES SEULEMENT)
    predictions = db.query(Prediction).join(Match).filter(
        Match.matchday == matchday,
        Prediction.actual_result != None,  # Verified predictions only
        Prediction.confidence >= 0.50     # SEUIL ÉLEVÉ : prédictions assurées uniquement
    )
    # Filter by season (active season by default)
    if season_id is not None:
        predictions = predictions.filter(Match.season_id == season_id)

    predictions = predictions.all()

    if not predictions:
        return MatchdayComparisonResult(
            matchday=matchday,
            predictions=[],
            total_stake=0,
            total_profit=0.0,
            correct_count=0,
            total_predictions=0,
            accuracy=0.0
        )

    total_stake = 0
    total_profit = 0.0
    correct_count = 0
    pred_list = []

    for pred in predictions:
        match = pred.match
        if not match or not match.result:
            continue

        total_stake += STAKE

        # Calculate profit
        if pred.is_correct:
            correct_count += 1
            if pred.predicted_result == 'V':
                odds = match.odd_home or 2.0
            elif pred.predicted_result == 'N':
                odds = match.odd_draw or 3.5
            else:
                odds = match.odd_away or 3.0
            profit = (STAKE * odds) - STAKE
            total_profit += profit
        else:
            profit = -STAKE
            total_profit -= STAKE

        pred_list.append({
            'match': f"{match.home_team_name} vs {match.away_team_name}",
            'matchday': match.matchday,
            'predicted': result_name.get(pred.predicted_result, pred.predicted_result),
            'actual': result_name.get(match.result, match.result),
            'actual_result': match.result,
            'is_correct': pred.is_correct,
            'stake': STAKE,
            'profit_loss': profit if pred.is_correct else -STAKE,
            'score': f"{match.score_home}-{match.score_away}",
        })

    return MatchdayComparisonResult(
        matchday=matchday,
        predictions=pred_list,
        total_stake=total_stake,
        total_profit=total_profit,
        correct_count=correct_count,
        total_predictions=len(pred_list),
        accuracy=correct_count / len(pred_list) if pred_list else 0.0
    )


@router.get("/comparison", response_model=ComparisonDataResponse)
async def get_all_comparisons(
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get all verified prediction results grouped by matchday.
    
    Returns comparison data for verified predictions of the ACTIVE season by default.
    """
    from datetime import datetime

    STAKE = 1000  # Fixed stake per prediction
    result_name = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}

    # Default to active season if not specified
    if season_id is None:
        season_id = _get_active_season_id(db)

    # Get HIGH CONFIDENCE verified predictions for active season (PRÉDICTIONS ASSURÉES SEULEMENT)
    predictions = db.query(Prediction).join(Match).filter(
        Prediction.actual_result != None,  # Verified predictions only
        Prediction.confidence >= 0.50     # SEUIL ÉLEVÉ : prédictions assurées uniquement
    )
    # Filter by season (active season by default)
    if season_id is not None:
        predictions = predictions.filter(Match.season_id == season_id)
    
    predictions = predictions.order_by(Match.matchday.asc()).all()

    if not predictions:
        return ComparisonDataResponse(
            predictions=[],
            by_matchday=[],
            total_stake=0,
            total_profit=0.0,
            correct_count=0,
            total_predictions=0,
            accuracy=0.0,
            timestamp=datetime.utcnow().isoformat()
        )

    # Group by matchday
    by_matchday = {}
    all_predictions = []
    total_stake = 0
    total_profit = 0.0
    total_correct = 0

    for pred in predictions:
        match = pred.match
        if not match or not match.result:
            continue

        total_stake += STAKE

        # Calculate profit
        if pred.is_correct:
            total_correct += 1
            if pred.predicted_result == 'V':
                odds = match.odd_home or 2.0
            elif pred.predicted_result == 'N':
                odds = match.odd_draw or 3.5
            else:
                odds = match.odd_away or 3.0
            profit = (STAKE * odds) - STAKE
            total_profit += profit
        else:
            profit = -STAKE
            total_profit -= STAKE

        pred_data = {
            'match': f"{match.home_team_name} vs {match.away_team_name}",
            'matchday': match.matchday,
            'predicted': result_name.get(pred.predicted_result, pred.predicted_result),
            'actual': result_name.get(match.result, match.result),
            'actual_result': match.result,
            'is_correct': pred.is_correct,
            'stake': STAKE,
            'profit_loss': profit if pred.is_correct else -STAKE,
            'score': f"{match.score_home}-{match.score_away}",
        }
        all_predictions.append(pred_data)

        # Group by matchday
        md = match.matchday
        if md not in by_matchday:
            by_matchday[md] = {
                'matchday': md,
                'predictions': [],
                'stake': 0,
                'profit': 0.0,
                'correct': 0
            }
        
        by_matchday[md]['predictions'].append(pred_data)
        by_matchday[md]['stake'] += STAKE
        by_matchday[md]['profit'] += profit if pred.is_correct else -STAKE
        if pred.is_correct:
            by_matchday[md]['correct'] += 1

    # Convert to list and calculate accuracy per matchday
    matchday_list = []
    for md_data in by_matchday.values():
        md_data['total_stake'] = md_data.pop('stake')
        md_data['total_profit'] = md_data.pop('profit')
        md_data['correct_count'] = md_data.pop('correct')
        md_data['total_predictions'] = len(md_data['predictions'])
        md_data['accuracy'] = md_data['correct_count'] / md_data['total_predictions'] if md_data['total_predictions'] > 0 else 0.0
        matchday_list.append(MatchdayComparisonResult(**md_data))

    # Sort by matchday
    matchday_list.sort(key=lambda x: x.matchday)

    return ComparisonDataResponse(
        predictions=all_predictions,
        by_matchday=matchday_list,
        total_stake=total_stake,
        total_profit=total_profit,
        correct_count=total_correct,
        total_predictions=len(all_predictions),
        accuracy=total_correct / len(all_predictions) if all_predictions else 0.0,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/verified-matchdays")
async def get_verified_matchdays(
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get list of matchdays that have verified predictions (results available).
    
    Returns matchdays for the ACTIVE season by default.
    """
    # Default to active season if not specified
    if season_id is None:
        season_id = _get_active_season_id(db)
    
    # MIN_CONFIDENCE filter: only predictions with confidence >= 50%
    # MIN_ODDS filter: only predictions with odds >= 1.80
    MIN_CONFIDENCE = 0.5
    MIN_ODDS = 1.80
    
    # Get distinct matchdays with verified predictions
    q = db.query(Match.matchday).join(Prediction).filter(
        Prediction.actual_result != None,  # Verified predictions only
        Prediction.confidence >= MIN_CONFIDENCE,
        # Odds filter based on predicted result
        ((Prediction.predicted_result == 'V') & (Match.odd_home >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'N') & (Match.odd_draw >= MIN_ODDS)) |
        ((Prediction.predicted_result == 'D') & (Match.odd_away >= MIN_ODDS))
    )
    # Filter by season (active season by default)
    if season_id is not None:
        q = q.filter(Match.season_id == season_id)
    
    matchdays = q.distinct().order_by(Match.matchday.desc()).limit(20).all()
    
    return {"matchdays": [m[0] for m in matchdays]}


@router.get("/season-history")
async def get_season_betting_history(db: Session = Depends(get_db)):
    """Get betting statistics for all seasons based on actual BETS placed.
    
    Uses the Bet table for accurate tracking of stakes, odds, and results.
    Falls back to predictions only if no bets exist.
    """
    from app.models import Season, Bet
    
    # Get all seasons
    seasons = db.query(Season).order_by(Season.season_number.desc()).all()
    
    if not seasons:
        return {"seasons": [], "message": "Aucune saison trouvée en base de données"}
    
    history = []
    for season in seasons:
        # First try to get stats from actual bets
        bets = db.query(Bet).filter(Bet.season_id == season.id).all()
        
        if bets:
            total_bets = len(bets)
            total_stake = sum(b.stake or 0 for b in bets)
            winning_bets = sum(1 for b in bets if b.status == 'won')
            losing_bets = sum(1 for b in bets if b.status == 'lost')
            settled_bets = winning_bets + losing_bets
            
            # Calculate profit from bets
            total_profit = sum(b.profit_loss or 0 for b in bets if b.is_settled)
            
            win_rate = winning_bets / settled_bets if settled_bets > 0 else 0
        else:
            # Fallback: calculate from predictions with is_selected_for_bet
            # Also include predictions that have actual_result (meaning match completed)
            verified_preds = db.query(Prediction).join(Match).filter(
                Prediction.season_id == season.id,
                Prediction.actual_result != None,
                Match.is_completed == True
            ).all()
            
            if verified_preds:
                STAKE = 1000  # Fixed stake per prediction
                total_stake = 0
                total_profit = 0.0
                winning_bets = 0
                
                for pred in verified_preds:
                    match = pred.match
                    if not match or not match.result:
                        continue
                        
                    total_stake += STAKE
                    
                    if pred.is_correct:
                        winning_bets += 1
                        # Get odds for the predicted outcome
                        if pred.predicted_result == 'V':
                            odds = match.odd_home or 2.0
                        elif pred.predicted_result == 'N':
                            odds = match.odd_draw or 3.5
                        else:
                            odds = match.odd_away or 3.0
                        profit = (STAKE * odds) - STAKE
                        total_profit += profit
                    else:
                        total_profit -= STAKE
                
                total_bets = len(verified_preds)
                win_rate = winning_bets / total_bets if total_bets > 0 else 0
            else:
                total_bets = 0
                total_stake = 0
                total_profit = 0
                win_rate = 0
                winning_bets = 0
        
        # Get match count
        match_count = db.query(Match).filter(Match.season_id == season.id).count()
        completed_matches = db.query(Match).filter(
            Match.season_id == season.id,
            Match.is_completed == True
        ).count()
        
        # Determine status
        if season.is_active and not season.is_completed:
            status = "En cours"
        elif season.is_completed:
            status = "Terminée"
        else:
            status = "Inactive"
        
        history.append({
            'season_number': season.season_number,
            'season_id': season.id,
            'is_active': season.is_active,
            'is_completed': season.is_completed,
            'status': status,
            'total_matches': match_count,
            'completed_matches': completed_matches,
            'total_bets': total_bets,
            'total_stake': round(total_stake, 2),
            'total_profit': round(total_profit, 2),
            'win_rate': round(win_rate, 4),
            'winning_bets': winning_bets,
            'roi': round((total_profit / total_stake) if total_stake > 0 else 0, 4),
            'started_at': season.start_date.isoformat() if season.start_date else None,
            'ended_at': season.end_date.isoformat() if season.end_date else None
        })
    
    return {
        "seasons": history,
        "total_seasons": len(seasons),
        "message": f"{len(seasons)} saisons trouvées"
    }


class ConservativePredictionOut(BaseModel):
    """Conservative prediction with quality scoring."""
    match_id: int
    home_team: str
    away_team: str
    matchday: int
    predicted_result: str
    predicted_result_name: str
    confidence: float
    model_agreement: float
    quality_score: float
    quality_level: str
    is_quality_bet: bool
    recommended_outcome: Optional[str] = None
    recommended_odds: Optional[float] = None
    value_edge: Optional[float] = None
    reason: str


class ConservativeDailySummary(BaseModel):
    """Summary of conservative predictions for a matchday."""
    matchday: int
    total_matches: int
    quality_bets_count: int
    quality_bets: List[ConservativePredictionOut]
    expected_profit: float
    win_probability: float
    strategy: str


@router.get("/conservative/{matchday}", response_model=ConservativeDailySummary)
async def get_conservative_predictions(
    matchday: int,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Get CONSERVATIVE predictions for a specific matchday.
    
    Only returns HIGH QUALITY predictions that meet strict criteria:
    - Confidence >= 75%
    - Model agreement >= 80%
    - Value edge >= 10%
    
    Goal: Guarantee daily profits with quality over quantity.
    """
    from app.services.daily_profit_manager import DailyProfitManager
    from app.core.config import settings
    
    if season_id is None:
        season_id = _get_active_season_id(db)
    
    # Get all matches for this matchday
    matches = db.query(Match).filter(
        Match.matchday == matchday
    )
    if season_id is not None:
        matches = matches.filter(Match.season_id == season_id)
    matches = matches.all()
    
    if not matches:
        return ConservativeDailySummary(
            matchday=matchday,
            total_matches=0,
            quality_bets_count=0,
            quality_bets=[],
            expected_profit=0,
            win_probability=0,
            strategy="Aucun match trouvé"
        )
    
    # Initialize conservative predictor
    predictor = ConservativePredictor()
    profit_manager = DailyProfitManager(initial_bankroll=settings.INITIAL_BANKROLL)
    
    quality_bets = []
    
    for match in matches:
        # Get prediction if exists
        prediction = db.query(Prediction).filter(
            Prediction.match_id == match.id
        ).first()
        
        if not prediction:
            continue
        
        # Get teams
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        # Evaluate match quality
        evaluation = predictor.evaluate_match_quality(
            match=match,
            prediction=prediction,
            home_team=home_team,
            away_team=away_team
        )
        
        quality_level = QualityScorer.categorize_quality(evaluation['quality_score'])
        
        pred_out = ConservativePredictionOut(
            match_id=match.id,
            home_team=match.home_team_name,
            away_team=match.away_team_name,
            matchday=match.matchday,
            predicted_result=prediction.predicted_result,
            predicted_result_name=prediction.predicted_result_name,
            confidence=prediction.confidence,
            model_agreement=prediction.model_agreement,
            quality_score=evaluation['quality_score'],
            quality_level=quality_level,
            is_quality_bet=evaluation['is_quality_bet'],
            recommended_outcome=evaluation.get('recommended_outcome'),
            recommended_odds=evaluation.get('recommended_odds'),
            value_edge=evaluation.get('value_edge'),
            reason=evaluation['reason']
        )
        
        quality_bets.append(pred_out)
    
    # Filter to quality bets only
    quality_only = [p for p in quality_bets if p.is_quality_bet]
    
    # Calculate expected profit
    expected_profit = 0
    win_prob = 0
    if quality_only:
        decisions = predictor.select_daily_bets(
            evaluations=[{
                'is_quality_bet': p.is_quality_bet,
                'quality_score': p.quality_score,
                'match_id': p.match_id,
                'home_team': p.home_team,
                'away_team': p.away_team,
                'recommended_outcome': p.recommended_outcome,
                'recommended_odds': p.recommended_odds,
                'value_edge': p.value_edge,
                'confidence': p.confidence,
                'model_agreement': p.model_agreement,
                'value_analysis': {p.recommended_outcome: {'prob': p.confidence, 'odds': p.recommended_odds}} if p.recommended_outcome else {}
            } for p in quality_only],
            bankroll=settings.INITIAL_BANKROLL,
            matchday=matchday
        )
        
        profit_info = predictor.calculate_expected_daily_profit(decisions)
        expected_profit = profit_info['expected_profit']
        win_prob = profit_info['win_probability']
    
    strategy = profit_manager.get_strategy_recommendation()
    
    return ConservativeDailySummary(
        matchday=matchday,
        total_matches=len(matches),
        quality_bets_count=len(quality_only),
        quality_bets=quality_only,
        expected_profit=expected_profit,
        win_probability=win_prob,
        strategy=strategy
    )


@router.get("/conservative/upcoming")
async def get_upcoming_conservative_predictions(
    limit: int = 20,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Get upcoming CONSERVATIVE predictions - QUALITY ONLY.
    
    Returns only predictions that meet strict quality criteria:
    - Confidence >= 75%
    - Model agreement >= 80%
    - Significant value edge
    
    This endpoint prioritizes QUALITY over QUANTITY to guarantee profits.
    """
    from app.core.config import settings
    
    if season_id is None:
        season_id = _get_active_season_id(db)
    
    # Get upcoming matches with predictions
    predictions = db.query(Prediction).join(Match).filter(
        Match.is_upcoming == True,
        Prediction.confidence >= settings.MIN_CONFIDENCE,
        Prediction.model_agreement >= getattr(settings, 'MIN_MODEL_AGREEMENT', 0.80)
    )
    if season_id is not None:
        predictions = predictions.filter(Prediction.season_id == season_id)
    
    predictions = predictions.order_by(Match.matchday.asc()).limit(limit).all()
    
    predictor = ConservativePredictor()
    results = []
    
    for pred in predictions:
        match = pred.match
        if not match.has_odds:
            continue
            
        home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
        
        evaluation = predictor.evaluate_match_quality(
            match=match,
            prediction=pred,
            home_team=home_team,
            away_team=away_team
        )
        
        if evaluation['is_quality_bet']:
            quality_level = QualityScorer.categorize_quality(evaluation['quality_score'])
            
            results.append({
                'match_id': match.id,
                'matchday': match.matchday,
                'home_team': match.home_team_name,
                'away_team': match.away_team_name,
                'predicted_result': pred.predicted_result,
                'predicted_result_name': pred.predicted_result_name,
                'confidence': pred.confidence,
                'model_agreement': pred.model_agreement,
                'quality_score': evaluation['quality_score'],
                'quality_level': quality_level,
                'recommended_outcome': evaluation.get('recommended_outcome'),
                'recommended_odds': evaluation.get('recommended_odds'),
                'value_edge': evaluation.get('value_edge'),
                'odds_home': match.odd_home,
                'odds_draw': match.odd_draw,
                'odds_away': match.odd_away
            })
    
    return {
        "total_quality_predictions": len(results),
        "strategy": "CONSERVATIVE - Qualité avant quantité",
        "min_confidence": settings.MIN_CONFIDENCE,
        "min_model_agreement": getattr(settings, 'MIN_MODEL_AGREEMENT', 0.80),
        "predictions": results
    }


class DynamicSelectionOut(BaseModel):
    """Dynamic selection output with full details."""
    match_id: int
    home_team: str
    away_team: str
    matchday: int
    outcome: str
    outcome_name: str
    odds: float
    adjusted_prob: float
    value: float
    confidence: float
    model_agreement: float
    score: float
    selection_rank: int
    selection_reason: str
    is_strong_draw: bool
    has_sequence_pattern: bool
    stake: float
    potential_return: float
    potential_profit: float


class DynamicSelectionSummary(BaseModel):
    """Summary of dynamic selection for a matchday."""
    matchday: int
    season_bets_placed: int
    season_bets_remaining: int
    can_bet: bool
    risk_status: str
    selections: List[DynamicSelectionOut]
    total_stake: float
    total_potential_profit: float


@router.get("/dynamic-selection/{matchday}", response_model=DynamicSelectionSummary)
async def get_dynamic_selection(
    matchday: int,
    season_id: Optional[int] = None,
    bankroll: float = 10000.0,
    db: Session = Depends(get_db),
):
    """
    Get DYNAMIC selection for a specific matchday.
    
    Implements:
    - Adjusted probability calculation
    - Advanced draw detection
    - Sequence pattern analysis
    - Season limit (max 70 bets)
    - Daily limit (0-5 bets)
    - Priority-based ranking
    - Risk management
    
    Only returns predictions with is_selected_for_bet = True.
    """
    from app.services.dynamic_selection_engine import DynamicSelectionEngine
    
    if season_id is None:
        season_id = _get_active_season_id(db)
    
    engine = DynamicSelectionEngine()
    
    # Run selection
    decisions = engine.select_bets_for_matchday(
        db=db,
        matchday=matchday,
        season_id=season_id,
        bankroll=bankroll
    )
    
    # Check season status
    can_bet, season_count = engine.check_season_limit(db, season_id)
    
    # Check risk
    can_proceed, risk_reason = engine.check_risk_management(db, season_id)
    
    selections = []
    for d in decisions:
        selections.append(DynamicSelectionOut(
            match_id=d['match_id'],
            home_team=d['home_team'],
            away_team=d['away_team'],
            matchday=d['matchday'],
            outcome=d['outcome'],
            outcome_name=d['outcome_name'],
            odds=d['odds'],
            adjusted_prob=d['adjusted_prob'],
            value=d['value'],
            confidence=d['confidence'],
            model_agreement=d['model_agreement'],
            score=d['score'],
            selection_rank=d['selection_rank'],
            selection_reason=d['reason'],
            is_strong_draw=d['is_strong_draw'],
            has_sequence_pattern=d['has_sequence_pattern'],
            stake=d['stake'],
            potential_return=d['potential_return'],
            potential_profit=d['potential_profit']
        ))
    
    total_stake = sum(s.stake for s in selections)
    total_potential = sum(s.potential_profit for s in selections)
    
    return DynamicSelectionSummary(
        matchday=matchday,
        season_bets_placed=season_count,
        season_bets_remaining=engine.MAX_BETS_PER_SEASON - season_count,
        can_bet=can_bet and can_proceed,
        risk_status=risk_reason,
        selections=selections,
        total_stake=total_stake,
        total_potential_profit=total_potential
    )


@router.post("/generate-with-realtime")
async def generate_predictions_with_realtime(
    limit: int = 5,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Generate predictions using real-time engine integration."""
    try:
        from app.core.orchestrator import PredictionOrchestrator
        
        orchestrator = PredictionOrchestrator()
        await orchestrator._generate_predictions()
        
        return {"status": "success", "message": f"Generated predictions with real-time engine integration"}
    except Exception as e:
        logger.error(f"Error generating predictions with realtime: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/selected")
async def get_selected_predictions(
    limit: int = 20,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Get all SELECTED predictions for betting.
    
    CRITICAL: This endpoint ONLY returns predictions where:
    - is_selected_for_bet = True
    
    The frontend should use this endpoint to display predictions.
    """
    if season_id is None:
        season_id = _get_active_season_id(db)
    
    # Query ONLY selected predictions
    q = db.query(Prediction).join(Match).filter(
        Prediction.is_selected_for_bet == True
    )
    
    if season_id is not None:
        q = q.filter(Prediction.season_id == season_id)
    
    predictions = q.order_by(
        Match.matchday.asc(),
        Prediction.selection_rank.asc()
    ).limit(limit).all()
    
    results = []
    for pred in predictions:
        match = pred.match
        results.append({
            'prediction_id': pred.id,
            'match_id': match.id,
            'matchday': match.matchday,
            'home_team': match.home_team_name,
            'away_team': match.away_team_name,
            'predicted_result': pred.predicted_result,
            'predicted_result_name': pred.predicted_result_name,
            'prob_home_win': pred.prob_home_win,
            'prob_draw': pred.prob_draw,
            'prob_away_win': pred.prob_away_win,
            'confidence': pred.confidence,
            'model_agreement': pred.model_agreement,
            'selection_rank': pred.selection_rank,
            'selection_reason': pred.selection_reason,
            'odds_home': match.odd_home,
            'odds_draw': match.odd_draw,
            'odds_away': match.odd_away,
            'is_upcoming': match.is_upcoming,
            'is_completed': match.is_completed,
            'actual_result': pred.actual_result,
            'is_correct': pred.is_correct
        })
    
    return {
        "total_selected": len(results),
        "message": "Only selected predictions are shown",
        "predictions": results
    }


@router.post("/recalculate-selection")
async def recalculate_values_and_selection(db: Session = Depends(get_db)):
    """Recalculate values for all predictions and run dynamic selection.
    
    This endpoint is used to fix predictions that were created without values
    or when the selection engine needs to be re-run.
    """
    from app.services.dynamic_selection_engine import DynamicSelectionEngine
    from loguru import logger
    
    # Get active season
    season = db.query(Season).filter(Season.is_active == True).first()
    if not season:
        return {"success": False, "message": "No active season"}
    
    # Get predictions without values
    predictions = db.query(Prediction).join(Match).filter(
        Match.season_id == season.id,
        Prediction.value_home == None
    ).all()
    
    updated = 0
    for pred in predictions:
        match = pred.match
        if match and match.odd_home and match.odd_home > 0:
            pred.calculate_value(match.odd_home, match.odd_draw, match.odd_away)
            updated += 1
    
    db.commit()
    logger.info(f"Updated {updated} predictions with values")
    
    # Get current matchday
    upcoming = db.query(Match).filter(
        Match.season_id == season.id,
        Match.is_upcoming == True
    ).order_by(Match.matchday.asc()).first()
    
    if not upcoming:
        return {
            "success": True,
            "values_updated": updated,
            "selections": 0,
            "message": "No upcoming matches"
        }
    
    matchday = upcoming.matchday
    
    # Run dynamic selection
    engine = DynamicSelectionEngine()
    decisions = engine.select_bets_for_matchday(
        db=db,
        matchday=matchday,
        season_id=season.id,
        bankroll=10000.0
    )
    
    # Get selected predictions
    selected = db.query(Prediction).filter(
        Prediction.season_id == season.id,
        Prediction.is_selected_for_bet == True
    ).count()
    
    return {
        "success": True,
        "values_updated": updated,
        "matchday": matchday,
        "selections": len(decisions),
        "total_selected": selected,
        "message": f"Updated {updated} values, selected {len(decisions)} new bets"
    }
