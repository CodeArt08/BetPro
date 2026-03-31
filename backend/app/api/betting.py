"""Betting API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from app.core.database import get_db
from app.models import Bet, Match, Prediction, Season


router = APIRouter()


class BetResponse(BaseModel):
    id: int
    match_id: int
    bet_outcome: str
    bet_outcome_name: str
    odds: float
    stake: float
    potential_return: float
    actual_return: Optional[float]
    profit_loss: Optional[float]
    bankroll_before: float
    bankroll_after: Optional[float]
    status: str
    value_edge: float
    confidence: float
    placed_at: datetime
    settled_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class BetWithMatchResponse(BaseModel):
    bet: BetResponse
    match: dict


class BettingSummaryResponse(BaseModel):
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    total_stake: float
    total_return: float
    total_profit: float
    roi: float
    avg_odds: float
    avg_stake: float


@router.get("/", response_model=List[BetResponse])
async def get_bets(
    season_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get bets with optional filters."""
    query = db.query(Bet)
    
    if season_id:
        query = query.filter(Bet.season_id == season_id)
    if status:
        query = query.filter(Bet.status == status)
    
    bets = query.order_by(Bet.placed_at.desc()).limit(limit).all()
    return bets


@router.get("/pending", response_model=List[BetWithMatchResponse])
async def get_pending_bets(db: Session = Depends(get_db)):
    """Get all pending (unsettled) bets."""
    bets = db.query(Bet).filter(Bet.is_settled == False).all()
    
    result = []
    for bet in bets:
        match = bet.match
        result.append({
            "bet": BetResponse.model_validate(bet),
            "match": {
                "id": match.id,
                "matchday": match.matchday,
                "home_team": match.home_team_name,
                "away_team": match.away_team_name,
                "score_home": match.score_home,
                "score_away": match.score_away,
                "result": match.result
            }
        })
    
    return result


@router.get("/match/{match_id}")
async def get_bet_for_match(match_id: int, db: Session = Depends(get_db)):
    """Get bet for a specific match if exists."""
    bet = db.query(Bet).filter(Bet.match_id == match_id).first()
    
    if not bet:
        return {
            "bet_outcome": None,
            "bet_odds": None,
            "stake": None,
            "profit_loss": None,
            "is_win": None
        }
    
    match = bet.match
    return {
        "bet_outcome": bet.bet_outcome_name,
        "bet_odds": bet.odds,
        "stake": bet.stake,
        "profit_loss": bet.profit_loss,
        "is_win": bet.status == 'won' if bet.is_settled else None,
        "match": {
            "id": match.id,
            "matchday": match.matchday,
            "home_team": match.home_team_name,
            "away_team": match.away_team_name,
            "score_home": match.score_home,
            "score_away": match.score_away,
            "result": match.result
        }
    }


@router.get("/summary", response_model=BettingSummaryResponse)
async def get_betting_summary(
    season_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get betting summary statistics."""
    query = db.query(Bet).filter(Bet.is_settled == True)
    
    if season_id:
        query = query.filter(Bet.season_id == season_id)
    
    bets = query.all()
    
    if not bets:
        return BettingSummaryResponse(
            total_bets=0, winning_bets=0, losing_bets=0,
            win_rate=0, total_stake=0, total_return=0,
            total_profit=0, roi=0, avg_odds=0, avg_stake=0
        )
    
    total = len(bets)
    winning = sum(1 for b in bets if b.status == 'won')
    losing = sum(1 for b in bets if b.status == 'lost')
    total_stake = sum(b.stake for b in bets)
    total_return = sum(b.actual_return or 0 for b in bets)
    total_profit = sum(b.profit_loss or 0 for b in bets)
    
    return BettingSummaryResponse(
        total_bets=total,
        winning_bets=winning,
        losing_bets=losing,
        win_rate=winning / total if total > 0 else 0,
        total_stake=total_stake,
        total_return=total_return,
        total_profit=total_profit,
        roi=total_profit / total_stake if total_stake > 0 else 0,
        avg_odds=sum(b.odds for b in bets) / total,
        avg_stake=total_stake / total
    )


@router.get("/summary/ariary")
async def get_betting_summary_ariary(
    season_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get betting summary in Ariary with fixed stake of 1000 Ar per bet."""
    FIXED_STAKE_ARIARY = 1000
    
    query = db.query(Bet).filter(Bet.is_settled == True)
    
    if season_id:
        query = query.filter(Bet.season_id == season_id)
    else:
        # Get active season
        active_season = db.query(Season).filter(Season.is_active == True).first()
        if active_season:
            query = query.filter(Bet.season_id == active_season.id)
    
    bets = query.all()
    
    if not bets:
        return {
            "total_bets": 0,
            "winning_bets": 0,
            "losing_bets": 0,
            "win_rate": 0,
            "total_stake_ariary": 0,
            "total_profit_ariary": 0,
            "roi": 0,
            "fixed_stake_ariary": FIXED_STAKE_ARIARY,
        }
    
    total = len(bets)
    winning = sum(1 for b in bets if b.status == 'won')
    losing = sum(1 for b in bets if b.status == 'lost')
    
    # Calculate with fixed stake
    total_stake_ariary = total * FIXED_STAKE_ARIARY
    
    # Calculate profit based on actual results with fixed stake
    total_profit_ariary = 0
    for b in bets:
        if b.status == 'won':
            # Profit = (stake * odds) - stake
            profit = (FIXED_STAKE_ARIARY * b.odds) - FIXED_STAKE_ARIARY
            total_profit_ariary += profit
        else:
            total_profit_ariary -= FIXED_STAKE_ARIARY
    
    return {
        "total_bets": total,
        "winning_bets": winning,
        "losing_bets": losing,
        "win_rate": winning / total if total > 0 else 0,
        "total_stake_ariary": total_stake_ariary,
        "total_profit_ariary": total_profit_ariary,
        "roi": total_profit_ariary / total_stake_ariary if total_stake_ariary > 0 else 0,
        "fixed_stake_ariary": FIXED_STAKE_ARIARY,
    }


@router.get("/{bet_id}", response_model=BetResponse)
async def get_bet(bet_id: int, db: Session = Depends(get_db)):
    """Get a specific bet."""
    bet = db.query(Bet).filter(Bet.id == bet_id).first()
    
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    
    return bet


@router.get("/bankroll/current")
async def get_current_bankroll():
    """Get current bankroll status."""
    from app.services.bankroll import BankrollManager
    
    manager = BankrollManager()
    stats = manager.get_statistics()
    
    return stats


@router.get("/bankroll/history")
async def get_bankroll_history(limit: int = 100):
    """Get bankroll history."""
    from app.services.bankroll import BankrollManager
    
    manager = BankrollManager()
    history = manager.get_history(limit)
    
    return {"history": history, "count": len(history)}


@router.get("/recommendations/today")
async def get_todays_recommendations(db: Session = Depends(get_db)):
    """Get today's betting recommendations."""
    try:
        from app.services.betting_engine import BettingDecisionEngine
        from app.services.bankroll import BankrollManager
        
        # Get upcoming matches with predictions (limit to 10)
        upcoming = db.query(Match).filter(
            Match.is_upcoming == True,
            Match.has_odds == True
        ).limit(10).all()
        
        if not upcoming:
            return {"recommendations": [], "count": 0, "bankroll": 1000.0}
        
        engine = BettingDecisionEngine()
        bankroll = BankrollManager()
        
        evaluations = []
        for match in upcoming:
            prediction = db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).first()
            
            if prediction:
                try:
                    eval_result = engine.evaluate_match(match, prediction)
                    if eval_result:
                        evaluations.append(eval_result)
                except Exception as e:
                    logger.warning(f"Error evaluating match {match.id}: {e}")
                    continue
        
        # Make decisions
        try:
            decisions = engine.make_betting_decisions(
                evaluations, 
                bankroll.get_current_bankroll()
            )
        except Exception as e:
            logger.error(f"Error making betting decisions: {e}")
            decisions = []
        
        return {
            "recommendations": decisions,
            "count": len(decisions),
            "bankroll": bankroll.get_current_bankroll()
        }
    except Exception as e:
        logger.error(f"Error in get_todays_recommendations: {e}")
        return {"recommendations": [], "count": 0, "bankroll": 1000.0, "error": str(e)}
