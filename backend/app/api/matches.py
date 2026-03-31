"""Matches API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.models import Match, Season


router = APIRouter()


class MatchResponse(BaseModel):
    id: int
    season_id: int
    matchday: int
    line_position: int
    home_team_name: str
    away_team_name: str
    score_home: Optional[int]
    score_away: Optional[int]
    result: Optional[str]
    odd_home: Optional[float]
    odd_draw: Optional[float]
    odd_away: Optional[float]
    is_completed: bool
    is_upcoming: bool
    
    class Config:
        from_attributes = True


class MatchListResponse(BaseModel):
    matches: List[MatchResponse]
    total: int


@router.get("/", response_model=MatchListResponse)
async def get_matches(
    season_id: Optional[int] = None,
    matchday: Optional[int] = None,
    is_completed: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get matches with optional filters. Default: only completed matches from active season."""
    query = db.query(Match)
    
    # Default to active season if not specified
    if season_id is None:
        active_season = db.query(Season).filter(Season.is_active == True).first()
        if active_season:
            season_id = active_season.id
    
    if season_id:
        query = query.filter(Match.season_id == season_id)
    if matchday:
        query = query.filter(Match.matchday == matchday)
    
    # Default: only show completed matches (results)
    if is_completed is None:
        query = query.filter(Match.is_completed == True)
    else:
        query = query.filter(Match.is_completed == is_completed)
    
    total = query.count()
    matches = query.order_by(Match.matchday.desc()).offset(offset).limit(limit).all()
    
    return {"matches": matches, "total": total}


@router.get("/upcoming", response_model=MatchListResponse)
async def get_upcoming_matches(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get upcoming matches with odds."""
    active_season = db.query(Season).filter(Season.is_active == True).first()
    query = db.query(Match).filter(
        Match.is_upcoming == True,
        Match.has_odds == True
    )
    if active_season:
        query = query.filter(Match.season_id == active_season.id)

    matches = query.order_by(Match.matchday).limit(limit).all()
    
    return {"matches": matches, "total": len(matches)}


@router.get("/latest-matchday")
async def get_latest_matchday(db: Session = Depends(get_db)):
    """Get the latest completed matchday for the active season."""
    active_season = db.query(Season).filter(Season.is_active == True).first()
    if not active_season:
        return {"matchday": 0, "season_id": None}
    
    last_match = db.query(Match).filter(
        Match.season_id == active_season.id,
        Match.is_completed == True
    ).order_by(Match.matchday.desc()).first()
    
    return {
        "matchday": last_match.matchday if last_match else 0,
        "season_id": active_season.id,
        "season_number": active_season.season_number
    }


@router.get("/recent", response_model=MatchListResponse)
async def get_recent_results(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get recent completed matches."""
    matches = db.query(Match).filter(
        Match.is_completed == True
    ).order_by(Match.result_recorded_at.desc()).limit(limit).all()
    
    return {"matches": matches, "total": len(matches)}


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(match_id: int, db: Session = Depends(get_db)):
    """Get a specific match."""
    match = db.query(Match).filter(Match.id == match_id).first()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    return match


@router.get("/matchday/{matchday}", response_model=MatchListResponse)
async def get_matchday_matches(
    matchday: int,
    season_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all matches for a specific matchday."""
    query = db.query(Match).filter(Match.matchday == matchday)
    
    if season_id:
        query = query.filter(Match.season_id == season_id)
    else:
        # Get active season
        active_season = db.query(Season).filter(Season.is_active == True).first()
        if active_season:
            query = query.filter(Match.season_id == active_season.id)
    
    matches = query.order_by(Match.line_position).all()
    
    return {"matches": matches, "total": len(matches)}
