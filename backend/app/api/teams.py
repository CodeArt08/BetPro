"""Teams API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.models import Team, Match


router = APIRouter()


class TeamResponse(BaseModel):
    id: int
    name: str
    elo_rating: float
    elo_home: float
    elo_away: float
    bayesian_rating: float
    attack_strength: float
    defense_strength: float
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_scored: int
    goals_conceded: int
    current_form: str
    winning_streak: int
    losing_streak: int
    
    class Config:
        from_attributes = True


class TeamStrengthResponse(BaseModel):
    name: str
    elo_rating: float
    attack_strength: float
    defense_strength: float
    attack_strength_home: float
    defense_strength_home: float
    attack_strength_away: float
    defense_strength_away: float


@router.get("/", response_model=List[TeamResponse])
async def get_teams(db: Session = Depends(get_db)):
    """Get all teams."""
    teams = db.query(Team).order_by(Team.elo_rating.desc()).all()
    return teams


@router.get("/strengths", response_model=List[TeamStrengthResponse])
async def get_team_strengths(db: Session = Depends(get_db)):
    """Get team strength ratings for prediction."""
    teams = db.query(Team).all()
    
    return [{
        "name": t.name,
        "elo_rating": t.elo_rating,
        "attack_strength": t.attack_strength,
        "defense_strength": t.defense_strength,
        "attack_strength_home": t.attack_strength_home,
        "defense_strength_home": t.defense_strength_home,
        "attack_strength_away": t.attack_strength_away,
        "defense_strength_away": t.defense_strength_away
    } for t in teams]


@router.get("/elo-rankings")
async def get_elo_rankings(db: Session = Depends(get_db)):
    """Get teams ranked by ELO rating."""
    teams = db.query(Team).order_by(Team.elo_rating.desc()).all()
    
    return {
        "rankings": [{
            "position": i + 1,
            "name": t.name,
            "elo_rating": round(t.elo_rating, 1),
            "elo_home": round(t.elo_home, 1),
            "elo_away": round(t.elo_away, 1),
            "matches_played": t.matches_played,
            "win_rate": round(t.win_rate, 3) if t.matches_played > 0 else 0
        } for i, t in enumerate(teams)]
    }


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(team_id: int, db: Session = Depends(get_db)):
    """Get a specific team."""
    team = db.query(Team).filter(Team.id == team_id).first()
    
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return team


@router.get("/name/{team_name}", response_model=TeamResponse)
async def get_team_by_name(team_name: str, db: Session = Depends(get_db)):
    """Get team by name."""
    team = db.query(Team).filter(Team.name == team_name).first()
    
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return team


@router.get("/{team_id}/matches")
async def get_team_matches(
    team_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent matches for a team."""
    matches = db.query(Match).filter(
        (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
        Match.is_completed == True
    ).order_by(Match.matchday.desc()).limit(limit).all()
    
    team = db.query(Team).filter(Team.id == team_id).first()
    
    return {
        "team": team.name if team else "Unknown",
        "matches": [{
            "matchday": m.matchday,
            "home_team": m.home_team_name,
            "away_team": m.away_team_name,
            "score_home": m.score_home,
            "score_away": m.score_away,
            "result": m.result,
            "is_home": m.home_team_id == team_id
        } for m in matches]
    }


@router.get("/comparison")
async def compare_teams(
    team1: str,
    team2: str,
    db: Session = Depends(get_db)
):
    """Compare two teams."""
    t1 = db.query(Team).filter(Team.name == team1).first()
    t2 = db.query(Team).filter(Team.name == team2).first()
    
    if not t1 or not t2:
        raise HTTPException(status_code=404, detail="One or both teams not found")
    
    # Get head-to-head
    h2h_matches = db.query(Match).filter(
        ((Match.home_team_id == t1.id) & (Match.away_team_id == t2.id)) |
        ((Match.home_team_id == t2.id) & (Match.away_team_id == t1.id)),
        Match.is_completed == True
    ).all()
    
    t1_wins = sum(1 for m in h2h_matches if 
        (m.home_team_id == t1.id and m.result == 'V') or
        (m.away_team_id == t1.id and m.result == 'D'))
    t2_wins = sum(1 for m in h2h_matches if 
        (m.home_team_id == t2.id and m.result == 'V') or
        (m.away_team_id == t2.id and m.result == 'D'))
    draws = sum(1 for m in h2h_matches if m.result == 'N')
    
    return {
        "team1": {
            "name": t1.name,
            "elo": t1.elo_rating,
            "attack": t1.attack_strength,
            "defense": t1.defense_strength,
            "form": t1.current_form
        },
        "team2": {
            "name": t2.name,
            "elo": t2.elo_rating,
            "attack": t2.attack_strength,
            "defense": t2.defense_strength,
            "form": t2.current_form
        },
        "comparison": {
            "elo_diff": t1.elo_rating - t2.elo_rating,
            "attack_diff": t1.attack_strength - t2.attack_strength,
            "defense_diff": t1.defense_strength - t2.defense_strength
        },
        "head_to_head": {
            "total_matches": len(h2h_matches),
            "team1_wins": t1_wins,
            "team2_wins": t2_wins,
            "draws": draws
        }
    }
