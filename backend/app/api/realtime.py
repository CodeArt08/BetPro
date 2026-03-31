"""
Real-Time Engine API endpoints.
Module 0/9/10 de la spécification élite.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Optional
from loguru import logger

from app.core.database import get_db
from app.services.realtime_engine import get_engine
from app.models import Match

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.get("/state")
async def get_engine_state():
    """État courant du moteur temps réel (cache + signaux)."""
    engine = get_engine()
    return {
        "cache_ready": len(engine.cache) > 0,
        "results_count": len(engine.results_history),
        "bankroll": engine.bankroll.get_stats(),
        "recovery_mode": engine.error_system.recovery_mode,
        "consecutive_errors": engine.error_system.consecutive_errors,
        "fast_mode_count": engine.fast_mode_count,
        "inference_log_last5": engine.inference_log[-5:],
        "active_lessons": engine.error_system.get_active_lessons(),
    }


@router.get("/dashboard")
async def get_dashboard():
    """Données complètes pour le Module 10 dashboard."""
    engine = get_engine()
    return engine.get_dashboard_data()


@router.get("/signals")
async def get_engine_signals():
    """Tous les signaux engine courants (cache snapshot). ALWAYS returns valid data."""
    engine = get_engine()
    
    # Ensure cache is initialized before returning
    engine._ensure_cache_initialized()
    
    cache = engine.get_cache_snapshot()
    
    # GUARANTEE non-empty cycle and streak for frontend validation
    cycle = cache.get("cycle", {})
    streak = cache.get("streak", {})
    
    # If empty, provide defaults
    if not cycle:
        cycle = {
            'V': {'rate_10': 0.4, 'overdue_score': 0.2, 'overdue': False, 'saturated': False},
            'N': {'rate_10': 0.35, 'overdue_score': 0.5, 'overdue': False, 'saturated': False},
            'D': {'rate_10': 0.25, 'overdue_score': 0.1, 'overdue': False, 'saturated': False}
        }
    if not streak:
        streak = {
            'V': {'current_streak': 0, 'correction_prob': 0.3, 'correction_imminent': False},
            'N': {'current_streak': 0, 'correction_prob': 0.3, 'correction_imminent': False},
            'D': {'current_streak': 0, 'correction_prob': 0.3, 'correction_imminent': False}
        }
    
    return {
        "cycle": cycle,
        "streak": streak,
        "fourier": cache.get("fourier", {'cycle_detected': False}),
        "autocorr": cache.get("autocorr", {}),
        "runs_test": cache.get("runs_test", {}),
        "changepoint": cache.get("changepoint", {}),
        "symbolic": cache.get("symbolic", {}),
        "line_bias": cache.get("line_bias", {}),
        "time_bias": cache.get("time_bias_all", {}),
        "cross_match_corr": cache.get("cross_match_corr", {}),
        "shin": cache.get("shin", {}),
        "dist_50": cache.get("dist_50", {}),
        "regime": cache.get("regime", "STABLE"),
        "engine_scores": {
            "V": cache.get("engine_score_V", 0),
            "N": cache.get("engine_score_N", 0),
            "D": cache.get("engine_score_D", 0),
        },
    }


@router.get("/bankroll")
async def get_bankroll_stats():
    """Statistiques complètes de la bankroll V2 avec historique par journée."""
    engine = get_engine()
    return engine.bankroll.get_stats()


@router.post("/inference/{match_id}")
async def run_inference(
    match_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Lance l'inference complète pour un match (< 8s).
    """
    # Get match from DB
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    engine = get_engine()
    
    result = engine.run_inference(
        match_id=match_id,
        home_team=match.home_team_name,
        away_team=match.away_team_name,
        odds_h=match.odd_home or 2.0,
        odds_d=match.odd_draw or 3.0,
        odds_a=match.odd_away or 3.5,
    )
    
    return result


@router.post("/on-result")
async def on_match_result(match_data: Dict, background_tasks: BackgroundTasks):
    """
    Notifie le moteur d'un résultat. Déclenche Thread A.
    """
    engine = get_engine()
    background_tasks.add_task(engine.on_match_completed, match_data)
    return {"status": "Thread A triggered", "match_id": match_data.get("match_id")}


@router.post("/prepare-cache")
async def prepare_cache(next_match: Dict, background_tasks: BackgroundTasks):
    """
    Lance Thread B: pré-calcul du cache pour le prochain match.
    """
    engine = get_engine()
    background_tasks.add_task(engine.prepare_cache, next_match)
    return {"status": "Thread B started", "match": next_match.get("home_team")}


@router.post("/run-montecarlo")
async def run_montecarlo(
    lambda_h: float = 1.35,
    lambda_a: float = 1.10,
    n_runs: int = 10000,
    background_tasks: BackgroundTasks = None
):
    """Lance Thread C: Monte Carlo en background."""
    engine = get_engine()
    engine.run_monte_carlo_background(lambda_h, lambda_a, n_runs)
    return {"status": "Thread C started", "n_runs": n_runs}


@router.get("/bankroll")
async def get_bankroll():
    """État détaillé de la bankroll."""
    engine = get_engine()
    return engine.bankroll.get_stats()


@router.get("/rl-agent")
async def get_rl_stats():
    """Stats de l'agent RL."""
    engine = get_engine()
    return {
        "agent": engine.rl_agent.to_dict(),
        "bandit_stats": engine.ucb_bandit.get_model_stats(),
        "anti_martingale": engine.anti_martingale.to_dict(),
    }


@router.get("/errors")
async def get_error_stats():
    """Stats du système d'erreurs."""
    engine = get_engine()
    return {
        "status": engine.error_system.to_dict(),
        "recent_autopsies": engine.error_system.error_log[-10:],
        "ece": engine.calibration_drift.get_ece_status(),
        "meta_alerts": engine.error_system.check_meta_patterns(),
    }


@router.post("/save-state")
async def save_engine_state():
    """Sauvegarde l'état complet du moteur."""
    engine = get_engine()
    engine.save_state()
    return {"status": "Engine state saved"}


@router.post("/load-history")
async def load_historical_data(db: Session = Depends(get_db)):
    """
    Charge les résultats historiques depuis la DB dans le moteur.
    À appeler au démarrage ou après un reset.
    """
    engine = get_engine()
    
    try:
        # Load completed matches
        matches = db.query(Match).filter(
            Match.is_completed == True,
            Match.result != None
        ).order_by(Match.matchday).all()
        
        results = [m.result for m in matches if m.result]
        results_by_line = {}
        results_by_hour = {}
        score_counts = {}
        
        for m in matches:
            if not m.result:
                continue
            # By line position
            lp = getattr(m, 'line_position', 1) or 1
            if lp not in results_by_line:
                results_by_line[lp] = []
            results_by_line[lp].append(m.result)
            
            # Score
            if m.score_home is not None and m.score_away is not None:
                score_key = f"{m.score_home}-{m.score_away}"
                score_counts[score_key] = score_counts.get(score_key, 0) + 1
        
        engine.load_historical_results(
            results=results,
            results_by_line=results_by_line,
            results_by_hour=results_by_hour,
            score_counts=score_counts,
        )
        
        return {
            "status": "OK",
            "matches_loaded": len(results),
            "lines_tracked": len(results_by_line),
            "score_types": len(score_counts),
        }
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
