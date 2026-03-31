"""API routes package."""
from fastapi import APIRouter
from . import matches, betting, dashboard, teams, predictions, scrape, realtime

api_router = APIRouter()

api_router.include_router(matches.router, prefix="/matches", tags=["matches"])
api_router.include_router(betting.router, prefix="/betting", tags=["betting"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(scrape.router, prefix="/scrape", tags=["scrape"])
api_router.include_router(realtime.router, tags=["realtime"])
