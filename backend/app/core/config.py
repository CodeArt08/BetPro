"""Application configuration settings."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database - use backend/data folder
    DATABASE_URL: str = "sqlite:///./backend/data/bet261_prediction.db"
    
    # Application
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True
    
    # Betting Parameters - CONSERVATIVE STRATEGY
    INITIAL_BANKROLL: float = 1000.0
    MIN_ODDS: float = 1.80  # Lowered to accept favorites
    MIN_CONFIDENCE: float = 0.75  # Raised for quality
    MIN_MODEL_AGREEMENT: float = 0.80  # New: require strong agreement
    MIN_VALUE_EDGE: float = 0.10  # New: require 10% value edge
    KELLY_FRACTION: float = 0.10  # Reduced for safety
    MAX_STAKE_PERCENT: float = 0.05
    MAX_BETS_PER_MATCHDAY: int = 2  # New: limit daily bets
    
    # Season Targets
    SEASON_TARGET_PROFIT: float = 10000.0  # 10,000 Ar target
    TOTAL_MATCHDAYS: int = 38
    
    # Browser Automation
    HEADLESS: bool = True  # Set to False to see browser for debugging
    BET261_URL: str = "https://bet261.mg/virtual/category/instant-league/8037/matches"
    SCRAPE_INTERVAL_SECONDS: int = 30
    RESULT_TRIGGER_SECONDS: int = 58
    
    # Model Parameters
    MODEL_RETRAIN_INTERVAL: int = 100
    ENSEMBLE_WEIGHTS_UPDATE_INTERVAL: int = 50
    MONTE_CARLO_SIMULATIONS: int = 10000
    
    # Feature Engineering
    FORM_MATCHES: int = 5
    SEQUENCE_LENGTH: int = 3
    
    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "models")
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "logs")
    
    # Spanish League Teams
    TEAMS: List[str] = [
        "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Betis",
        "Celta Vigo", "Eibar", "Espanyol", "Getafe", "Girona",
        "Leganes", "Levante", "Mallorca", "Osasuna", "Rayo Vallecano",
        "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [settings.DATA_DIR, settings.MODELS_DIR, settings.LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
