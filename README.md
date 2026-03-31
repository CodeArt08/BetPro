# Bet261 Prediction Engine

Autonomous AI Prediction Engine for Bet261 Virtual Spanish League.

## Quick Start

### Double-click to start:
- `start_all.bat` - Start Backend + Frontend together
- `start_backend.bat` - Start API only
- `start_frontend.bat` - Start Dashboard only
- `start_orchestrator.bat` - Start autonomous engine

### Access:
- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Setup (First Time)

### Backend
```bash
cd backend
py -m pip install python-dotenv pydantic pydantic-settings sqlalchemy playwright pandas numpy scikit-learn xgboost lightgbm scipy fastapi uvicorn python-multipart aiohttp httpx loguru python-dateutil joblib
py -m playwright install chromium
py init_db.py
```

### Frontend
```bash
cd frontend
npm install
```

## Overview

This system continuously monitors Bet261 virtual Spanish league matches, collects historical data, builds predictive models, and identifies value betting opportunities.

## Architecture

The system consists of 14 integrated subsystems:

1. **Browser Automation Engine** - Playwright-based Chrome automation
2. **Data Extraction Engine** - Parses and deduplicates match data
3. **SQLite Database** - Stores teams, matches, predictions, bets
4. **Feature Engineering Pipeline** - Computes 50+ predictive features
5. **Team Strength Models** - ELO, Bayesian, Poisson ratings
6. **Sequence Pattern Analysis** - Markov chains, HMM
7. **Machine Learning Ensemble** - Logistic Regression, Random Forest, XGBoost, LightGBM
8. **Monte Carlo Simulation** - Match outcome probabilities
9. **Odds Analysis** - Value detection, Kelly criterion
10. **Betting Decision Engine** - Confidence thresholds, value requirements
11. **Bankroll Management** - Fractional Kelly, risk metrics
12. **Profit Tracking** - Bet history, ROI calculation
13. **Season Lifecycle Manager** - 38 matchdays, standings
14. **Continuous Learning Engine** - Model updates, weight adaptation

## Project Structure

```
Bet261PredictionEngine/
├── start_all.bat           # Start all services
├── start_backend.bat       # Start API
├── start_frontend.bat      # Start Dashboard
├── start_orchestrator.bat   # Start autonomous engine
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Config, database, orchestrator
│   │   ├── models/        # SQLAlchemy ORM models
│   │   └── services/      # Business logic modules
│   ├── init_db.py         # Initialize database
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── pages/         # React components
│   │   └── services/      # API client
│   ├── package.json
│   └── vite.config.ts
├── data/                  # SQLite database
├── logs/                  # Application logs
└── models/                # Saved ML models
```

## Configuration

Key settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | sqlite:///./data/bet261_prediction.db | SQLite database path |
| `INITIAL_BANKROLL` | 1000 | Starting bankroll |
| `MIN_ODDS` | 2.0 | Minimum odds to bet |
| `MIN_CONFIDENCE` | 0.65 | Minimum prediction confidence |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |
| `MAX_STAKE_PERCENT` | 0.05 | Max bankroll per bet |
| `RESULT_TRIGGER_SECONDS` | 58 | Countdown trigger for scraping |
| `MODEL_RETRAIN_INTERVAL` | 100 | Matches between retraining |

## API Endpoints

### Dashboard
- `GET /api/dashboard/overview` - Main dashboard data
- `GET /api/dashboard/standings` - League standings
- `GET /api/dashboard/statistics` - League statistics

### Matches
- `GET /api/matches/` - List matches
- `GET /api/matches/upcoming` - Upcoming matches with odds
- `GET /api/matches/recent` - Recent results

### Predictions
- `GET /api/predictions/` - List predictions
- `GET /api/predictions/upcoming` - Predictions for upcoming matches
- `GET /api/predictions/accuracy/stats` - Accuracy statistics

### Betting
- `GET /api/betting/` - Bet history
- `GET /api/betting/pending` - Pending bets
- `GET /api/betting/summary` - Betting summary
- `GET /api/betting/bankroll/current` - Current bankroll

### Teams
- `GET /api/teams/` - All teams
- `GET /api/teams/elo-rankings` - ELO rankings
- `GET /api/teams/{id}/matches` - Team match history

## Spanish League Teams

The system tracks 20 teams:
- Real Madrid, Barcelona, Atlético Madrid, Athletic Bilbao
- Sevilla, Real Sociedad, Villarreal, Real Betis
- Valencia, Getafe, Rayo Vallecano, Osasuna
- Celta Vigo, Mallorca, Girona, Almería
- Cádiz, Las Palmas, Alavés, Espanyol

## How It Works

1. **Data Collection**: Browser automation monitors Bet261 for upcoming matches and results
2. **Feature Engineering**: Each match gets 50+ computed features (form, H2H, strength, momentum)
3. **Prediction**: ML ensemble + Monte Carlo + ELO combine for final probabilities
4. **Value Detection**: Compare model probabilities to bookmaker odds
5. **Betting Decision**: Apply confidence/value thresholds, Kelly criterion
6. **Continuous Learning**: Update models after each match result

## License

MIT
