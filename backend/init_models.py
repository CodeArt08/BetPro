"""Initialize ML models with synthetic training data."""
import numpy as np
import pandas as pd
from app.services.ml_ensemble import MachineLearningEnsemble
from app.services.feature_engineering import FeatureEngineeringPipeline
from app.core.database import SessionLocal
from app.models import Team, MatchFeatures
from pathlib import Path

def create_synthetic_training_data():
    """Create synthetic training data based on team strengths."""
    db = SessionLocal()
    
    teams = db.query(Team).all()
    team_names = [t.name for t in teams]
    
    # Create synthetic matches
    np.random.seed(42)
    n_samples = 500
    
    data = []
    for _ in range(n_samples):
        home_idx = np.random.randint(0, len(teams))
        away_idx = np.random.randint(0, len(teams))
        while away_idx == home_idx:
            away_idx = np.random.randint(0, len(teams))
        
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        # Simulate features based on ELO
        elo_diff = home_team.elo_rating - away_team.elo_rating
        
        # Higher ELO = higher chance of home win
        home_win_prob = 0.45 + (elo_diff / 1000) * 0.15
        draw_prob = 0.28
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Determine outcome
        rand = np.random.random()
        if rand < home_win_prob:
            result = 'V'
        elif rand < home_win_prob + draw_prob:
            result = 'N'
        else:
            result = 'D'
        
        # Create synthetic features
        features = {
            'home_elo': home_team.elo_rating,
            'away_elo': away_team.elo_rating,
            'elo_diff': elo_diff,
            'home_elo_home': home_team.elo_home,
            'away_elo_away': away_team.elo_away,
            'home_bayesian_rating': home_team.bayesian_rating,
            'away_bayesian_rating': away_team.bayesian_rating,
            'bayesian_diff': home_team.bayesian_rating - away_team.bayesian_rating,
            'home_attack_strength': home_team.attack_strength,
            'away_attack_strength': away_team.attack_strength,
            'home_defense_strength': home_team.defense_strength,
            'away_defense_strength': away_team.defense_strength,
            'home_form_points': np.random.uniform(3, 12),
            'away_form_points': np.random.uniform(3, 12),
            'home_form_goals_scored': np.random.uniform(3, 10),
            'away_form_goals_scored': np.random.uniform(3, 10),
            'home_form_goals_conceded': np.random.uniform(2, 8),
            'away_form_goals_conceded': np.random.uniform(2, 8),
            'home_form_wins': np.random.randint(0, 4),
            'away_form_wins': np.random.randint(0, 4),
            'home_form_draws': np.random.randint(0, 3),
            'away_form_draws': np.random.randint(0, 3),
            'home_form_losses': np.random.randint(0, 3),
            'away_form_losses': np.random.randint(0, 3),
            'home_home_form_points': np.random.uniform(2, 8),
            'away_away_form_points': np.random.uniform(2, 8),
            'home_home_form_goals': np.random.uniform(1, 5),
            'away_away_form_goals': np.random.uniform(1, 5),
            'home_home_form_conceded': np.random.uniform(1, 4),
            'away_away_form_conceded': np.random.uniform(1, 4),
            'home_gd_trend': np.random.uniform(-2, 2),
            'away_gd_trend': np.random.uniform(-2, 2),
            'home_scoring_freq': np.random.uniform(0.8, 2.0),
            'away_scoring_freq': np.random.uniform(0.8, 2.0),
            'home_conceding_freq': np.random.uniform(0.6, 1.5),
            'away_conceding_freq': np.random.uniform(0.6, 1.5),
            'h2h_home_wins': np.random.randint(0, 5),
            'h2h_away_wins': np.random.randint(0, 5),
            'h2h_draws': np.random.randint(0, 4),
            'h2h_avg_goals': np.random.uniform(1.5, 3.5),
            'h2h_home_avg_goals': np.random.uniform(0.5, 2.0),
            'h2h_away_avg_goals': np.random.uniform(0.5, 2.0),
            'h2h_draw_rate': np.random.uniform(0.15, 0.35),
            'home_winning_streak': np.random.randint(0, 5),
            'away_winning_streak': np.random.randint(0, 5),
            'home_losing_streak': np.random.randint(0, 4),
            'away_losing_streak': np.random.randint(0, 4),
            'home_unbeaten_streak': np.random.randint(0, 8),
            'away_unbeaten_streak': np.random.randint(0, 8),
            'home_advantage': np.random.uniform(0.05, 0.2),
            'away_disadvantage': np.random.uniform(-0.2, -0.05),
            'line_position': np.random.randint(1, 11),
            'line_home_win_rate': 0.45,
            'line_draw_rate': 0.28,
            'line_away_win_rate': 0.27,
            'line_avg_goals': 2.6,
            'home_sequence_prob_V': 0.45,
            'home_sequence_prob_N': 0.28,
            'home_sequence_prob_D': 0.27,
            'away_sequence_prob_V': 0.35,
            'away_sequence_prob_N': 0.28,
            'away_sequence_prob_D': 0.37,
            'home_xg': np.random.uniform(0.8, 2.2),
            'away_xg': np.random.uniform(0.6, 1.8),
            'home_win_rate': np.random.uniform(0.3, 0.6),
            'away_win_rate': np.random.uniform(0.25, 0.55),
            'home_draw_rate': np.random.uniform(0.2, 0.35),
            'away_draw_rate': np.random.uniform(0.2, 0.35),
            'home_avg_goals_scored': np.random.uniform(1.0, 2.2),
            'away_avg_goals_scored': np.random.uniform(0.8, 2.0),
            'home_avg_goals_conceded': np.random.uniform(0.8, 1.8),
            'away_avg_goals_conceded': np.random.uniform(0.9, 2.0),
            'matchday': np.random.randint(1, 39),
            'is_early_matchday': np.random.choice([0, 1]),
            'is_late_matchday': np.random.choice([0, 1]),
        }
        
        data.append({**features, 'result': result})
    
    db.close()
    
    df = pd.DataFrame(data)
    X = df.drop('result', axis=1)
    y = df['result']
    
    return X, y

def initialize_models():
    """Initialize and train ML models."""
    print("Creating synthetic training data...")
    X, y = create_synthetic_training_data()
    print(f"Training data: {len(X)} samples, {len(X.columns)} features")
    
    print("\nTraining ML ensemble...")
    ml = MachineLearningEnsemble()
    ml.train(X, y)
    
    print("\nSaving models...")
    ml.save_models()
    
    print("\nModels initialized successfully!")
    print(f"Model weights: {ml.model_weights}")

if __name__ == "__main__":
    initialize_models()
