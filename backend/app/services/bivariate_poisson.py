"""Bivariate Poisson Model for enhanced match prediction."""
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger
from scipy import stats
from scipy.optimize import minimize


class BivariatePoissonModel:
    """
    Implements bivariate Poisson distribution for correlated goal scoring.
    Allows for correlation between home and away goals.
    """
    
    def __init__(self, n_simulations: int = 20000):
        self.n_simulations = n_simulations
        np.random.seed(42)
        
        # League averages - balanced to reduce home bias
        self.league_avg_home_goals = 1.35
        self.league_avg_away_goals = 1.20
    
    def estimate_parameters(self, home_attack: float, home_defense: float,
                           away_attack: float, away_defense: float,
                           h2h_data: Optional[Dict] = None) -> Tuple[float, float, float]:
        """
        Estimate bivariate Poisson parameters.
        Returns (lambda_home, lambda_away, covariance).
        """
        # Base expected goals
        lambda_home = home_attack * away_defense * self.league_avg_home_goals
        lambda_away = away_attack * home_defense * self.league_avg_away_goals
        
        # Estimate covariance from H2H data if available
        covariance = 0.0
        if h2h_data and h2h_data.get('h2h_last_5_results'):
            # Higher correlation when teams have history
            # Positive covariance = both score or both don't
            # Negative covariance = one scores, other doesn't
            last_5 = h2h_data['h2h_last_5_results']
            draw_count = last_5.count('N')
            # More draws = higher correlation in scoring patterns
            covariance = (draw_count / len(last_5)) * 0.3 if last_5 else 0.0
        
        # Ensure non-negative parameters
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)
        
        return lambda_home, lambda_away, covariance
    
    def simulate_bivariate_poisson(self, lambda1: float, lambda2: float, 
                                   covariance: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate from bivariate Poisson distribution.
        Uses trivariate reduction method: X = Z1 + Z3, Y = Z2 + Z3
        where Z3 introduces correlation.
        """
        # Covariance parameter (lambda3 in trivariate reduction)
        lambda3 = max(0, covariance)
        
        # Independent components
        lambda1_ind = max(0, lambda1 - lambda3)
        lambda2_ind = max(0, lambda2 - lambda3)
        
        # Generate Poisson samples
        Z1 = np.random.poisson(lambda1_ind, self.n_simulations)
        Z2 = np.random.poisson(lambda2_ind, self.n_simulations)
        Z3 = np.random.poisson(lambda3, self.n_simulations)
        
        # Correlated goals
        home_goals = Z1 + Z3
        away_goals = Z2 + Z3
        
        return home_goals, away_goals
    
    def predict_match(self, home_attack: float, home_defense: float,
                      away_attack: float, away_defense: float,
                      home_elo: float = 1500, away_elo: float = 1500,
                      h2h_data: Optional[Dict] = None,
                      form_data: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive prediction using bivariate Poisson.
        Integrates form, ELO, and H2H into the model.
        """
        # Get base parameters
        lambda_home, lambda_away, covariance = self.estimate_parameters(
            home_attack, home_defense, away_attack, away_defense, h2h_data
        )
        
        # Adjust for ELO difference
        elo_diff = home_elo - away_elo
        elo_adjustment = 1 + (elo_diff / 400) * 0.1  # 10% adjustment per 400 ELO
        lambda_home *= elo_adjustment
        lambda_away /= elo_adjustment
        
        # Adjust for form if provided
        if form_data:
            home_form_rating = form_data.get('home_form_rating', 0.5)
            away_form_rating = form_data.get('away_form_rating', 0.5)
            
            form_diff = home_form_rating - away_form_rating
            lambda_home *= (1 + form_diff * 0.2)
            lambda_away *= (1 - form_diff * 0.2)
        
        # Simulate
        home_goals, away_goals = self.simulate_bivariate_poisson(
            lambda_home, lambda_away, covariance
        )
        
        # Calculate outcome probabilities
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        prob_home_win = home_wins / self.n_simulations
        prob_draw = draws / self.n_simulations
        prob_away_win = away_wins / self.n_simulations
        
        # Calculate goal expectation
        exp_home_goals = float(np.mean(home_goals))
        exp_away_goals = float(np.mean(away_goals))
        
        # Most likely score
        from collections import Counter
        scores = list(zip(home_goals, away_goals))
        score_counts = Counter(scores)
        most_likely_score = score_counts.most_common(1)[0][0]
        
        # Probability distribution for over/under
        total_goals = home_goals + away_goals
        over_2_5 = np.sum(total_goals > 2.5) / self.n_simulations
        over_1_5 = np.sum(total_goals > 1.5) / self.n_simulations
        
        # Both teams to score
        bts = np.sum((home_goals > 0) & (away_goals > 0)) / self.n_simulations
        
        # Clean sheet probabilities
        home_clean_sheet = np.sum(away_goals == 0) / self.n_simulations
        away_clean_sheet = np.sum(home_goals == 0) / self.n_simulations
        
        return {
            'prob_home_win': float(prob_home_win),
            'prob_draw': float(prob_draw),
            'prob_away_win': float(prob_away_win),
            'expected_home_goals': exp_home_goals,
            'expected_away_goals': exp_away_goals,
            'most_likely_score': (int(most_likely_score[0]), int(most_likely_score[1])),
            'over_2_5_prob': float(over_2_5),
            'over_1_5_prob': float(over_1_5),
            'both_teams_score_prob': float(bts),
            'home_clean_sheet_prob': float(home_clean_sheet),
            'away_clean_sheet_prob': float(away_clean_sheet),
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'covariance': covariance,
            'n_simulations': self.n_simulations
        }
    
    def calculate_value_bets(self, prediction: Dict, odds: Dict[str, float]) -> Dict:
        """
        Calculate value bets based on model vs market probabilities.
        """
        value_bets = []
        
        # Check home win
        if odds.get('home', 0) > 0:
            implied_prob = 1 / odds['home']
            value = prediction['prob_home_win'] - implied_prob
            if value > 0.10:  # 10% threshold
                value_bets.append({
                    'outcome': 'V',
                    'outcome_name': 'Home Win',
                    'model_prob': prediction['prob_home_win'],
                    'implied_prob': implied_prob,
                    'value': value,
                    'odds': odds['home']
                })
        
        # Check draw
        if odds.get('draw', 0) > 0:
            implied_prob = 1 / odds['draw']
            value = prediction['prob_draw'] - implied_prob
            if value > 0.10:
                value_bets.append({
                    'outcome': 'N',
                    'outcome_name': 'Draw',
                    'model_prob': prediction['prob_draw'],
                    'implied_prob': implied_prob,
                    'value': value,
                    'odds': odds['draw']
                })
        
        # Check away win
        if odds.get('away', 0) > 0:
            implied_prob = 1 / odds['away']
            value = prediction['prob_away_win'] - implied_prob
            if value > 0.10:
                value_bets.append({
                    'outcome': 'D',
                    'outcome_name': 'Away Win',
                    'model_prob': prediction['prob_away_win'],
                    'implied_prob': implied_prob,
                    'value': value,
                    'odds': odds['away']
                })
        
        return {
            'value_bets': value_bets,
            'best_value': max(value_bets, key=lambda x: x['value']) if value_bets else None
        }


class EnhancedMonteCarlo:
    """
    Enhanced Monte Carlo simulator with 20,000 simulations and correlation.
    """
    
    def __init__(self, n_simulations: int = 20000):
        self.n_simulations = n_simulations
        np.random.seed(42)
    
    def simulate_match(self, expected_home_goals: float, expected_away_goals: float,
                       correlation: float = 0.0, uncertainty: float = 0.15) -> Dict:
        """
        Simulate match with uncertainty and correlation.
        """
        # Add uncertainty to expected goals
        home_exp_dist = np.random.normal(expected_home_goals, expected_home_goals * uncertainty, self.n_simulations)
        away_exp_dist = np.random.normal(expected_away_goals, expected_away_goals * uncertainty, self.n_simulations)
        
        # Clip to valid range
        home_exp_dist = np.clip(home_exp_dist, 0.1, 5.0)
        away_exp_dist = np.clip(away_exp_dist, 0.1, 5.0)
        
        # Generate correlated Poisson samples
        if abs(correlation) > 0.01:
            # Use Gaussian copula for correlation
            rho = correlation
            Z = np.random.multivariate_normal(
                [0, 0], [[1, rho], [rho, 1]], self.n_simulations
            )
            U = stats.norm.cdf(Z)
            home_goals = stats.poisson.ppf(U[:, 0], home_exp_dist).astype(int)
            away_goals = stats.poisson.ppf(U[:, 1], away_exp_dist).astype(int)
        else:
            home_goals = np.array([np.random.poisson(h) for h in home_exp_dist])
            away_goals = np.array([np.random.poisson(a) for a in away_exp_dist])
        
        # Calculate probabilities
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        return {
            'prob_home_win': float(home_wins / self.n_simulations),
            'prob_draw': float(draws / self.n_simulations),
            'prob_away_win': float(away_wins / self.n_simulations),
            'avg_home_goals': float(np.mean(home_goals)),
            'avg_away_goals': float(np.mean(away_goals)),
            'avg_total_goals': float(np.mean(home_goals + away_goals)),
            'both_teams_scored_pct': float(np.sum((home_goals > 0) & (away_goals > 0)) / self.n_simulations),
            'over_2_5_pct': float(np.sum(home_goals + away_goals > 2.5) / self.n_simulations),
            'n_simulations': self.n_simulations
        }
