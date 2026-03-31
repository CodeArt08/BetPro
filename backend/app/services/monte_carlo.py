"""Monte Carlo Simulation for match prediction."""
import numpy as np
from typing import Dict, Tuple, List
from loguru import logger
from scipy import stats


class MonteCarloSimulator:
    """
    Simulates matches using Poisson goal distributions.
    """
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        np.random.seed(42)
    
    def simulate_match(self, expected_home_goals: float, expected_away_goals: float) -> Dict:
        """
        Simulate a match using Poisson distributions.
        Returns detailed simulation results.
        """
        # Simulate goals using Poisson distribution
        home_goals = np.random.poisson(expected_home_goals, self.n_simulations)
        away_goals = np.random.poisson(expected_away_goals, self.n_simulations)
        
        # Count outcomes
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        # Calculate probabilities
        prob_home_win = home_wins / self.n_simulations
        prob_draw = draws / self.n_simulations
        prob_away_win = away_wins / self.n_simulations
        
        # Additional statistics
        total_goals = home_goals + away_goals
        both_teams_scored = np.sum((home_goals > 0) & (away_goals > 0))
        over_2_5 = np.sum(total_goals > 2.5)
        
        return {
            'home_wins': int(home_wins),
            'draws': int(draws),
            'away_wins': int(away_wins),
            'prob_home_win': float(prob_home_win),
            'prob_draw': float(prob_draw),
            'prob_away_win': float(prob_away_win),
            'avg_home_goals': float(np.mean(home_goals)),
            'avg_away_goals': float(np.mean(away_goals)),
            'avg_total_goals': float(np.mean(total_goals)),
            'both_teams_scored_pct': float(both_teams_scored / self.n_simulations),
            'over_2_5_pct': float(over_2_5 / self.n_simulations),
            'most_likely_score': self._get_most_likely_score(home_goals, away_goals),
            'score_distribution': self._get_score_distribution(home_goals, away_goals)
        }
    
    def _get_most_likely_score(self, home_goals: np.ndarray, away_goals: np.ndarray) -> Tuple[int, int]:
        """Get the most likely exact score."""
        from collections import Counter
        scores = list(zip(home_goals, away_goals))
        score_counts = Counter(scores)
        score = score_counts.most_common(1)[0][0]
        return (int(score[0]), int(score[1]))
    
    def _get_score_distribution(self, home_goals: np.ndarray, away_goals: np.ndarray, top_n: int = 10) -> Dict:
        """Get distribution of most common scores."""
        from collections import Counter
        scores = list(zip(home_goals, away_goals))
        score_counts = Counter(scores)
        
        total = len(scores)
        distribution = {}
        
        for score, count in score_counts.most_common(top_n):
            distribution[f"{score[0]}-{score[1]}"] = count / total
        
        return distribution
    
    def simulate_with_uncertainty(self, expected_home_goals: float, expected_away_goals: float,
                                  uncertainty: float = 0.2) -> Dict:
        """
        Simulate with uncertainty in expected goals.
        """
        # Add uncertainty to expected goals
        home_goals_range = expected_home_goals * (1 + np.random.uniform(-uncertainty, uncertainty, self.n_simulations))
        away_goals_range = expected_away_goals * (1 + np.random.uniform(-uncertainty, uncertainty, self.n_simulations))
        
        # Simulate each match with slightly different expected goals
        home_goals = np.array([np.random.poisson(hg) for hg in home_goals_range])
        away_goals = np.array([np.random.poisson(ag) for ag in away_goals_range])
        
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        return {
            'prob_home_win': home_wins / self.n_simulations,
            'prob_draw': draws / self.n_simulations,
            'prob_away_win': away_wins / self.n_simulations
        }
    
    def calculate_value_from_simulation(self, simulation_results: Dict,
                                        odds: Dict[str, float]) -> Dict:
        """
        Calculate value bets from simulation results.
        """
        value = {}
        
        if odds.get('home', 0) > 0:
            implied_home = 1 / odds['home']
            value['home'] = simulation_results['prob_home_win'] - implied_home
        
        if odds.get('draw', 0) > 0:
            implied_draw = 1 / odds['draw']
            value['draw'] = simulation_results['prob_draw'] - implied_draw
        
        if odds.get('away', 0) > 0:
            implied_away = 1 / odds['away']
            value['away'] = simulation_results['prob_away_win'] - implied_away
        
        return value
    
    def simulate_season(self, team_strengths: Dict[str, Dict], fixture_list: List[Dict]) -> Dict:
        """
        Simulate an entire season.
        """
        standings = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                           'goals_for': 0, 'goals_against': 0} 
                    for team in team_strengths}
        
        for fixture in fixture_list:
            home_team = fixture['home']
            away_team = fixture['away']
            
            home_strength = team_strengths[home_team]
            away_strength = team_strengths[away_team]
            
            # Expected goals - balanced multipliers
            exp_home = home_strength['attack_home'] * away_strength['defense_away'] * 1.35
            exp_away = away_strength['attack_away'] * home_strength['defense_home'] * 1.20
            
            # Simulate
            result = self.simulate_match(exp_home, exp_away)
            
            # Sample one outcome
            outcome = np.random.choice(['V', 'N', 'D'], p=[
                result['prob_home_win'],
                result['prob_draw'],
                result['prob_away_win']
            ])
            
            # Sample goals
            home_goals = np.random.poisson(exp_home)
            away_goals = np.random.poisson(exp_away)
            
            # Update standings
            standings[home_team]['goals_for'] += home_goals
            standings[home_team]['goals_against'] += away_goals
            standings[away_team]['goals_for'] += away_goals
            standings[away_team]['goals_against'] += home_goals
            
            if outcome == 'V':
                standings[home_team]['points'] += 3
                standings[home_team]['wins'] += 1
                standings[away_team]['losses'] += 1
            elif outcome == 'N':
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
                standings[home_team]['draws'] += 1
                standings[away_team]['draws'] += 1
            else:
                standings[away_team]['points'] += 3
                standings[away_team]['wins'] += 1
                standings[home_team]['losses'] += 1
        
        # Sort by points
        sorted_standings = sorted(standings.items(), key=lambda x: (-x[1]['points'], 
                                      -(x[1]['goals_for'] - x[1]['goals_against'])))
        
        return dict(sorted_standings)
    
    def run_multiple_season_simulations(self, team_strengths: Dict, fixture_list: List[Dict],
                                        n_seasons: int = 1000) -> Dict:
        """
        Run multiple season simulations for statistical analysis.
        """
        all_standings = []
        champion_counts = {}
        
        for _ in range(n_seasons):
            standings = self.simulate_season(team_strengths, fixture_list)
            all_standings.append(standings)
            
            champion = list(standings.keys())[0]
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
        
        # Calculate average positions
        avg_positions = {}
        for team in team_strengths:
            positions = [list(s.keys()).index(team) + 1 for s in all_standings]
            avg_positions[team] = {
                'avg_position': np.mean(positions),
                'std_position': np.std(positions),
                'champion_prob': champion_counts.get(team, 0) / n_seasons
            }
        
        return {
            'avg_positions': avg_positions,
            'champion_probabilities': {k: v / n_seasons for k, v in champion_counts.items()}
        }
