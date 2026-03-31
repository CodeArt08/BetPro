"""Odds and Market Analysis module."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class OddsAnalyzer:
    """
    Analyzes bookmaker odds and calculates value opportunities.
    """
    
    def __init__(self, min_odds: float = 2.0):
        self.min_odds = min_odds
        self.historical_margins: List[float] = []
    
    def convert_odds_to_implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 1.0:
            return 0.0
        return 1.0 / odds
    
    def calculate_bookmaker_margin(self, odds_home: float, odds_draw: float, 
                                   odds_away: float) -> float:
        """
        Calculate the bookmaker's margin (overround).
        Lower margin = better for bettor.
        """
        if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
            return 0.0
        
        implied_home = 1 / odds_home
        implied_draw = 1 / odds_draw
        implied_away = 1 / odds_away
        
        margin = (implied_home + implied_draw + implied_away) - 1
        self.historical_margins.append(margin)
        
        return margin
    
    def normalize_probabilities(self, odds_home: float, odds_draw: float,
                               odds_away: float) -> Dict[str, float]:
        """
        Normalize implied probabilities to remove bookmaker margin.
        """
        if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        raw_home = 1 / odds_home
        raw_draw = 1 / odds_draw
        raw_away = 1 / odds_away
        
        total = raw_home + raw_draw + raw_away
        
        return {
            'V': raw_home / total,
            'N': raw_draw / total,
            'D': raw_away / total
        }
    
    def calculate_value(self, model_probability: float, odds: float) -> float:
        """
        Calculate value: Model Prob - Implied Prob.
        Positive value = potential profitable bet.
        """
        if odds <= 1.0:
            return 0.0
        
        implied_prob = 1 / odds
        return model_probability - implied_prob
    
    def calculate_value_percent(self, model_probability: float, odds: float) -> float:
        """Calculate value as a percentage."""
        if odds <= 1.0:
            return 0.0
        
        implied_prob = 1 / odds
        if implied_prob == 0:
            return 0.0
        
        return (model_probability / implied_prob - 1) * 100
    
    def find_value_bets(self, predictions: Dict[str, float], 
                       odds: Dict[str, float],
                       min_value: float = 0.05) -> List[Dict]:
        """
        Find value betting opportunities.
        """
        value_bets = []
        
        outcomes = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}
        
        for outcome, prob in predictions.items():
            if outcome not in odds or odds[outcome] <= 1.0:
                continue
            
            if odds[outcome] < self.min_odds:
                continue
            
            value = self.calculate_value(prob, odds[outcome])
            value_pct = self.calculate_value_percent(prob, odds[outcome])
            
            if value >= min_value:
                value_bets.append({
                    'outcome': outcome,
                    'outcome_name': outcomes[outcome],
                    'model_probability': prob,
                    'odds': odds[outcome],
                    'implied_probability': 1 / odds[outcome],
                    'value': value,
                    'value_percent': value_pct
                })
        
        # Sort by value
        value_bets.sort(key=lambda x: x['value'], reverse=True)
        
        return value_bets
    
    def calculate_expected_value(self, model_probability: float, odds: float) -> float:
        """
        Calculate expected value of a bet.
        EV = (Prob * (Odds - 1)) - ((1 - Prob) * 1)
        """
        if odds <= 1.0:
            return -1.0
        
        win_amount = odds - 1  # Profit if win
        lose_amount = 1  # Loss if lose (stake)
        
        ev = (model_probability * win_amount) - ((1 - model_probability) * lose_amount)
        return ev
    
    def calculate_edge(self, model_probability: float, odds: float) -> float:
        """
        Calculate edge over the bookmaker.
        """
        if odds <= 1.0:
            return 0.0
        
        implied_prob = 1 / odds
        edge = (model_probability / implied_prob - 1) * 100
        return edge
    
    def analyze_odds_movement(self, opening_odds: Dict, current_odds: Dict) -> Dict:
        """
        Analyze odds movement to detect market sentiment.
        """
        movements = {}
        
        for outcome in ['V', 'N', 'D']:
            if outcome in opening_odds and outcome in current_odds:
                opening = opening_odds[outcome]
                current = current_odds[outcome]
                
                change = current - opening
                change_pct = (change / opening) * 100 if opening > 0 else 0
                
                movements[outcome] = {
                    'opening': opening,
                    'current': current,
                    'change': change,
                    'change_percent': change_pct,
                    'direction': 'shortening' if change < 0 else 'drifting',
                    'implied_prob_change': (1/current - 1/opening) if current > 0 and opening > 0 else 0
                }
        
        return movements
    
    def calculate_optimal_stake_kelly(self, probability: float, odds: float,
                                      kelly_fraction: float = 1.0) -> float:
        """
        Calculate optimal stake using Kelly Criterion.
        Kelly = (p * b - 1) / (b - 1) where b = odds
        """
        if odds <= 1.0:
            return 0.0
        
        b = odds
        p = probability
        
        # Full Kelly
        kelly = (p * b - 1) / (b - 1)
        
        # Apply fraction (e.g., half Kelly for more conservative)
        kelly = kelly * kelly_fraction
        
        # Never bet more than available
        return max(0, min(kelly, 1.0))
    
    def calculate_optimal_stake_fractional_kelly(self, probability: float, odds: float,
                                                  fraction: float = 0.25) -> float:
        """Calculate fractional Kelly stake."""
        return self.calculate_optimal_stake_kelly(probability, odds, fraction)
    
    def detect_arbitrage(self, odds_home: float, odds_draw: float, 
                        odds_away: float) -> Optional[Dict]:
        """
        Detect arbitrage opportunity (very rare in practice).
        """
        if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
            return None
        
        total_implied = (1/odds_home + 1/odds_draw + 1/odds_away)
        
        if total_implied < 1.0:
            # Arbitrage exists
            profit_pct = (1 - total_implied) * 100
            
            # Calculate stakes for each outcome
            stakes = {
                'V': (1/odds_home) / total_implied,
                'N': (1/odds_draw) / total_implied,
                'D': (1/odds_away) / total_implied
            }
            
            return {
                'is_arbitrage': True,
                'profit_percent': profit_pct,
                'total_implied': total_implied,
                'recommended_stakes': stakes
            }
        
        return None
    
    def get_average_margin(self) -> float:
        """Get average bookmaker margin from history."""
        if not self.historical_margins:
            return 0.0
        return np.mean(self.historical_margins)
    
    def assess_market_efficiency(self, historical_data: List[Dict]) -> Dict:
        """
        Assess market efficiency by comparing odds to actual results.
        """
        total_matches = len(historical_data)
        if total_matches == 0:
            return {'efficiency': 0, 'favorite_win_rate': 0}
        
        favorite_wins = 0
        favorite_losses = 0
        
        for match in historical_data:
            odds = {
                'V': match.get('odd_home', 0),
                'N': match.get('odd_draw', 0),
                'D': match.get('odd_away', 0)
            }
            result = match.get('result')
            
            # Find favorite (lowest odds)
            favorite = min(odds, key=odds.get)
            
            if favorite == result:
                favorite_wins += 1
            else:
                favorite_losses += 1
        
        favorite_win_rate = favorite_wins / total_matches
        
        return {
            'total_matches': total_matches,
            'favorite_wins': favorite_wins,
            'favorite_win_rate': favorite_win_rate,
            'efficiency': favorite_win_rate  # Higher = more efficient market
        }
