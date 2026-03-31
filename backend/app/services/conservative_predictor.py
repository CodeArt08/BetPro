"""Conservative Prediction System - Quality over Quantity.

This system is designed to guarantee daily profits by:
1. Only predicting on HIGH CONFIDENCE matches (≥75%)
2. Requiring strong model agreement (≥80%)
3. Seeking significant value edge (≥10%)
4. Limiting bets to 1-2 per matchday maximum
5. Preferring safer outcomes (favorites with good odds)

Goal: Small but consistent daily gains that compound over 38 matchdays.
Target: Up to 10,000 Ar cumulative profit by end of season.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from app.models import Match, Prediction, Team


class ConservativePredictor:
    """
    Conservative prediction system focused on guaranteed daily profits.
    
    Key principles:
    - Only bet when VERY confident
    - Skip uncertain matches entirely
    - Prefer favorites with decent odds (2.0-3.0)
    - One quality bet > multiple risky bets
    """
    
    # Strict thresholds for quality predictions
    MIN_CONFIDENCE = 0.75  # 75% minimum confidence
    MIN_MODEL_AGREEMENT = 0.80  # 80% of models must agree
    MIN_VALUE_EDGE = 0.10  # 10% minimum value edge
    MIN_ODDS = 1.80  # Minimum odds to consider
    MAX_ODDS = 4.0  # Maximum odds (avoid risky long shots)
    
    # Preferred odds range for favorites
    FAVORITE_ODDS_RANGE = (1.80, 2.80)
    
    # Maximum bets per matchday
    MAX_BETS_PER_MATCHDAY = 2
    
    def __init__(self):
        self.prediction_history = []
        self.daily_profits = {}
    
    def evaluate_match_quality(self, match: Match, prediction: Prediction,
                               home_team: Team, away_team: Team) -> Dict:
        """
        Evaluate if a match meets quality criteria for betting.
        Returns quality score and recommendation.
        """
        evaluation = {
            'match_id': match.id,
            'home_team': match.home_team_name,
            'away_team': match.away_team_name,
            'is_quality_bet': False,
            'quality_score': 0.0,
            'confidence': prediction.confidence,
            'model_agreement': prediction.model_agreement,
            'predicted_result': prediction.predicted_result,
            'recommended_outcome': None,
            'recommended_odds': 0.0,
            'value_edge': 0.0,
            'risk_level': 'high',
            'reason': ''
        }
        
        # Step 1: Check confidence threshold
        if prediction.confidence < self.MIN_CONFIDENCE:
            evaluation['reason'] = f"Confidence {prediction.confidence:.1%} below {self.MIN_CONFIDENCE:.0%}"
            return evaluation
        
        # Step 2: Check model agreement
        if prediction.model_agreement < self.MIN_MODEL_AGREEMENT:
            evaluation['reason'] = f"Model agreement {prediction.model_agreement:.1%} below {self.MIN_MODEL_AGREEMENT:.0%}"
            return evaluation
        
        # Step 3: Check odds availability
        if not match.has_odds:
            evaluation['reason'] = "No odds available"
            return evaluation
        
        # Step 4: Find the best value outcome
        odds = {
            'V': match.odd_home,
            'N': match.odd_draw,
            'D': match.odd_away
        }
        
        probs = {
            'V': prediction.prob_home_win,
            'N': prediction.prob_draw,
            'D': prediction.prob_away_win
        }
        
        # Calculate value for each outcome
        value_analysis = {}
        for outcome, prob in probs.items():
            odd = odds[outcome]
            if odd > 0:
                implied = 1 / odd
                value = prob - implied
                value_pct = (prob / implied - 1) if implied > 0 else 0
                value_analysis[outcome] = {
                    'prob': prob,
                    'odds': odd,
                    'value': value,
                    'value_pct': value_pct
                }
        
        # Step 5: Find best value bet that aligns with prediction
        best_outcome = None
        best_value = 0
        best_score = 0
        
        for outcome, analysis in value_analysis.items():
            # Skip if odds out of range
            if analysis['odds'] < self.MIN_ODDS or analysis['odds'] > self.MAX_ODDS:
                continue
            
            # Skip if value edge too low
            if analysis['value'] < self.MIN_VALUE_EDGE:
                continue
            
            # Calculate quality score
            # Higher confidence + higher value + good odds = better score
            score = (
                probs[outcome] * 0.4 +  # Confidence weight
                analysis['value'] * 0.3 +  # Value weight
                (1 / analysis['odds']) * 0.2 +  # Probability of winning
                prediction.model_agreement * 0.1  # Agreement weight
            )
            
            # Bonus for favorite in good odds range
            if self.FAVORITE_ODDS_RANGE[0] <= analysis['odds'] <= self.FAVORITE_ODDS_RANGE[1]:
                score *= 1.15  # 15% bonus
            
            # Bonus if outcome matches prediction
            if outcome == prediction.predicted_result:
                score *= 1.10  # 10% bonus
            
            if score > best_score:
                best_score = score
                best_outcome = outcome
                best_value = analysis['value']
        
        if best_outcome is None:
            evaluation['reason'] = "No outcome meets quality criteria"
            return evaluation
        
        # Step 6: Determine risk level
        best_odds = value_analysis[best_outcome]['odds']
        best_prob = value_analysis[best_outcome]['prob']
        
        if best_prob >= 0.80 and best_odds <= 2.0:
            risk_level = 'low'
        elif best_prob >= 0.70 and best_odds <= 2.5:
            risk_level = 'medium'
        else:
            risk_level = 'moderate'
        
        # Final evaluation
        evaluation['is_quality_bet'] = True
        evaluation['quality_score'] = best_score
        evaluation['recommended_outcome'] = best_outcome
        evaluation['recommended_odds'] = best_odds
        evaluation['value_edge'] = best_value
        evaluation['risk_level'] = risk_level
        evaluation['value_analysis'] = value_analysis
        evaluation['reason'] = f"Quality bet: {best_outcome} @ {best_odds:.2f} (conf: {best_prob:.1%}, value: {best_value:.1%})"
        
        return evaluation
    
    def select_daily_bets(self, evaluations: List[Dict], bankroll: float,
                         matchday: int = None) -> List[Dict]:
        """
        Select the best bets for a matchday.
        Maximum 2 bets, prioritizing highest quality.
        """
        # Filter quality bets only
        quality_bets = [e for e in evaluations if e['is_quality_bet']]
        
        if not quality_bets:
            logger.info(f"No quality bets found for matchday {matchday}")
            return []
        
        # Sort by quality score (best first)
        quality_bets.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Select top bets (max 2)
        selected = quality_bets[:self.MAX_BETS_PER_MATCHDAY]
        
        # Calculate stakes using conservative Kelly
        decisions = []
        for eval_data in selected:
            # Very conservative Kelly: 10% of full Kelly
            odds = eval_data['recommended_odds']
            prob = eval_data['value_analysis'][eval_data['recommended_outcome']]['prob']
            
            # Full Kelly
            kelly_full = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
            kelly_full = max(0, kelly_full)
            
            # Conservative: 10% of Kelly
            kelly_conservative = kelly_full * 0.10
            
            # Calculate stake
            stake = bankroll * kelly_conservative
            
            # Minimum stake: 1000 Ar, Maximum: 5% of bankroll
            stake = max(1000, min(stake, bankroll * 0.05))
            
            # Round to nearest 100
            stake = round(stake / 100) * 100
            
            # Ensure potential profit is meaningful (at least 500 Ar)
            potential_profit = stake * (odds - 1)
            if potential_profit < 500:
                stake = round(500 / (odds - 1) / 100) * 100
                stake = max(1000, stake)
            
            decision = {
                'match_id': eval_data['match_id'],
                'home_team': eval_data['home_team'],
                'away_team': eval_data['away_team'],
                'outcome': eval_data['recommended_outcome'],
                'outcome_name': {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}[eval_data['recommended_outcome']],
                'odds': odds,
                'model_probability': prob,
                'value_edge': eval_data['value_edge'],
                'confidence': eval_data['confidence'],
                'model_agreement': eval_data['model_agreement'],
                'quality_score': eval_data['quality_score'],
                'risk_level': eval_data['risk_level'],
                'stake': stake,
                'potential_return': stake * odds,
                'potential_profit': stake * (odds - 1),
                'kelly_fraction': kelly_conservative,
                'bankroll_before': bankroll
            }
            
            decisions.append(decision)
            
            # Update bankroll for next bet
            bankroll -= stake
        
        logger.info(f"Selected {len(decisions)} quality bets for matchday {matchday}")
        for d in decisions:
            logger.info(f"  {d['home_team']} vs {d['away_team']}: {d['outcome_name']} @ {d['odds']:.2f} - Stake: {d['stake']} Ar")
        
        return decisions
    
    def calculate_expected_daily_profit(self, decisions: List[Dict]) -> Dict:
        """
        Calculate expected profit for the day based on selected bets.
        """
        if not decisions:
            return {
                'expected_profit': 0,
                'worst_case': 0,
                'best_case': 0,
                'win_probability': 0
            }
        
        total_stake = sum(d['stake'] for d in decisions)
        expected_profit = 0
        worst_case = -total_stake
        best_case = sum(d['potential_profit'] for d in decisions)
        
        # Calculate expected value
        for d in decisions:
            prob = d['model_probability']
            profit_if_win = d['potential_profit']
            loss_if_lose = d['stake']
            
            ev = prob * profit_if_win - (1 - prob) * loss_if_lose
            expected_profit += ev
        
        # Overall win probability (at least one bet wins)
        if len(decisions) == 1:
            win_prob = decisions[0]['model_probability']
        else:
            # Probability all lose
            prob_all_lose = 1
            for d in decisions:
                prob_all_lose *= (1 - d['model_probability'])
            win_prob = 1 - prob_all_lose
        
        return {
            'expected_profit': expected_profit,
            'worst_case': worst_case,
            'best_case': best_case,
            'win_probability': win_prob,
            'total_stake': total_stake
        }
    
    def should_skip_matchday(self, evaluations: List[Dict]) -> Tuple[bool, str]:
        """
        Determine if we should skip betting this matchday entirely.
        Better to skip than to make low-quality bets.
        """
        quality_bets = [e for e in evaluations if e['is_quality_bet']]
        
        if not quality_bets:
            return True, "No quality bets available - skip to preserve bankroll"
        
        # Check if best quality bet is good enough
        best = max(quality_bets, key=lambda x: x['quality_score'])
        
        if best['quality_score'] < 0.5:
            return True, "Best quality score too low - skip to preserve bankroll"
        
        if best['confidence'] < 0.78:
            return True, "Best confidence too low - skip to preserve bankroll"
        
        return False, "Quality bets available"
    
    def get_prediction_summary(self, decisions: List[Dict]) -> str:
        """
        Generate a human-readable summary of predictions.
        """
        if not decisions:
            return "Aucun pari de qualité aujourd'hui - préservation du capital"
        
        lines = ["=== PRÉDICTIONS DU JOUR ===", ""]
        
        for i, d in enumerate(decisions, 1):
            lines.append(f"Par {i}: {d['home_team']} vs {d['away_team']}")
            lines.append(f"  → {d['outcome_name']} @ {d['odds']:.2f}")
            lines.append(f"  → Confiance: {d['confidence']:.1%}")
            lines.append(f"  → Valeur: +{d['value_edge']:.1%}")
            lines.append(f"  → Mise: {d['stake']:.0f} Ar")
            lines.append(f"  → Profit potentiel: +{d['potential_profit']:.0f} Ar")
            lines.append("")
        
        expected = self.calculate_expected_daily_profit(decisions)
        lines.append(f"Espérance de gain: +{expected['expected_profit']:.0f} Ar")
        lines.append(f"Probabilité de gain: {expected['win_probability']:.1%}")
        
        return "\n".join(lines)


class QualityScorer:
    """
    Scores prediction quality based on multiple factors.
    """
    
    @staticmethod
    def score_prediction(prediction: Prediction, match: Match) -> float:
        """
        Calculate overall quality score for a prediction.
        Score from 0 to 1.
        """
        score = 0.0
        
        # Factor 1: Confidence (40% weight)
        confidence_score = prediction.confidence
        score += confidence_score * 0.40
        
        # Factor 2: Model Agreement (25% weight)
        agreement_score = prediction.model_agreement
        score += agreement_score * 0.25
        
        # Factor 3: Value Edge (20% weight)
        best_value = max(
            prediction.value_home or 0,
            prediction.value_draw or 0,
            prediction.value_away or 0
        )
        value_score = min(1.0, best_value * 5)  # Cap at 1.0
        score += value_score * 0.20
        
        # Factor 4: Probability Gap (15% weight)
        # Larger gap = more decisive prediction
        probs = [prediction.prob_home_win, prediction.prob_draw, prediction.prob_away_win]
        probs_sorted = sorted(probs, reverse=True)
        gap = probs_sorted[0] - probs_sorted[1]
        gap_score = min(1.0, gap * 3)
        score += gap_score * 0.15
        
        return score
    
    @staticmethod
    def categorize_quality(score: float) -> str:
        """Categorize prediction quality."""
        if score >= 0.80:
            return "EXCELLENT"
        elif score >= 0.70:
            return "GOOD"
        elif score >= 0.60:
            return "ACCEPTABLE"
        elif score >= 0.50:
            return "MARGINAL"
        else:
            return "POOR"
