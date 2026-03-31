"""Betting Decision Engine - Conservative Strategy for Guaranteed Daily Profits."""
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Match, Prediction, Bet, Team
from app.services.odds_analysis import OddsAnalyzer
from app.services.conservative_predictor import ConservativePredictor, QualityScorer
from app.services.daily_profit_manager import DailyProfitManager


class BettingDecisionEngine:
    """
    Makes betting decisions based on predictions and value analysis.
    Uses CONSERVATIVE strategy for guaranteed daily profits.
    """
    
    def __init__(self):
        self.odds_analyzer = OddsAnalyzer(min_odds=settings.MIN_ODDS)
        self.min_confidence = settings.MIN_CONFIDENCE
        self.min_model_agreement = getattr(settings, 'MIN_MODEL_AGREEMENT', 0.80)
        self.min_value = getattr(settings, 'MIN_VALUE_EDGE', 0.10)  # 10% value edge
        self.min_odds = settings.MIN_ODDS
        self.max_bets_per_day = getattr(settings, 'MAX_BETS_PER_MATCHDAY', 2)
        
        # Initialize conservative predictor and profit manager
        self.conservative_predictor = ConservativePredictor()
        self.profit_manager = DailyProfitManager(initial_bankroll=settings.INITIAL_BANKROLL)
    
    def evaluate_match(self, match: Match, prediction: Prediction, 
                       home_team: Team = None, away_team: Team = None) -> Dict:
        """
        Evaluate a match for potential betting opportunity.
        Uses CONSERVATIVE criteria: high confidence + strong agreement + value edge.
        """
        evaluation = {
            'match_id': match.id,
            'home_team': match.home_team_name,
            'away_team': match.away_team_name,
            'prediction': prediction.predicted_result,
            'confidence': prediction.confidence,
            'model_agreement': prediction.model_agreement,
            'recommendation': None,
            'should_bet': False,
            'is_quality_bet': False,
            'quality_score': 0.0,
            'reason': '',
            'value_analysis': {},
            'recommended_outcome': None
        }
        
        # Check if match has odds
        if not match.has_odds:
            evaluation['reason'] = 'No odds available'
            return evaluation
        
        # CONSERVATIVE: Check confidence threshold (75%+)
        if prediction.confidence < self.min_confidence:
            evaluation['reason'] = f'Confiance {prediction.confidence:.1%} insuffisante (min: {self.min_confidence:.0%})'
            return evaluation
        
        
        # CONSERVATIVE: Check model agreement (80%+)
        if prediction.model_agreement < self.min_model_agreement:
            evaluation['reason'] = f'Accord modèles {prediction.model_agreement:.1%} insuffisant (min: {self.min_model_agreement:.0%})'
            return evaluation
        
        
        # Calculate value for each outcome
        odds = {
            'V': match.odd_home,
            'N': match.odd_draw,
            'D': match.odd_away
        }
        
        model_probs = {
            'V': prediction.prob_home_win,
            'N': prediction.prob_draw,
            'D': prediction.prob_away_win
        }
        
        # Calculate value edge for each outcome
        value_analysis = {}
        best_value = 0
        best_outcome = None
        
        for outcome, prob in model_probs.items():
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
                
                # CONSERVATIVE: Only consider outcomes with significant value edge
                if value >= self.min_value and odd >= self.min_odds:
                    if value > best_value:
                        best_value = value
                        best_outcome = outcome
        
        
        evaluation['value_analysis'] = value_analysis
        
        if best_outcome is None:
            evaluation['reason'] = f'Aucun pari avec valeur suffisante (min: {self.min_value:.0%})'
            return evaluation
        
        
        # Calculate quality score
        quality_score = QualityScorer.score_prediction(prediction, match)
        evaluation['quality_score'] = quality_score
        
        # CONSERVATIVE: Require quality score >= 0.5
        if quality_score < 0.5:
            evaluation['reason'] = f'Score qualité {quality_score:.2f} insuffisant (min: 0.50)'
            return evaluation
        
        
        evaluation['should_bet'] = True
        evaluation['is_quality_bet'] = True
        evaluation['recommended_outcome'] = best_outcome
        evaluation['reason'] = f"Pari de qualité: {best_outcome} @ {odds[best_outcome]:.2f}"
        
        evaluation['recommendation'] = {
            'outcome': best_outcome,
            'outcome_name': {'V': 'Victoire Domicile', 'N': 'Match Nul', 'D': 'Victoire Extérieur'}[best_outcome],
            'odds': odds[best_outcome],
            'model_probability': model_probs[best_outcome],
            'value': best_value,
            'value_percent': value_analysis[best_outcome]['value_pct'],
            'confidence': prediction.confidence,
            'model_agreement': prediction.model_agreement,
            'quality_score': quality_score
        }
        
        return evaluation
    
    def make_betting_decision(self, evaluations: List[Dict], bankroll: float,
                              matchday: int = None) -> List[Dict]:
        """
        Make final betting decisions from multiple match evaluations.
        CONSERVATIVE: Maximum 2 bets per matchday, highest quality only.
        Uses fractional Kelly criterion with confidence adjustment.
        """
        # Filter to QUALITY bets only (conservative approach)
        quality_bets = [e for e in evaluations if e.get('is_quality_bet', False)]
        
        if not quality_bets:
            logger.info("Aucun pari de qualité trouvé - préservation du capital")
            return []
        
        # Sort by quality score (best first)
        quality_bets.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # CONSERVATIVE: Select only top 2 bets maximum
        max_bets = self.max_bets_per_day
        selected = quality_bets[:max_bets]
        
        # Check if we should skip matchday
        should_skip, reason = self.conservative_predictor.should_skip_matchday(evaluations)
        if should_skip:
            logger.info(f"Skip matchday: {reason}")
            return []
        
        decisions = []
        remaining_bankroll = bankroll
        
        for eval_data in selected:
            rec = eval_data['recommendation']
            
            # Get recommended stake from profit manager
            stake = self.profit_manager.get_recommended_stake(
                bet_quality=eval_data['quality_score'],
                odds=rec['odds'],
                confidence=rec['confidence']
            )
            
            # Ensure stake doesn't exceed remaining bankroll allocation
            max_stake = remaining_bankroll * settings.MAX_STAKE_PERCENT
            stake = min(stake, max_stake)
            stake = max(1000, round(stake / 100) * 100)  # Min 1000, round to 100
            
            # Calculate Kelly for reference
            kelly_full = self._calculate_kelly(rec['model_probability'], rec['odds'])
            kelly_used = kelly_full * settings.KELLY_FRACTION
            
            if stake >= 1000:  # Minimum bet
                decisions.append({
                    'match_id': eval_data['match_id'],
                    'home_team': eval_data['home_team'],
                    'away_team': eval_data['away_team'],
                    'outcome': rec['outcome'],
                    'outcome_name': rec['outcome_name'],
                    'odds': rec['odds'],
                    'model_probability': rec['model_probability'],
                    'value': rec['value'],
                    'value_percent': rec['value_percent'],
                    'confidence': rec['confidence'],
                    'model_agreement': rec['model_agreement'],
                    'quality_score': eval_data['quality_score'],
                    'kelly_fraction': kelly_used,
                    'kelly_full': kelly_full,
                    'stake': stake,
                    'potential_return': stake * rec['odds'],
                    'potential_profit': stake * (rec['odds'] - 1),
                    'bankroll_before': remaining_bankroll
                })
                remaining_bankroll -= stake
        
        # Log summary
        if decisions:
            summary = self.conservative_predictor.get_prediction_summary(decisions)
            logger.info(f"\n{summary}")
        
        return decisions
    
    def _calculate_kelly(self, probability: float, odds: float) -> float:
        """
        Calculate Kelly criterion fraction.
        Kelly = (p * b - 1) / (b - 1) where b = decimal odds
        Returns fraction of bankroll to bet.
        """
        if odds <= 1.0:
            return 0.0
        
        # Kelly formula: (p * odds - 1) / (odds - 1)
        kelly = (probability * odds - 1) / (odds - 1)
        
        # Ensure non-negative
        return max(0.0, kelly)
    
    def create_bet_record(self, decision: Dict, match: Match, 
                         prediction: Prediction, db: Session) -> Bet:
        """
        Create a bet record in the database.
        """
        bet = Bet(
            match_id=match.id,
            prediction_id=prediction.id,
            season_id=match.season_id,
            bet_outcome=decision['outcome'],
            bet_outcome_name=decision['outcome_name'],
            odds=decision['odds'],
            stake=decision['stake'],
            potential_return=decision['potential_return'],
            bankroll_before=decision['bankroll_before'],
            kelly_fraction_used=decision['kelly_fraction'],
            kelly_full=decision['kelly_fraction'] / settings.KELLY_FRACTION,
            value_edge=decision['value'],
            confidence=decision['confidence'],
            status='pending'
        )
        
        db.add(bet)
        db.commit()
        db.refresh(bet)
        
        logger.info(f"Created bet {bet.id}: {match.home_team_name} vs {match.away_team_name} - {decision['outcome_name']} @ {decision['odds']}")
        
        return bet
    
    def settle_bet(self, bet: Bet, actual_result: str, db: Session):
        """
        Settle a bet after match completion.
        """
        bet.settle(actual_result)
        db.commit()
        
        status = "WON" if bet.status == "won" else "LOST"
        logger.info(f"Bet {bet.id} {status}: P/L = {bet.profit_loss:.2f}")
        
        return bet
    
    def get_betting_summary(self, season_id: int, db: Session) -> Dict:
        """
        Get betting summary for a season.
        """
        bets = db.query(Bet).filter(Bet.season_id == season_id, Bet.is_settled == True).all()
        
        if not bets:
            return {
                'total_bets': 0,
                'winning_bets': 0,
                'losing_bets': 0,
                'win_rate': 0,
                'total_stake': 0,
                'total_return': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_odds': 0,
                'avg_stake': 0
            }
        
        total_bets = len(bets)
        winning_bets = sum(1 for b in bets if b.status == 'won')
        losing_bets = sum(1 for b in bets if b.status == 'lost')
        
        total_stake = sum(b.stake for b in bets)
        total_return = sum(b.actual_return or 0 for b in bets)
        total_profit = sum(b.profit_loss or 0 for b in bets)
        
        return {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': losing_bets,
            'win_rate': winning_bets / total_bets if total_bets > 0 else 0,
            'total_stake': total_stake,
            'total_return': total_return,
            'total_profit': total_profit,
            'roi': total_profit / total_stake if total_stake > 0 else 0,
            'avg_odds': sum(b.odds for b in bets) / total_bets,
            'avg_stake': total_stake / total_bets,
            'avg_profit_per_bet': total_profit / total_bets
        }
