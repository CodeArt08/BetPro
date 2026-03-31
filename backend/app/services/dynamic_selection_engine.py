"""Dynamic Selection Engine - Intelligent bet selection with draw integration.

This system implements:
1. Adjusted probability calculation (ELO, attack/defense, H2H, form)
2. Advanced draw detection
3. Sequence pattern analysis integration
4. Season filter (max 70 bets)
5. Daily dynamic selection (0-5 bets per matchday)
6. Priority-based ranking system
7. Risk management (Kelly 0.25, max 5% bankroll, stop at 3 losses)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session
from datetime import datetime

from app.models import Match, Prediction, Team, Bet, Season
from app.services.realtime_engine import get_engine
from app.services.sequence_analysis import SequencePatternAnalyzer


class DynamicSelectionEngine:
    """
    Intelligent bet selection engine with dynamic daily selection.
    
    Key principles:
    - Ultra-selective: 50-70 bets per season maximum
    - Dynamic per matchday: 0 to 5 bets
    - Intelligent draw integration
    - Sequence pattern awareness
    - Perfect backend/frontend synchronization
    """
    
    # Season limits
    MAX_BETS_PER_SEASON = 70
    MIN_BETS_PER_SEASON = 50
    
    # Daily limits
    MAX_BETS_PER_MATCHDAY = 5
    
    # Selection thresholds - adjusted for realistic confidence levels
    MIN_ODDS = 2.0
    MIN_VALUE = 0.08  # 8% value edge (lowered from 12%)
    MIN_CONFIDENCE = 0.45  # 45% minimum confidence (lowered from 50% to allow N/D)
    MIN_MODEL_AGREEMENT = 0.55  # 55% model agreement (lowered from 60%)
    
    # Draw detection thresholds
    DRAW_PROB_THRESHOLD = 0.25  # Lowered from 0.28 to allow more draws
    ELO_BALANCE_THRESHOLD = 60  # Increased from 50
    H2H_DRAW_RATIO_THRESHOLD = 0.25  # Lowered from 0.30
    MIN_DRAW_ODDS = 2.2  # Lowered from 2.5
    
    # Away win detection thresholds (NOUVEAU)
    AWAY_PROB_THRESHOLD = 0.25  # Minimum prob for away win
    MIN_AWAY_ODDS = 2.0  # Minimum odds for away win
    AWAY_VALUE_BONUS = 0.03  # Bonus for away value
    
    # Risk management
    KELLY_FRACTION = 0.25
    MAX_BANKROLL_PERCENT = 0.05
    MAX_CONSECUTIVE_LOSSES = 3
    
    # Selection reasons
    REASON_STRONG_DRAW = "STRONG_DRAW"
    REASON_AWAY_VALUE = "AWAY_VALUE"  # NOUVEAU
    REASON_HIGH_VALUE = "HIGH_VALUE"
    REASON_MODEL_CONSENSUS = "MODEL_CONSENSUS"
    REASON_SEQUENCE_PATTERN = "SEQUENCE_PATTERN"
    
    def __init__(self):
        self.sequence_analyzer = SequencePatternAnalyzer(sequence_length=5)
        self.consecutive_losses = 0
        self.season_bet_count = 0
        
        # Charger les seuils optimisés depuis le RealTimeEngine
        try:
            engine = get_engine()
            optimized_thresholds = engine.get_optimized_thresholds()
            self.MIN_CONFIDENCE_V = optimized_thresholds.get('V', 0.45)
            self.MIN_CONFIDENCE_N = optimized_thresholds.get('N', 0.35)
            self.MIN_CONFIDENCE_D = optimized_thresholds.get('D', 0.35)
            logger.info(f"Seuils optimisés chargés: V={self.MIN_CONFIDENCE_V}, N={self.MIN_CONFIDENCE_N}, D={self.MIN_CONFIDENCE_D}")
        except Exception as e:
            logger.warning(f"Impossible de charger les seuils optimisés: {e}")
            self.MIN_CONFIDENCE_V = 0.45
            self.MIN_CONFIDENCE_N = 0.35
            self.MIN_CONFIDENCE_D = 0.35
    
    def calculate_adjusted_probabilities(
        self, 
        prediction: Prediction, 
        match: Match,
        home_team: Team,
        away_team: Team
    ) -> Dict[str, float]:
        """
        Calculate adjusted probabilities based on multiple factors.
        
        Factors:
        - ELO balance
        - Attack/defense balance
        - H2H weighted
        - Home/away form
        """
        base_probs = {
            'V': prediction.prob_home_win,
            'N': prediction.prob_draw,
            'D': prediction.prob_away_win
        }
        
        adjustments = {'V': 0.0, 'N': 0.0, 'D': 0.0}
        
        # 1. ELO Balance adjustment
        elo_diff = abs(home_team.elo_rating - away_team.elo_rating)
        if elo_diff < self.ELO_BALANCE_THRESHOLD:
            # Teams are balanced - increase draw probability
            elo_balance_factor = 1 - (elo_diff / self.ELO_BALANCE_THRESHOLD)
            adjustments['N'] += 0.05 * elo_balance_factor
        else:
            # One team stronger - adjust toward that team
            if home_team.elo_rating > away_team.elo_rating:
                adjustments['V'] += min(0.05, elo_diff / 500)
            else:
                adjustments['D'] += min(0.05, elo_diff / 500)
        
        # 2. Attack/Defense balance
        home_attack = home_team.attack_strength or 1.0
        home_defense = home_team.defense_strength or 1.0
        away_attack = away_team.attack_strength or 1.0
        away_defense = away_team.defense_strength or 1.0
        
        # Expected goals balance
        home_expected = home_attack * away_defense
        away_expected = away_attack * home_defense
        goal_balance = abs(home_expected - away_expected)
        
        if goal_balance < 0.3:
            # Similar expected goals - draw more likely
            adjustments['N'] += 0.03 * (1 - goal_balance / 0.3)
        
        # 3. H2H weighted adjustment
        h2h_draw_ratio = getattr(match, 'h2h_draw_ratio', 0.25)
        if h2h_draw_ratio >= self.H2H_DRAW_RATIO_THRESHOLD:
            adjustments['N'] += 0.02 * (h2h_draw_ratio - 0.25)
        
        # 4. Home/Away form balance
        # Parse form string to get form rating
        def parse_form(form_str):
            if not form_str:
                return 0.5
            # Form string like "WDLWW" - count wins
            wins = form_str.count('V') + form_str.count('W')
            draws = form_str.count('N') + form_str.count('D')
            total = len(form_str) if form_str else 1
            return (wins * 1.0 + draws * 0.5) / total if total > 0 else 0.5
        
        home_form = parse_form(home_team.current_form) if home_team.current_form else 0.5
        away_form = parse_form(away_team.current_form) if away_team.current_form else 0.5
        form_diff = abs(home_form - away_form)
        
        if form_diff < 0.15:
            # Similar form - draw more likely
            adjustments['N'] += 0.02 * (1 - form_diff / 0.15)
        
        # Apply adjustments
        adjusted = {}
        for outcome in ['V', 'N', 'D']:
            adjusted[outcome] = base_probs[outcome] + adjustments[outcome]
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def detect_strong_draw(
        self,
        adjusted_probs: Dict[str, float],
        match: Match,
        home_team: Team,
        away_team: Team,
        sequence_compatible: bool = False
    ) -> Tuple[bool, float]:
        """
        Advanced draw detection with multiple conditions.
        
        Conditions:
        - prob_draw >= 0.28
        - |elo_home - elo_away| < 50
        - h2h_draw_ratio >= 0.30
        - odds_draw >= 2.5
        - sequence compatible → bonus +5%
        
        Returns (is_strong_draw, confidence).
        """
        conditions_met = 0
        confidence = 0.0
        
        # Condition 1: Draw probability
        if adjusted_probs['N'] >= self.DRAW_PROB_THRESHOLD:
            conditions_met += 1
            confidence += 0.25
        
        # Condition 2: ELO balance
        elo_diff = abs(home_team.elo_rating - away_team.elo_rating)
        if elo_diff < self.ELO_BALANCE_THRESHOLD:
            conditions_met += 1
            confidence += 0.25
        
        # Condition 3: H2H draw ratio
        h2h_draw_ratio = getattr(match, 'h2h_draw_ratio', 0.0)
        if h2h_draw_ratio >= self.H2H_DRAW_RATIO_THRESHOLD:
            conditions_met += 1
            confidence += 0.20
        
        # Condition 4: Draw odds
        if match.odd_draw and match.odd_draw >= self.MIN_DRAW_ODDS:
            conditions_met += 1
            confidence += 0.15
        
        # Condition 5: Sequence compatibility bonus
        if sequence_compatible:
            conditions_met += 1
            confidence += 0.15
        
        # Need at least 4 conditions for strong draw
        is_strong_draw = conditions_met >= 3 and confidence >= 0.50  # Lowered from 4/0.60
        
        return is_strong_draw, confidence
    
    def detect_away_win_opportunity(
        self,
        adjusted_probs: Dict[str, float],
        match: Match,
        home_team: Team,
        away_team: Team
    ) -> Tuple[bool, float]:
        """
        Détecte les opportunités de victoire extérieur (D).
        
        Conditions:
        - prob_away >= 0.25
        - away_team.elo > home_team.elo (ou proche)
        - odds_away >= 2.0
        - value positif
        
        Returns (is_away_opportunity, confidence).
        """
        conditions_met = 0
        confidence = 0.0
        
        # Condition 1: Away probability
        if adjusted_probs['D'] >= self.AWAY_PROB_THRESHOLD:
            conditions_met += 1
            confidence += 0.25
        
        # Condition 2: ELO advantage or close
        elo_diff = away_team.elo_rating - home_team.elo_rating
        if elo_diff > 0:
            conditions_met += 1
            confidence += 0.25
        elif elo_diff > -50:  # Close match
            conditions_met += 1
            confidence += 0.15
        
        # Condition 3: Away odds value
        if match.odd_away and match.odd_away >= self.MIN_AWAY_ODDS:
            conditions_met += 1
            confidence += 0.20
        
        # Condition 4: Away form better than home
        def parse_form(form_str):
            if not form_str:
                return 0.5
            wins = form_str.count('V') + form_str.count('W')
            total = len(form_str) if form_str else 1
            return wins / total if total > 0 else 0.5
        
        away_form = parse_form(away_team.current_form) if away_team.current_form else 0.5
        home_form = parse_form(home_team.current_form) if home_team.current_form else 0.5
        if away_form > home_form:
            conditions_met += 1
            confidence += 0.15
        
        # Need at least 3 conditions
        is_away_opportunity = conditions_met >= 3 and confidence >= 0.45
        
        return is_away_opportunity, confidence
    
    def analyze_sequence_pattern(
        self,
        db: Session,
        match: Match,
        current_line_sequence: str
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Analyze sequence pattern for conditional probability boost.
        
        Returns (has_strong_pattern, boosted_probabilities).
        """
        self.sequence_analyzer.load_line_sequences(db)
        
        line_pos = match.line_position or 1
        predicted, confidence, seq_probs = self.sequence_analyzer.predict_line_next_result(
            line_pos, 
            current_line_sequence
        )
        
        # Check if pattern is strong (confidence > 0.45)
        has_strong_pattern = confidence > 0.45
        
        # Calculate boost (5-12% based on confidence)
        boost = 0.0
        if has_strong_pattern:
            boost = min(0.12, 0.05 + (confidence - 0.45) * 0.35)
        
        boosted = seq_probs.copy()
        if predicted in boosted:
            boosted[predicted] = min(0.85, boosted[predicted] + boost)
            # Normalize
            total = sum(boosted.values())
            if total > 0:
                boosted = {k: v / total for k, v in boosted.items()}
        
        return has_strong_pattern, boosted
    
    def calculate_selection_score(
        self,
        prediction: Prediction,
        adjusted_probs: Dict[str, float],
        value: float,
        match: Match
    ) -> float:
        """
        Calculate selection score for ranking.
        
        score = (value * 0.4) + (confidence * 0.3) + (model_agreement * 0.2) + (probability_strength * 0.1)
        """
        value_score = min(1.0, value * 3)  # Normalize value
        confidence_score = prediction.confidence
        agreement_score = prediction.model_agreement
        prob_strength = max(adjusted_probs.values())
        
        score = (
            value_score * 0.4 +
            confidence_score * 0.3 +
            agreement_score * 0.2 +
            prob_strength * 0.1
        )
        
        return score
    
    def check_season_limit(self, db: Session, season_id: int) -> Tuple[bool, int]:
        """
        Check if season bet limit is reached.
        
        Returns (can_bet, current_count).
        """
        settled_bets = db.query(Bet).filter(
            Bet.season_id == season_id,
            Bet.is_settled == True
        ).count()
        
        pending_bets = db.query(Bet).filter(
            Bet.season_id == season_id,
            Bet.is_settled == False
        ).count()
        
        total_bets = settled_bets + pending_bets
        
        return total_bets < self.MAX_BETS_PER_SEASON, total_bets
    
    def check_risk_management(self, db: Session, season_id: int) -> Tuple[bool, str]:
        """
        Check risk management rules.
        
        Returns (can_proceed, reason).
        """
        # Check consecutive losses
        recent_bets = db.query(Bet).filter(
            Bet.season_id == season_id,
            Bet.is_settled == True
        ).order_by(Bet.settled_at.desc()).limit(5).all()
        
        consecutive_losses = 0
        for bet in recent_bets:
            if bet.status == 'lost':
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            return False, f"Stop: {consecutive_losses} consecutive losses"
        
        return True, "Risk check passed"
    
    def select_bets_for_matchday(
        self,
        db: Session,
        matchday: int,
        season_id: int,
        bankroll: float
    ) -> List[Dict]:
        """
        Main selection method for a matchday.
        
        Process:
        1. Get all matches with predictions
        2. Calculate adjusted probabilities
        3. Detect strong draws
        4. Analyze sequence patterns
        5. Calculate selection scores
        6. Filter by thresholds
        7. Rank and select top 0-5
        8. Update predictions with selection data
        """
        # Check season limit
        can_bet, season_count = self.check_season_limit(db, season_id)
        if not can_bet:
            logger.info(f"Season limit reached: {season_count}/{self.MAX_BETS_PER_SEASON}")
            return []
        
        # Check risk management
        can_proceed, risk_reason = self.check_risk_management(db, season_id)
        if not can_proceed:
            logger.warning(f"Risk management stop: {risk_reason}")
            return []
        
        # Get matches for matchday
        matches = db.query(Match).filter(
            Match.matchday == matchday,
            Match.season_id == season_id,
            Match.has_odds == True
        ).all()
        
        if not matches:
            logger.info(f"No matches with odds for matchday {matchday}")
            return []
        
        candidates = []
        
        for match in matches:
            prediction = db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).first()
            
            if not prediction:
                continue
            
            home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
            away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
            
            if not home_team or not away_team:
                continue
            
            # Step 1: Adjusted probabilities
            adjusted_probs = self.calculate_adjusted_probabilities(
                prediction, match, home_team, away_team
            )
            
            # Step 2: Calculate value for each outcome
            odds = {'V': match.odd_home, 'N': match.odd_draw, 'D': match.odd_away}
            value_analysis = {}
            best_value = 0
            best_outcome = None
            
            for outcome in ['V', 'N', 'D']:
                odd = odds.get(outcome, 0)
                if odd and odd > 0:
                    implied = 1 / odd
                    value = adjusted_probs[outcome] - implied
                    value_analysis[outcome] = {
                        'prob': adjusted_probs[outcome],
                        'odds': odd,
                        'value': value,
                        'implied': implied
                    }
                    if value > best_value:
                        best_value = value
                        best_outcome = outcome
            
            # Step 3: Check thresholds
            if best_outcome is None:
                continue
            
            if odds.get(best_outcome, 0) < self.MIN_ODDS:
                continue
            
            if best_value < self.MIN_VALUE:
                continue
            
            # Step 3b: Apply optimized confidence thresholds by outcome type
            min_confidence = self.MIN_CONFIDENCE
            if best_outcome == 'V':
                min_confidence = self.MIN_CONFIDENCE_V
            elif best_outcome == 'N':
                min_confidence = self.MIN_CONFIDENCE_N
            elif best_outcome == 'D':
                min_confidence = self.MIN_CONFIDENCE_D
            
            if prediction.confidence < min_confidence:
                continue
            
            if prediction.model_agreement < self.MIN_MODEL_AGREEMENT:
                continue
            
            # Step 4: Detect strong draw
            is_strong_draw, draw_confidence = self.detect_strong_draw(
                adjusted_probs, match, home_team, away_team
            )
            
            # Step 4b: Detect away win opportunity
            is_away_opportunity, away_confidence = self.detect_away_win_opportunity(
                adjusted_probs, match, home_team, away_team
            )
            
            # Step 5: Analyze sequence pattern
            line_sequence = self._get_line_sequence(db, match.line_position or 1)
            has_pattern, boosted_probs = self.analyze_sequence_pattern(
                db, match, line_sequence
            )
            
            # Step 6: Calculate selection score
            score = self.calculate_selection_score(
                prediction, adjusted_probs, best_value, match
            )
            
            # Determine selection reason
            reason = self._determine_reason(
                is_strong_draw, is_away_opportunity, best_value, prediction, has_pattern, best_outcome
            )
            
            candidates.append({
                'match_id': match.id,
                'prediction_id': prediction.id,
                'home_team': match.home_team_name,
                'away_team': match.away_team_name,
                'matchday': matchday,
                'outcome': best_outcome,
                'outcome_name': {'V': 'Victoire Domicile', 'N': 'Match Nul', 'D': 'Victoire Extérieur'}[best_outcome],
                'odds': odds[best_outcome],
                'adjusted_prob': adjusted_probs[best_outcome],
                'value': best_value,
                'confidence': prediction.confidence,
                'model_agreement': prediction.model_agreement,
                'score': score,
                'reason': reason,
                'is_strong_draw': is_strong_draw,
                'has_sequence_pattern': has_pattern,
                'value_analysis': value_analysis
            })
        
        if not candidates:
            logger.info(f"No candidates for matchday {matchday}")
            return []
        
        # Step 7: Sort by priority and score
        # Priority: STRONG_DRAW > AWAY_VALUE > HIGH_VALUE > MODEL_CONSENSUS > SEQUENCE_PATTERN
        priority_order = {
            self.REASON_STRONG_DRAW: 0,
            self.REASON_AWAY_VALUE: 1,
            self.REASON_HIGH_VALUE: 2,
            self.REASON_MODEL_CONSENSUS: 3,
            self.REASON_SEQUENCE_PATTERN: 4
        }
        
        candidates.sort(key=lambda x: (
            priority_order.get(x['reason'], 4),
            -x['score']
        ))
        
        # Step 8: Select top candidates (max 5)
        # Only select if score is high enough
        min_score_threshold = 0.55
        selected = [c for c in candidates if c['score'] >= min_score_threshold]
        selected = selected[:self.MAX_BETS_PER_MATCHDAY]
        
        # If no strong candidates, select 0 or 1
        if not selected:
            if candidates and candidates[0]['score'] >= 0.50:
                selected = [candidates[0]]
            else:
                logger.info(f"No strong candidates for matchday {matchday} - skip")
                return []
        
        # Step 9: Calculate stakes and update predictions
        decisions = []
        for rank, candidate in enumerate(selected, 1):
            # Calculate stake using fractional Kelly
            stake = self._calculate_stake(
                bankroll, 
                candidate['adjusted_prob'],
                candidate['odds']
            )
            
            decision = {
                **candidate,
                'selection_rank': rank,
                'stake': stake,
                'potential_return': stake * candidate['odds'],
                'potential_profit': stake * (candidate['odds'] - 1),
                'kelly_fraction': self.KELLY_FRACTION
            }
            
            decisions.append(decision)
            
            # Update prediction with selection data
            prediction = db.query(Prediction).filter(
                Prediction.id == candidate['prediction_id']
            ).first()
            
            if prediction:
                prediction.is_selected_for_bet = True
                prediction.selection_rank = rank
                prediction.selection_reason = candidate['reason']
        
        db.commit()
        
        logger.info(f"Selected {len(decisions)} bets for matchday {matchday}")
        for d in decisions:
            logger.info(f"  #{d['selection_rank']}: {d['home_team']} vs {d['away_team']} - {d['outcome_name']} @ {d['odds']:.2f} ({d['reason']})")
        
        return decisions
    
    def _determine_reason(
        self,
        is_strong_draw: bool,
        is_away_opportunity: bool,
        best_value: float,
        prediction: Prediction,
        has_pattern: bool,
        outcome: str
    ) -> str:
        """Determine selection reason based on factors."""
        if is_strong_draw and outcome == 'N':
            return self.REASON_STRONG_DRAW
        
        if is_away_opportunity and outcome == 'D':
            return self.REASON_AWAY_VALUE
        
        if best_value >= 0.20:
            return self.REASON_HIGH_VALUE
        
        if prediction.model_agreement >= 0.85:
            return self.REASON_MODEL_CONSENSUS
        
        if has_pattern:
            return self.REASON_SEQUENCE_PATTERN
        
        # Default to high value if above threshold
        if best_value >= self.MIN_VALUE:
            return self.REASON_HIGH_VALUE
        
        return self.REASON_MODEL_CONSENSUS
    
    def _calculate_stake(
        self,
        bankroll: float,
        probability: float,
        odds: float
    ) -> float:
        """Calculate stake using fractional Kelly criterion."""
        if odds <= 1.0:
            return 0.0
        
        # Full Kelly
        kelly_full = (probability * odds - 1) / (odds - 1)
        kelly_full = max(0.0, kelly_full)
        
        # Fractional Kelly (0.25)
        kelly_fraction = kelly_full * self.KELLY_FRACTION
        
        # Calculate stake
        stake = bankroll * kelly_fraction
        
        # Cap at max bankroll percent
        max_stake = bankroll * self.MAX_BANKROLL_PERCENT
        stake = min(stake, max_stake)
        
        # Minimum stake
        stake = max(1000, stake)
        
        # Round to nearest 100
        stake = round(stake / 100) * 100
        
        return stake
    
    def _get_line_sequence(self, db: Session, line_position: int) -> str:
        """Get current sequence for a line position."""
        self.sequence_analyzer.load_line_sequences(db)
        
        if line_position in self.sequence_analyzer.line_sequences:
            seq = self.sequence_analyzer.line_sequences[line_position]
            return ''.join(seq[-5:])  # Last 5 results
        
        return ""
    
    def update_after_result(
        self,
        db: Session,
        bet: Bet,
        actual_result: str
    ) -> Dict:
        """
        Update system after bet result.
        
        - Update accuracy tracking
        - Recalculate weights
        - Adjust future selection
        """
        # Settle the bet
        bet.settle(actual_result)
        db.commit()
        
        # Update consecutive losses counter
        if bet.status == 'lost':
            self.consecutive_losses += 1
            logger.warning(f"Bet lost. Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"Bet won! Profit: {bet.profit_loss}")
        
        # Get updated season stats
        season_bets = db.query(Bet).filter(
            Bet.season_id == bet.season_id,
            Bet.is_settled == True
        ).all()
        
        wins = sum(1 for b in season_bets if b.status == 'won')
        total = len(season_bets)
        
        return {
            'bet_result': bet.status,
            'profit_loss': bet.profit_loss,
            'consecutive_losses': self.consecutive_losses,
            'season_win_rate': wins / total if total > 0 else 0,
            'season_total_bets': total
        }
    
    def get_selection_summary(self, decisions: List[Dict]) -> str:
        """Generate human-readable summary of selections."""
        if not decisions:
            return "Aucune sélection pour cette journée - préservation du capital"
        
        lines = ["=== SÉLECTIONS DU JOUR ===", ""]
        
        for d in decisions:
            lines.append(f"#{d['selection_rank']}: {d['home_team']} vs {d['away_team']}")
            lines.append(f"  → {d['outcome_name']} @ {d['odds']:.2f}")
            lines.append(f"  → Confiance: {d['confidence']:.1%}")
            lines.append(f"  → Valeur: +{d['value']:.1%}")
            lines.append(f"  → Score: {d['score']:.2f}")
            lines.append(f"  → Raison: {d['reason']}")
            lines.append(f"  → Mise: {d['stake']:.0f} Ar")
            lines.append(f"  → Profit potentiel: +{d['potential_profit']:.0f} Ar")
            lines.append("")
        
        total_stake = sum(d['stake'] for d in decisions)
        total_potential = sum(d['potential_profit'] for d in decisions)
        
        lines.append(f"Total misé: {total_stake:.0f} Ar")
        lines.append(f"Profit potentiel: +{total_potential:.0f} Ar")
        
        return "\n".join(lines)


class SeasonBetTracker:
    """Tracks season betting limits and statistics."""
    
    def __init__(self, max_bets: int = 70):
        self.max_bets = max_bets
        self.bets_placed = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
    
    def can_place_bet(self) -> bool:
        """Check if more bets can be placed."""
        return self.bets_placed < self.max_bets
    
    def record_bet(self, is_win: bool, profit: float):
        """Record a bet result."""
        self.bets_placed += 1
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        self.total_profit += profit
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'bets_placed': self.bets_placed,
            'bets_remaining': self.max_bets - self.bets_placed,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / self.bets_placed if self.bets_placed > 0 else 0,
            'total_profit': self.total_profit,
            'roi': self.total_profit / (self.bets_placed * 1000) if self.bets_placed > 0 else 0
        }
