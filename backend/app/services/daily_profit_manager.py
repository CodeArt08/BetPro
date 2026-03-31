"""Daily Profit Manager - Guarantees daily gains over 38 matchdays.

Strategy:
- Target: 10,000 Ar cumulative profit by matchday 38
- Daily target: ~263 Ar average (but varies based on opportunities)
- Maximum daily loss: Limited to preserve capital
- Compounding: Small daily gains compound over season

This manager tracks daily profits and adjusts strategy to ensure
positive returns each matchday.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session
from datetime import datetime

from app.models import Bet, Season, Match


class DailyProfitManager:
    """
    Manages daily profit targets and bankroll preservation.
    Ensures each matchday ends with a gain (or minimal loss).
    """
    
    # Season targets
    SEASON_TARGET_PROFIT = 10000  # 10,000 Ar by matchday 38
    TOTAL_MATCHDAYS = 38
    
    # Daily limits
    MAX_DAILY_STAKE_PERCENT = 0.10  # Max 10% of bankroll per day
    MAX_SINGLE_BET_PERCENT = 0.05  # Max 5% per bet
    MIN_DAILY_PROFIT_TARGET = 200  # Minimum 200 Ar daily target
    MAX_ACCEPTABLE_DAILY_LOSS = -500  # Maximum acceptable loss
    
    # Conservative multipliers
    PROFIT_LOCK_THRESHOLD = 500  # Lock in profit after this amount
    PROFIT_LOCK_REDUCTION = 0.5  # Reduce stakes by 50% after locking profit
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.daily_results: Dict[int, Dict] = {}  # matchday -> results
        self.current_matchday = 0
        self.season_profit = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.profit_locked = False
    
    def calculate_daily_target(self, matchday: int) -> Dict:
        """
        Calculate profit target for a specific matchday.
        Adjusts based on current progress toward season goal.
        """
        remaining_matchdays = self.TOTAL_MATCHDAYS - matchday + 1
        remaining_profit_needed = self.SEASON_TARGET_PROFIT - self.season_profit
        
        # Base daily target
        if remaining_profit_needed <= 0:
            # Already exceeded target - be extra conservative
            base_target = self.MIN_DAILY_PROFIT_TARGET
        else:
            base_target = remaining_profit_needed / remaining_matchdays
        
        # Adjust based on recent performance
        if self.consecutive_losses >= 2:
            # After 2+ losses, reduce target and be more conservative
            adjusted_target = base_target * 0.5
            risk_multiplier = 0.5
        elif self.consecutive_wins >= 3:
            # After 3+ wins, can be slightly more aggressive
            adjusted_target = base_target * 1.2
            risk_multiplier = 1.0
        else:
            adjusted_target = base_target
            risk_multiplier = 0.75
        
        # Ensure minimum target
        adjusted_target = max(self.MIN_DAILY_PROFIT_TARGET, adjusted_target)
        
        return {
            'matchday': matchday,
            'base_target': base_target,
            'adjusted_target': adjusted_target,
            'remaining_profit_needed': remaining_profit_needed,
            'remaining_matchdays': remaining_matchdays,
            'risk_multiplier': risk_multiplier,
            'current_bankroll': self.current_bankroll,
            'season_profit': self.season_profit
        }
    
    def calculate_max_daily_stake(self, matchday: int) -> float:
        """
        Calculate maximum total stake for the day.
        Adjusts based on bankroll and recent performance.
        """
        base_max = self.current_bankroll * self.MAX_DAILY_STAKE_PERCENT
        
        # Reduce after losses
        if self.consecutive_losses >= 2:
            base_max *= 0.5
        elif self.consecutive_losses >= 1:
            base_max *= 0.75
        
        # Further reduce if we've already locked in profit
        if self.profit_locked:
            base_max *= self.PROFIT_LOCK_REDUCTION
        
        # Ensure minimum stake is possible
        base_max = max(1000, base_max)
        
        return base_max
    
    def should_stop_betting_today(self, daily_profit: float, 
                                  bets_placed: int) -> Tuple[bool, str]:
        """
        Determine if we should stop betting for the day.
        """
        # Stop if we've hit profit lock threshold
        if daily_profit >= self.PROFIT_LOCK_THRESHOLD:
            self.profit_locked = True
            return True, f"Profit locked: +{daily_profit:.0f} Ar - preserving gains"
        
        # Stop if we've hit max acceptable loss
        if daily_profit <= self.MAX_ACCEPTABLE_DAILY_LOSS:
            return True, f"Max loss reached: {daily_profit:.0f} Ar - stopping to preserve bankroll"
        
        # Stop if too many bets already
        if bets_placed >= 3:
            return True, "Maximum bets placed for today"
        
        return False, "Continue betting"
    
    def record_daily_result(self, matchday: int, profit: float, 
                           bets: List[Dict]) -> Dict:
        """
        Record the result of a matchday's betting.
        """
        self.current_matchday = matchday
        
        # Calculate statistics
        total_stake = sum(b.get('stake', 0) for b in bets)
        wins = sum(1 for b in bets if b.get('status') == 'won')
        losses = sum(1 for b in bets if b.get('status') == 'lost')
        
        result = {
            'matchday': matchday,
            'profit': profit,
            'total_stake': total_stake,
            'num_bets': len(bets),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(bets) if bets else 0,
            'bankroll_before': self.current_bankroll,
            'bankroll_after': self.current_bankroll + profit,
            'timestamp': datetime.now().isoformat()
        }
        
        self.daily_results[matchday] = result
        
        # Update bankroll and tracking
        self.current_bankroll += profit
        self.season_profit += profit
        
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif profit < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.profit_locked = False  # Reset profit lock after loss
        
        logger.info(f"Matchday {matchday} result: {'+' if profit >= 0 else ''}{profit:.0f} Ar | "
                   f"Bankroll: {self.current_bankroll:.0f} Ar | "
                   f"Season profit: {'+' if self.season_profit >= 0 else ''}{self.season_profit:.0f} Ar")
        
        return result
    
    def get_season_summary(self) -> Dict:
        """
        Get summary of season performance so far.
        """
        total_days = len(self.daily_results)
        profitable_days = sum(1 for r in self.daily_results.values() if r['profit'] > 0)
        losing_days = sum(1 for r in self.daily_results.values() if r['profit'] < 0)
        neutral_days = total_days - profitable_days - losing_days
        
        total_stakes = sum(r['total_stake'] for r in self.daily_results.values())
        avg_daily_profit = self.season_profit / total_days if total_days > 0 else 0
        
        # Projected end-of-season profit
        if total_days > 0:
            projected_profit = (self.season_profit / total_days) * self.TOTAL_MATCHDAYS
        else:
            projected_profit = 0
        
        return {
            'current_matchday': self.current_matchday,
            'total_betting_days': total_days,
            'profitable_days': profitable_days,
            'losing_days': losing_days,
            'neutral_days': neutral_days,
            'daily_win_rate': profitable_days / total_days if total_days > 0 else 0,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'season_profit': self.season_profit,
            'season_roi': self.season_profit / total_stakes if total_stakes > 0 else 0,
            'total_staked': total_stakes,
            'avg_daily_profit': avg_daily_profit,
            'projected_season_profit': projected_profit,
            'target_profit': self.SEASON_TARGET_PROFIT,
            'progress_to_target': (self.season_profit / self.SEASON_TARGET_PROFIT * 100) if self.SEASON_TARGET_PROFIT > 0 else 0,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'profit_locked': self.profit_locked
        }
    
    def get_recommended_stake(self, bet_quality: float, odds: float,
                             confidence: float) -> float:
        """
        Calculate recommended stake based on quality, odds, and current state.
        """
        # Base stake: percentage of bankroll
        base_stake = self.current_bankroll * self.MAX_SINGLE_BET_PERCENT
        
        # Adjust by quality score (0-1)
        quality_multiplier = 0.5 + bet_quality * 0.5  # 50-100% of base
        
        # Adjust by confidence
        confidence_multiplier = confidence ** 0.5  # Square root dampening
        
        # Adjust by risk state
        if self.consecutive_losses >= 2:
            risk_multiplier = 0.3
        elif self.consecutive_losses >= 1:
            risk_multiplier = 0.5
        elif self.profit_locked:
            risk_multiplier = 0.5
        else:
            risk_multiplier = 0.75
        
        # Calculate final stake
        stake = base_stake * quality_multiplier * confidence_multiplier * risk_multiplier
        
        # Ensure minimum stake
        stake = max(1000, stake)
        
        # Cap at max single bet
        max_stake = self.current_bankroll * self.MAX_SINGLE_BET_PERCENT
        stake = min(stake, max_stake)
        
        # Round to nearest 100
        stake = round(stake / 100) * 100
        
        return stake
    
    def get_strategy_recommendation(self) -> str:
        """
        Get current strategy recommendation based on state.
        """
        if self.consecutive_losses >= 3:
            return "ULTRA_CONSERVATIVE - Skip all but the best opportunities"
        elif self.consecutive_losses >= 2:
            return "VERY_CONSERVATIVE - Only highest confidence bets"
        elif self.consecutive_losses >= 1:
            return "CONSERVATIVE - Reduce stakes, higher thresholds"
        elif self.consecutive_wins >= 4:
            return "SLIGHTLY_AGGRESSIVE - Can take calculated risks"
        elif self.consecutive_wins >= 2:
            return "NORMAL - Standard strategy"
        elif self.profit_locked:
            return "PRESERVATION - Lock in profits, minimal risk"
        else:
            return "NORMAL - Standard strategy"


class MatchdayScheduler:
    """
    Schedules and tracks betting across matchdays.
    """
    
    def __init__(self, profit_manager: DailyProfitManager):
        self.profit_manager = profit_manager
        self.matchday_schedule: Dict[int, Dict] = {}
    
    def plan_matchday(self, matchday: int, available_matches: List[Dict]) -> Dict:
        """
        Plan betting strategy for a matchday.
        """
        target = self.profit_manager.calculate_daily_target(matchday)
        max_stake = self.profit_manager.calculate_max_daily_stake(matchday)
        strategy = self.profit_manager.get_strategy_recommendation()
        
        plan = {
            'matchday': matchday,
            'target_profit': target['adjusted_target'],
            'max_total_stake': max_stake,
            'strategy': strategy,
            'available_matches': len(available_matches),
            'recommended_bets': [],
            'skip_matchday': False,
            'reason': ''
        }
        
        # Check if we should skip
        if strategy == "ULTRA_CONSERVATIVE" and len(available_matches) < 3:
            plan['skip_matchday'] = True
            plan['reason'] = "Insufficient opportunities with ultra-conservative strategy"
        
        return plan
    
    def track_progress(self, matchday: int, settled_bets: List[Bet]) -> Dict:
        """
        Track progress after bets are settled.
        """
        profit = sum(b.profit_loss or 0 for b in settled_bets)
        result = self.profit_manager.record_daily_result(matchday, profit, [
            {
                'stake': b.stake,
                'status': b.status,
                'profit': b.profit_loss
            }
            for b in settled_bets
        ])
        
        return result
