"""Bankroll Management module."""
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
import json
from pathlib import Path

from app.core.config import settings
from app.models import Bet


class BankrollManager:
    """
    Manages bankroll and stake calculations.
    """
    
    def __init__(self, initial_bankroll: float = None):
        self.initial_bankroll = initial_bankroll or settings.INITIAL_BANKROLL
        self.current_bankroll = self.initial_bankroll
        self.max_stake_percent = settings.MAX_STAKE_PERCENT
        self.kelly_fraction = settings.KELLY_FRACTION
        
        # History tracking
        self.history: List[Dict] = []
        self.daily_snapshots: Dict[str, float] = {}
        
        # Load saved state
        self._load_state()
    
    def _load_state(self):
        """Load bankroll state from file."""
        state_file = settings.DATA_DIR / "bankroll_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.current_bankroll = data.get('current_bankroll', self.initial_bankroll)
                    self.history = data.get('history', [])
                    logger.info(f"Loaded bankroll state: {self.current_bankroll:.2f}")
            except Exception as e:
                logger.warning(f"Could not load bankroll state: {e}")
    
    def _save_state(self):
        """Save bankroll state to file."""
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        state_file = settings.DATA_DIR / "bankroll_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    'current_bankroll': self.current_bankroll,
                    'initial_bankroll': self.initial_bankroll,
                    'history': self.history[-1000:]  # Keep last 1000 records
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save bankroll state: {e}")
    
    def get_current_bankroll(self) -> float:
        """Get current bankroll."""
        return self.current_bankroll
    
    def calculate_stake(self, probability: float, odds: float, 
                       method: str = 'kelly') -> float:
        """
        Calculate stake based on method.
        """
        if odds <= 1.0:
            return 0.0
        
        if method == 'kelly':
            # Kelly criterion
            b = odds
            p = probability
            
            kelly = (p * b - 1) / (b - 1)
            kelly = max(0, kelly) * self.kelly_fraction
            
            stake = kelly * self.current_bankroll
            
        elif method == 'flat':
            # Flat staking
            stake = self.current_bankroll * self.max_stake_percent
            
        elif method == 'percentage':
            # Fixed percentage
            stake = self.current_bankroll * self.max_stake_percent
            
        else:
            stake = self.current_bankroll * self.max_stake_percent
        
        # Cap at max stake
        max_stake = self.current_bankroll * self.max_stake_percent
        stake = min(stake, max_stake)
        
        # Minimum stake
        stake = max(stake, 0)
        
        return round(stake, 2)
    
    def place_bet(self, stake: float) -> bool:
        """
        Record a bet placement (deduct from bankroll).
        """
        if stake > self.current_bankroll:
            logger.warning(f"Insufficient bankroll: {stake:.2f} > {self.current_bankroll:.2f}")
            return False
        
        if stake <= 0:
            return False
        
        self.current_bankroll -= stake
        
        self.history.append({
            'type': 'bet_placed',
            'amount': stake,
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self._save_state()
        
        return True
    
    def settle_bet(self, profit_loss: float):
        """
        Record bet settlement (add/subtract from bankroll).
        """
        self.current_bankroll += profit_loss
        
        self.history.append({
            'type': 'bet_settled',
            'profit_loss': profit_loss,
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self._save_state()
        
        logger.info(f"Bankroll updated: {self.current_bankroll:.2f} (P/L: {profit_loss:.2f})")
    
    def deposit(self, amount: float):
        """Add funds to bankroll."""
        self.current_bankroll += amount
        
        self.history.append({
            'type': 'deposit',
            'amount': amount,
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self._save_state()
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw funds from bankroll."""
        if amount > self.current_bankroll:
            return False
        
        self.current_bankroll -= amount
        
        self.history.append({
            'type': 'withdrawal',
            'amount': amount,
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self._save_state()
        return True
    
    def take_snapshot(self, date: str = None):
        """Take a daily snapshot of bankroll."""
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        
        self.daily_snapshots[date] = self.current_bankroll
    
    def get_statistics(self) -> Dict:
        """Get bankroll statistics."""
        profit = self.current_bankroll - self.initial_bankroll
        roi = profit / self.initial_bankroll if self.initial_bankroll > 0 else 0
        
        # Calculate drawdown
        peak = self.initial_bankroll
        max_drawdown = 0
        current_drawdown = 0
        
        for record in self.history:
            if record['type'] == 'bet_settled':
                bal = record['bankroll_after']
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                current_drawdown = (peak - self.current_bankroll) / peak if peak > 0 else 0
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'profit': profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'total_bets': sum(1 for r in self.history if r['type'] == 'bet_placed'),
            'total_settled': sum(1 for r in self.history if r['type'] == 'bet_settled')
        }
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get bankroll history."""
        return self.history[-limit:]
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics."""
        if not self.history:
            return {'var_95': 0, 'expected_shortfall': 0}
        
        # Get profit/loss history
        pl_history = [r['profit_loss'] for r in self.history if r['type'] == 'bet_settled']
        
        if not pl_history:
            return {'var_95': 0, 'expected_shortfall': 0}
        
        import numpy as np
        
        pl_array = np.array(pl_history)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(pl_array, 5)
        
        # Expected Shortfall (Conditional VaR)
        es_mask = pl_array <= var_95
        expected_shortfall = np.mean(pl_array[es_mask]) if np.any(es_mask) else 0
        
        return {
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'std_pl': np.std(pl_array),
            'mean_pl': np.mean(pl_array),
            'win_rate': np.sum(pl_array > 0) / len(pl_array)
        }
    
    def reset(self, new_bankroll: float = None):
        """Reset bankroll to initial or specified amount."""
        self.current_bankroll = new_bankroll or self.initial_bankroll
        self.history = [{
            'type': 'reset',
            'bankroll_after': self.current_bankroll,
            'timestamp': datetime.utcnow().isoformat()
        }]
        self._save_state()
        logger.info(f"Bankroll reset to {self.current_bankroll:.2f}")
