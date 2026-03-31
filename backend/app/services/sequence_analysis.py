"""Sequence Pattern Analysis using Markov Chains and HMM."""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sqlalchemy.orm import Session

from app.models import Match, Team


class SequencePatternAnalyzer:
    """
    Analyzes result sequences using Markov chains.
    """
    
    def __init__(self, sequence_length: int = 3):
        self.sequence_length = sequence_length
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.team_sequences: Dict[int, List[str]] = {}
        
        # States: V (home win), N (draw), D (away win) - from team perspective
        # When team plays at home: V=win, N=draw, D=loss
        # When team plays away: V=loss, N=draw, D=win
        self.states = ['V', 'N', 'D']
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
    
    def build_transition_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        Build a transition probability matrix from sequences.
        """
        n_states = len(self.states)
        transition_counts = np.zeros((n_states, n_states))
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                from_state = seq[i]
                to_state = seq[i + 1]
                if from_state in self.state_to_idx and to_state in self.state_to_idx:
                    transition_counts[self.state_to_idx[from_state], self.state_to_idx[to_state]] += 1
        
        # Normalize to probabilities
        transition_matrix = np.zeros_like(transition_counts, dtype=float)
        for i in range(n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
            else:
                transition_matrix[i] = 1.0 / n_states  # Uniform if no data
        
        return transition_matrix
    
    def build_higher_order_transition_matrix(self, sequences: List[str], order: int = 2) -> Dict[str, np.ndarray]:
        """
        Build higher-order Markov transition probabilities.
        P(next_state | previous_n_states)
        """
        transitions = defaultdict(lambda: np.zeros(len(self.states)))
        
        for seq in sequences:
            for i in range(order, len(seq)):
                context = tuple(seq[i-order:i])  # Previous n states as tuple (hashable)
                next_state = seq[i]
                
                if next_state in self.state_to_idx:
                    transitions[context][self.state_to_idx[next_state]] += 1
        
        # Normalize
        probabilities = {}
        for context, counts in transitions.items():
            total = counts.sum()
            if total > 0:
                probabilities[context] = counts / total
            else:
                probabilities[context] = np.ones(len(self.states)) / len(self.states)
        
        return probabilities
    
    def load_team_sequences(self, db: Session):
        """Load all team result sequences from database."""
        teams = db.query(Team).all()
        
        for team in teams:
            # Get all matches for this team
            home_matches = db.query(Match).filter(
                Match.home_team_id == team.id,
                Match.is_completed == True
            ).order_by(Match.matchday).all()
            
            away_matches = db.query(Match).filter(
                Match.away_team_id == team.id,
                Match.is_completed == True
            ).order_by(Match.matchday).all()
            
            # Combine and sort by matchday
            all_matches = list(home_matches) + list(away_matches)
            all_matches.sort(key=lambda m: m.matchday)
            
            # Build sequence from team perspective
            sequence = []
            for match in all_matches:
                if match.home_team_id == team.id:
                    # Home game: V=win, N=draw, D=loss
                    if match.result:
                        sequence.append(match.result)
                else:
                    # Away game: D=win, N=draw, V=loss (inverted)
                    if match.result:
                        inverted = {'V': 'D', 'N': 'N', 'D': 'V'}
                        sequence.append(inverted.get(match.result, 'N'))
            
            self.team_sequences[team.id] = sequence
        
        logger.info(f"Loaded sequences for {len(self.team_sequences)} teams")
    
    def get_sequence_probability(self, team_id: int, context: str) -> Dict[str, float]:
        """
        Get probability distribution for next result given context.
        """
        if team_id not in self.team_sequences:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        sequence = self.team_sequences[team_id]
        
        # Find all occurrences of context in sequence
        matches = []
        for i in range(len(sequence) - len(context)):
            if sequence[i:i+len(context)] == context:
                if i + len(context) < len(sequence):
                    matches.append(sequence[i + len(context)])
        
        if not matches:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        # Calculate probabilities
        counts = {'V': 0, 'N': 0, 'D': 0}
        for m in matches:
            counts[m] = counts.get(m, 0) + 1
        
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
    
    def analyze_patterns(self, db: Session) -> Dict:
        """
        Analyze patterns across all teams and seasons.
        """
        self.load_team_sequences(db)
        
        # Build global transition matrix
        all_sequences = list(self.team_sequences.values())
        global_transitions = self.build_transition_matrix(all_sequences)
        
        # Build higher-order transitions
        higher_order = self.build_higher_order_transition_matrix(all_sequences, order=2)
        
        # Analyze common patterns
        pattern_counts = defaultdict(int)
        for seq in all_sequences:
            for i in range(len(seq) - 2):
                pattern = tuple(seq[i:i+3])  # Convert to tuple for hashable key
                pattern_counts[pattern] += 1
        
        # Calculate pattern probabilities
        total_patterns = sum(pattern_counts.values())
        pattern_probs = {k: v / total_patterns for k, v in pattern_counts.items()}
        
        return {
            'global_transitions': global_transitions,
            'higher_order_transitions': higher_order,
            'pattern_probabilities': dict(sorted(pattern_probs.items(), key=lambda x: -x[1])[:20]),
            'total_sequences': len(all_sequences)
        }
    
    def predict_next_result(self, team_id: int, current_sequence: str) -> Tuple[str, float]:
        """
        Predict the next result for a team given their current sequence.
        Returns (predicted_result, confidence).
        """
        # Get context (last n results)
        context = current_sequence[-self.sequence_length:] if len(current_sequence) >= self.sequence_length else current_sequence
        
        probs = self.get_sequence_probability(team_id, context)
        
        # Predict most likely outcome
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]
        
        return predicted, confidence
    
    def update_sequence(self, team_id: int, result: str):
        """Update team sequence with new result."""
        if team_id not in self.team_sequences:
            self.team_sequences[team_id] = []
        self.team_sequences[team_id].append(result)
    
    def get_team_form_analysis(self, team_id: int) -> Dict:
        """
        Analyze a team's recent form patterns.
        """
        if team_id not in self.team_sequences:
            return {'form': '', 'patterns': {}}
        
        sequence = self.team_sequences[team_id]
        recent = sequence[-10:] if len(sequence) >= 10 else sequence
        
        # Count patterns in recent form
        patterns = defaultdict(int)
        for i in range(len(recent) - 1):
            pattern = recent[i:i+2]
            patterns[pattern] += 1
        
        return {
            'form': ''.join(recent[-5:]),
            'full_sequence': ''.join(recent),
            'patterns': dict(patterns),
            'win_rate': recent.count('V') / len(recent) if recent else 0,
            'draw_rate': recent.count('N') / len(recent) if recent else 0,
            'loss_rate': recent.count('D') / len(recent) if recent else 0
        }
    
    # ========================================
    # LINE POSITION SEQUENCE ANALYSIS
    # ========================================
    
    def load_line_sequences(self, db: Session):
        """Load result sequences for each line position from database."""
        self.line_sequences: Dict[int, List[str]] = {}
        
        # Get all completed matches ordered by matchday
        matches = db.query(Match).filter(
            Match.is_completed == True
        ).order_by(Match.matchday).all()
        
        # Group by line position
        for match in matches:
            line_pos = match.line_position or 1
            if line_pos not in self.line_sequences:
                self.line_sequences[line_pos] = []
            if match.result:
                self.line_sequences[line_pos].append(match.result)
        
        logger.info(f"Loaded sequences for {len(self.line_sequences)} line positions")
    
    def get_line_sequence_probability(self, line_position: int, context: str) -> Dict[str, float]:
        """
        Get probability distribution for next result given context for a line position.
        
        Example: line_position=1, context="VVVN" 
        Returns probabilities for next result based on historical patterns for line 1.
        """
        if not hasattr(self, 'line_sequences') or line_position not in self.line_sequences:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        sequence = self.line_sequences[line_position]
        
        if len(sequence) < len(context) + 1:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        # Find all occurrences of context in sequence
        matches = []
        context_list = list(context)
        for i in range(len(sequence) - len(context)):
            if sequence[i:i+len(context)] == context_list:
                if i + len(context) < len(sequence):
                    matches.append(sequence[i + len(context)])
        
        if not matches:
            # Fallback: use shorter context or uniform
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        # Calculate probabilities
        counts = {'V': 0, 'N': 0, 'D': 0}
        for m in matches:
            counts[m] = counts.get(m, 0) + 1
        
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
    
    def get_line_form_analysis(self, line_position: int) -> Dict:
        """
        Analyze recent form for a line position.
        """
        if not hasattr(self, 'line_sequences') or line_position not in self.line_sequences:
            return {'sequence': '', 'recent_form': '', 'stats': {}}
        
        sequence = self.line_sequences[line_position]
        recent = sequence[-10:] if len(sequence) >= 10 else sequence
        
        return {
            'sequence': ''.join(sequence),
            'recent_form': ''.join(recent[-5:]),
            'total_matches': len(sequence),
            'stats': {
                'V_count': sequence.count('V'),
                'N_count': sequence.count('N'),
                'D_count': sequence.count('D'),
                'V_rate': sequence.count('V') / len(sequence) if sequence else 0,
                'N_rate': sequence.count('N') / len(sequence) if sequence else 0,
                'D_rate': sequence.count('D') / len(sequence) if sequence else 0,
            }
        }
    
    def predict_line_next_result(self, line_position: int, current_sequence: str) -> Tuple[str, float, Dict]:
        """
        Predict the next result for a line position given current sequence.
        
        Returns (predicted_result, confidence, probabilities).
        """
        # Get context (last n results)
        context_length = min(self.sequence_length, len(current_sequence))
        context = current_sequence[-context_length:] if current_sequence else ""
        
        probs = self.get_line_sequence_probability(line_position, context)
        
        # Predict most likely outcome
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]
        
        return predicted, confidence, probs
    
    def analyze_line_patterns(self, db: Session) -> Dict:
        """
        Analyze patterns across all line positions.
        """
        self.load_line_sequences(db)
        
        results = {}
        for line_pos, sequence in self.line_sequences.items():
            if len(sequence) < 5:
                continue
            
            # Build transition matrix for this line
            transitions = self.build_transition_matrix([sequence])
            
            # Build higher-order transitions
            higher_order = self.build_higher_order_transition_matrix([sequence], order=2)
            
            # Analyze common patterns
            pattern_counts = defaultdict(int)
            for i in range(len(sequence) - 2):
                pattern = tuple(sequence[i:i+3])
                pattern_counts[pattern] += 1
            
            # Most common patterns
            total_patterns = sum(pattern_counts.values())
            top_patterns = dict(sorted(pattern_counts.items(), key=lambda x: -x[1])[:10])
            pattern_probs = {k: v / total_patterns for k, v in top_patterns.items()}
            
            results[line_pos] = {
                'sequence_length': len(sequence),
                'transitions': transitions.tolist(),
                'top_patterns': {str(k): v for k, v in pattern_probs.items()},
                'stats': {
                    'V_rate': sequence.count('V') / len(sequence),
                    'N_rate': sequence.count('N') / len(sequence),
                    'D_rate': sequence.count('D') / len(sequence),
                }
            }
        
        return results


class HiddenMarkovModel:
    """
    Simple HMM for result sequence modeling.
    States represent underlying team conditions (good form, bad form, neutral).
    """
    
    def __init__(self, n_hidden_states: int = 3):
        self.n_hidden_states = n_hidden_states
        self.n_observations = 3  # V, N, D
        
        # Initialize parameters
        self.pi = np.ones(n_hidden_states) / n_hidden_states  # Initial state distribution
        self.A = np.ones((n_hidden_states, n_hidden_states)) / n_hidden_states  # Transition matrix
        self.B = np.ones((n_hidden_states, self.n_observations)) / self.n_observations  # Emission matrix
    
    def train(self, sequences: List[str], n_iterations: int = 10):
        """
        Train HMM using Baum-Welch algorithm.
        """
        # Convert sequences to observation indices
        obs_map = {'V': 0, 'N': 1, 'D': 2}
        obs_sequences = [[obs_map[s] for s in seq if s in obs_map] for seq in sequences]
        
        # Baum-Welch (EM) algorithm
        for _ in range(n_iterations):
            for obs in obs_sequences:
                if len(obs) < 2:
                    continue
                
                # Forward pass
                alpha = self._forward(obs)
                
                # Backward pass
                beta = self._backward(obs)
                
                # Update parameters (simplified)
                self._update_parameters(obs, alpha, beta)
        
        logger.info("HMM training completed")
    
    def _forward(self, observations: List[int]) -> np.ndarray:
        """Forward algorithm."""
        T = len(observations)
        alpha = np.zeros((T, self.n_hidden_states))
        
        # Initialize
        alpha[0] = self.pi * self.B[:, observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_hidden_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]
        
        return alpha
    
    def _backward(self, observations: List[int]) -> np.ndarray:
        """Backward algorithm."""
        T = len(observations)
        beta = np.zeros((T, self.n_hidden_states))
        
        # Initialize
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            for i in range(self.n_hidden_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
        
        return beta
    
    def _update_parameters(self, observations: List[int], alpha: np.ndarray, beta: np.ndarray):
        """Update HMM parameters (simplified update)."""
        # This is a simplified version - full Baum-Welch would be more complex
        pass
    
    def predict_state(self, observations: List[str]) -> int:
        """Predict most likely hidden state given observations."""
        obs_map = {'V': 0, 'N': 1, 'D': 2}
        obs_indices = [obs_map[s] for s in observations if s in obs_map]
        
        if not obs_indices:
            return 0
        
        alpha = self._forward(obs_indices)
        return np.argmax(alpha[-1])
    
    def predict_next_observation(self, observations: List[str]) -> Dict[str, float]:
        """Predict probability distribution for next observation."""
        obs_map = {'V': 0, 'N': 1, 'D': 2}
        obs_indices = [obs_map[s] for s in observations if s in obs_map]
        
        if not obs_indices:
            return {'V': 0.33, 'N': 0.33, 'D': 0.33}
        
        # Get current state distribution
        alpha = self._forward(obs_indices)
        state_probs = alpha[-1] / np.sum(alpha[-1])
        
        # Predict next observation
        next_obs_probs = np.zeros(self.n_observations)
        for obs in range(self.n_observations):
            for state in range(self.n_hidden_states):
                next_obs_probs[obs] += state_probs[state] * self.B[state, obs]
        
        # Normalize
        next_obs_probs = next_obs_probs / np.sum(next_obs_probs)
        
        return {
            'V': next_obs_probs[0],
            'N': next_obs_probs[1],
            'D': next_obs_probs[2]
        }
