"""
MatchAnalyzer - Analyse INDIVIDUELLE de chaque match.
Combine forme, H2H, cycles, ML, signaux pour prédire de manière VARIÉE.
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path


class MatchAnalyzer:
    """
    Analyse approfondie d'un match pour prédire de manière intelligente.
    
    Utilise:
    - Forme récente (5 derniers matchs)
    - H2H (confrontations directes)
    - Cycles et patterns
    - Position dans la saison
    - Signaux de surprise
    - Apprentissage des erreurs passées (AggressiveLearner)
    """
    
    def __init__(self, db_path: str = "data/bet261_prediction.db"):
        self.db_path = db_path
        self.team_history = {}  # Cache historique équipes
        self.h2h_cache = {}     # Cache H2H
        self.patterns = self._load_patterns()
        
        # Charger l'AggressiveLearner pour apprendre des erreurs passées
        from app.services.aggressive_learner import AggressiveLearner
        self.learner = AggressiveLearner()
        logger.info(f"MatchAnalyzer initialized with learner weights: V={self.learner.outcome_weights.get('V', 1.0):.2f}, N={self.learner.outcome_weights.get('N', 1.0):.2f}, D={self.learner.outcome_weights.get('D', 1.0):.2f}")
        
    def _load_patterns(self) -> Dict:
        """Charge les patterns identifiés depuis 9000+ matchs."""
        patterns_file = Path("data/patterns_analysis.json")
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_team_recent_form(self, team_name: str, limit: int = 5) -> Dict:
        """Analyse la forme récente d'une équipe."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Matchs récents de l'équipe (domicile + extérieur)
        cur.execute("""
            SELECT 
                CASE 
                    WHEN home_team_name = ? THEN 'H'
                    ELSE 'A'
                END as venue,
                CASE 
                    WHEN home_team_name = ? THEN 
                        CASE WHEN result = 'V' THEN 'W' WHEN result = 'N' THEN 'D' ELSE 'L' END
                    ELSE 
                        CASE WHEN result = 'D' THEN 'W' WHEN result = 'N' THEN 'D' ELSE 'L' END
                END as outcome,
                matchday, season_id
            FROM matches
            WHERE (home_team_name = ? OR away_team_name = ?)
            AND is_completed = 1
            ORDER BY season_id DESC, matchday DESC
            LIMIT ?
        """, (team_name, team_name, team_name, team_name, limit))
        
        matches = cur.fetchall()
        conn.close()
        
        if not matches:
            return {'form_score': 0.5, 'wins': 0, 'draws': 0, 'losses': 0, 'trend': 'unknown'}
        
        wins = sum(1 for m in matches if m[1] == 'W')
        draws = sum(1 for m in matches if m[1] == 'D')
        losses = sum(1 for m in matches if m[1] == 'L')
        
        # Score de forme (0-1)
        form_score = (wins * 3 + draws) / (len(matches) * 3)
        
        # Tendance (derniers 3 matchs)
        recent = matches[:3]
        recent_wins = sum(1 for m in recent if m[1] == 'W')
        recent_losses = sum(1 for m in recent if m[1] == 'L')
        
        if recent_wins >= 2:
            trend = 'hot'
        elif recent_losses >= 2:
            trend = 'cold'
        else:
            trend = 'neutral'
        
        return {
            'form_score': form_score,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'trend': trend,
            'matches': len(matches)
        }
    
    def get_h2h_analysis(self, home_team: str, away_team: str, limit: int = 5) -> Dict:
        """Analyse les confrontations directes."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                result,
                matchday,
                season_id
            FROM matches
            WHERE home_team_name = ? AND away_team_name = ?
            AND is_completed = 1
            ORDER BY season_id DESC, matchday DESC
            LIMIT ?
        """, (home_team, away_team, limit))
        
        h2h_matches = cur.fetchall()
        conn.close()
        
        if not h2h_matches:
            return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'dominance': 'unknown', 'total': 0}
        
        home_wins = sum(1 for m in h2h_matches if m[0] == 'V')
        draws = sum(1 for m in h2h_matches if m[0] == 'N')
        away_wins = sum(1 for m in h2h_matches if m[0] == 'D')
        
        # Dominance
        if home_wins > away_wins + 2:
            dominance = 'home'
        elif away_wins > home_wins + 2:
            dominance = 'away'
        else:
            dominance = 'balanced'
        
        return {
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'dominance': dominance,
            'total': len(h2h_matches)
        }
    
    def detect_surprise_signals(
        self,
        home_team: str,
        away_team: str,
        odds: Tuple[float, float, float],
        matchday: int
    ) -> Dict:
        """Détecte les signaux de surprise potentielle."""
        signals = []
        surprise_boost = {'V': 0.0, 'N': 0.0, 'D': 0.0}
        
        odd_h, odd_d, odd_a = odds
        
        # 1. Forme des équipes
        home_form = self.get_team_recent_form(home_team)
        away_form = self.get_team_recent_form(away_team)
        
        # Favori en mauvaise forme → risque de surprise
        if odd_h < 1.8 and home_form['trend'] == 'cold':
            signals.append('FAVORI_COLD')
            surprise_boost['N'] += 0.08
            surprise_boost['D'] += 0.05
        
        if odd_a < 2.5 and away_form['trend'] == 'hot':
            signals.append('UNDERDOG_HOT')
            surprise_boost['D'] += 0.06
        
        # 2. Différence de forme
        form_diff = home_form['form_score'] - away_form['form_score']
        
        # Favori avec forme similaire à l'underdog → risque de nul
        if odd_h < 2.0 and abs(form_diff) < 0.15:
            signals.append('FORM_SIMILAR')
            surprise_boost['N'] += 0.07
        
        # 3. H2H
        h2h = self.get_h2h_analysis(home_team, away_team)
        
        # Historique de nuls
        if h2h['total'] >= 3 and h2h['draws'] / h2h['total'] > 0.3:
            signals.append('H2H_HIGH_DRAWS')
            surprise_boost['N'] += 0.12  # Augmenté
        
        # Historique de victoires extérieur
        if h2h['total'] >= 3 and h2h['away_wins'] > h2h['home_wins'] + 1:
            signals.append('H2H_AWAY_STRONG')
            surprise_boost['D'] += 0.15  # Augmenté
        
        # 4. Position dans la saison (fin de saison = plus de surprises)
        if matchday >= 30:
            signals.append('LATE_SEASON')
            surprise_boost['N'] += 0.03
            surprise_boost['D'] += 0.03
        elif matchday <= 5:
            signals.append('EARLY_SEASON')
            surprise_boost['N'] += 0.02
        
        # 5. Cotes indicatrices
        # Match équilibré (cotes proches)
        if max(odd_h, odd_a) / min(odd_h, odd_a) < 1.5:
            signals.append('ODDS_BALANCED')
            surprise_boost['N'] += 0.08  # Augmenté
        
        # 6. Pattern de la journée (depuis 9000 matchs)
        md_patterns = self.patterns.get('matchday_patterns', {}).get(str(matchday), {})
        if md_patterns:
            if md_patterns.get('draw_rate', 0) > 0.35:
                signals.append('HIGH_DRAW_MATCHDAY')
                surprise_boost['N'] += 0.10  # Augmenté
            if md_patterns.get('away_win_rate', 0) > 0.30:
                signals.append('HIGH_AWAY_MATCHDAY')
                surprise_boost['D'] += 0.08  # Augmenté
        
        return {
            'signals': signals,
            'surprise_boost': surprise_boost,
            'home_form': home_form,
            'away_form': away_form,
            'h2h': h2h
        }
    
    def compute_intelligent_prediction(
        self,
        home_team: str,
        away_team: str,
        odds: Tuple[float, float, float],
        matchday: int,
        ml_probs: Optional[Dict] = None,
        engine_probs: Optional[Dict] = None
    ) -> Dict:
        """
        Calcule une prédiction INTELLIGENTE et VARIÉE.
        
        Combine:
        - Cotes (Shin)
        - ML (si disponible)
        - Signaux de surprise
        - Forme
        - H2H
        """
        odd_h, odd_d, odd_a = odds
        
        # 1. Probabilités de base depuis les cotes (Shin)
        from app.services.signal_detectors import compute_shin_probabilities
        shin = compute_shin_probabilities(odd_h, odd_d, odd_a)
        base_probs = shin['shin'].copy()
        
        # 2. Détecter les signaux de surprise
        surprise = self.detect_surprise_signals(home_team, away_team, odds, matchday)
        
        # 3. Appliquer les boosts de surprise
        adjusted_probs = base_probs.copy()
        for outcome in ['V', 'N', 'D']:
            adjusted_probs[outcome] += surprise['surprise_boost'][outcome]
        
        # 4. Intégrer ML si disponible
        if ml_probs and isinstance(ml_probs, dict):
            # Blend avec ML (poids 0.3)
            for outcome in ['V', 'N', 'D']:
                adjusted_probs[outcome] = (
                    0.7 * adjusted_probs[outcome] + 
                    0.3 * ml_probs.get(outcome, 0.33)
                )
        
        # 5. Intégrer Engine si disponible
        if engine_probs and isinstance(engine_probs, dict):
            # Blend avec Engine (poids 0.25)
            for outcome in ['V', 'N', 'D']:
                adjusted_probs[outcome] = (
                    0.75 * adjusted_probs[outcome] + 
                    0.25 * engine_probs.get(outcome, 0.33)
                )
        
        # 6. Normaliser
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}
        
        # 7. Appliquer les corrections de l'AggressiveLearner (apprentissage des erreurs passées)
        learner_context = {'odd_home': odd_h, 'odd_draw': odd_d, 'odd_away': odd_a}
        adjusted_probs = self.learner.apply_corrections_to_prediction(adjusted_probs, learner_context)
        
        # 8. Sélectionner le résultat
        predicted = max(adjusted_probs, key=adjusted_probs.get)
        confidence = adjusted_probs[predicted]
        
        return {
            'final_probs': adjusted_probs,
            'predicted': predicted,
            'confidence': confidence,
            'surprise_signals': surprise['signals'],
            'home_form': surprise['home_form'],
            'away_form': surprise['away_form'],
            'h2h': surprise['h2h'],
            'reasoning': self._generate_reasoning(
                predicted, surprise, adjusted_probs, base_probs
            )
        }
    
    def _generate_reasoning(
        self,
        predicted: str,
        surprise: Dict,
        final_probs: Dict,
        base_probs: Dict
    ) -> str:
        """Génère une explication de la prédiction."""
        reasons = []
        
        signals = surprise.get('signals', [])
        home_form = surprise.get('home_form', {})
        away_form = surprise.get('away_form', {})
        h2h = surprise.get('h2h', {})
        
        # Forme
        if home_form.get('trend') == 'cold':
            reasons.append(f"Domicile en méforme ({home_form.get('wins', 0)}V/{home_form.get('losses', 0)}D sur 5)")
        if away_form.get('trend') == 'hot':
            reasons.append(f"Extérieur en forme ({away_form.get('wins', 0)}V sur 5)")
        
        # H2H
        if h2h.get('dominance') == 'home':
            reasons.append("H2H favorable au domicile")
        elif h2h.get('dominance') == 'away':
            reasons.append("H2H favorable à l'extérieur")
        elif h2h.get('draws', 0) > 0:
            reasons.append(f"H2H: {h2h['draws']} nuls sur {h2h['total']} confrontations")
        
        # Signaux
        if 'FAVORI_COLD' in signals:
            reasons.append("Favori en déclin → surprise possible")
        if 'FORM_SIMILAR' in signals:
            reasons.append("Formes équivalentes → match serré")
        if 'H2H_HIGH_DRAWS' in signals:
            reasons.append("Historique de nuls fréquent")
        
        # Changement vs cotes
        if predicted == 'N' and base_probs['N'] < 0.25:
            reasons.append("Nul détecté malgré les cotes (surprise)")
        elif predicted == 'D' and base_probs['D'] < 0.25:
            reasons.append("Victoire extérieur détectée (surprise)")
        
        if not reasons:
            reasons.append(f"Prédiction basée sur les cotes et forme")
        
        return " | ".join(reasons)


def test_match_analyzer():
    """Test l'analyseur sur différents matchs."""
    analyzer = MatchAnalyzer()
    
    test_cases = [
        ("Bilbao", "Levante", 1.19, 5.94, 18.90, 15),
        ("Barca", "Getafe", 1.25, 5.20, 14.78, 15),
        ("Espanyol", "R. Madrid", 5.94, 4.53, 1.49, 15),
        ("Osasuna", "Valencia", 2.16, 3.00, 3.77, 15),
        ("Girona", "Barca", 2.41, 3.84, 2.60, 15),
        ("Betis", "Sevilla", 1.84, 3.34, 4.60, 15),
    ]
    
    print("=" * 80)
    print("TEST MATCH ANALYZER - PRÉDICTIONS VARIÉES")
    print("=" * 80)
    print(f"{'Match':<25} {'Cotes':<15} {'Pred':<4} {'Probs V/N/D':<20} {'Signaux'}")
    print("-" * 80)
    
    for home, away, oh, od, oa, md in test_cases:
        result = analyzer.compute_intelligent_prediction(
            home, away, (oh, od, oa), md
        )
        
        probs = result['final_probs']
        pred = result['predicted']
        signals = result['surprise_signals']
        
        probs_str = f"{probs['V']:.2f}/{probs['N']:.2f}/{probs['D']:.2f}"
        signals_str = ', '.join(signals[:2]) if signals else '-'
        
        print(f"{home} vs {away:<15} {oh:.2f}/{od:.2f}/{oa:.2f}  {pred}    {probs_str:<20} {signals_str}")
        print(f"  → {result['reasoning'][:70]}...")


if __name__ == "__main__":
    test_match_analyzer()
