"""
Aggressive Learner - Apprentissage agressif sur 9000+ résultats.
Met à jour dynamiquement les poids et corrections basés sur les erreurs passées.
"""

import sqlite3
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from loguru import logger
from pathlib import Path
from datetime import datetime


class AggressiveLearner:
    """
    Système d'apprentissage agressif qui:
    - Analyse les erreurs par contexte
    - Met à jour les poids des modèles
    - Ajuste les corrections de biais
    - Maximise le ROI long terme
    """
    
    def __init__(self, db_path: str = "data/bet261_prediction.db"):
        self.db_path = db_path
        self.model_weights = {
            'engine': 0.25,
            'ml': 0.45,
            'odds': 0.22,
            'h2h': 0.08
        }
        self.context_corrections = {}
        self.outcome_weights = {'V': 1.0, 'N': 1.0, 'D': 1.0}
        self.learning_rate = 0.15  # Taux d'apprentissage agressif
        
    def load_all_history(self) -> Tuple[List[Dict], List[Dict]]:
        """Charge tout l'historique des matchs et prédictions."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Charger les prédictions avec résultats
        cur.execute("""
            SELECT 
                p.id, p.match_id, p.predicted_result, p.actual_result,
                p.prob_home_win, p.prob_draw, p.prob_away_win,
                p.confidence, p.model_agreement,
                m.odd_home, m.odd_draw, m.odd_away,
                m.home_team_name, m.away_team_name,
                m.score_home, m.score_away,
                m.season_id, m.matchday
            FROM predictions p
            JOIN matches m ON m.id = p.match_id
            WHERE p.actual_result IS NOT NULL
            ORDER BY m.season_id, m.matchday
        """)
        
        predictions = []
        for row in cur.fetchall():
            predictions.append({
                'id': row[0],
                'match_id': row[1],
                'predicted': row[2],
                'actual': row[3],
                'prob_v': row[4],
                'prob_n': row[5],
                'prob_d': row[6],
                'confidence': row[7],
                'model_agreement': row[8],
                'odd_home': row[9],
                'odd_draw': row[10],
                'odd_away': row[11],
                'home_team': row[12],
                'away_team': row[13],
                'home_score': row[14],
                'away_score': row[15],
                'season': row[16],
                'matchday': row[17]
            })
        
        conn.close()
        logger.info(f"Chargé {len(predictions)} prédictions avec résultats")
        return predictions
    
    def analyze_errors_by_context(self, predictions: List[Dict]) -> Dict:
        """Analyse les erreurs par contexte pour identifier les patterns."""
        analysis = {
            'by_odds_range': {},
            'by_confidence_range': {},
            'by_outcome': {},
            'by_prob_range': {},
            'roi_by_outcome': {}
        }
        
        # Initialiser les structures
        for outcome in ['V', 'N', 'D']:
            analysis['by_outcome'][outcome] = {
                'total': 0, 'correct': 0, 'roi': 0.0, 'total_stake': 0.0, 'total_return': 0.0
            }
        
        # Analyser chaque prédiction
        for pred in predictions:
            predicted = pred['predicted']
            actual = pred['actual']
            confidence = pred['confidence']
            
            # Odds range
            odd_home = pred['odd_home'] or 2.0
            if odd_home < 1.5:
                odds_bin = 'H<1.5'
            elif odd_home < 2.0:
                odds_bin = 'H1.5-2.0'
            elif odd_home < 2.5:
                odds_bin = 'H2.0-2.5'
            else:
                odds_bin = 'H>2.5'
            
            if odds_bin not in analysis['by_odds_range']:
                analysis['by_odds_range'][odds_bin] = {'V': {'total': 0, 'correct': 0}, 
                                                        'N': {'total': 0, 'correct': 0}, 
                                                        'D': {'total': 0, 'correct': 0}}
            
            analysis['by_odds_range'][odds_bin][predicted]['total'] += 1
            if predicted == actual:
                analysis['by_odds_range'][odds_bin][predicted]['correct'] += 1
            
            # Confidence range
            if confidence < 0.4:
                conf_bin = '0-0.4'
            elif confidence < 0.6:
                conf_bin = '0.4-0.6'
            else:
                conf_bin = '0.6-1.0'
            
            if conf_bin not in analysis['by_confidence_range']:
                analysis['by_confidence_range'][conf_bin] = {'total': 0, 'correct': 0}
            
            analysis['by_confidence_range'][conf_bin]['total'] += 1
            if predicted == actual:
                analysis['by_confidence_range'][conf_bin]['correct'] += 1
            
            # By outcome
            analysis['by_outcome'][predicted]['total'] += 1
            if predicted == actual:
                analysis['by_outcome'][predicted]['correct'] += 1
            
            # ROI calculation
            odds = {'V': pred['odd_home'], 'N': pred['odd_draw'], 'D': pred['odd_away']}
            odd = odds.get(predicted, 2.0) or 2.0
            analysis['by_outcome'][predicted]['total_stake'] += 1.0
            if predicted == actual:
                analysis['by_outcome'][predicted]['total_return'] += odd
                analysis['by_outcome'][predicted]['roi'] = (
                    analysis['by_outcome'][predicted]['total_return'] / 
                    analysis['by_outcome'][predicted]['total_stake']
                )
        
        return analysis
    
    def compute_model_performance(self, predictions: List[Dict]) -> Dict:
        """Calcule la performance de chaque composant du modèle."""
        performance = {
            'prob_accuracy': {'V': [], 'N': [], 'D': []},
            'confidence_reliability': [],
            'model_agreement_value': []
        }
        
        for pred in predictions:
            predicted = pred['predicted']
            actual = pred['actual']
            
            # Probabilité assignée au résultat réel
            prob_actual = {'V': pred['prob_v'], 'N': pred['prob_n'], 'D': pred['prob_d']}.get(actual, 0.33)
            performance['prob_accuracy'][actual].append(prob_actual)
            
            # Fiabilité de la confiance
            is_correct = 1.0 if predicted == actual else 0.0
            performance['confidence_reliability'].append({
                'confidence': pred['confidence'],
                'correct': is_correct
            })
        
        # Calculer les moyennes
        for outcome in ['V', 'N', 'D']:
            if performance['prob_accuracy'][outcome]:
                performance['prob_accuracy'][outcome] = np.mean(performance['prob_accuracy'][outcome])
            else:
                performance['prob_accuracy'][outcome] = 0.33
        
        return performance
    
    def compute_aggressive_corrections(self, analysis: Dict, performance: Dict) -> Dict:
        """Calcule les corrections agressives basées sur l'analyse."""
        corrections = {
            'outcome_multipliers': {},
            'confidence_adjustments': {},
            'odds_range_adjustments': {},
            'model_weight_adjustments': {}
        }
        
        # 1. Multiplicateurs par outcome (pour rééquilibrer) - DOUX pour éviter absurdités
        for outcome in ['V', 'N', 'D']:
            stats = analysis['by_outcome'][outcome]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                # Si accuracy < 30%, ajustement doux de 10%
                # Si accuracy > 55%, ajustement doux de 10%
                if accuracy < 0.30:
                    corrections['outcome_multipliers'][outcome] = 0.90  # Réduire doucement
                elif accuracy > 0.55:
                    corrections['outcome_multipliers'][outcome] = 1.10  # Augmenter doucement
                else:
                    corrections['outcome_multipliers'][outcome] = 1.0
            else:
                corrections['outcome_multipliers'][outcome] = 1.0
        
        # 2. Ajustements par range de cotes - DOUX
        for odds_bin, outcomes in analysis['by_odds_range'].items():
            best_outcome = None
            best_accuracy = 0.0
            
            for outcome in ['V', 'N', 'D']:
                if outcomes[outcome]['total'] > 10:  # Minimum 10 samples
                    accuracy = outcomes[outcome]['correct'] / outcomes[outcome]['total']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_outcome = outcome
            
            # Boost max de 10% pour éviter les absurdités
            if best_outcome and best_accuracy > 0.45:
                corrections['odds_range_adjustments'][odds_bin] = {
                    'boost_outcome': best_outcome,
                    'boost_factor': 1.10  # Max 10% boost
                }
        
        # 3. Ajustements des poids de modèle - DOUX
        # Si prob_accuracy pour N est faible, boost doux de 15%
        prob_n = performance['prob_accuracy']['N']
        prob_d = performance['prob_accuracy']['D']
        
        if prob_n < 0.25:
            corrections['model_weight_adjustments']['draw_signal_boost'] = 1.15
        if prob_d < 0.25:
            corrections['model_weight_adjustments']['away_signal_boost'] = 1.15
        
        return corrections
    
    def update_weights_from_history(self) -> Dict:
        """Met à jour les poids depuis tout l'historique."""
        logger.info("=== DÉBUT APPRENTISSAGE AGRESSIF ===")
        
        # Charger l'historique
        predictions = self.load_all_history()
        
        if len(predictions) < 100:
            logger.warning("Pas assez de données pour l'apprentissage agressif")
            return {}
        
        # Analyser les erreurs
        analysis = self.analyze_errors_by_context(predictions)
        logger.info(f"Analyse par outcome: {analysis['by_outcome']}")
        
        # Performance des modèles
        performance = self.compute_model_performance(predictions)
        logger.info(f"Performance prob_accuracy: {performance['prob_accuracy']}")
        
        # Calculer les corrections
        corrections = self.compute_aggressive_corrections(analysis, performance)
        logger.info(f"Corrections calculées: {corrections}")
        
        # Mettre à jour les poids internes
        for outcome, mult in corrections['outcome_multipliers'].items():
            self.outcome_weights[outcome] = mult
        
        # Sauvegarder
        self.context_corrections = corrections
        
        return corrections
    
    def apply_corrections_to_prediction(
        self, 
        probs: Dict[str, float],
        context: Dict = None
    ) -> Dict[str, float]:
        """Applique les corrections agressives à une prédiction."""
        corrected = probs.copy()
        
        # 1. Appliquer les multiplicateurs par outcome
        for outcome in ['V', 'N', 'D']:
            corrected[outcome] *= self.outcome_weights.get(outcome, 1.0)
        
        # 2. Appliquer les ajustements par cotes si disponible
        if context and 'odd_home' in context:
            odd_home = context['odd_home']
            if odd_home < 1.5:
                odds_bin = 'H<1.5'
            elif odd_home < 2.0:
                odds_bin = 'H1.5-2.0'
            elif odd_home < 2.5:
                odds_bin = 'H2.0-2.5'
            else:
                odds_bin = 'H>2.5'
            
            if odds_bin in self.context_corrections.get('odds_range_adjustments', {}):
                adj = self.context_corrections['odds_range_adjustments'][odds_bin]
                boost_outcome = adj['boost_outcome']
                boost_factor = adj['boost_factor']
                corrected[boost_outcome] *= boost_factor
        
        # 3. Normaliser
        total = sum(corrected.values())
        if total > 0:
            corrected = {k: v/total for k, v in corrected.items()}
        
        return corrected
    
    def save_learning_state(self, filepath: str = "data/aggressive_learning_state.json"):
        """Sauvegarde l'état d'apprentissage."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'outcome_weights': self.outcome_weights,
            'context_corrections': self.context_corrections,
            'model_weights': self.model_weights,
            'last_updated': str(datetime.now())
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"État d'apprentissage sauvegardé dans {filepath}")
    
    def load_learning_state(self, filepath: str = "data/aggressive_learning_state.json"):
        """Charge l'état d'apprentissage sauvegardé."""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.outcome_weights = state.get('outcome_weights', self.outcome_weights)
                self.context_corrections = state.get('context_corrections', {})
                self.model_weights = state.get('model_weights', self.model_weights)
                
                logger.info(f"État d'apprentissage chargé: outcome_weights={self.outcome_weights}")
                return True
        except Exception as e:
            logger.warning(f"Impossible de charger l'état d'apprentissage: {e}")
        
        return False
    
    def get_learning_stats(self) -> Dict:
        """Retourne les statistiques d'apprentissage."""
        return {
            'outcome_weights': self.outcome_weights,
            'model_weights': self.model_weights,
            'context_corrections_count': len(self.context_corrections.get('odds_range_adjustments', {}))
        }


def run_aggressive_learning():
    """Exécute l'apprentissage agressif complet."""
    learner = AggressiveLearner()
    
    # Essayer de charger l'état existant
    learner.load_learning_state()
    
    # Mettre à jour depuis l'historique
    corrections = learner.update_weights_from_history()
    
    # Sauvegarder
    learner.save_learning_state()
    
    print("=== RÉSULTATS APPRENTISSAGE AGRESSIF ===")
    print(f"Outcome weights: {learner.outcome_weights}")
    print(f"Model weights: {learner.model_weights}")
    print(f"Context corrections: {learner.context_corrections}")
    
    return learner


if __name__ == "__main__":
    run_aggressive_learning()
