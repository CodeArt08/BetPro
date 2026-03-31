"""
Prediction Optimizer - Apprentissage agressif sur 9000+ matchs.
Corrige les biais systématiques et améliore drastiquement N et D.
"""

import sqlite3
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


class PredictionOptimizer:
    """
    Optimiseur qui analyse 9000+ matchs et corrige les biais:
    - Calibration des probabilités
    - Rééquilibrage V/N/D
    - Pondération adaptative par contexte
    """
    
    def __init__(self, db_path: str = "data/bet261_prediction.db"):
        self.db_path = db_path
        self.calibration_matrix = {}
        self.bias_corrections = {}
        self.context_weights = {}
        
    def analyze_biases(self) -> Dict:
        """Analyse complète des biais sur 9000+ matchs."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        analysis = {}
        
        # 1. Distribution réelle vs prédite
        cur.execute("""
            SELECT result, COUNT(*) FROM matches 
            WHERE result IS NOT NULL 
            GROUP BY result
        """)
        real_dist = dict(cur.fetchall())
        
        cur.execute("""
            SELECT predicted_result, COUNT(*) FROM predictions 
            GROUP BY predicted_result
        """)
        pred_dist = dict(cur.fetchall())
        
        analysis['real_distribution'] = real_dist
        analysis['pred_distribution'] = pred_dist
        
        # 2. Accuracy par type
        cur.execute("""
            SELECT p.predicted_result, COUNT(*) as total,
                   SUM(CASE WHEN p.actual_result = p.predicted_result THEN 1 ELSE 0 END) as correct,
                   ROUND(SUM(CASE WHEN p.actual_result = p.predicted_result THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy
            FROM predictions p 
            WHERE p.actual_result IS NOT NULL 
            GROUP BY p.predicted_result
        """)
        accuracy_by_type = {}
        for pred_type, total, correct, acc in cur.fetchall():
            accuracy_by_type[pred_type] = {'total': total, 'correct': correct, 'accuracy': acc}
        analysis['accuracy_by_type'] = accuracy_by_type
        
        # 3. Calibration matrix (probabilités moyennes vs réalité)
        cur.execute("""
            SELECT p.predicted_result,
                   AVG(p.prob_home_win) as avg_v,
                   AVG(p.prob_draw) as avg_n,
                   AVG(p.prob_away_win) as avg_d,
                   COUNT(*) as total
            FROM predictions p 
            WHERE p.actual_result IS NOT NULL 
            GROUP BY p.predicted_result
        """)
        calibration = {}
        for pred_type, avg_v, avg_n, avg_d, total in cur.fetchall():
            calibration[pred_type] = {
                'avg_probs': {'V': avg_v, 'N': avg_n, 'D': avg_d},
                'total': total
            }
        analysis['calibration'] = calibration
        
        # 4. Performance par tranche de cotes
        cur.execute("""
            SELECT 
                CASE 
                    WHEN m.odd_home < 1.5 THEN 'H<1.5'
                    WHEN m.odd_home < 2.0 THEN 'H1.5-2.0'
                    WHEN m.odd_home < 2.5 THEN 'H2.0-2.5'
                    ELSE 'H>2.5'
                END as odds_bin,
                COUNT(*) as total,
                ROUND(AVG(p.prob_home_win), 3) as avg_home_prob,
                ROUND(SUM(CASE WHEN m.result = 'V' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as real_home_rate
            FROM predictions p 
            JOIN matches m ON m.id = p.match_id 
            WHERE p.actual_result IS NOT NULL AND m.odd_home IS NOT NULL
            GROUP BY odds_bin
        """)
        odds_performance = {}
        for odds_bin, total, avg_prob, real_rate in cur.fetchall():
            odds_performance[odds_bin] = {
                'avg_prob': avg_prob,
                'real_rate': real_rate,
                'bias': avg_prob - real_rate/100
            }
        analysis['odds_performance'] = odds_performance
        
        conn.close()
        return analysis
    
    def compute_calibration_corrections(self, analysis: Dict) -> Dict:
        """Calcule les corrections de calibration basées sur 9000+ matchs."""
        corrections = {}
        real_dist = analysis['real_distribution']
        total_real = sum(real_dist.values())
        
        # Distribution réelle normalisée
        real_probs = {k: v/total_real for k, v in real_dist.items()}
        
        # Corrections par type de prédiction - DOUCES pour éviter les absurdités
        for pred_type in ['V', 'N', 'D']:
            if pred_type in analysis['calibration']:
                avg_probs = analysis['calibration'][pred_type]['avg_probs']
                
                # Correction multiplicative SUBTILE pour chaque outcome
                # On limite à +/- 20% pour éviter les prédictions absurdes
                correction = {}
                for outcome in ['V', 'N', 'D']:
                    if avg_probs[outcome] > 0:
                        ratio = real_probs[outcome] / avg_probs[outcome]
                        # Limiter entre 0.8 et 1.2 (max 20% d'ajustement)
                        correction[outcome] = max(0.8, min(1.2, ratio))
                    else:
                        correction[outcome] = 1.0
                
                corrections[pred_type] = correction
        
        return corrections
    
    def compute_bias_adjustments(self, analysis: Dict) -> Dict:
        """Calcule les ajustements pour corriger les biais systématiques."""
        adjustments = {}
        
        # 1. Ajustement distributionnel (trop de V prédits) - DOUX
        real_dist = analysis['real_distribution']
        pred_dist = analysis['pred_distribution']
        total_real = sum(real_dist.values())
        total_pred = sum(pred_dist.values())
        
        real_probs = {k: v/total_real for k, v in real_dist.items()}
        pred_probs = {k: v/total_pred for k, v in pred_dist.items()}
        
        # Ajustement DOUX pour rééquilibrer vers la réalité (max 20%)
        distribution_adjustment = {}
        for outcome in ['V', 'N', 'D']:
            if pred_probs.get(outcome, 0) > 0:
                ratio = real_probs[outcome] / pred_probs[outcome]
                # Limiter entre 0.8 et 1.2 (max 20% d'ajustement)
                distribution_adjustment[outcome] = max(0.8, min(1.2, ratio))
            else:
                distribution_adjustment[outcome] = 1.0
        
        adjustments['distribution'] = distribution_adjustment
        
        # 2. Ajustement par cotes (correction du biais domicile)
        odds_perf = analysis['odds_performance']
        odds_adjustment = {}
        
        for odds_bin, perf in odds_perf.items():
            bias = perf['bias']
            # Si le système sur-estime domicile, réduire V
            if odds_bin in ['H<1.5', 'H1.5-2.0'] and bias > 0.05:
                odds_adjustment[odds_bin] = {
                    'V': 0.85,  # Réduire V
                    'N': 1.15,  # Augmenter N
                    'D': 1.20   # Augmenter D
                }
            # Si le système sous-estime domicile, augmenter V
            elif odds_bin in ['H>2.5'] and bias < -0.05:
                odds_adjustment[odds_bin] = {
                    'V': 1.20,
                    'N': 0.90,
                    'D': 0.85
                }
        
        adjustments['odds_based'] = odds_adjustment
        
        return adjustments
    
    def generate_optimization_config(self) -> Dict:
        """Génère la configuration optimisée pour le moteur de prédiction."""
        logger.info("Analyse des 9000+ matchs pour optimisation...")
        analysis = self.analyze_biases()
        
        logger.info(f"Distribution réelle: {analysis['real_distribution']}")
        logger.info(f"Distribution prédite: {analysis['pred_distribution']}")
        logger.info(f"Accuracy: V={analysis['accuracy_by_type'].get('V', {}).get('accuracy', 0)}%, "
                   f"N={analysis['accuracy_by_type'].get('N', {}).get('accuracy', 0)}%, "
                   f"D={analysis['accuracy_by_type'].get('D', {}).get('accuracy', 0)}%")
        
        # Calculer les corrections
        calibration_corrections = self.compute_calibration_corrections(analysis)
        bias_adjustments = self.compute_bias_adjustments(analysis)
        
        # Configuration optimisée
        config = {
            'calibration_corrections': calibration_corrections,
            'bias_adjustments': bias_adjustments,
            'target_distribution': {
                'V': 0.48,  # Légèrement réduit
                'N': 0.29,  # Augmenté
                'D': 0.23   # Augmenté
            },
            'confidence_thresholds': {
                'V': 0.45,  # Plus strict pour V
                'N': 0.35,  # Plus permissif pour N
                'D': 0.35   # Plus permissif pour D
            },
            'model_weights_adjustment': {
                # Réduire poids des modèles qui favorisent V
                'engine_weight': 0.20,  # Réduit de 0.25
                'ml_weight': 0.55,      # Augmenté de 0.45
                'odds_weight': 0.25     # Augmenté de 0.22
            },
            'signal_weights_optimized': {
                'draw_signal_strength': 0.15,  # Augmenté de 0.10
                'away_signal_strength': 0.15,  # Augmenté de 0.10
                'cycle_overdue_score': 0.12,    # Augmenté pour N/D
                'distribution_deviation': 0.10, # Augmenté
                'line_bias': 0.10,              # Augmenté
                'odds_calib_edge': 0.08         # Augmenté
            }
        }
        
        # Sauvegarder pour utilisation
        self.calibration_matrix = calibration_corrections
        self.bias_corrections = bias_adjustments
        
        logger.info("Configuration optimisée générée")
        return config
    
    def apply_corrections_to_probabilities(
        self, 
        probs: Dict[str, float], 
        context: Dict = None
    ) -> Dict[str, float]:
        """Applique les corrections de calibration aux probabilités."""
        corrected = probs.copy()
        
        # 1. Correction distributionnelle
        if 'distribution' in self.bias_corrections:
            dist_adj = self.bias_corrections['distribution']
            for outcome in ['V', 'N', 'D']:
                corrected[outcome] *= dist_adj.get(outcome, 1.0)
        
        # 2. Correction par cotes si disponible
        if context and 'odds_bin' in context:
            odds_bin = context['odds_bin']
            if 'odds_based' in self.bias_corrections and odds_bin in self.bias_corrections['odds_based']:
                odds_adj = self.bias_corrections['odds_based'][odds_bin]
                for outcome in ['V', 'N', 'D']:
                    corrected[outcome] *= odds_adj.get(outcome, 1.0)
        
        # 3. Normalisation
        total = sum(corrected.values())
        if total > 0:
            corrected = {k: v/total for k, v in corrected.items()}
        
        return corrected
    
    def save_optimization_state(self, filepath: str = "data/optimization_state.json"):
        """Sauvegarde l'état d'optimisation."""
        import json
        import os
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'calibration_corrections': self.calibration_matrix,
            'bias_corrections': self.bias_corrections,
            'last_updated': str(np.datetime64('now'))
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"État d'optimisation sauvegardé dans {filepath}")


if __name__ == "__main__":
    # Test et génération
    optimizer = PredictionOptimizer()
    config = optimizer.generate_optimization_config()
    optimizer.save_optimization_state()
    
    print("=== CONFIGURATION OPTIMISÉE ===")
    import json
    print(json.dumps(config, indent=2))
