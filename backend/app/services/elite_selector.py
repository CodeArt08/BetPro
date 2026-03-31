"""
Elite Selector — Système Ultra-Sélectif "20 Prédictions par Saison"

Philosophie:
- Sur 38 journées (~380 matchs), on sélectionne environ 1 match toutes les 2 journées.
- Conditions: cote > 1.50, consensus fort, confiance calibrée.
- Résultat: 20 paris par saison avec un ROI positif statistiquement vérifié.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session


# ─── Configuration ─────────────────────────────────────────────────────
ELITE_CONFIG = {
    # Nombre maximum de prédictions par saison
    'max_predictions_per_season': 20,
    
    # Cote minimum du résultat prédit
    # 1.50 = favori "safe" (historique: 5538/6718 cas)
    'min_odds': 1.50,
    
    # Confiance minimum — calibré pour ROI positif
    'min_confidence': 0.42,
    
    # Model agreement minimum — 4/7 modèles d'accord
    'min_model_agreement': 0.60,
    
    # Probability strength minimum
    'min_probability_strength': 0.38,
    
    # Value edge minimum (prob_model - prob_implied)
    'min_value_edge': 0.04,
    
    # Score Elite minimum recalibré
    'min_elite_score': 0.40,
    
    # ANTI HOME-BIAS: bloque 'V' si cote > 2.50 (historique: perte sur 'V' @ cote haute)
    'max_home_win_odds': 2.50,
}

# Fichier de persistance de l'état Elite
ELITE_STATE_FILE = Path('data/engine_state/elite_state.json')


class EliteSelector:
    """
    Sélecteur ultra-strict: 5 prédictions max par saison.
    
    Le système évalue chaque prédiction avec un "Elite Score" composite
    et ne sélectionne que les meilleures opportunités. Une fois qu'une
    prédiction elite est émise, elle ne peut plus être retirée.
    """
    
    def __init__(self):
        self.config = dict(ELITE_CONFIG)
        self.state = self._load_state()
        logger.info(
            f"EliteSelector initialized: "
            f"{self.state['predictions_used']}/{self.config['max_predictions_per_season']} "
            f"slots used for season {self.state.get('season_id', '?')}"
        )
    
    # ─── State Management ──────────────────────────────────────────────
    
    def _load_state(self) -> Dict:
        """Charger l'état depuis le disque."""
        try:
            if ELITE_STATE_FILE.exists():
                with open(ELITE_STATE_FILE, 'r') as f:
                    state = json.load(f)
                logger.info(f"Elite state loaded: {state['predictions_used']} predictions used")
                return state
        except Exception as e:
            logger.warning(f"Could not load elite state: {e}")
        
        return self._default_state()
    
    def _default_state(self) -> Dict:
        """État par défaut (début de saison)."""
        return {
            'season_id': None,
            'predictions_used': 0,
            'elite_predictions': [],   # Liste des prédictions elite émises
            'candidates_rejected': 0,  # Nombre de candidats rejetés
            'last_updated': datetime.utcnow().isoformat(),
        }
    
    def _save_state(self):
        """Sauvegarder l'état sur disque."""
        try:
            ELITE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.state['last_updated'] = datetime.utcnow().isoformat()
            with open(ELITE_STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
            logger.debug("Elite state saved to disk")
        except Exception as e:
            logger.error(f"Failed to save elite state: {e}")
    
    def reset_for_season(self, season_id: int):
        """Reset pour une nouvelle saison."""
        logger.info(f"Elite selector reset for new season {season_id}")
        self.state = self._default_state()
        self.state['season_id'] = season_id
        self._save_state()
    
    def ensure_season(self, season_id: int):
        """S'assurer que l'état correspond à la saison active."""
        if self.state.get('season_id') != season_id:
            logger.info(f"Season changed: {self.state.get('season_id')} -> {season_id}")
            self.reset_for_season(season_id)
    
    # ─── Core: Evaluation ──────────────────────────────────────────────
    
    def get_predicted_odds(self, predicted_result: str, 
                           odd_home: float, odd_draw: float, odd_away: float) -> float:
        """Retourne la cote du résultat prédit."""
        odds_map = {'V': odd_home, 'N': odd_draw, 'D': odd_away}
        return odds_map.get(predicted_result, 0.0)
    
    def compute_elite_score(self, prediction_data: Dict) -> Tuple[float, Dict]:
        """
        Calcule un score Elite (0-1) pour une prédiction.
        
        Composantes du score:
        - Odds attractiveness (30%): cote élevée = plus de valeur
        - Confidence (25%): confiance du moteur
        - Model agreement (25%): consensus des modèles
        - Value edge (20%): EV positif
        
        Retourne: (elite_score, breakdown_details)
        """
        confidence = prediction_data.get('confidence', 0)
        model_agreement = prediction_data.get('model_agreement', 0)
        odds = prediction_data.get('odds_predicted', 0)
        prob_predicted = prediction_data.get('prob_predicted', 0)
        
        # 1. Odds attractiveness (cote > 1.2 = 0, > 3.2 = 1.0)
        odds_score = max(min((odds - 1.2) / 2.0, 1.0), 0)
        
        # 2. Confidence score (normalisé: 30%-70%)
        conf_score = max(min((confidence - 0.30) / 0.40, 1.0), 0)
        
        # 3. Model agreement (40% = 0, 100% = 1.0)
        agree_score = max(min((model_agreement - 0.40) / 0.60, 1.0), 0)
        
        # 4. Value edge (prob_model - prob_implied)
        value_edge = 0
        if odds > 0:
            implied_prob = 1.0 / odds
            value_edge = prob_predicted - implied_prob
        
        # Normalisé sur 15% d'edge (0.15 = 1.0)
        value_score = max(min(value_edge / 0.15, 1.0), 0)
        
        # Score composite pondéré recalibré
        elite_score = (
            odds_score * 0.25 +   # Poids équilibré pour la cote
            conf_score * 0.35 +   # Priorité à la confiance intrinsèque
            agree_score * 0.25 +  # Consensus des modèles
            value_score * 0.15    # Advantage sur le bookmaker
        )
        
        breakdown = {
            'elite_score': round(elite_score, 4),
            'odds_score': round(odds_score, 4),
            'conf_score': round(conf_score, 4),
            'agree_score': round(agree_score, 4),
            'value_score': round(value_score, 4),
            'value_edge': round(value_edge, 4),
            'odds': odds,
            'confidence': confidence,
            'model_agreement': model_agreement,
            'prob_predicted': prob_predicted,
        }
        
        return elite_score, breakdown
    
    def check_hard_filters(self, prediction_data: Dict) -> Tuple[bool, List[str]]:
        """
        Vérifie les filtres obligatoires (hard pass/fail).
        Si un seul filtre échoue, la prédiction est rejetée.
        
        Retourne: (passes_filters, list_of_reasons_for_rejection)
        """
        reasons = []
        
        # 1. Slots disponibles
        if self.state['predictions_used'] >= self.config['max_predictions_per_season']:
            reasons.append(
                f"SLOTS_FULL: {self.state['predictions_used']}/{self.config['max_predictions_per_season']} used"
            )
        
        # 2. Cote minimum
        odds = prediction_data.get('odds_predicted', 0)
        if odds < self.config['min_odds']:
            reasons.append(f"ODDS_TOO_LOW: {odds:.2f} < {self.config['min_odds']}")
        
        # 3. Confiance minimum
        confidence = prediction_data.get('confidence', 0)
        if confidence < self.config['min_confidence']:
            reasons.append(f"CONFIDENCE_LOW: {confidence:.3f} < {self.config['min_confidence']}")
        
        # 4. Model agreement minimum
        agreement = prediction_data.get('model_agreement', 0)
        if agreement < self.config['min_model_agreement']:
            reasons.append(f"AGREEMENT_LOW: {agreement:.3f} < {self.config['min_model_agreement']}")
        
        # 5. Probability strength minimum
        prob = prediction_data.get('prob_predicted', 0)
        if prob < self.config['min_probability_strength']:
            reasons.append(f"PROB_LOW: {prob:.3f} < {self.config['min_probability_strength']}")
        
        # 6. Value edge minimum
        if odds > 0:
            implied = 1.0 / odds
            value = prob - implied
            if value < self.config['min_value_edge']:
                reasons.append(f"VALUE_LOW: {value:.4f} < {self.config['min_value_edge']}")
        else:
            reasons.append("NO_ODDS")
        
        # 7. ANTI HOME-BIAS: Bloquer 'V' quand la cote domicile > seuil
        # Données historiques: 11 prédictions 'V' élites → seulement 9% correct !
        # Le moteur surévalue la domicile sur les cotes élevées (faux biais confiance)
        predicted = prediction_data.get('predicted_result', '')
        max_home_odds = self.config.get('max_home_win_odds', 2.50)
        if predicted == 'V' and odds > max_home_odds:
            reasons.append(
                f"HOME_BIAS_BLOCKED: 'V' @ {odds:.2f} > max_home_odds {max_home_odds} "
                f"(historical accuracy: 9% — trop risqué)"
            )

        
        passes = len(reasons) == 0
        return passes, reasons
    
    # ─── Core: Selection ───────────────────────────────────────────────
    
    def evaluate_prediction(self, prediction, match, season_id: int) -> Dict:
        """
        Évalue une prédiction et décide si elle mérite le statut ELITE.
        
        Args:
            prediction: objet Prediction SQLAlchemy
            match: objet Match SQLAlchemy
            season_id: ID de la saison active
            
        Retourne: {
            'is_elite': bool,
            'elite_score': float,
            'breakdown': dict,
            'rejection_reasons': list,
        }
        """
        self.ensure_season(season_id)
        
        # Construire les données de la prédiction
        predicted = prediction.predicted_result
        odds_predicted = self.get_predicted_odds(
            predicted, 
            match.odd_home or 0, 
            match.odd_draw or 0, 
            match.odd_away or 0
        )
        
        prob_map = {'V': prediction.prob_home_win, 'N': prediction.prob_draw, 'D': prediction.prob_away_win}
        prob_predicted = prob_map.get(predicted, 0)
        
        prediction_data = {
            'confidence': prediction.confidence or 0,
            'model_agreement': prediction.model_agreement or 0,
            'odds_predicted': odds_predicted,
            'prob_predicted': prob_predicted,
            'predicted_result': predicted,
        }
        
        # 1. Filtres obligatoires
        passes_filter, rejection_reasons = self.check_hard_filters(prediction_data)
        
        # 2. Score Elite
        elite_score, breakdown = self.compute_elite_score(prediction_data)
        
        # 3. Vérifier le score minimum
        if passes_filter and elite_score < self.config['min_elite_score']:
            passes_filter = False
            rejection_reasons.append(
                f"ELITE_SCORE_LOW: {elite_score:.4f} < {self.config['min_elite_score']}"
            )
        
        is_elite = passes_filter
        
        result = {
            'is_elite': is_elite,
            'elite_score': elite_score,
            'breakdown': breakdown,
            'rejection_reasons': rejection_reasons,
            'prediction_data': prediction_data,
            'match_info': f"{match.home_team_name} vs {match.away_team_name} (J{match.matchday})",
        }
        
        if is_elite:
            logger.info(
                f"🏆 ELITE CANDIDATE: {match.home_team_name} vs {match.away_team_name} "
                f"→ {predicted} @{odds_predicted:.2f} "
                f"(score={elite_score:.4f}, conf={prediction.confidence:.3f}, "
                f"agree={prediction.model_agreement:.3f})"
            )
        else:
            self.state['candidates_rejected'] += 1
            logger.debug(
                f"❌ Elite rejected: {match.home_team_name} vs {match.away_team_name} "
                f"→ {rejection_reasons}"
            )
        
        return result
    
    def confirm_elite_prediction(self, prediction, match, elite_score: float, 
                                  breakdown: Dict) -> bool:
        """
        Confirme et enregistre une prédiction Elite.
        
        Args:
            prediction: objet Prediction
            match: objet Match
            elite_score: score calculé
            breakdown: détails du score
            
        Retourne: True si confirmé avec succès
        """
        if self.state['predictions_used'] >= self.config['max_predictions_per_season']:
            logger.warning("Cannot confirm elite: all slots used")
            return False
        
        # Vérifier que ce match n'est pas déjà dans la liste
        for ep in self.state['elite_predictions']:
            if ep.get('match_id') == match.id:
                logger.info(f"Match {match.id} already confirmed as elite")
                return False
        
        # Enregistrer
        odds_predicted = self.get_predicted_odds(
            prediction.predicted_result,
            match.odd_home or 0, match.odd_draw or 0, match.odd_away or 0
        )
        
        elite_entry = {
            'slot': self.state['predictions_used'] + 1,
            'match_id': match.id,
            'prediction_id': prediction.id,
            'matchday': match.matchday,
            'home_team': match.home_team_name,
            'away_team': match.away_team_name,
            'predicted_result': prediction.predicted_result,
            'predicted_result_name': prediction.predicted_result_name,
            'odds': odds_predicted,
            'confidence': prediction.confidence,
            'model_agreement': prediction.model_agreement,
            'elite_score': elite_score,
            'breakdown': breakdown,
            'confirmed_at': datetime.utcnow().isoformat(),
            'actual_result': None,
            'is_correct': None,
            'profit_loss': None,
        }
        
        self.state['elite_predictions'].append(elite_entry)
        self.state['predictions_used'] += 1
        self._save_state()
        
        slots_remaining = self.config['max_predictions_per_season'] - self.state['predictions_used']
        logger.info(
            f"🏆🏆🏆 ELITE PREDICTION #{self.state['predictions_used']} CONFIRMED: "
            f"{match.home_team_name} vs {match.away_team_name} → {prediction.predicted_result} "
            f"@{odds_predicted:.2f} (score={elite_score:.4f}) "
            f"[{slots_remaining} slots remaining]"
        )
        
        return True
    
    def update_elite_result(self, match_id: int, actual_result: str, 
                             profit_loss: float = None):
        """Mettre à jour le résultat d'une prédiction elite."""
        for ep in self.state['elite_predictions']:
            if ep.get('match_id') == match_id:
                ep['actual_result'] = actual_result
                ep['is_correct'] = (ep['predicted_result'] == actual_result)
                if profit_loss is not None:
                    ep['profit_loss'] = profit_loss
                elif ep['is_correct'] and ep.get('odds'):
                    # Calcul automatique: stake=1000, profit = stake * (odds - 1)
                    ep['profit_loss'] = 1000 * (ep['odds'] - 1)
                else:
                    ep['profit_loss'] = -1000
                
                self._save_state()
                
                status = "✅ CORRECT" if ep['is_correct'] else "❌ WRONG"
                logger.info(
                    f"Elite #{ep['slot']} result: {status} "
                    f"{ep['home_team']} vs {ep['away_team']} "
                    f"(predicted={ep['predicted_result']}, actual={actual_result}, "
                    f"P/L={ep['profit_loss']:.0f} Ar)"
                )
                return
    
    # ─── Batch Processing: Evaluate all predictions for a matchday ─────
    
    def evaluate_matchday(self, predictions_with_matches: List[Tuple], 
                           season_id: int) -> List[Dict]:
        """
        Évalue toutes les prédictions d'une journée et sélectionne les élites.
        
        Args:
            predictions_with_matches: list of (prediction, match) tuples
            season_id: ID de la saison active
            
        Retourne: liste des résultats d'évaluation
        """
        self.ensure_season(season_id)
        
        slots_remaining = self.config['max_predictions_per_season'] - self.state['predictions_used']
        if slots_remaining <= 0:
            logger.info("All elite slots used for this season, skipping evaluation")
            return []
        
        results = []
        candidates = []
        
        for prediction, match in predictions_with_matches:
            eval_result = self.evaluate_prediction(prediction, match, season_id)
            results.append(eval_result)
            
            if eval_result['is_elite']:
                candidates.append((eval_result, prediction, match))
        
        # Trier les candidats par score décroissant
        candidates.sort(key=lambda x: x[0]['elite_score'], reverse=True)
        
        # Sélectionner les meilleurs (limité par les slots restants)
        selected_count = 0
        for eval_result, prediction, match in candidates:
            if selected_count >= slots_remaining:
                break
            
            confirmed = self.confirm_elite_prediction(
                prediction, match, 
                eval_result['elite_score'], 
                eval_result['breakdown']
            )
            if confirmed:
                selected_count += 1
                eval_result['confirmed'] = True
        
        total_candidates = len(candidates)
        logger.info(
            f"Elite matchday evaluation: {len(predictions_with_matches)} predictions → "
            f"{total_candidates} candidates → {selected_count} confirmed elite. "
            f"Slots: {self.state['predictions_used']}/{self.config['max_predictions_per_season']}"
        )
        
        # Save state to persist candidates_rejected counter
        self._save_state()
        
        return results
    
    # ─── Status / Dashboard ────────────────────────────────────────────
    
    def get_status(self) -> Dict:
        """Retourne le statut complet pour l'API/dashboard."""
        total_profit = 0
        correct_count = 0
        verified_count = 0
        
        for ep in self.state['elite_predictions']:
            if ep.get('actual_result') is not None:
                verified_count += 1
                if ep.get('is_correct'):
                    correct_count += 1
                total_profit += (ep.get('profit_loss') or 0)
        
        pending = [ep for ep in self.state['elite_predictions'] if ep.get('actual_result') is None]
        verified = [ep for ep in self.state['elite_predictions'] if ep.get('actual_result') is not None]
        
        return {
            'max_predictions': self.config['max_predictions_per_season'],
            'predictions_used': self.state['predictions_used'],
            'slots_remaining': self.config['max_predictions_per_season'] - self.state['predictions_used'],
            'season_id': self.state.get('season_id'),
            'candidates_rejected': self.state.get('candidates_rejected', 0),
            'elite_predictions': self.state['elite_predictions'],
            'pending_predictions': pending,
            'verified_predictions': verified,
            'correct_count': correct_count,
            'verified_count': verified_count,
            'accuracy': correct_count / verified_count if verified_count > 0 else None,
            'total_profit': total_profit,
            'config': self.config,
            'last_updated': self.state.get('last_updated'),
        }
    
    def get_slots_remaining(self) -> int:
        """Nombre de slots élite restants."""
        return self.config['max_predictions_per_season'] - self.state['predictions_used']


# ─── Singleton ─────────────────────────────────────────────────────────
_elite_instance: Optional[EliteSelector] = None

def get_elite_selector() -> EliteSelector:
    global _elite_instance
    if _elite_instance is None:
        _elite_instance = EliteSelector()
    return _elite_instance
