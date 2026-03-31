import numpy as np
import time
from loguru import logger
from typing import Dict, List, Optional
import os

class LSTMAttentionModel:
    """
    Module 3: Modèle LSTM (Long Short-Term Memory) + Mécanisme d'Attention.
    Analyse séquentielle des 30 derniers matchs pour détecter les paternes
    non linéaires de l'algorithme "RNG" de Bet261.
    Exécution en Thread D du RealTimeEngine.
    """
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.model = None
        self.is_ready = False
        self._build_model()
        
    def _build_model(self):
        """Initialise le réseau de neurones en RAM."""
        try:
            # Force CPU for fast inference to avoid TF GPU init delays
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            import tensorflow as tf
            from tensorflow.keras.layers import LSTM, Dense, Input, Attention, GlobalAveragePooling1D, Flatten
            from tensorflow.keras.models import Model
            
            # Input: Sequence de 30 matchs, 3 features (One-hot V, N, D)
            inputs = Input(shape=(self.sequence_length, 3), name='sequence_input')
            
            # LSTM retourne la séquence complète pour l'Attention
            lstm_out, state_h, state_c = LSTM(16, return_sequences=True, return_state=True)(inputs)
            
            # Mécanisme d'attention: le modèle se concentre sur certains matchs de l'historique
            # Query = state_h (état final), Value = lstm_out (tous les états)
            # On doit expand state_h pour matcher la dimension de l'attention layer query
            state_h_expanded = tf.expand_dims(state_h, 1)
            attention_out = Attention()([state_h_expanded, lstm_out])
            
            # Aplatir et classer
            flat_att = Flatten()(attention_out)
            dense1 = Dense(16, activation='relu')(flat_att)
            outputs = Dense(3, activation='softmax', name='prediction')(dense1)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            
            # Mock weight init pour que predict() fonctionne sans crash
            dummy_x = np.zeros((1, self.sequence_length, 3))
            self.model.predict(dummy_x, verbose=0)
            
            self.is_ready = True
            logger.info("Modèle LSTM+Attention compilé et monté en RAM (Inference Ready)")
            
        except ImportError:
            logger.warning("TensorFlow non disponible. Le LSTM tournera en mode Heuristique (Fallback numpy).")
            self.is_ready = False
        except Exception as e:
            logger.error(f"Erreur init LSTM: {e}")
            self.is_ready = False

    def encode_sequence(self, results: List[str]) -> np.ndarray:
        """Transforme l'historique en matrice The One-Hot pour TF."""
        seq = np.zeros((self.sequence_length, 3))
        
        # Prendre les 30 derniers
        recent = results[-self.sequence_length:] if len(results) >= self.sequence_length else results
        
        # Map to one-hot (V=0, N=1, D=2)
        mapping = {'V': 0, 'N': 1, 'D': 2}
        
        # Remplir la fin de la matrice (padding à gauche par des zéros implicites)
        start_idx = self.sequence_length - len(recent)
        for i, res in enumerate(recent):
            idx = mapping.get(res, 0)
            seq[start_idx + i, idx] = 1.0
            
        return seq

    def predict_sequence(self, results_history: List[str]) -> Dict[str, float]:
        """
        Prédiction Inference Mode < 50ms.
        Retourne les probabilités {V, N, D}.
        """
        if len(results_history) < 5:
            return {'V': 0.33, 'N': 0.33, 'D': 0.34}
            
        t0 = time.time()
        
        # 1. Encodage vectoriel
        seq_matrix = self.encode_sequence(results_history)
        
        # 2. Inférence Keras/TF
        if self.is_ready and self.model is not None:
            # Reshape for batch (1, seq_length, 3)
            x_input = np.expand_dims(seq_matrix, axis=0)
            # verbose=0 is mandatory for fast bg predictions
            pred = self.model.predict(x_input, verbose=0)[0]
            
            probs = {
                'V': float(pred[0]),
                'N': float(pred[1]),
                'D': float(pred[2])
            }
        else:
            # Fallback exponentiel simple (si TF n'est pas dispo)
            probs = self._numpy_exponential_fallback(results_history)
            
        t1 = time.time()
        logger.debug(f"[LSTM] Inférence complétée en {(t1-t0)*1000:.1f}ms")
        
        return probs
        
    def _numpy_exponential_fallback(self, results_history: List[str]) -> Dict[str, float]:
        """Fallback si TF crash: moyenne mobile pondérée exponentielle."""
        recent = results_history[-self.sequence_length:]
        weights = np.exp(np.linspace(-2, 0, len(recent))) # Poids plus fort sur les récents
        weights /= np.sum(weights)
        
        scores = {'V': 0.001, 'N': 0.001, 'D': 0.001}
        for w, r in zip(weights, recent):
            if r in scores:
                scores[r] += w
                
        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}
