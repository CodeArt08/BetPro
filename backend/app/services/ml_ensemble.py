"""Machine Learning Ensemble for match prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from pathlib import Path
import joblib
from sqlalchemy.orm import Session

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb

# Try to import CatBoost (optional but highly recommended)
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed. Install with: pip install catboost")

from app.core.config import settings
from app.models import MatchFeatures, ModelMetrics


class MachineLearningEnsemble:
    """
    Ensemble of ML models for match outcome prediction.
    """
    
    def __init__(self):
        self.models: Dict = {}
        self.model_weights: Dict = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names: List[str] = []
        self.calibrators: Dict = {}
        self.use_calibration = True
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize all models with optimized hyperparameters."""
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=2000,
                C=0.5,
                penalty='l2',
                solver='lbfgs',
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=250,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1,
                tree_method='hist'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.07,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=10,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                boosting_type='dart'
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.01,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42
            )
        }
        
        # Add CatBoost if available (excellent for tabular data)
        if HAS_CATBOOST:
            self.models['catboost'] = cb.CatBoostClassifier(
                iterations=300,
                depth=7,
                learning_rate=0.08,
                l2_leaf_reg=3.0,
                border_count=128,
                random_state=42,
                verbose=0,
                allow_writing_files=False,
                auto_class_weights='Balanced'
            )
        
        # Initialize equal weights
        n_models = len(self.models)
        self.model_weights = {name: 1.0 / n_models for name in self.models}
    
    def prepare_features(self, features: MatchFeatures) -> np.ndarray:
        """Prepare features for prediction."""
        feature_dict = features.to_dict()
        self.feature_names = list(feature_dict.keys())
        return np.array([list(feature_dict.values())])
    
    def train(self, X: pd.DataFrame, y: pd.Series, db: Session = None):
        """
        Train all models on the dataset with advanced techniques.
        """
        logger.info(f"Training ensemble on {len(X)} samples with {len(X.columns)} features...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use stratified k-fold for better validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        best_scores = {}
        
        # Train each model with cross-validation
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                logger.info(f"{name} CV accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
                
                # Fit on full data
                model.fit(X_scaled, y_encoded)
                best_scores[name] = mean_score
                
                # Calibrate probabilities if enabled
                if self.use_calibration and name not in ['mlp', 'catboost']:
                    try:
                        calibrator = CalibratedClassifierCV(
                            model, 
                            method='isotonic' if len(X) > 500 else 'sigmoid',
                            cv='prefit'
                        )
                        calibrator.fit(X_scaled, y_encoded)
                        self.calibrators[name] = calibrator
                        logger.info(f"{name} calibrated successfully")
                    except Exception as e:
                        logger.warning(f"Calibration failed for {name}: {e}")
                
                # Update metrics in database
                if db:
                    self._update_model_metrics(name, model, X_scaled, y_encoded, db)
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                best_scores[name] = 0.0
        
        # Update weights based on performance (weighted by CV score)
        self._update_weights_advanced(best_scores)
        
        self.is_trained = True
        logger.info(f"Ensemble training completed. Best model: {max(best_scores, key=best_scores.get)} ({max(best_scores.values()):.4f})")
    
    def _update_weights_advanced(self, scores: Dict[str, float]):
        """Update ensemble weights using softmax over CV scores."""
        import math
        
        # Temperature parameter for softmax (lower = more concentrated on best)
        temperature = 0.1
        
        # Shift scores to avoid overflow in exp
        max_score = max(scores.values()) if scores else 0.5
        
        exp_scores = {}
        for name, score in scores.items():
            # Boost score slightly to avoid zero weights
            adjusted_score = max(score, 0.3)
            exp_scores[name] = math.exp(adjusted_score / temperature)
        
        total = sum(exp_scores.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in exp_scores.items()}
        else:
            n = len(self.models)
            self.model_weights = {k: 1.0 / n for k in self.models}
        
        logger.info(f"Updated weights (softmax): {self.model_weights}")
    
    def _update_weights(self, X: np.ndarray, y: np.ndarray):
        """Update ensemble weights based on model performance."""
        performances = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                performances[name] = accuracy
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
                performances[name] = 0.5
        
        # Normalize weights
        total_perf = sum(performances.values())
        if total_perf > 0:
            self.model_weights = {k: v / total_perf for k, v in performances.items()}
        else:
            n = len(self.models)
            self.model_weights = {k: 1.0 / n for k in self.models}
        
        logger.info(f"Updated weights: {self.model_weights}")
    
    def _update_model_metrics(self, model_name: str, model, X: np.ndarray, y: np.ndarray, db: Session):
        """Update model metrics in database."""
        metrics = db.query(ModelMetrics).filter(
            ModelMetrics.model_name == model_name
        ).first()
        
        if not metrics:
            metrics = ModelMetrics(
                model_name=model_name,
                model_version='1.0'
            )
            db.add(metrics)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        metrics.total_predictions = len(y)
        metrics.correct_predictions = int(accuracy_score(y, y_pred) * len(y))
        metrics.accuracy = accuracy_score(y, y_pred)
        
        # Per-class metrics
        for i, cls in enumerate(self.label_encoder.classes_):
            mask = (y == i)
            pred_mask = (y_pred == i)
            
            if cls == 'V':
                metrics.home_win_predictions = int(pred_mask.sum())
                metrics.home_win_correct = int((mask & pred_mask).sum())
                metrics.home_win_accuracy = metrics.home_win_correct / metrics.home_win_predictions if metrics.home_win_predictions > 0 else 0
            elif cls == 'N':
                metrics.draw_predictions = int(pred_mask.sum())
                metrics.draw_correct = int((mask & pred_mask).sum())
                metrics.draw_accuracy = metrics.draw_correct / metrics.draw_predictions if metrics.draw_predictions > 0 else 0
            elif cls == 'D':
                metrics.away_win_predictions = int(pred_mask.sum())
                metrics.away_win_correct = int((mask & pred_mask).sum())
                metrics.away_win_accuracy = metrics.away_win_correct / metrics.away_win_predictions if metrics.away_win_predictions > 0 else 0
        
        db.commit()
    
    def predict(self, features: MatchFeatures) -> Dict[str, Dict]:
        """
        Generate predictions from all models with calibration.
        Returns model outputs and ensemble prediction.
        """
        if not self.is_trained:
            logger.warning("Models not trained, returning uniform probabilities")
            return {
                'ensemble': {'V': 0.33, 'N': 0.33, 'D': 0.33},
                'model_outputs': {},
                'predicted_result': 'N',
                'confidence': 0.33
            }
        
        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        model_outputs = {}
        weighted_probs = np.zeros(3)
        total_weight = 0.0
        
        for name, model in self.models.items():
            try:
                # Use calibrated probabilities if available
                if name in self.calibrators:
                    probs = self.calibrators[name].predict_proba(X_scaled)[0]
                else:
                    probs = model.predict_proba(X_scaled)[0]
                
                classes = self.label_encoder.classes_
                
                output = {cls: float(probs[i]) for i, cls in enumerate(classes)}
                model_outputs[name] = output
                
                # Weight the probabilities
                weight = self.model_weights.get(name, 1.0 / len(self.models))
                weighted_probs += probs * weight
                total_weight += weight
                
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        # Normalize weighted probabilities
        if total_weight > 0:
            weighted_probs = weighted_probs / total_weight
        else:
            weighted_probs = np.array([0.33, 0.33, 0.34])
        
        # Apply temperature scaling to sharpen predictions slightly
        temperature = 1.2
        weighted_probs = np.exp(weighted_probs / temperature)
        weighted_probs = weighted_probs / weighted_probs.sum()
        
        # Create ensemble output
        classes = self.label_encoder.classes_
        ensemble_output = {cls: float(weighted_probs[i]) for i, cls in enumerate(classes)}
        
        # Predicted result
        predicted_idx = np.argmax(weighted_probs)
        predicted_result = classes[predicted_idx]
        confidence = float(weighted_probs[predicted_idx])
        
        return {
            'ensemble': ensemble_output,
            'model_outputs': model_outputs,
            'predicted_result': predicted_result,
            'confidence': confidence
        }
    
    def predict_batch(self, features_list: List[MatchFeatures]) -> List[Dict]:
        """Predict for multiple matches."""
        return [self.predict(f) for f in features_list]
    
    def calculate_model_agreement(self, model_outputs: Dict) -> float:
        """Calculate how many models agree on the prediction."""
        if not model_outputs:
            return 0.0
        
        predictions = [max(output, key=output.get) for output in model_outputs.values()]
        
        # Most common prediction
        if predictions:
            most_common = max(set(predictions), key=predictions.count)
            agreement = predictions.count(most_common) / len(predictions)
            return agreement
        
        return 0.0
    
    def save_models(self, path: Path = None):
        """Save all models to disk."""
        if path is None:
            path = settings.MODELS_DIR
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = path / f"{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save scaler and encoder
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.label_encoder, path / "label_encoder.joblib")
        joblib.dump(self.model_weights, path / "weights.joblib")
        joblib.dump(self.calibrators, path / "calibrators.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        
        logger.info("All models saved successfully")
    
    def load_models(self, path: Path = None):
        """Load models from disk."""
        if path is None:
            path = settings.MODELS_DIR
        
        try:
            # Load each model
            for name in self.models.keys():
                model_path = path / f"{name}.joblib"
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name}")
            
            # Load scaler and encoder
            self.scaler = joblib.load(path / "scaler.joblib")
            self.label_encoder = joblib.load(path / "label_encoder.joblib")
            self.model_weights = joblib.load(path / "weights.joblib")
            
            # Load calibrators if available
            calibrators_path = path / "calibrators.joblib"
            if calibrators_path.exists():
                self.calibrators = joblib.load(calibrators_path)
            
            # Load feature names if available
            feature_names_path = path / "feature_names.joblib"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
            
            self.is_trained = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
    
    def update_weights_from_metrics(self, db: Session):
        """Update ensemble weights from database metrics."""
        metrics_list = db.query(ModelMetrics).all()
        
        for metrics in metrics_list:
            if metrics.model_name in self.models:
                self.model_weights[metrics.model_name] = metrics.ensemble_weight
        
        # Normalize
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}
        
        logger.info(f"Updated weights from metrics: {self.model_weights}")


class NeuralNetworkPredictor:
    """
    Optional neural network predictor using TensorFlow/Keras.
    """
    
    def __init__(self, input_dim: int):
        self.model = None
        self.input_dim = input_dim
    
    def build_model(self):
        """Build the neural network."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            self.model = Sequential([
                Dense(128, activation='relu', input_dim=self.input_dim),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Neural network built successfully")
            
        except ImportError:
            logger.warning("TensorFlow not available, skipping neural network")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the neural network."""
        if self.model is None:
            self.build_model()
        
        if self.model:
            from tensorflow.keras.utils import to_categorical
            
            y_cat = to_categorical(y, num_classes=3)
            
            self.model.fit(
                X, y_cat,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with neural network."""
        if self.model:
            return self.model.predict(X, verbose=0)
        return np.ones((len(X), 3)) / 3
