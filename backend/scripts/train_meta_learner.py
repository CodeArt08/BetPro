"""
Script d'initialisation et d'entraînement du StackedMetaLearner.
À exécuter une fois depuis le répertoire backend/ pour entraîner le meta-model.

Usage:
    cd backend
    python -m app.services.init_meta_learner
    
    OR
    python scripts/train_meta_learner.py
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from loguru import logger


def main():
    logger.info("=== StackedMetaLearner Training Script ===")
    
    try:
        from app.services.stacked_meta_learner import StackedMetaLearner
        
        learner = StackedMetaLearner()
        
        if learner.is_trained:
            logger.info(f"Meta-learner already trained. Stats: {learner.get_stats()}")
            response = input("Retrain? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping retraining.")
                return
        
        logger.info("Starting meta-learner training from DB history...")
        stats = learner.train_from_history(min_samples=100)
        
        if 'error' in stats:
            logger.error(f"Training failed: {stats['error']}")
            if 'samples' in stats:
                logger.error(f"Only {stats['samples']} samples available (need >= 100)")
        else:
            logger.success(f"✅ Meta-learner trained successfully!")
            logger.info(f"   Samples: {stats.get('n_samples', 'N/A')}")
            logger.info(f"   CV Accuracy: {stats.get('cv_accuracy', 0):.4f} ± {stats.get('cv_std', 0):.4f}")
            logger.info(f"   Train Accuracy: {stats.get('train_accuracy', 0):.4f}")
            logger.info(f"   Class distribution: {stats.get('class_distribution', {})}")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure sklearn is installed: pip install scikit-learn")
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
