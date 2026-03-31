"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import asyncio

from app.core.config import settings, ensure_directories
from app.core.database import init_db
from app.api import api_router
from app.core.socket_manager import sio_app


if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    settings.LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level=settings.LOG_LEVEL
)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Ensure directories exist
    ensure_directories()
    
    app = FastAPI(
        title="Bet261 Prediction Engine",
        description="Autonomous AI Prediction Engine for Bet261 Virtual Spanish League",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/api")
    
    # Mount Socket.IO
    app.mount("/socket.io", sio_app)
    
    @app.on_event("startup")
    async def startup_event():
        """Run on application startup."""
        logger.info("Starting Bet261 Prediction Engine...")
        
        # Initialize database
        try:
            init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
        
        # Load saved models
        try:
            from app.services.continuous_learning import ContinuousLearningEngine
            engine = ContinuousLearningEngine()
            engine.load_models()
            logger.info("ML models loaded")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
        
        # ── Initialize Real-Time Engine ──────────────────────────────
        try:
            from app.services.realtime_engine import get_engine
            from app.core.database import SessionLocal
            
            rt_engine = get_engine()
            rt_engine.start()
            logger.info("Real-Time Engine started")
            
            # Load historical results into RAM
            db = SessionLocal()
            try:
                from app.models import Match
                matches = db.query(Match).filter(
                    Match.is_completed == True,
                    Match.result != None
                ).order_by(Match.matchday).all()
                
                results = [m.result for m in matches if m.result]
                results_by_line = {}
                results_by_hour = {}
                score_counts = {}
                
                for m in matches:
                    if not m.result:
                        continue
                    lp = getattr(m, 'line_position', 1) or 1
                    if lp not in results_by_line:
                        results_by_line[lp] = []
                    results_by_line[lp].append(m.result)
                    if m.score_home is not None and m.score_away is not None:
                        sk = f"{m.score_home}-{m.score_away}"
                        score_counts[sk] = score_counts.get(sk, 0) + 1
                
                rt_engine.load_historical_results(
                    results=results,
                    results_by_line=results_by_line,
                    results_by_hour=results_by_hour,
                    score_counts=score_counts,
                )
                logger.info(f"Loaded {len(results)} historical results into RT engine")
            finally:
                db.close()
                
        except Exception as e:
            logger.warning(f"Real-Time Engine initialization warning: {e}")
            
        # Orchestrator disabled - manual control via API buttons
        # User controls scraping manually via "Capturer Résultats" and "Capturer Matchs" buttons
        logger.info("Orchestrator disabled - manual scraping control enabled")
            
        logger.info("Application started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run on application shutdown."""
        logger.info("Shutting down Bet261 Prediction Engine...")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Bet261 Prediction Engine",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
