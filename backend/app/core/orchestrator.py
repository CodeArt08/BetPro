"""Main orchestration loop for the autonomous system."""
import asyncio
import re
from typing import Optional, Dict, List
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db_context
from app.core.socket_manager import socket_manager
from app.services.bet261_scraper import Bet261Scraper
from app.services.data_extraction import DataExtractionEngine
from app.services.feature_engineering import FeatureEngineeringPipeline
from app.services.team_strength import TeamStrengthEngine
from app.services.sequence_analysis import SequencePatternAnalyzer
from app.services.ml_ensemble import MachineLearningEnsemble
from app.services.monte_carlo import MonteCarloSimulator
from app.services.odds_analysis import OddsAnalyzer
from app.services.betting_engine import BettingDecisionEngine
from app.services.bankroll import BankrollManager
from app.services.season_manager import SeasonManager
from app.services.continuous_learning import ContinuousLearningEngine
# Ultra-precise prediction imports
from app.services.enhanced_learning import EnhancedContinuousLearning
from app.services.adaptive_ensemble import AdaptiveEnsemble, PredictionFilter
from app.services.bivariate_poisson import BivariatePoissonModel, EnhancedMonteCarlo
from app.services.advanced_features import AdvancedFeatureEngine
from app.services.dynamic_selection_engine import DynamicSelectionEngine
from app.services.realtime_engine import get_engine
from app.services.elite_selector import get_elite_selector

from app.models import Match, Prediction, MatchFeatures, Season, Bet, Team


class PredictionOrchestrator:
    """
    Main orchestrator that coordinates all system components.
    Runs the autonomous prediction loop.
    
    State Machine:
    - Phase 1 (Startup): Season detection and backfill
    - Phase 2 (LIVE): Wait for match to end
    - Phase 3 (UPCOMING): Extract matches with odds, generate predictions
    - Phase 4 (RESULT): Compare predictions with actual results, show modal
    """
    
    # Fixed stake in Ariary
    FIXED_STAKE_ARIARY = 1000
    UPCOMING_PREDICTION_COUNT = 10
    
    def __init__(self):
        # Initialize all services
        self.scraper = Bet261Scraper()
        self.data_extractor = DataExtractionEngine()
        self.feature_engineer = FeatureEngineeringPipeline()
        self.team_strength = TeamStrengthEngine()
        self.sequence_analyzer = SequencePatternAnalyzer()
        self.ml_ensemble = MachineLearningEnsemble()
        self.monte_carlo = MonteCarloSimulator()
        self.odds_analyzer = OddsAnalyzer()
        self.betting_engine = BettingDecisionEngine()
        self.bankroll = BankrollManager()
        self.season_manager = SeasonManager()
        # Dynamic selection engine for intelligent bet selection
        self.dynamic_selection = DynamicSelectionEngine()
        # Elite selector: 5 predictions per season ultra-selective system
        self.elite_selector = get_elite_selector()
        # Real-time engine for M1-M15 signals
        self.realtime_engine = get_engine()
        
        self.is_running = False
        self.cycle_count = 0
        self._season_ready = False
        
        # State tracking for the prediction cycle
        self._current_upcoming_matchday: Optional[int] = None
        self._predictions_for_comparison: List[Dict] = []
        self._last_comparison_results: Optional[Dict] = None
    
    async def start(self):
        """Start the autonomous system with simplified startup flow.
        
        Flow:
        1. Go to RESULTS once, load all results (click 'Afficher plus' until J1)
        2. Compare J1 match#1 with DB to detect new season
        3. Backfill missing results if needed
        4. Go to MATCHES once
        5. If LIVE -> wait, else -> extract upcoming matches with odds
        """
        logger.info("=" * 60)
        logger.info("Starting Bet261 Prediction Engine")
        logger.info("=" * 60)
        
        self.is_running = True
        
        # Load saved models (use enhanced learning)
        self.enhanced_learning.load_models()
        # Ensure models are trained at least once with ALL historical data
        with get_db_context() as db:
            self.enhanced_learning.ensure_models_trained(db)
        
        # Start scraper
        await self.scraper.start()
        
        # Set up callbacks
        self.scraper.on_results = self._on_new_result
        self.scraper.on_results_fast = self._on_new_result_fast
        self.scraper.on_upcoming_matches = self._on_upcoming_matches
        self.scraper.on_prediction_result = self._on_prediction_result

        # === SIMPLIFIED STARTUP FLOW ===
        # Step 1: Go to RESULTS page ONCE and get all data
        logger.info("Step 1: Going to RESULTS page...")
        await self.scraper._safe_goto(self.scraper.RESULTS_URL)
        
        # Wait longer for page to fully load (cookie banner, dynamic content)
        await asyncio.sleep(8)
        
        # Accept cookies if banner appears - try multiple selectors
        cookie_accepted = False
        cookie_selectors = [
            'button:has-text("Autoriser tous les cookies")',
            'button:has-text("Autoriser")',
            'button:has-text("Autoriser tous")',
            'button:has-text("Accepter")',
            'button:has-text("Tout accepter")',
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("Allow")',
            '#accept-cookies',
            '.cookie-accept',
            '[class*="cookie"] button',
            'button[id*="accept"]',
        ]
        
        for selector in cookie_selectors:
            try:
                cookie_btn = await self.scraper.page.query_selector(selector)
                if cookie_btn:
                    is_visible = await cookie_btn.is_visible()
                    if is_visible:
                        await cookie_btn.click()
                        logger.info(f"Accepted cookies via: {selector}")
                        cookie_accepted = True
                        await asyncio.sleep(2)
                        break
            except Exception as e:
                logger.debug(f"Cookie selector {selector} failed: {e}")
        
        if not cookie_accepted:
            logger.debug("No cookie banner found or already accepted")
        
        # Verify we're on results page
        current_url = self.scraper.page.url
        logger.info(f"Current URL after navigation: {current_url}")
        
        # If not on results page, navigate again
        if "results" not in current_url:
            logger.warning("Not on results page, navigating again...")
            await self.scraper._safe_goto(self.scraper.RESULTS_URL)
            await asyncio.sleep(5)
        
        # Wait for results content to load - retry if page is empty
        for attempt in range(3):
            try:
                await self.scraper.page.wait_for_selector('[class*="result"], [class*="match"], .score', timeout=15000)
                logger.info("Results content detected on page")
                break
            except Exception as e:
                logger.warning(f"Timeout waiting for results content (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await self.scraper.page.reload()
                    await asyncio.sleep(5)
        
        # Verify page has enough content
        page_text = await self.scraper.page.inner_text('body')
        if len(page_text) < 1000:
            logger.warning(f"Page content too small ({len(page_text)} chars), reloading...")
            await self.scraper.page.reload()
            await asyncio.sleep(5)
        
        # Load all results by clicking 'Afficher plus' until J1
        clicks = await self.scraper._load_more_results(max_clicks=30)
        logger.info(f"Clicked 'Afficher plus' {clicks} times")
        
        # Wait after loading more
        await asyncio.sleep(2)
        
        # Step 2: Detect season and backfill in ONE pass
        logger.info("Step 2: Detecting season and backfilling...")
        season_ok = await self._detect_and_backfill_season()
        
        if not season_ok:
            logger.error("Season verification failed - will retry in main loop")
            self._season_ready = False
        else:
            self._season_ready = True
            self.scraper.matches_enabled = True
            
            # Step 3: Go to MATCHES page
            logger.info("Step 3: Going to MATCHES page...")
            await self.scraper._safe_goto(self.scraper.MATCHES_URL)
            await asyncio.sleep(1)  # Réduit de 3s à 1s
            
            # Check if LIVE or UPCOMING
            is_live = await self.scraper._is_live_phase()
            
            if is_live:
                logger.info("MATCHES page is in LIVE phase - waiting...")
                self.scraper._matches_was_live = True
            else:
                logger.info("MATCHES page shows UPCOMING matches - extracting...")
                # Wait for content to load
                try:
                    await self.scraper.page.wait_for_selector('[class*="match"], [class*="odds"], [class*="outcome"]', timeout=10000)
                except:
                    pass
                await asyncio.sleep(2)
                
                matches = await self.scraper._extract_matches(fallback_matchday=1)
                if matches and self.scraper.on_upcoming_matches:
                    await self.scraper.on_upcoming_matches(matches)
                    await self._generate_predictions()
                    await self._store_predictions_for_comparison()
        
        # Cleanup stale data
        await self._cleanup_stale_data()
        
        # Main loop
        await self._main_loop()
    
    async def _detect_and_backfill_season(self) -> bool:
        """Detect season and backfill in one pass (called after loading all results).
        
        This is called AFTER the scraper has already loaded all results via 'Afficher plus'.
        No additional navigation needed.
        """
        logger.info("Detecting season from current results page...")
        
        try:
            # First, log what's on the page for debugging
            page_text = await self.scraper.page.inner_text('body')
            logger.info(f"Page content length: {len(page_text)} chars")
            logger.debug(f"First 500 chars: {page_text[:500]}")
            
            # Extract all results from current page (already loaded)
            results = await self.scraper._extract_results()
            
            if not results:
                logger.error("No results found on page")
                return False
            
            
            # Find J1 match#1 from extracted results
            site_first = None
            for r in results:
                if r.get('matchday') == 1 and r.get('line_position') == 1:
                    site_first = r
                    break
            
            
            if not site_first:
                logger.warning("J1 match#1 not found in results - may need more 'Afficher plus' clicks")
                # Try to find the oldest matchday
                min_matchday = min(r.get('matchday', 1) for r in results)
                logger.info(f"Oldest matchday found: J{min_matchday}")
            else:
                logger.info(f"Site J1 match#1: {site_first.get('home_team')} {site_first.get('score_home')}-{site_first.get('score_away')} {site_first.get('away_team')}")
            
            # Get DB state
            with get_db_context() as db:
                current_season = db.query(Season).filter(Season.is_active == True).first()
                
                if not current_season:
                    logger.info("No active season - creating new one and backfilling...")
                    self.season_manager.create_new_season(db)
                    db.commit()
                    current_season = db.query(Season).filter(Season.is_active == True).first()
                
                
                # Get DB's J1 match#1
                db_first_match = db.query(Match).filter(
                    Match.season_id == current_season.id,
                    Match.is_completed == True,
                    Match.matchday == 1,
                    Match.line_position == 1
                ).first()
                
                db_sig = None
                if db_first_match:
                    db_sig = (
                        db_first_match.home_team_name,
                        db_first_match.away_team_name,
                        db_first_match.score_home,
                        db_first_match.score_away,
                    )
                    logger.info(f"DB J1 match#1: {db_sig[0]} {db_sig[2]}-{db_sig[3]} {db_sig[1]}")
                
                
                # Compare signatures
                if site_first and db_sig:
                    site_sig = (
                        site_first.get('home_team'),
                        site_first.get('away_team'),
                        site_first.get('score_home'),
                        site_first.get('score_away'),
                    )
                    
                    if site_sig != db_sig:
                        # NEW SEASON!
                        logger.info("NEW SEASON DETECTED! Signatures don't match.")
                        await self._handle_new_season(db, current_season)
                    else:
                        logger.info("Same season - backfilling missing results...")
                
                
                # Backfill all results from current page
                logger.info(f"Processing {len(results)} results from page...")
                for result in results:
                    await self._on_new_result_fast(result)
                
                
                return True
                
        except Exception as e:
            logger.error(f"Error in season detection/backfill: {e}")
            return False
    
    async def _handle_new_season(self, db, old_season):
        """Handle new season reset."""
        logger.info("Handling new season...")
        
        # Deactivate old season
        old_season.is_active = False
        old_season.ended_at = datetime.utcnow()
        
        # Create new season
        new_season = self.season_manager.create_new_season(db)
        db.commit()
        
        logger.info(f"Created new season: {new_season.id}")
    
    async def _detect_and_handle_new_season(self) -> bool:
        """Legacy method - now just calls the simplified version."""
        return await self._detect_and_backfill_season()

    async def _cleanup_stale_data(self):
        """Cleanup stale upcoming matches and predictions on startup."""
        logger.info("Cleaning up stale data...")
        try:
            with get_db_context() as db:
                # 1. Detect current matchday from site
                site_matchday = await self.scraper.detect_current_matchday()
                if site_matchday == 0:
                    # Fallback to DB state if site fails
                    last_match = db.query(Match).filter(Match.is_completed == True).order_by(Match.matchday.desc()).first()
                    site_matchday = last_match.matchday if last_match else 1
                
                logger.info(f"System identified current matchday as: J{site_matchday}")
                
                # 2. Mark matches for PAST matchdays as not upcoming
                stale_matches = db.query(Match).filter(
                    Match.is_upcoming == True,
                    Match.matchday < site_matchday
                ).all()
                
                if stale_matches:
                    logger.info(f"Marking {len(stale_matches)} stale matches from J<{site_matchday} as not upcoming")
                    for m in stale_matches:
                        m.is_upcoming = False
                    db.commit()
                
                # 3. Handle predictions for current/future matchdays that might be stale
                # If a match is upcoming but its odds have changed or it was from a previous app run,
                # the user wants us to "reinitialize". 
                # We'll remove predictions for matches that are upcoming so they're re-generated.
                upcoming_matches = db.query(Match).filter(Match.is_upcoming == True).all()
                for match in upcoming_matches:
                    # Optional: Remove predictions to force regeneration on startup
                    # This ensures we have the "prediction actuelle" the user asked for
                    stale_pred = db.query(Prediction).filter(Prediction.match_id == match.id).first()
                    if stale_pred:
                        # First delete ALL bets linked to this prediction (FK constraint)
                        stale_bets = db.query(Bet).filter(Bet.prediction_id == stale_pred.id).all()
                        for stale_bet in stale_bets:
                            logger.info(f"Removing stale bet for {match.home_team_name} vs {match.away_team_name}")
                            db.delete(stale_bet)
                        db.commit()  # Commit bet deletions before deleting prediction
                        logger.info(f"Removing stale prediction for {match.home_team_name} vs {match.away_team_name} (J{match.matchday})")
                        db.delete(stale_pred)
                
                db.commit()
                logger.info("Stale data cleanup complete.")
                
                # Broadcast update so frontend reflects the cleanup
                await socket_manager.broadcast("upcoming_matches_updated", {"matchday": site_matchday})
                await socket_manager.broadcast("prediction_generated", {"matchday": site_matchday})
        except Exception as e:
            logger.error(f"Error during stale data cleanup: {e}")
    
    async def _reset_for_new_season(self):
        """Reset system for a new season."""
        logger.info("=" * 50)
        logger.info("RESETTING FOR NEW SEASON")
        logger.info("=" * 50)
        
        with get_db_context() as db:
            # Close old season (this creates/activates the next season internally)
            new_season = self.season_manager.close_season(db)

            # Reset teams for the newly created season
            self.season_manager.reset_teams_for_new_season(db)

            logger.info(f"Created new season: {new_season.id}")
        
        # Backfill all results from site
        logger.info("Backfilling all results from site...")
        await self.scraper.backfill_all_results()
        
        logger.info("New season setup complete!")
    
    async def _main_loop(self):
        """Main loop - minimal navigation, no repeated cycles.
        
        Optimized to stay on MATCHES page and only go to RESULTS when needed.
        """
        logger.info("Entering optimized prediction loop...")
        
        while self.is_running:
            try:
                # 1. Stay on MATCHES page as the main observation post
                current_url = self.scraper.page.url if self.scraper.page else ""
                if self.scraper.MATCHES_URL not in current_url and self.scraper.RESULTS_URL not in current_url:
                    logger.info("Navigating to MATCHES page...")
                    await self.scraper._safe_goto(self.scraper.MATCHES_URL)
                    await asyncio.sleep(2)

                # 2. Check if currently LIVE
                is_live = await self.scraper._is_live_phase()
                
                if is_live:
                    # LIVE phase detection
                    elapsed = await self._get_match_elapsed_time()
                    if elapsed and elapsed < 90:
                        remaining_real = int((90 - elapsed) * (50 / 90))
                        logger.info(f"LIVE {elapsed}' - Staying on page for ~{remaining_real}s")
                        # Wait most of the time but don't block forever
                        await asyncio.sleep(min(remaining_real + 2, 15))  # Réduit max de 25s à 15s
                    else:
                        logger.info("LIVE phase ending. Waiting 8s for transition...")
                        await asyncio.sleep(3)  # Réduit de 8s à 3s
                    
                    # Transition: LIVE -> END
                    # Fetch results once after live ends
                    logger.info("Cycle finished. Fetching official results...")
                    await self._fetch_results_once()
                    
                    if self._predictions_for_comparison:
                        await self._compare_predictions_with_results()
                    
                    # Return to MATCHES page immediately
                    await self.scraper._safe_goto(self.scraper.MATCHES_URL)
                    await asyncio.sleep(2)
                    
                else:
                    # NOT LIVE -> UPCOMING matches
                    # We should already be on MATCHES page
                    if self.scraper.RESULTS_URL in current_url:
                        await self.scraper._safe_goto(self.scraper.MATCHES_URL)
                        await asyncio.sleep(2)
                        
                    # Extract matches and odds. This will also trigger predictions.
                    await self._extract_upcoming_once()
                    
                    # If extraction was successful and we predicted, wait for kickoff
                    if self._predictions_for_comparison:
                        countdown = await self.scraper._get_countdown()
                        if countdown and countdown > 0:
                            logger.info(f"J{self._current_upcoming_matchday} starts in {countdown}s. Staying on page.")
                            # Long sleep if countdown is high
                            if countdown > 20:
                                await asyncio.sleep(max(countdown - 10, 5))  # Réduit attente
                            else:
                                await asyncio.sleep(2)  # Réduit de 5s à 2s
                        else:
                            # Waiting for kickoff transition
                            logger.info("Waiting for kickoff - checking state in 10s...")
                            await asyncio.sleep(5)  # Réduit de 10s à 5s
                            # Reload occasionally to see state change
                            await self.scraper.page.reload()
                            await asyncio.sleep(0.5)  # Réduit de 2s à 0.5s
                    else:
                        # No matches captured yet or missing odds
                        logger.info("Waiting for upcoming matches with odds... (15s)")
                        await asyncio.sleep(8)  # Réduit de 15s à 8s
                        await self.scraper.page.reload()
                        await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(15)
    
    async def _get_match_elapsed_time(self) -> Optional[int]:
        """Get elapsed minutes from LIVE match."""
        try:
            page_text = await self.scraper.page.inner_text('body')
            match = re.search(r"(\d{1,2})'", page_text)
            if match:
                return int(match.group(1))
        except:
            pass
        return None
    
    async def _fetch_results_once(self):
        """RESULTS once, fetch, return - let _scrape_results handle navigation and clicks."""
        # _scrape_results handles its own navigation + load_more clicks internally
        await self.scraper._scrape_results()
    
    def _get_db_last_matchday(self) -> int:
        """Get the last completed matchday from database."""
        return self.scraper._get_db_last_matchday()
        
    def _get_next_matchday_from_db(self) -> int:
        """Get the next expected matchday based on DB state."""
        last_md = self._get_db_last_matchday()
        return last_md + 1

    async def _extract_upcoming_once(self):
        """MATCHES once, extract, generate predictions - FAST (cycle 30s)."""
        # Use only_if_different to avoid reload if already there
        await self.scraper._safe_goto(self.scraper.MATCHES_URL, only_if_different=True)
        await asyncio.sleep(0.5)
        
        # Click on MATCHS tab if needed (often required on Bet261 mobile/new UI)
        try:
            matchs_tab = await self.scraper.page.query_selector('button:has-text("MATCHS"), a:has-text("MATCHS"), [role="tab"]:has-text("MATCHS"), text=MATCHS')
            if matchs_tab:
                await matchs_tab.click()
                await asyncio.sleep(0.5)
        except Exception:
            pass
        
        upcoming_matchday = self._get_next_matchday_from_db()
        logger.info(f"Upcoming should be: J{upcoming_matchday}")
        
        # Quick extraction
        matches = await self.scraper._extract_matches(fallback_matchday=upcoming_matchday)
        
        # Always save extracted matches (even if partial)
        if matches and len(matches) > 0:
            logger.info(f"Extracted {len(matches)} upcoming matches for J{matches[0]['matchday']}")
            self._current_upcoming_matchday = matches[0]['matchday']
            if self.scraper.on_upcoming_matches:
                await self.scraper.on_upcoming_matches(matches)

            # Keep DB in a tight window to ensure frontend always shows exactly N upcoming.
            with get_db_context() as db:
                self._trim_upcoming_matches(db, limit=self.UPCOMING_PREDICTION_COUNT)
                db.commit()

            # Only generate predictions if we have enough matches with odds
            matches_with_odds = [m for m in matches if m.get('has_odds')]
            if len(matches_with_odds) >= 3:
                await self._generate_predictions()
                await self._store_predictions_for_comparison()
            else:
                logger.warning(f"Only {len(matches_with_odds)} matches with odds, skipping predictions")
        else:
            logger.warning(f"No matches extracted")

    def _trim_upcoming_matches(self, db: Session, limit: int) -> None:
        """Ensure we only keep the next `limit` upcoming matches (ordered by matchday/line_position).

        This keeps the UI consistent (always shows UPCOMING_PREDICTION_COUNT = 10 matches).
        """
        upcoming = db.query(Match).filter(
            Match.is_upcoming == True,
            Match.has_odds == True,
            Match.is_completed == False,
        ).order_by(Match.matchday.asc(), Match.line_position.asc()).all()

        if len(upcoming) <= limit:
            return

        to_disable = upcoming[limit:]
        for m in to_disable:
            m.is_upcoming = False

            stale_pred = db.query(Prediction).filter(Prediction.match_id == m.id).first()
            if stale_pred and stale_pred.actual_result is None:
                db.delete(stale_pred)
    
    async def _store_predictions_for_comparison(self):
        """Store current predictions for later comparison with results."""
        with get_db_context() as db:
            # Get predictions for upcoming matches
            predictions = db.query(Prediction).join(Match).filter(
                Match.is_upcoming == True,
                Match.has_odds == True,
                Prediction.actual_result == None
            ).order_by(Match.matchday.asc(), Match.line_position.asc()).limit(self.UPCOMING_PREDICTION_COUNT).all()
            
            self._predictions_for_comparison = [{
                'prediction_id': p.id,
                'match_id': p.match_id,
                'matchday': p.match.matchday,
                'home_team': p.match.home_team_name,
                'away_team': p.match.away_team_name,
                'predicted_result': p.predicted_result,
                'predicted_result_name': p.predicted_result_name,
                'confidence': p.confidence,
                'odd_home': p.match.odd_home,
                'odd_draw': p.match.odd_draw,
                'odd_away': p.match.odd_away,
                'stake': self.FIXED_STAKE_ARIARY,
            } for p in predictions]
            
            logger.info(f"Stored {len(self._predictions_for_comparison)} predictions for comparison")
    
    async def _compare_predictions_with_results(self):
        """Compare stored predictions with actual results and broadcast comparison."""
        if not self._predictions_for_comparison:
            return
        
        with get_db_context() as db:
            comparison_results = []
            total_stake = 0
            total_profit = 0
            correct_count = 0
            remaining_queue = []
            
            for pred_data in self._predictions_for_comparison:
                # Find the match result
                match = db.query(Match).filter(Match.id == pred_data['match_id']).first()
                
                if not match or not match.is_completed:
                    remaining_queue.append(pred_data)
                    continue
                
                # Get the prediction
                prediction = db.query(Prediction).filter(Prediction.id == pred_data['prediction_id']).first()
                
                if prediction and prediction.actual_result is None:
                    # Verify prediction
                    prediction.actual_result = match.result
                    prediction.is_correct = (prediction.predicted_result == match.result)
                    prediction.verified_at = datetime.utcnow()
                    db.commit()
                
                # Calculate profit/loss
                stake = pred_data['stake']
                total_stake += stake
                
                is_correct = prediction.is_correct if prediction else False
                
                if is_correct:
                    correct_count += 1
                    # Get the odds for the predicted outcome
                    if pred_data['predicted_result'] == 'V':
                        odds = pred_data['odd_home']
                    elif pred_data['predicted_result'] == 'N':
                        odds = pred_data['odd_draw']
                    else:
                        odds = pred_data['odd_away']
                    
                    profit = (stake * odds) - stake
                    total_profit += profit
                else:
                    total_profit -= stake
                
                
                comparison_results.append({
                    'match': f"{pred_data['home_team']} vs {pred_data['away_team']}",
                    'matchday': pred_data['matchday'],
                    'predicted': pred_data['predicted_result_name'],
                    'actual': match.result_description if match else 'Unknown',
                    'actual_result': match.result if match else None,
                    'is_correct': is_correct,
                    'stake': stake,
                    'profit_loss': profit if is_correct else -stake,
                    'score': f"{match.score_home}-{match.score_away}" if match else None,
                })
            
            if comparison_results:
                # === DN Filter: D/N predictions with odds > 2 ===
                dn_filter_results = []
                dn_total_stake = 0
                dn_total_profit = 0
                dn_correct_count = 0
                
                for pred_data in self._predictions_for_comparison:
                    match = db.query(Match).filter(Match.id == pred_data['match_id']).first()
                    if not match or not match.is_completed:
                        continue
                    
                    # Check if prediction is D or N with odds > 2
                    predicted = pred_data['predicted_result']
                    if predicted not in ['D', 'N']:
                        continue
                    
                    # Get odds for predicted outcome
                    if predicted == 'N':
                        odds = pred_data['odd_draw']
                    else:  # D
                        odds = pred_data['odd_away']
                    
                    if odds <= 2.0:
                        continue
                    
                    # This prediction matches DN filter
                    prediction = db.query(Prediction).filter(Prediction.id == pred_data['prediction_id']).first()
                    is_correct = prediction.is_correct if prediction else False
                    stake = pred_data['stake']
                    
                    dn_total_stake += stake
                    if is_correct:
                        dn_correct_count += 1
                        profit = (stake * odds) - stake
                        dn_total_profit += profit
                    else:
                        dn_total_profit -= stake
                        profit = -stake
                    
                    dn_filter_results.append({
                        'match': f"{pred_data['home_team']} vs {pred_data['away_team']}",
                        'matchday': pred_data['matchday'],
                        'predicted': pred_data['predicted_result_name'],
                        'predicted_result': predicted,
                        'odds': odds,
                        'actual': match.result_description if match else 'Unknown',
                        'actual_result': match.result if match else None,
                        'is_correct': is_correct,
                        'stake': stake,
                        'profit_loss': profit if is_correct else -stake,
                        'score': f"{match.score_home}-{match.score_away}" if match else None,
                    })
                
                # Store comparison results
                self._last_comparison_results = {
                    'predictions': comparison_results,
                    'total_stake': total_stake,
                    'total_profit': total_profit,
                    'correct_count': correct_count,
                    'total_predictions': len(comparison_results),
                    'accuracy': correct_count / len(comparison_results) if comparison_results else 0,
                    'timestamp': datetime.utcnow().isoformat(),
                    # DN Filter results
                    'dn_filter': {
                        'predictions': dn_filter_results,
                        'total_stake': dn_total_stake,
                        'total_profit': dn_total_profit,
                        'correct_count': dn_correct_count,
                        'total_predictions': len(dn_filter_results),
                        'accuracy': dn_correct_count / len(dn_filter_results) if dn_filter_results else 0,
                        'description': 'D/N predictions with odds > 2.00',
                    },
                }
                
                # Broadcast comparison modal to frontend
                await socket_manager.broadcast('prediction_comparison', self._last_comparison_results)
                
                logger.info(f"Prediction comparison: {correct_count}/{len(comparison_results)} correct, Profit: {total_profit} Ar")

            # Keep rolling queue: remove verified ones, keep the rest for next result.
            self._predictions_for_comparison = remaining_queue
    
    async def _on_prediction_result(self, comparison_data: Dict):
        """Handle prediction result comparison (callback from scraper)."""
        logger.info(f"Prediction result: {comparison_data}")
        await socket_manager.broadcast('prediction_comparison', comparison_data)
    
    async def _on_new_result(self, result_data: Dict):
        """Handle new match result - verify predictions and update."""
        logger.info(f"Processing new result: {result_data}")
        
        with get_db_context() as db:
            # Store result
            match = self.data_extractor.process_result(result_data, db)
            
            if match:
                # Find and verify prediction for this match
                prediction = db.query(Prediction).filter(
                    Prediction.match_id == match.id
                ).first()
                
                if prediction and prediction.actual_result is None:
                    # Verify prediction against actual result
                    prediction.actual_result = match.result
                    prediction.is_correct = (prediction.predicted_result == match.result)
                    prediction.verified_at = datetime.utcnow()
                    db.commit()
                    
                    if prediction.is_correct:
                        logger.info(f"✓ PREDICTION CORRECT: {match.home_team_name} vs {match.away_team_name} - Predicted {prediction.predicted_result}, Got {match.result}")
                    else:
                        logger.info(f"✗ PREDICTION WRONG: {match.home_team_name} vs {match.away_team_name} - Predicted {prediction.predicted_result}, Got {match.result}")
                    
                    # ── Update Elite Selector state if this was an elite prediction ────
                    try:
                        odds_map = {'V': match.odd_home, 'N': match.odd_draw, 'D': match.odd_away}
                        odds_used = odds_map.get(prediction.predicted_result, 1.0) or 1.0
                        profit_loss = 1000 * (odds_used - 1) if prediction.is_correct else -1000
                        self.elite_selector.update_elite_result(
                            match_id=match.id,
                            actual_result=match.result,
                            profit_loss=profit_loss
                        )
                        # Broadcast updated elite status
                        elite_status = self.elite_selector.get_status()
                        await socket_manager.broadcast('elite_updated', {
                            'match_id': match.id,
                            'is_correct': prediction.is_correct,
                            'profit_loss': profit_loss,
                            'elite_status': {
                                'predictions_used': elite_status['predictions_used'],
                                'slots_remaining': elite_status['slots_remaining'],
                                'total_profit': elite_status['total_profit'],
                            }
                        })
                    except Exception as e:
                        logger.debug(f"Elite update skipped (not an elite prediction or error): {e}")
                
                # Update team strengths
                self.team_strength.update_all_ratings(match, db)
                
                # Update features if not exist
                existing_features = db.query(MatchFeatures).filter(
                    MatchFeatures.match_id == match.id
                ).first()
                
                if not existing_features:
                    features = self.feature_engineer.compute_features(match, db)
                    db.add(features)
                    db.commit()
                
                # Update sequences
                self.sequence_analyzer.load_team_sequences(db)
                self.sequence_analyzer.load_line_sequences(db)
                
                # Continuous learning
                self.learning_engine.on_match_completed(match, db)
                
                # Settle any pending bets
                await self._settle_bets_for_match(match, db)
                
                # Broadcast real-time update
                await socket_manager.broadcast('match_completed', {
                    "match_id": match.id,
                    "matchday": match.matchday,
                    "home_team": match.home_team_name,
                    "away_team": match.away_team_name,
                    "score": f"{match.score_home}-{match.score_away}",
                    "result": match.result
                })

                # Keep upcoming list tight (always 9) after result entry
                self._trim_upcoming_matches(db, limit=self.UPCOMING_PREDICTION_COUNT)
                db.commit()
                
    async def _on_new_result_fast(self, result_data: Dict):
        """Fast result processing for backfill - store result and settle bets."""
        with get_db_context() as db:
            match = self.data_extractor.process_result(result_data, db)
            if match:
                logger.debug(f"Stored: {match.home_team_name} {match.score_home}-{match.score_away} {match.away_team_name}")
                
                # Settle any pending bets for this match
                bets = db.query(Bet).filter(
                    Bet.match_id == match.id,
                    Bet.is_settled == False
                ).all()
                
                for bet in bets:
                    self.betting_engine.settle_bet(bet, match.result, db)
                    self.bankroll.settle_bet(bet.profit_loss)
                    logger.info(f"Settled bet {bet.id}: {bet.status}, P/L={bet.profit_loss:.2f}")
                
                # Check season completion (38 matchdays)
                if match.matchday >= 38:
                    if self.season_manager.check_season_completion(db):
                        logger.info("Season 38 complete! Starting new season...")
                        self.season_manager.close_season(db)
                        self.season_manager.reset_teams_for_new_season(db)
    
    async def _on_upcoming_matches(self, matches_data: List[Dict]):
        """Handle upcoming matches with odds."""
        logger.info(f"Processing {len(matches_data)} upcoming matches")
        
        with get_db_context() as db:
            for match_data in matches_data:
                match = self.data_extractor.process_upcoming_match(match_data, db)
                
                if match:
                    # Check if features already exist
                    existing_features = db.query(MatchFeatures).filter(
                        MatchFeatures.match_id == match.id
                    ).first()
                    
                    if not existing_features:
                        # Compute features
                        features = self.feature_engineer.compute_features(match, db)
                        try:
                            db.add(features)
                            db.commit()
                        except Exception as e:
                            db.rollback()
                            # Features might have been created by another process
                            logger.debug(f"Features already exist for match {match.id}: {e}")
            
            # Broadcast update for upcoming matches
            await socket_manager.broadcast('upcoming_matches_updated', {
                "count": len(matches_data),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _generate_predictions(self):
        """Generate ultra-precise predictions for all upcoming matches using ALL methods."""
        with get_db_context() as db:
            # Get upcoming matches (with or without odds)
            matches = db.query(Match).filter(
                Match.is_upcoming == True
            ).order_by(Match.matchday.asc(), Match.line_position.asc()).limit(self.UPCOMING_PREDICTION_COUNT).all()
            
            for match in matches:
                # Check if prediction exists
                existing = db.query(Prediction).filter(
                    Prediction.match_id == match.id
                ).first()
                
                if existing:
                    continue
                
                # Get features
                features = db.query(MatchFeatures).filter(
                    MatchFeatures.match_id == match.id
                ).first()
                
                if not features:
                    features = self.feature_engineer.compute_features(match, db)
                    try:
                        db.add(features)
                        db.commit()
                    except Exception as e:
                        db.rollback()
                        features = db.query(MatchFeatures).filter(
                            MatchFeatures.match_id == match.id
                        ).first()
                        if not features:
                            logger.warning(f"Could not create features for match {match.id}: {e}")
                            continue
                
                # Update features with advanced computations
                features = self.advanced_features.update_features(features, match, db)
                try:
                    db.commit()
                except:
                    db.rollback()
                
                # Get teams
                from app.models import Team
                home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
                away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
                
                # === METHOD 1: ML ENSEMBLE ===
                ml_result = self.ml_ensemble.predict(features)
                ml_probs = ml_result['ensemble']
                
                # === METHOD 2: MONTE CARLO (20k simulations) ===
                exp_home, exp_away = self.team_strength.predict_poisson_goals(home_team, away_team)
                mc_result = self.enhanced_monte_carlo.simulate_match(exp_home, exp_away, correlation=0.15)
                mc_probs = {
                    'prob_home_win': mc_result['prob_home_win'],
                    'prob_draw': mc_result['prob_draw'],
                    'prob_away_win': mc_result['prob_away_win']
                }
                
                # === METHOD 3: BIVARIATE POISSON ===
                # Get form data for Poisson
                extra = features.extra_features or {}
                form_data = {
                    'home_form_rating': extra.get('home_weighted_form_rating', 0.5),
                    'away_form_rating': extra.get('away_weighted_form_rating', 0.5)
                }
                h2h_data = {
                    'h2h_dominance_score': features.h2h_dominance_score,
                    'h2h_home_win_rate': features.h2h_home_win_rate,
                    'h2h_away_win_rate': features.h2h_away_win_rate,
                    'h2h_last_5_results': features.h2h_last_5_results
                }
                poisson_result = self.bivariate_poisson.predict_match(
                    home_team.attack_strength_home or home_team.attack_strength,
                    home_team.defense_strength_home or home_team.defense_strength,
                    away_team.attack_strength_away or away_team.attack_strength,
                    away_team.defense_strength_away or away_team.defense_strength,
                    home_team.elo_home or home_team.elo_rating,
                    away_team.elo_away or away_team.elo_rating,
                    h2h_data=h2h_data,
                    form_data=form_data
                )
                poisson_probs = {
                    'prob_home_win': poisson_result['prob_home_win'],
                    'prob_draw': poisson_result['prob_draw'],
                    'prob_away_win': poisson_result['prob_away_win']
                }
                
                # === METHOD 4: ELO ===
                elo_probs = self.team_strength.predict_elo_probabilities(home_team, away_team)
                
                # === METHOD 5: HEAD-TO-HEAD (with strict home/away) ===
                h2h_probs = self.adaptive_ensemble.compute_h2h_probabilities(
                    features.h2h_dominance_score,
                    features.h2h_home_win_rate,
                    features.h2h_away_win_rate,
                    features.h2h_strict_home_win_rate,
                    features.h2h_strict_away_win_rate,
                    features.h2h_strict_total_matches
                )
                
                # === METHOD 6: REAL-TIME ENGINE (M1-M15 signals) ===
                try:
                    # Prepare match info for real-time engine
                    next_match = {
                        'home_team': match.home_team_name,
                        'away_team': match.away_team_name,
                        'odd_home': match.odd_home or 2.0,
                        'odd_draw': match.odd_draw or 3.0,
                        'odd_away': match.odd_away or 3.5,
                        'ligne': match.line_position or 1,
                        'heure': 12,  # Default hour
                        'odds_implied': {
                            'V': 1/(match.odd_home or 2.0),
                            'N': 1/(match.odd_draw or 3.0),
                            'D': 1/(match.odd_away or 3.5)
                        }
                    }
                    
                    # Prepare cache for this match (updates M1-M15 signals)
                    self.realtime_engine.prepare_cache(next_match)
                    
                    # Run real-time inference
                    rt_result = self.realtime_engine.run_inference(
                        match_id=match.id,
                        home_team=match.home_team_name,
                        away_team=match.away_team_name,
                        odds_h=match.odd_home or 2.0,
                        odds_d=match.odd_draw or 3.0,
                        odds_a=match.odd_away or 3.5
                    )
                    
                    # Extract draw signal strength from real-time engine cache
                    rt_cache = self.realtime_engine.get_cache_snapshot()
                    draw_signals = rt_cache.get('draw_detection', {})
                    goal_signals = rt_cache.get('goal_expectation', {})
                    
                    rt_probs = {
                        'prob_home_win': rt_result.get('final_probs', {}).get('V', 0.33),
                        'prob_draw': rt_result.get('final_probs', {}).get('N', 0.33),
                        'prob_away_win': rt_result.get('final_probs', {}).get('D', 0.33),
                        # NEW: Pass draw signal strength for enhanced combination
                        'draw_signal_strength': draw_signals.get('draw_signal_strength', 0),
                        'goal_expectation': goal_signals.get('low_goal_expectation', 0),
                        'draw_factors': draw_signals.get('draw_factors', {}),
                        'draw_recommendation': draw_signals.get('draw_recommendation', 'LOW')
                    }
                    
                    logger.info(f"Real-time engine prediction for {match.home_team_name} vs {match.away_team_name}: {rt_probs}")
                    logger.info(f"Draw signal strength: {draw_signals.get('draw_signal_strength', 0):.3f}, Recommendation: {draw_signals.get('draw_recommendation', 'LOW')}")
                    
                except Exception as e:
                    logger.warning(f"Real-time engine failed for match {match.id}: {e}")
                    rt_probs = {'prob_home_win': 0.33, 'prob_draw': 0.33, 'prob_away_win': 0.34}
                
                # === COMBINE ALL METHODS WITH DYNAMIC WEIGHTS ===
                # Pass odds so the ensemble can use odds-implied probabilities as base
                odds = {'home': match.odd_home, 'draw': match.odd_draw, 'away': match.odd_away}
                final_probs = self.adaptive_ensemble.combine_predictions(
                    ml_probs, mc_probs, poisson_probs, elo_probs, h2h_probs, rt_probs, db, odds
                )
                
                # Apply filtering
                should_predict, filter_reason = self.prediction_filter.should_predict(final_probs, odds)
                
                if not should_predict:
                    logger.info(f"Filtered out: {match.home_team_name} vs {match.away_team_name} - {filter_reason}")
                    continue
                
                # Find value bets
                value_bets = self.prediction_filter.find_value_bets(final_probs, odds)
                
                # Evaluate prediction quality
                model_agreement = self.ml_ensemble.calculate_model_agreement(ml_result.get('model_outputs', {}))
                quality = self.prediction_filter.evaluate_prediction_quality(final_probs, model_agreement)

                def _to_jsonable(obj):
                    try:
                        import numpy as np
                    except Exception:
                        np = None
                    if np is not None:
                        if isinstance(obj, (np.integer,)):
                            return int(obj)
                        if isinstance(obj, (np.floating,)):
                            return float(obj)
                        if isinstance(obj, (np.ndarray,)):
                            return [_to_jsonable(x) for x in obj.tolist()]
                    if isinstance(obj, dict):
                        return {k: _to_jsonable(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_to_jsonable(v) for v in obj]
                    return obj

                model_outputs_json = _to_jsonable(ml_result.get('model_outputs', {}))
                mc_result_json = _to_jsonable(mc_result)
                
                # Create prediction record with all method outputs
                prediction = Prediction(
                    match_id=match.id,
                    season_id=match.season_id,
                    prob_home_win=final_probs['V'],
                    prob_draw=final_probs['N'],
                    prob_away_win=final_probs['D'],
                    model_outputs=model_outputs_json,
                    monte_carlo_results=mc_result_json,
                    predicted_result=max(final_probs, key=final_probs.get),
                    predicted_result_name={'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}[max(final_probs, key=final_probs.get)],
                    confidence=max(final_probs.values()),
                    model_agreement=model_agreement,
                    probability_strength=max(final_probs.values())
                )
                
                # Add to session FIRST so SQLAlchemy tracks changes
                try:
                    db.add(prediction)
                    db.flush()  # Flush to get ID and track changes
                    
                    # Calculate value AFTER adding to session (ensures persistence)
                    if match.odd_home and match.odd_home > 0:
                        prediction.calculate_value(match.odd_home, match.odd_draw, match.odd_away)
                    
                    # Store ensemble weights
                    prediction.ensemble_weights = self.adaptive_ensemble.method_weights
                    
                    db.commit()
                except Exception as e:
                    db.rollback()
                    # Prediction already exists - fetch it and ensure values are calculated
                    logger.debug(f"Prediction already exists for match {match.id}, fetching existing")
                    prediction = db.query(Prediction).filter(Prediction.match_id == match.id).first()
                    if prediction:
                        # Calculate values if missing
                        if prediction.value_home is None and match.odd_home and match.odd_home > 0:
                            prediction.calculate_value(match.odd_home, match.odd_draw, match.odd_away)
                            db.commit()
                            logger.info(f"Updated values for existing prediction: {match.home_team_name} vs {match.away_team_name}")
                    continue
                
                # Broadcast prediction update
                await socket_manager.broadcast('prediction_generated', {
                    "match_id": match.id,
                    "prediction_id": prediction.id,
                    "predicted_result": prediction.predicted_result,
                    "confidence": prediction.confidence,
                    "quality": quality['quality_level'],
                    "value_bets": len(value_bets)
                })
                
                log_msg = f"Prediction: {match.home_team_name} vs {match.away_team_name} -> {prediction.predicted_result_name} ({prediction.confidence:.1%}) [Quality: {quality['quality_level']}]"
                if value_bets:
                    log_msg += f" [Value: {value_bets[0]['outcome_name']} @ {value_bets[0]['odds']:.2f}]"
                logger.info(log_msg)
                
                # Predictions are generated but NOT auto-selected for betting.
                # The Elite Selector below will decide which ones to bet on.
                logger.info(
                    f"Prediction generated: {match.home_team_name} vs {match.away_team_name} "
                    f"-> {prediction.predicted_result} (conf={prediction.confidence:.3f}, "
                    f"agree={prediction.model_agreement:.3f})"
                )
            
            # === ELITE SELECTION: Ultra-selective 5 predictions per season ===
            await self._apply_elite_selection(db)
    
    def _combine_predictions(self, ml_probs: Dict, mc_result: Dict, elo_probs: Dict) -> Dict:
        """Combine predictions from multiple sources."""
        # Weight: ML 50%, Monte Carlo 30%, ELO 20%
        combined = {
            'V': ml_probs['V'] * 0.5 + mc_result['prob_home_win'] * 0.3 + elo_probs['V'] * 0.2,
            'N': ml_probs['N'] * 0.5 + mc_result['prob_draw'] * 0.3 + elo_probs['N'] * 0.2,
            'D': ml_probs['D'] * 0.5 + mc_result['prob_away_win'] * 0.3 + elo_probs['D'] * 0.2
        }
        
        # Normalize
        total = sum(combined.values())
        return {k: v / total for k, v in combined.items()}
    
    async def _evaluate_bets(self):
        """Evaluate betting opportunities."""
        with get_db_context() as db:
            # Get predictions for upcoming matches
            predictions = db.query(Prediction).join(Match).filter(
                Match.is_upcoming == True,
                Match.has_odds == True
            ).all()
            
            evaluations = []
            for pred in predictions:
                eval_result = self.betting_engine.evaluate_match(pred.match, pred)
                evaluations.append(eval_result)
            
            # Make betting decisions
            decisions = self.betting_engine.make_betting_decision(
                evaluations,
                self.bankroll.get_current_bankroll()
            )
            
            for decision in decisions:
                # Place bet
                match = db.query(Match).filter(Match.id == decision['match_id']).first()
                pred = db.query(Prediction).filter(Prediction.match_id == match.id).first()
                
                if self.bankroll.place_bet(decision['stake']):
                    bet = self.betting_engine.create_bet_record(decision, match, pred, db)
                    logger.info(f"Bet placed: {match.home_team_name} vs {match.away_team_name} - {decision['outcome_name']} @ {decision['odds']}")
    
    async def _settle_bets_for_match(self, match: Match, db: Session):
        """Settle any pending bets for a completed match and update bankroll V2."""
        bets = db.query(Bet).filter(
            Bet.match_id == match.id,
            Bet.is_settled == False
        ).all()
        
        for bet in bets:
            # Settle via betting engine (updates bet record)
            self.betting_engine.settle_bet(bet, match.result, db)
            
            # Update bankroll V2 with actual stake and result
            won = bet.bet_outcome == match.result
            self.realtime_engine.bankroll.settle_bet(
                stake=bet.stake,
                odds=bet.odds,
                won=won,
                matchday=match.matchday,
                season_id=match.season_id
            )
            
            # Also update legacy bankroll
            self.bankroll.settle_bet(bet.profit_loss)
            
            logger.info(f"Settled bet {bet.id}: {'WON' if won else 'LOST'} - {match.home_team_name} vs {match.away_team_name} | Bankroll: {self.realtime_engine.bankroll.bankroll:.0f} Ar")
    
    async def _create_automatic_bet(self, match: Match, prediction: Prediction, db: Session):
        """Create an automatic bet with fixed stake of 1000 Ariary for each prediction."""
        # Check if bet already exists for this match (prevent duplicates)
        existing_bets = db.query(Bet).filter(Bet.match_id == match.id).count()
        if existing_bets > 0:
            logger.debug(f"Bet already exists for match {match.id} ({existing_bets} bets)")
            return
        
        # Get odds for the predicted outcome
        odds_map = {
            'V': match.odd_home,
            'N': match.odd_draw,
            'D': match.odd_away
        }
        odds = odds_map.get(prediction.predicted_result)
        
        if not odds or odds <= 0:
            logger.warning(f"No valid odds for match {match.id}, skipping bet creation")
            return
        
        # Fixed stake in Ariary
        stake = self.FIXED_STAKE_ARIARY
        
        # Create bet record
        bet = Bet(
            match_id=match.id,
            prediction_id=prediction.id,
            season_id=match.season_id,
            bet_outcome=prediction.predicted_result,
            bet_outcome_name=prediction.predicted_result_name,
            odds=odds,
            stake=stake,
            potential_return=stake * odds,
            bankroll_before=self.bankroll.get_current_bankroll(),
            kelly_fraction_used=0.0,  # Fixed stake, no Kelly
            kelly_full=0.0,
            value_edge=prediction.best_value_amount or 0.0,
            confidence=prediction.confidence,
            status='pending'
        )
        
        db.add(bet)
        db.commit()
        db.refresh(bet)
        
        logger.info(f"Created automatic bet: {match.home_team_name} vs {match.away_team_name} -> {prediction.predicted_result_name} @ {odds:.2f} (Stake: {stake} Ar)")
    
    async def _apply_elite_selection(self, db: Session):
        """
        Apply ELITE ultra-selective system: only 5 predictions per entire season.
        
        Rules:
        - Odds of predicted result > 2.00
        - Confidence >= 50%
        - Model agreement >= 80%
        - Value edge >= 8%
        - Elite score >= 60%
        - Max 5 total per season
        
        Most matchdays will have ZERO selections. This is normal.
        """
        if not self._current_upcoming_matchday:
            return
        
        # Get active season
        active_season = db.query(Season).filter(Season.is_active == True).first()
        if not active_season:
            return
        
        # Ensure elite selector knows the current season
        self.elite_selector.ensure_season(active_season.id)
        
        # Check if we have slots remaining
        slots_remaining = self.elite_selector.get_slots_remaining()
        logger.info(
            f"Elite selection for J{self._current_upcoming_matchday}: "
            f"{slots_remaining} slots remaining this season"
        )
        
        if slots_remaining <= 0:
            logger.info("All 5 elite slots used for this season — no more bets")
            return
        
        # Get ALL predictions for upcoming matches
        predictions_query = db.query(Prediction, Match).join(Match).filter(
            Match.matchday == self._current_upcoming_matchday,
            Match.is_upcoming == True,
            Match.season_id == active_season.id
        ).all()
        
        if not predictions_query:
            return
        
        # Reset all selections first (no prediction is selected by default)
        for pred, match in predictions_query:
            pred.is_selected_for_bet = False
            pred.selection_rank = None
            pred.selection_reason = None
        
        # Run elite evaluation on all predictions
        eval_results = self.elite_selector.evaluate_matchday(
            [(pred, match) for pred, match in predictions_query],
            active_season.id
        )
        
        # Mark confirmed elite predictions
        elite_count = 0
        for pred, match in predictions_query:
            for ep in self.elite_selector.state['elite_predictions']:
                if ep.get('match_id') == match.id and ep.get('actual_result') is None:
                    pred.is_selected_for_bet = True
                    pred.selection_rank = ep['slot']
                    pred.selection_reason = f"ELITE_#{ep['slot']}"
                    elite_count += 1
                    
                    # Create automatic bet for elite prediction
                    await self._create_automatic_bet(match, pred, db)
                    break
        
        db.commit()
        
        # Broadcast update
        elite_status = self.elite_selector.get_status()
        await socket_manager.broadcast('elite_selection', {
            'matchday': self._current_upcoming_matchday,
            'elite_count': elite_count,
            'slots_used': elite_status['predictions_used'],
            'slots_remaining': elite_status['slots_remaining'],
        })
        
        if elite_count > 0:
            logger.info(
                f"🏆 {elite_count} ELITE prediction(s) confirmed for J{self._current_upcoming_matchday}! "
                f"Season total: {elite_status['predictions_used']}/{elite_status['max_predictions']}"
            )
        else:
            logger.info(
                f"No elite prediction for J{self._current_upcoming_matchday} — "
                f"waiting for better opportunity. "
                f"({elite_status['candidates_rejected']} total rejected this season)"
            )

    async def _apply_dynamic_selection(self, db: Session):
        """Legacy method — now delegates to elite selection."""
        await self._apply_elite_selection(db)
    
    async def _check_season_status(self):
        """Check and manage season lifecycle."""
        with get_db_context() as db:
            if self.season_manager.check_season_completion(db):
                logger.info("Season complete! Closing and starting new season...")
                self.season_manager.close_season(db)
                self.season_manager.reset_teams_for_new_season(db)
    
    async def stop(self):
        """Stop the autonomous system."""
        logger.info("Stopping Prediction Engine...")
        self.is_running = False
        await self.scraper.close()
        
        # Save models
        self.ml_ensemble.save_models()
        
        logger.info("Prediction Engine stopped")
    
    def get_status(self) -> Dict:
        """Get current system status."""
        with get_db_context() as db:
            season_progress = self.season_manager.get_season_progress(db)
            bankroll_stats = self.bankroll.get_statistics()
            learning_status = self.learning_engine.get_learning_status(db)
            
            return {
                "is_running": self.is_running,
                "cycle_count": self.cycle_count,
                "season": season_progress,
                "bankroll": bankroll_stats,
                "learning": learning_status,
                "timestamp": datetime.utcnow().isoformat()
            }


# CLI entry point
async def main():
    """Main entry point."""
    orchestrator = PredictionOrchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
