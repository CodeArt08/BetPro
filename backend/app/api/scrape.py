"""Manual scrape API routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import List, Optional
from datetime import datetime
import asyncio
import concurrent.futures

router = APIRouter()

# Global browser instance - stays open between captures
_global_browser = None
_global_context = None
_global_page = None
_results_page_loaded = False  # Track if results page is already loaded

# SINGLETON THREAD FOR PLAYWRIGHT: Empêche l'ouverture de multiples fenêtres
_scrape_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _get_or_create_browser():
    """Get existing browser or create new one."""
    global _global_browser, _global_context, _global_page
    from playwright.sync_api import sync_playwright
    from app.core.config import settings
    
    # Check if existing browser/context is still valid
    if _global_page is not None:
        try:
            # Test if page is still connected
            _global_page.evaluate('1')
        except Exception:
            # Browser was closed, reset everything
            logger.info("Browser was closed, recreating...")
            global _results_page_loaded
            _global_browser = None
            _global_context = None
            _global_page = None
            _results_page_loaded = False
    
    if _global_browser is None:
        settings.HEADLESS = False  # Always visible for manual control
        _global_browser = sync_playwright().start()
    
    if _global_context is None:
        browser = _global_browser.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        _global_context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        _global_page = _global_context.new_page()
    
    return _global_page


class ScrapeResultResponse(BaseModel):
    success: bool
    message: str
    matchday: Optional[int] = None
    results_count: int = 0
    matches_count: int = 0


class NewSeasonResponse(BaseModel):
    success: bool
    message: str
    season_number: Optional[int] = None


@router.post("/new-season", response_model=NewSeasonResponse)
async def create_new_season():
    """Create a new season manually.
    
    Closes the current active season and creates a new one.
    Preserves historical data from previous seasons.
    """
    from app.core.database import SessionLocal
    from app.models import Season, Prediction, Match, MatchFeatures, Bet
    from app.services.season_manager import SeasonManager
    
    db = SessionLocal()
    try:
        # Get season manager
        from app.services.season_manager import SeasonManager
        season_manager = SeasonManager()
        
        # Get current active season to report in message
        current_season = db.query(Season).filter(Season.is_active == True).first()
        current_num = current_season.season_number if current_season else "N/A"
        
        # Close season correctly (includes stats update and marking as completed)
        new_season = season_manager.close_season(db)
        
        # Get new season number
        season_number = new_season.season_number
        
        # Notify frontend to refresh all data
        from app.core.socket_manager import socket_manager
        import asyncio
        asyncio.create_task(socket_manager.broadcast('season_created', {
            'season_number': season_number,
            'message': f'Nouvelle saison {season_number} créée',
            'reset_all': True  # Indicate that frontend should reset all data
        }))
        
        # Also broadcast a reset event for safety
        asyncio.create_task(socket_manager.broadcast('reset_all_data', {
            'season_number': season_number,
            'message': 'Reset all frontend data'
        }))
        
        return NewSeasonResponse(
            success=True,
            message=f"Nouvelle saison {season_number} créée. La saison {current_num} a été archivée avec ses statistiques.",
            season_number=season_number
        )
        
    except Exception as e:
        logger.error(f"Error creating new season: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        return NewSeasonResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )
    finally:
        db.close()


@router.post("/cleanup-old-predictions")
async def cleanup_old_predictions():
    """Clean up old predictions from previous seasons.
    
    Removes predictions that are not associated with the current active season
    and haven't been compared yet.
    """
    from app.core.database import SessionLocal
    from app.models import Season, Prediction
    
    db = SessionLocal()
    try:
        # Get current active season
        current_season = db.query(Season).filter(Season.is_active == True).first()
        
        if not current_season:
            return {"success": False, "message": "No active season found"}
        
        # Count old predictions
        old_count = db.query(Prediction).filter(
            Prediction.season_id != current_season.id,
            Prediction.actual_result == None
        ).count()
        
        if old_count == 0:
            return {"success": True, "message": "No old predictions to clean up", "deleted": 0}
        
        # Delete old predictions
        deleted = db.query(Prediction).filter(
            Prediction.season_id != current_season.id,
            Prediction.actual_result == None
        ).delete(synchronize_session=False)
        
        db.commit()
        
        logger.info(f"Cleaned up {deleted} old predictions from previous seasons")
        
        return {
            "success": True,
            "message": f"Cleaned up {deleted} old predictions",
            "deleted": deleted
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old predictions: {e}")
        db.rollback()
        return {"success": False, "message": f"Error: {str(e)}"}
    finally:
        db.close()


@router.post("/results", response_model=ScrapeResultResponse)
async def scrape_results():
    """Manually trigger scraping of results page.
    
    Browser stays open - user controls 'Afficher plus' clicks.
    Each call extracts from current page state.
    """
    from app.core.database import SessionLocal
    from app.models import Match, Season, Team, Prediction
    from app.core.config import settings
    from app.core.socket_manager import socket_manager
    import re
    
    async def _compare_predictions_and_notify():
        """Compare predictions with actual results and broadcast to frontend.
        
        Handles 40+ results by grouping by matchday and sending organized data.
        """
        db = SessionLocal()
        try:
            # Find predictions that have been verified (match completed with result)
            verified_predictions = db.query(Prediction).join(Match).filter(
                Match.is_completed == True,
                Prediction.actual_result == None  # Not yet compared
            ).order_by(Match.matchday.asc()).all()
            
            if not verified_predictions:
                logger.info("No predictions to compare")
                return
            
            STAKE = 1000  # Fixed stake per prediction
            result_name = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}
            
            # Group by matchday
            by_matchday = {}
            total_stake = 0
            total_profit = 0
            total_correct = 0
            
            for pred in verified_predictions:
                match = pred.match
                if not match or not match.result:
                    continue
                
                # Mark prediction as verified
                pred.actual_result = match.result
                pred.is_correct = (pred.predicted_result == match.result)
                pred.verified_at = datetime.utcnow()
                
                total_stake += STAKE
                
                # Calculate profit
                if pred.is_correct:
                    total_correct += 1
                    if pred.predicted_result == 'V':
                        odds = match.odd_home or 2.0
                    elif pred.predicted_result == 'N':
                        odds = match.odd_draw or 3.5
                    else:
                        odds = match.odd_away or 3.0
                    profit = (STAKE * odds) - STAKE
                    total_profit += profit
                else:
                    profit = -STAKE
                    total_profit -= STAKE
                
                # Group by matchday
                md = match.matchday
                if md not in by_matchday:
                    by_matchday[md] = {
                        'matchday': md,
                        'predictions': [],
                        'stake': 0,
                        'profit': 0,
                        'correct': 0
                    }
                
                by_matchday[md]['predictions'].append({
                    'match': f"{match.home_team_name} vs {match.away_team_name}",
                    'matchday': match.matchday,
                    'predicted': result_name.get(pred.predicted_result, pred.predicted_result),
                    'actual': result_name.get(match.result, match.result),
                    'actual_result': match.result,
                    'is_correct': pred.is_correct,
                    'stake': STAKE,
                    'profit_loss': profit,
                    'score': f"{match.score_home}-{match.score_away}",
                })
                by_matchday[md]['stake'] += STAKE
                by_matchday[md]['profit'] += profit
                if pred.is_correct:
                    by_matchday[md]['correct'] += 1
            
            db.commit()
            
            # Convert to list sorted by matchday
            matchday_results = list(by_matchday.values())
            
            if matchday_results:
                comparison_data = {
                    'predictions': [p for md in matchday_results for p in md['predictions']],  # Flat list for compatibility
                    'by_matchday': matchday_results,  # Grouped by matchday
                    'total_stake': total_stake,
                    'total_profit': total_profit,
                    'correct_count': total_correct,
                    'total_predictions': len(verified_predictions),
                    'accuracy': total_correct / len(verified_predictions),
                    'timestamp': datetime.utcnow().isoformat(),
                }
                
                # Broadcast to frontend
                await socket_manager.broadcast('prediction_comparison', comparison_data)
                logger.info(f"Prediction comparison sent: {total_correct}/{len(verified_predictions)} correct across {len(matchday_results)} matchdays, Profit: {total_profit} Ar")
        except Exception as e:
            logger.error(f"Error comparing predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            db.rollback()
        finally:
            db.close()
    
    def _scrape_results_sync() -> list[dict]:
        results: list[dict] = []
        global _results_page_loaded
        
        RESULTS_URL = "https://bet261.mg/virtual/category/instant-league/8037/results"
        
        try:
            page = _get_or_create_browser()
            
            # Only navigate if page not already loaded
            if not _results_page_loaded:
                # Navigate to results page
                page.goto(RESULTS_URL, timeout=60000)
                page.wait_for_timeout(3000)
                _results_page_loaded = True
                logger.info("Results page loaded - user can click 'Afficher plus' manually")
            else:
                logger.info("Scraping from current page state (no refresh)")
            
            # User clicks 'Afficher plus' manually, then we scrape
            logger.info("Scraping results from current page state...")
            
            # Extract text from current visible content
            text = page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            
            logger.info(f"Results page: {len(lines)} lines")
            
            # Parse matches following Bet261 format:
            # HomeTeam\nGoalMinutes\nScore\nMT: Score\nAwayTeam\nGoalMinutes
            # Matchday markers: "Journée 1", "Jornada 1", "Matchday 1"
            
            current_matchday = 1
            last_matchday_line = -10
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Update current matchday if we see a matchday marker
                match = re.search(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', line)
                if match:
                    md = int(match.group(1))
                    if 1 <= md <= 38:
                        current_matchday = md
                        last_matchday_line = i
                        logger.info(f"Found matchday J{md}")
                
                # Check if this line is a score (format: "3:1", "2:0", "1:2")
                # Must NOT start with "MT:" (that's halftime score)
                if ':' in line and not line.startswith('MT') and not line.startswith('MT:'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            p1 = parts[0].strip().replace('\xa0', '')
                            p2 = parts[1].strip().replace('\xa0', '')
                            
                            if p1.isdigit() and p2.isdigit():
                                score_home = int(p1)
                                score_away = int(p2)
                                
                                # Look backwards for home team (skip goal minutes with ')
                                home_team = None
                                for j in range(i-1, max(i-5, 0), -1):
                                    prev_line = lines[j]
                                    # Skip goal minutes (contains ') and MT: scores
                                    if "'" in prev_line or 'MT:' in prev_line or ':' in prev_line:
                                        continue
                                    if prev_line and not prev_line.isdigit():
                                        home_team = prev_line
                                        break
                                
                                # Look forward for away team (skip MT: score and goal minutes)
                                away_team = None
                                for j in range(i+1, min(i+6, len(lines))):
                                    next_line = lines[j]
                                    # Skip MT: halftime score and goal minutes
                                    if 'MT:' in next_line or "'" in next_line:
                                        continue
                                    if ':' in next_line:
                                        # This might be another score, stop
                                        break
                                    if next_line and not next_line.isdigit():
                                        away_team = next_line
                                        break
                                
                                if home_team and away_team and not home_team.isdigit() and not away_team.isdigit():
                                    # Validate team names (should be alphabetic)
                                    if any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                                        # Determine result
                                        if score_home > score_away:
                                            result = 'V'
                                        elif score_home < score_away:
                                            result = 'D'
                                        else:
                                            result = 'N'
                                        
                                        # Count position within this matchday
                                        md_matches = [r for r in results if r['matchday'] == current_matchday]
                                        line_pos = len(md_matches) + 1
                                        
                                        results.append({
                                            'matchday': current_matchday,
                                            'line_position': line_pos,
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'score_home': score_home,
                                            'score_away': score_away,
                                            'result': result
                                        })
                                        logger.debug(f"J{current_matchday} Match {line_pos}: {home_team} {score_home}-{score_away} {away_team}")
                        except Exception as e:
                            pass
                
                i += 1
            
            # Sort by matchday descending (most recent first)
            results.sort(key=lambda x: (-x['matchday'], x['line_position']))
            
            # Log summary by matchday
            if results:
                matchdays_found = sorted(set(r['matchday'] for r in results), reverse=True)
                logger.info(f"Extracted {len(results)} results from {len(matchdays_found)} matchdays")
                for md in matchdays_found[:5]:
                    md_matches = [r for r in results if r['matchday'] == md]
                    logger.info(f"  J{md}: {len(md_matches)} matches")
                if len(matchdays_found) > 5:
                    logger.info(f"  ... and {len(matchdays_found) - 5} more matchdays")
            
            # Browser stays open for user to verify
            logger.info(f"Extraction complete. Browser stays open for verification.")
            
        except Exception as e:
            import traceback
            logger.error(f"Error during results scraping: {e}")
            logger.error(traceback.format_exc())
        
        return results
    
    try:
        scraped_results = await asyncio.get_running_loop().run_in_executor(_scrape_executor, _scrape_results_sync)
        
        # Persist to DB (async endpoint, sync DB session)
        saved_count = 0
        updated_count = 0
        new_count = 0
        
        # Group results by matchday for better logging
        results_by_matchday = {}
        for result in scraped_results:
            md = result['matchday']
            if md not in results_by_matchday:
                results_by_matchday[md] = []
            results_by_matchday[md].append(result)
        
        logger.info(f"Processing results from matchdays: {sorted(results_by_matchday.keys())}")
        
        for result in scraped_results:
            db = SessionLocal()
            try:
                active_season = db.query(Season).filter(Season.is_active == True).first()
                if not active_season:
                    logger.warning("No active season found")
                    continue
                season_id = active_season.id
                
                home_team = db.query(Team).filter(Team.name == result['home_team']).first()
                if not home_team:
                    home_team = Team(name=result['home_team'])
                    db.add(home_team)
                    db.flush()
                away_team = db.query(Team).filter(Team.name == result['away_team']).first()
                if not away_team:
                    away_team = Team(name=result['away_team'])
                    db.add(away_team)
                    db.flush()
                
                existing = db.query(Match).filter(
                    Match.season_id == season_id,
                    Match.matchday == result['matchday'],
                    Match.home_team_name == result['home_team'],
                    Match.away_team_name == result['away_team']
                ).first()
                
                if existing:
                    # Update existing match
                    old_score = f"{existing.score_home}-{existing.score_away}"
                    new_score = f"{result['score_home']}-{result['score_away']}"
                    existing.score_home = result['score_home']
                    existing.score_away = result['score_away']
                    existing.result = result['result']
                    existing.is_completed = True
                    existing.is_upcoming = False
                    existing.result_recorded_at = datetime.utcnow()
                    updated_count += 1
                    logger.debug(f"Updated J{result['matchday']}: {result['home_team']} vs {result['away_team']} - {old_score} -> {new_score}")
                else:
                    # Create new match
                    match = Match(
                        season_id=season_id,
                        matchday=result['matchday'],
                        line_position=result['line_position'],
                        home_team_name=result['home_team'],
                        away_team_name=result['away_team'],
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        score_home=result['score_home'],
                        score_away=result['score_away'],
                        result=result['result'],
                        is_completed=True,
                        is_upcoming=False,
                        result_recorded_at=datetime.utcnow()
                    )
                    db.add(match)
                    new_count += 1
                    logger.debug(f"New J{result['matchday']}: {result['home_team']} {result['score_home']}-{result['score_away']} {result['away_team']}")
                
                db.commit()
                saved_count += 1
                
                # ── Thread A: notify RealTime Engine ─────────────────
                try:
                    from app.services.realtime_engine import get_engine
                    rt_engine = get_engine()
                    match_data_rt = {
                        'match_id': existing.id if existing else None,
                        'matchday': result['matchday'],
                        'line_position': result['line_position'],
                        'home_team': result['home_team'],
                        'away_team': result['away_team'],
                        'result': result['result'],
                        'score_home': result['score_home'],
                        'score_away': result['score_away'],
                        'odd_home': existing.odd_home if existing else None,
                        'odd_draw': existing.odd_draw if existing else None,
                        'odd_away': existing.odd_away if existing else None,
                        'predicted': getattr(existing, '_rt_predicted', None),  # if cached
                        'confidence': None,
                    }
                    asyncio.create_task(
                        asyncio.to_thread(rt_engine.on_match_completed, match_data_rt)
                    )
                    logger.debug(f"RT Engine Thread A triggered for J{result['matchday']} match")
                except Exception as rt_err:
                    logger.debug(f"RT Engine Thread A (non-blocking): {rt_err}")
                # ─────────────────────────────────────────────────────
                    
            except Exception as e:
                logger.error(f"Error saving result: {e}")
                db.rollback()
            finally:
                db.close()
        
        logger.info(f"Save summary: {new_count} new, {updated_count} updated, {saved_count} total")
        
        # Compare predictions with results and send notification
        if saved_count > 0:
            await _compare_predictions_and_notify()
            
            # Broadcast match_completed event to trigger real-time engine signals update
            from app.core.socket_manager import socket_manager
            await socket_manager.broadcast('match_completed', {
                'message': f'{saved_count} résultats capturés',
                'results_count': saved_count,
                'new_count': new_count,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return ScrapeResultResponse(
            success=True,
            message=f"Capturé {saved_count} résultats ({new_count} nouveaux, {updated_count} mis à jour)",
            results_count=saved_count
        )
    except Exception as e:
        import traceback
        logger.error(f"Error scraping results: {e}")
        logger.error(traceback.format_exc())
        return ScrapeResultResponse(success=False, message=f"Erreur: {str(e) or type(e).__name__}")


@router.post("/matches", response_model=ScrapeResultResponse)
async def scrape_matches():
    """Manually trigger scraping of upcoming matches page.
    
    Browser stays open - user controls navigation.
    Each call extracts from current page state.
    """
    from app.core.database import SessionLocal
    from app.models import Match, Season, Team
    from app.core.config import settings
    import re
    
    def _scrape_matches_sync() -> dict:
        MATCHES_URL = "https://bet261.mg/virtual/category/instant-league/8037/matches"
        
        try:
            page = _get_or_create_browser()
            
            # Navigate to matches page
            page.goto(MATCHES_URL, timeout=60000)
            
            # Wait for match content to load - look for match containers
            try:
                # Wait for match elements to appear (various possible selectors)
                page.wait_for_selector('[class*="match"]', timeout=10000)
                logger.info("Found match elements with class*='match'")
            except:
                logger.warning("No match elements found with class*='match', trying other selectors")
                try:
                    page.wait_for_selector('[class*="event"]', timeout=5000)
                    logger.info("Found event elements")
                except:
                    logger.warning("No event elements found either")
            
            page.wait_for_timeout(2000)  # Extra wait for dynamic content
            
            # Try to extract match-specific content instead of whole body
            # Look for the main content area containing matches
            match_selectors = [
                '[class*="match-list"]',
                '[class*="matches-container"]',
                '[class*="events-list"]',
                '[class*="fixture"]',
                '[class*="upcoming"]',
                '.category-matches',
                '.match-container',
                '[class*="category-content"]',
                'main',
                '[role="main"]'
            ]
            
            text = None
            for selector in match_selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        text = element.inner_text()
                        if text and len(text) > 100:
                            logger.info(f"Found content with selector: {selector}")
                            break
                except:
                    continue
            
            # Fallback to body if no specific container found
            if not text or len(text) < 100:
                logger.warning("No specific match container found, falling back to body")
                text = page.inner_text('body')
            
            
            # Debug: Log first 30 lines to see what we're working with
            lines_for_debug = text.split('\n')[:30]
            logger.info("First 30 lines of matches content:")
            for i, line in enumerate(lines_for_debug):
                logger.info(f"  {i}: '{line}'")
            
            # Check for odds - if we find decimal odds, these are upcoming matches, not LIVE
            odds_found = []
            # Look for individual decimal odds (like 2.45, 3.20, 2.80, or comma separated like 2,50)
            decimal_odds_pattern = r'\b\d+[\.,]\d{1,2}\b'
            for line in text.split('\n'):
                odds_in_line = re.findall(decimal_odds_pattern, line)
                if len(odds_in_line) >= 3:  # At least 3 odds (1, X, 2)
                    odds_found.append(line.strip())
                    logger.info(f"Found odds line: '{line.strip()}' -> {odds_in_line}")
            
            has_odds = len(odds_found) > 0
            logger.info(f"Odds detection: found {len(odds_found)} lines with odds, has_odds={has_odds}")
            
            # Smarter LIVE detection:
            # Only treat as LIVE if we have clear LIVE indicators AND no betting odds
            # AND we see signs of a match actually being played (score, time elapsed, etc.)
            live_indicators = ['LIVE', 'EN DIRECT', 'live', 'EN COURS']
            has_live_text = any(indicator in text for indicator in live_indicators)
            
            # Check for signs that a match is actually being played (not just scheduled)
            match_in_progress_indicators = [
                r'\d+:\d+',  # Score like 1:0
                r'\d+\'',    # Minutes like 45'
                r'MT:',      # Half time
                'minute', 'minutes', 'min',
                'temps', 'période'
            ]
            match_in_progress = any(re.search(ind, text, re.IGNORECASE) for ind in match_in_progress_indicators)
            
            logger.info(f"LIVE indicators: {has_live_text}, match in progress: {match_in_progress}")
            logger.info(f"Indicators found: {[ind for ind in live_indicators if ind in text]}")
            
            # Bypass logic to avoid false positives:
            if has_odds:
                logger.info("Upcoming matches detected with odds - proceeding with extraction")
            else:
                logger.warning("No odds found, but continuing extraction anyway")
            
            logger.info("Proceeding with upcoming matches extraction")
            
            def _extract_matches_from_text(lines: list[str], known_teams: list[str], matchday: int) -> list[dict]:
                """Extract upcoming matches from text lines.
                
                Bet261 format: Team1, Team2, odd1, oddX, odd2 on separate lines
                """
                matches = []
                i = 0
                
                # Pattern for decimal odds like "1,33" or "4.90"
                odd_pattern = re.compile(r'^[\d]+[.,][\d]{1,2}$')
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check if this line contains a team name
                    home_team = None
                    for team in known_teams:
                        if team.lower() in line.lower() or line.lower() in team.lower():
                            home_team = line if len(line) > 2 else team
                            break
                    
                    if home_team:
                        # Look for away team in next line
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            away_team = None
                            for team in known_teams:
                                if team.lower() in next_line.lower() or next_line.lower() in team.lower():
                                    away_team = next_line if len(next_line) > 2 else team
                                    break
                            
                            if away_team and home_team != away_team:
                                # Found both teams, now look for 3 odds on next lines
                                # Pattern: odd1, oddX, odd2 on 3 consecutive lines
                                if i + 4 < len(lines):
                                    odd1_line = lines[i + 2].strip()
                                    oddX_line = lines[i + 3].strip()
                                    odd2_line = lines[i + 4].strip()
                                    
                                    # Check if all 3 are decimal odds
                                    if odd_pattern.match(odd1_line) and odd_pattern.match(oddX_line) and odd_pattern.match(odd2_line):
                                        odd_home = float(odd1_line.replace(',', '.'))
                                        odd_draw = float(oddX_line.replace(',', '.'))
                                        odd_away = float(odd2_line.replace(',', '.'))
                                        
                                        # Validate odds (reasonable range)
                                        if 1.0 <= odd_home <= 20 and 1.0 <= odd_draw <= 20 and 1.0 <= odd_away <= 20:
                                            matches.append({
                                                'matchday': matchday,
                                                'line_position': len(matches) + 1,
                                                'home_team': home_team,
                                                'away_team': away_team,
                                                'odd_home': odd_home,
                                                'odd_draw': odd_draw,
                                                'odd_away': odd_away,
                                                'has_odds': True
                                            })
                                            logger.info(f"Found match: {home_team} vs {away_team} - Odds: {odd_home}/{odd_draw}/{odd_away}")
                                            i += 4  # Skip to after odds
                    i += 1
                
                return matches

            # Load known teams from database first
            from app.core.database import SessionLocal
            from app.models import Team
            db_teams = SessionLocal()
            try:
                db_team_names = [t.name for t in db_teams.query(Team).all()]
                logger.info(f"Loaded {len(db_team_names)} teams from database: {db_team_names[:10]}...")
            except Exception as e:
                logger.warning(f"Could not load teams from DB: {e}")
                db_team_names = []
            finally:
                db_teams.close()
            
            # Combine with hardcoded teams (fallback)
            known_teams = list(set(db_team_names + ["Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Betis",
                          "Celta Vigo", "Eibar", "Elche", "Espanyol", "Getafe", "Girona",
                          "Granada", "Leganes", "Levante", "Malaga", "Mallorca", "Osasuna",
                          "Rayo Vallecano", "Real Madrid", "Real Sociedad", "Sevilla",
                          "Valencia", "Villarreal", "Almeria", "Cadiz", "Las Palmas",
                          "Real Oviedo", "Valladolid", "Huesca", "Zaragoza", "Tenerife",
                          "R. Madrid", "R. Sociedad", "R. Vallecano", "Villareal", "Bilbao",
                          "Vigo", "Vallecano", "Barca", "A. Madrid"]))
            
            logger.info(f"Total known teams for matching: {len(known_teams)}")
            
            # Extract matches from text
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            logger.info(f"Matches page: {len(lines)} lines")
            
            # Log all lines for debugging
            logger.info("=== ALL LINES FROM PAGE ===")
            for idx, line in enumerate(lines):
                logger.info(f"  L{idx}: '{line}'")
            logger.info("=== END OF LINES ===")
            
            # Determine matchday from latest completed results in ACTIVE SEASON only
            # Upcoming matches should be for matchday = last_completed_matchday + 1
            db_check = SessionLocal()
            try:
                active_season = db_check.query(Season).filter(Season.is_active == True).first()
                if active_season:
                    last_completed = db_check.query(Match).filter(
                        Match.is_completed == True,
                        Match.season_id == active_season.id
                    ).order_by(Match.matchday.desc()).first()
                    
                    if last_completed:
                        current_matchday = last_completed.matchday + 1
                        logger.info(f"Last completed matchday in active season (ID={active_season.id}): J{last_completed.matchday}, assigning upcoming to J{current_matchday}")
                    else:
                        # No completed matches in active season yet
                        current_matchday = 1
                        logger.info(f"No completed matches in active season, starting at J1")
                else:
                    current_matchday = 1
                    logger.warning("No active season found, defaulting to J1")
            finally:
                db_check.close()
            
            # Ensure matchday is within valid range
            if current_matchday > 38:
                current_matchday = 38
            
            matches = _extract_matches_from_text(lines, known_teams, current_matchday)
            
            logger.info(f"Extracted {len(matches)} matches for J{current_matchday}")
            if matches:
                logger.info(f"Sample matches:")
                for m in matches[:3]:
                    logger.info(f"  {m['home_team']} vs {m['away_team']} - {m['odd_home']}/{m['odd_draw']}/{m['odd_away']}")
            
            # Browser stays open
            
            return {"status": "OK", "matches": matches}
            
        except Exception as e:
            import traceback
            logger.error(f"Error during matches scraping: {e}")
            logger.error(traceback.format_exc())
            return {"status": "ERROR", "matches": []}
    
    try:
        payload = await asyncio.get_running_loop().run_in_executor(_scrape_executor, _scrape_matches_sync)
        
        if payload.get("status") == "LIVE":
            return ScrapeResultResponse(
                success=False,
                message="Match en cours (LIVE). Réessayez dans quelques minutes.",
                matches_count=0
            )
        
        matches = payload.get("matches") or []
        detected_matchday = matches[0]['matchday'] if matches else None
        saved_count = 0
        
        # Clean up old upcoming matches and predictions before adding new ones
        from app.models import Prediction as PredictionModel
        db = SessionLocal()
        try:
            active_season = db.query(Season).filter(Season.is_active == True).first()
            if active_season:
                season_id = active_season.id
                
                # Find old upcoming matches that haven't been played yet
                # We clean up EVERYTHING that is marked as upcoming before adding new ones
                # to prevent ghost matches from previous seasons from blocking the engine
                old_upcoming = db.query(Match).filter(
                    Match.is_upcoming == True,
                    Match.is_completed == False
                ).all()
                
                if old_upcoming:
                    # Delete predictions for old upcoming matches
                    old_match_ids = [m.id for m in old_upcoming]
                    deleted_preds = db.query(PredictionModel).filter(
                        PredictionModel.match_id.in_(old_match_ids)
                    ).delete(synchronize_session=False)
                    
                    # Delete old upcoming matches
                    deleted_matches = db.query(Match).filter(
                        Match.id.in_(old_match_ids)
                    ).delete(synchronize_session=False)
                    
                    db.commit()
                    logger.info(f"Cleaned up {deleted_matches} ghost matches and {deleted_preds} predictions across ALL seasons")

        except Exception as e:
            logger.error(f"Error cleaning up old matches: {e}")
            db.rollback()
        finally:
            db.close()
        
        for match_data in matches:
            db = SessionLocal()
            try:
                active_season = db.query(Season).filter(Season.is_active == True).first()
                if not active_season:
                    logger.warning("No active season found")
                    continue
                season_id = active_season.id
                
                home_team = db.query(Team).filter(Team.name == match_data['home_team']).first()
                if not home_team:
                    home_team = Team(name=match_data['home_team'])
                    # Initialize ratings for new team
                    from app.services.team_strength import TeamStrengthEngine
                    TeamStrengthEngine().initialize_team_ratings(home_team)
                    db.add(home_team)
                    db.flush()
                away_team = db.query(Team).filter(Team.name == match_data['away_team']).first()
                if not away_team:
                    away_team = Team(name=match_data['away_team'])
                    # Initialize ratings for new team
                    from app.services.team_strength import TeamStrengthEngine
                    TeamStrengthEngine().initialize_team_ratings(away_team)
                    db.add(away_team)
                    db.flush()
                
                existing = db.query(Match).filter(
                    Match.season_id == season_id,
                    Match.matchday == match_data['matchday'],
                    Match.home_team_name == match_data['home_team'],
                    Match.away_team_name == match_data['away_team']
                ).first()
                
                if existing:
                    existing.odd_home = match_data.get('odd_home')
                    existing.odd_draw = match_data.get('odd_draw')
                    existing.odd_away = match_data.get('odd_away')
                    existing.has_odds = match_data.get('has_odds', True)
                    existing.is_upcoming = True
                    # Also update team IDs in case teams were recreated
                    existing.home_team_id = home_team.id
                    existing.away_team_id = away_team.id
                else:
                    m = Match(
                        season_id=season_id,
                        matchday=match_data['matchday'],
                        line_position=match_data['line_position'],
                        home_team_name=match_data['home_team'],
                        away_team_name=match_data['away_team'],
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        odd_home=match_data.get('odd_home'),
                        odd_draw=match_data.get('odd_draw'),
                        odd_away=match_data.get('odd_away'),
                        has_odds=match_data.get('has_odds', True),
                        is_upcoming=True,
                        is_completed=False
                    )
                    db.add(m)
                
                db.commit()
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving matches: {e}")
                db.rollback()
            finally:
                db.close()
        
        predictions_count = await generate_predictions_for_matches()
        
        # ── Thread B: pré-calcul cache pour le prochain match ────────
        if matches:
            try:
                from app.services.realtime_engine import get_engine
                rt_engine = get_engine()
                # Use first upcoming match for cache prep
                next_match = matches[0]
                asyncio.create_task(
                    asyncio.to_thread(rt_engine.prepare_cache, {
                        'matchday': next_match['matchday'],
                        'home_team': next_match['home_team'],
                        'away_team': next_match['away_team'],
                        'odd_home': next_match.get('odd_home', 2.0),
                        'odd_draw': next_match.get('odd_draw', 3.2),
                        'odd_away': next_match.get('odd_away', 3.5),
                        'line_position': next_match.get('line_position', 1),
                    })
                )
                logger.info(f"RT Engine Thread B triggered: cache prep for {next_match['home_team']} vs {next_match['away_team']}")
            except Exception as rt_err:
                logger.debug(f"RT Engine Thread B (non-blocking): {rt_err}")
        # ───────────────────────────────────────────────────────────
        
        # Broadcast upcoming_matches_updated to refresh engine signals panel
        from app.core.socket_manager import socket_manager
        await socket_manager.broadcast('upcoming_matches_updated', {
            'message': f'{saved_count} matchs synchronisés',
            'matchday': detected_matchday,
            'predictions_count': predictions_count
        })
        
        return ScrapeResultResponse(
            success=True,
            message=f"Capturé {saved_count} matchs et généré {predictions_count} prédictions",
            matchday=detected_matchday,
            matches_count=saved_count
        )
    except Exception as e:
        import traceback
        logger.error(f"Error scraping matches: {e}")
        logger.error(traceback.format_exc())
        return ScrapeResultResponse(success=False, message=f"Erreur: {str(e) or type(e).__name__}")


async def generate_predictions_for_matches():
    """Generate predictions for all upcoming matches with odds."""
    from app.core.database import SessionLocal
    from app.models import Match, Prediction, MatchFeatures, Team, Season

    from app.services.feature_engineering import FeatureEngineeringPipeline
    from app.services.team_strength import TeamStrengthEngine
    from app.services.monte_carlo import MonteCarloSimulator
    from app.services.continuous_learning import ContinuousLearningEngine
    from app.core.socket_manager import socket_manager
    import traceback
    
    logger.info("Generating predictions for captured matches...")
    
    db = SessionLocal()
    
    try:
        # Load ML models
        learning_engine = ContinuousLearningEngine()
        learning_engine.load_models()
        ml_ensemble = learning_engine.ml_ensemble
        
        logger.info(f"ML ensemble is_trained: {ml_ensemble.is_trained}")
        if ml_ensemble.is_trained:
            scaler_features = getattr(ml_ensemble.scaler, 'n_features_in_', 'unknown')
            logger.info(f"Scaler expects {scaler_features} features")
        
        # Ensure models are trained (handles feature mismatch)
        learning_engine.ensure_models_trained(db)
        ml_ensemble = learning_engine.ml_ensemble
        logger.info(f"After ensure_models_trained: is_trained={ml_ensemble.is_trained}")
        
        # Initialize engines
        feature_engineer = FeatureEngineeringPipeline()
        team_strength = TeamStrengthEngine()
        monte_carlo = MonteCarloSimulator(n_simulations=1000)
        
        from app.services.realtime_engine import get_engine
        rt_engine = get_engine()
        base_cache = rt_engine.get_cache_snapshot()
        
        # Get active season
        active_season = db.query(Season).filter(Season.is_active == True).first()
        if not active_season:
            logger.warning("No active season found for predictions")
            return 0
            
        # Get upcoming matches without predictions FOR ACTIVE SEASON ONLY
        matches = db.query(Match).filter(
            Match.season_id == active_season.id,
            Match.is_upcoming == True,
            Match.has_odds == True
        ).order_by(Match.matchday.asc(), Match.line_position.asc()).limit(20).all()
        
        logger.info(f"Found {len(matches)} upcoming matches with odds for Season {active_season.season_number}")

        predictions_created = 0
        
        for match in matches:
            # Check if prediction already exists
            existing = db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).first()
            
            if existing:
                logger.info(f"Updating existing prediction for match {match.id}: {match.home_team_name} vs {match.away_team_name}")
            
            # Get or create features
            features = db.query(MatchFeatures).filter(
                MatchFeatures.match_id == match.id
            ).first()
            
            if not features:
                logger.info(f"Computing features for match {match.id}: {match.home_team_name} vs {match.away_team_name}")
                try:
                    features = feature_engineer.compute_features(match, db)
                    if features:
                        db.add(features)
                        db.commit()
                        logger.info(f"Features created successfully for match {match.id}")
                except Exception as fe_error:
                    logger.error(f"Error computing features for match {match.id}: {fe_error}")
                    logger.error(traceback.format_exc())
                    continue
            
            if not features:
                logger.warning(f"Could not compute features for match {match.id}")
                continue
            
            # Generate ML prediction
            logger.debug(f"Generating ML prediction for match {match.id}")
            try:
                ml_result = ml_ensemble.predict(features)
                logger.debug(f"ML result: {ml_result['ensemble']}")
            except Exception as ml_error:
                logger.error(f"ML prediction failed for match {match.id}: {ml_error}")
                logger.error(traceback.format_exc())
                # Use uniform probabilities as fallback
                ml_result = {
                    'ensemble': {'V': 0.33, 'N': 0.34, 'D': 0.33},
                    'model_outputs': {}
                }
            
            # Get teams
            home_team = db.query(Team).filter(Team.id == match.home_team_id).first()
            away_team = db.query(Team).filter(Team.id == match.away_team_id).first()
            
            if not home_team or not away_team:
                logger.error(f"Teams not found for match {match.id}: home_id={match.home_team_id}, away_id={match.away_team_id}")
                continue
            
            # Monte Carlo simulation inputs (Engine will do MC)
            exp_home, exp_away = team_strength.predict_poisson_goals(home_team, away_team)
            
            # ELO probabilities
            elo_probs = team_strength.predict_elo_probabilities(home_team, away_team)
            
            # ── UTILISER LE REALTIMEENGINE CORRIGÉ + MATCHANALYZER ────────────────────
            # Le RealTimeEngine intègre maintenant le MatchAnalyzer avec des poids équilibrés
            match_cache = {
                'elo': elo_probs,
                'lambda_h': exp_home,
                'lambda_a': exp_away,
                'ml_probs': ml_result.get('ensemble', {'V': 0.40, 'N': 0.30, 'D': 0.30}),
                'model_outputs': ml_result.get('model_outputs', {}),
                'line_position': match.line_position,
                'matchday': match.matchday
            }
            
            rt_result = rt_engine.run_inference(
                match_id=match.id,
                home_team=match.home_team_name,
                away_team=match.away_team_name,
                odds_h=match.odd_home or 2.0,
                odds_d=match.odd_draw or 3.2,
                odds_a=match.odd_away or 3.5,
                override_cache=match_cache
            )
            
            final_probs = rt_result['final_probs']
            predicted_result = rt_result['predicted']
            confidence = rt_result['confidence']
            model_agreement = rt_result.get('model_agreement', 0.7)
            
            # Evaluate quality
            is_good_quality = confidence >= 0.30
            is_good_odds = (match.odd_home and match.odd_home >= 1.1)
            
            logger.debug(f"MatchAnalyzer prediction for {match.home_team_name}: {predicted_result} (conf {confidence:.2f})")
            # ───────────────────────────────────────────────────
            # Create or update prediction
            if existing:
                existing.season_id = match.season_id
                existing.prob_home_win = final_probs['V']
                existing.prob_draw = final_probs['N']
                existing.prob_away_win = final_probs['D']
                existing.predicted_result = predicted_result
                existing.predicted_result_name = {'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}[predicted_result]
                existing.confidence = confidence
                existing.model_agreement = model_agreement
                existing.probability_strength = confidence
                db.commit()
                predictions_created += 1
                logger.info(f"Prediction updated: {match.home_team_name} vs {match.away_team_name} -> {existing.predicted_result_name} ({confidence:.1%})")
            else:
                prediction = Prediction(
                    match_id=match.id,
                    season_id=match.season_id,
                    prob_home_win=final_probs['V'],
                    prob_draw=final_probs['N'],
                    prob_away_win=final_probs['D'],
                    predicted_result=predicted_result,
                    predicted_result_name={'V': 'Home Win', 'N': 'Draw', 'D': 'Away Win'}[predicted_result],
                    confidence=confidence,
                    model_agreement=model_agreement,
                    probability_strength=confidence
                )
                
                db.add(prediction)
                db.commit()
                
                predictions_created += 1
                logger.info(f"Prediction: {match.home_team_name} vs {match.away_team_name} -> {prediction.predicted_result_name} ({confidence:.1%})")
        
        logger.info(f"Created {predictions_created} predictions")
        return predictions_created
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        logger.error(traceback.format_exc())
        db.rollback()
        return 0
    finally:
        db.close()
