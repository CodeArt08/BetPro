"""Bet261.mg Scraper - Alternates between Results and Matches."""
import asyncio
import re
from typing import Optional, Dict, List, Callable
from datetime import datetime
from loguru import logger
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Match, Season


class Bet261Scraper:
    """
    Scraper for Bet261.mg virtual league.
    Alternates between Results and Matches pages.
    Detects matchday from page title (Jornada XX / Journée XX).
    Waits for next matchday results based on DB state.
    
    State machine:
    - LIVE: Match in progress, wait for transition to UPCOMING
    - UPCOMING: Shows next matches with odds, extract and predict
    - After LIVE ends: Wait 10s, fetch results, compare with predictions
    """
    
    RESULTS_URL = "https://bet261.mg/virtual/category/instant-league/8037/results"
    MATCHES_URL = "https://bet261.mg/virtual/category/instant-league/8037/matches"
    MATCHES_PER_MATCHDAY = 10  # 10 matches per matchday
    SEASON_MATCHDAYS = 38
    
    # Fixed stake in Ariary
    FIXED_STAKE = 1000
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.is_running = False

        # Control whether we are allowed to scrape the Matches page.
        # The orchestrator enables this only after season verification/backfill.
        self.matches_enabled = True
        self._matches_was_live = False
        
        # Callbacks
        self.on_results: Optional[Callable] = None
        self.on_upcoming_matches: Optional[Callable] = None
        self.on_prediction_result: Optional[Callable] = None  # New: called after result comparison
        
        # State
        self.current_matchday = 0
        self.last_result_matchday = 0
        self.db_last_matchday = 0  # Last completed matchday in DB
        self.known_results: set = set()
        self.known_matches: set = set()
        
        # Track current match state for the cycle
        self._current_upcoming_matchday: Optional[int] = None
        self._waiting_for_live_result = False
        
    async def _safe_goto(self, url: str, timeout=30000, only_if_different=False) -> bool:
        """Resilient navigation with fallbacks for slow pages.
        
        Automatically accepts cookies if banner appears after navigation.
        """
        if not self.page:
            return False

        # Avoid redundant navigation if already on target URL
        if only_if_different and url in self.page.url:
            return True
            
        for wait_type in ["domcontentloaded", "commit"]:
            try:
                await self.page.goto(url, wait_until=wait_type, timeout=timeout) # type: ignore
                await asyncio.sleep(0.5)  # Réduit de 1.5s à 0.5s
                
                # Auto-accept cookies after each navigation
                await self._try_accept_cookies()
                
                return True
            except Exception as e:
                logger.warning(f"Goto ({wait_type}) failed for {url}: {e}")
                timeout = 15000  # Shorter for fallbacks
        return False
    
    async def _try_accept_cookies(self):
        """Try to accept cookies if banner is present. Non-blocking."""
        try:
            # Wait a moment for banner to appear
            await asyncio.sleep(0.5)
            
            # Quick check for cookie banner - common selectors
            cookie_selectors = [
                # French
                'button:has-text("Autoriser")',
                'button:has-text("Autoriser tous")',
                'button:has-text("Accepter")',
                'button:has-text("Tout accepter")',
                # English
                'button:has-text("Accept")',
                'button:has-text("Accept all")',
                'button:has-text("Allow")',
                'button:has-text("Allow all")',
                # Generic cookie selectors
                '[class*="cookie"] button:has-text("Autoriser")',
                '[class*="cookie"] button:has-text("Accepter")',
                '[class*="cookie"] button:has-text("Accept")',
                '.cookie-banner button',
                '#cookie-accept',
                '[data-cookie="accept"]',
                'button[id*="accept"]',
            ]
            
            for selector in cookie_selectors:
                try:
                    btn = await self.page.query_selector(selector)
                    if btn:
                        is_visible = await btn.is_visible()
                        if is_visible:
                            await btn.click(timeout=3000)
                            logger.info(f"Cookie banner accepted via: {selector}")
                            await asyncio.sleep(1)
                            return
                except Exception as e:
                    logger.debug(f"Cookie selector {selector} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Cookie check: {e}")
        
    def _get_db_last_matchday(self) -> int:
        """Get the last completed matchday from database."""
        try:
            db = SessionLocal()
            active_season = db.query(Season).filter(Season.is_active == True).first()
            if not active_season:
                self.db_last_matchday = 0
                logger.info("No active season found when reading DB last matchday")
                db.close()
                return 0

            # Get max matchday where matches are completed
            last_match = db.query(Match).filter(
                Match.season_id == active_season.id,
                Match.is_completed == True
            ).order_by(Match.matchday.desc()).first()
            
            if last_match:
                self.db_last_matchday = last_match.matchday
                logger.info(f"Last completed matchday in DB: J{self.db_last_matchday}")
            else:
                self.db_last_matchday = 0
                logger.info("No completed matches in DB yet")
            
            db.close()
            return self.db_last_matchday
        except Exception as e:
            logger.error(f"Error getting DB last matchday: {e}")
            return 0
        
    async def start(self):
        """Start browser."""
        logger.info("Starting Bet261 scraper...")
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=settings.HEADLESS,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            self.page = await self.context.new_page()
            self.is_running = True
            
            # Handle cookie banner on first load
            await self._handle_cookies()
            
            logger.info("Bet261 scraper started")
        except Exception as e:
            import traceback
            logger.error(f"Error starting browser: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def stop(self):
        """Stop browser and cleanup."""
        logger.info("Stopping Bet261 scraper...")
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            self.is_running = False
            logger.info("Bet261 scraper stopped")
        except Exception as e:
            logger.error(f"Error stopping scraper: {e}")
    
    async def _handle_cookies(self):
        """Accept cookies if banner appears."""
        try:
            # Go to site once to trigger cookie banner
            if not await self._safe_goto("https://bet261.mg"):
                return
            
            # Wait for page to fully load
            await asyncio.sleep(3)
            
            # Look for "Autoriser tous les cookies" button - try multiple approaches
            cookie_selectors = [
                # French - most common on bet261.mg
                'button:has-text("Autoriser tous les cookies")',
                'button:has-text("Autoriser")',
                'button:has-text("Autoriser tous")',
                'button:has-text("Accepter")',
                'button:has-text("Tout accepter")',
                # English
                'button:has-text("Accept")',
                'button:has-text("Accept all")',
                'button:has-text("Allow")',
                'button:has-text("Allow all")',
                # Generic cookie selectors
                '[class*="cookie"] button:has-text("Autoriser")',
                '[class*="cookie"] button:has-text("Accepter")',
                '[class*="cookie"] button:has-text("Accept")',
                '.cookie-banner button',
                '#cookie-accept',
                '[data-cookie="accept"]',
                # Common consent manager buttons
                'button[id*="accept"]',
                'a:has-text("Accepter")',
                'a:has-text("Autoriser")',
            ]
            
            for selector in cookie_selectors:
                try:
                    btn = await self.page.query_selector(selector)
                    if btn:
                        is_visible = await btn.is_visible()
                        if is_visible:
                            await btn.click(timeout=5000)
                            logger.info(f"Cookie banner accepted via: {selector}")
                            # Wait longer after clicking cookie
                            await asyncio.sleep(3)
                            return
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            # Try clicking any visible button that might be a cookie accept
            try:
                # Look for modal/popup that might contain cookie consent
                modal = await self.page.query_selector('[class*="modal"], [class*="popup"], [class*="consent"]')
                if modal:
                    buttons = await modal.query_selector_all('button')
                    for btn in buttons:
                        text = await btn.inner_text()
                        if any(word in text.lower() for word in ['autoriser', 'accepter', 'accept', 'allow', 'ok', 'oui', 'yes']):
                            await btn.click()
                            logger.info(f"Cookie accepted via modal button: {text}")
                            await asyncio.sleep(3)
                            return
            except Exception as e:
                logger.debug(f"Modal cookie search failed: {e}")
            
            logger.debug("No cookie banner found or already accepted")
            
        except Exception as e:
            logger.debug(f"Cookie handling: {e}")
    
    async def monitor_cycle(self):
        """One complete cycle: Results -> Matches."""
        if not self.is_running:
            return
            
        try:
            # 1. Go to Results page first
            await self._scrape_results()

            # 2. Then go to Matches page (only when enabled by orchestrator)
            if not self.matches_enabled:
                logger.info("Skipping MATCHES scrape (matches_enabled=False)")
                return

            matches_status = await self._scrape_matches()

            # If we just exited LIVE, fetch results again (last LIVE result) before scraping upcoming.
            if matches_status == "NEED_RESULTS_AFTER_LIVE":
                await self._scrape_results()
                await self._scrape_matches()
            
        except Exception as e:
            logger.error(f"Error in monitor_cycle: {e}")

    async def _load_more_results(self, max_clicks: int = 25) -> int:
        """Click 'Afficher plus' / 'Show more' on results page to load older matchdays."""
        selectors = [
            'button:has-text("Afficher plus")',
            'span:has-text("Afficher plus")',
            'text=Afficher plus',
            'button:has-text("Show more")',
            'text=Show more',
        ]

        clicks = 0
        for _ in range(max_clicks):
            btn = None
            for sel in selectors:
                try:
                    btn = await self.page.query_selector(sel)
                    if btn:
                        break
                except Exception:
                    continue

            if not btn:
                break

            try:
                await btn.scroll_into_view_if_needed()
                await btn.click()
                clicks += 1
                await asyncio.sleep(0.8)  # Réduit de 2s à 0.8s
            except Exception:
                break

        if clicks > 0:
            logger.info(f"Loaded more results: clicked {clicks} times")
        return clicks
    
    async def _detect_matchday_from_page(self) -> int:
        """Detect current matchday from page title/header/body."""
        try:
            import re
            text = await self.page.inner_text('body')
            # Find ALL matchday patterns
            matches = re.findall(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', text)
            
            if matches:
                # Convert to integers and filter valid range
                valid_matchdays = []
                for m in matches:
                    try:
                        val = int(m)
                        if 1 <= val <= 38:
                            valid_matchdays.append(val)
                    except ValueError:
                        continue
                
                if valid_matchdays:
                    # Take the FIRST matchday found - this is the most recent on results page
                    # Results page shows most recent first
                    matchday = valid_matchdays[0]
                    logger.info(f"Detected matchdays {sorted(set(valid_matchdays))}, selected FIRST: J{matchday}")
                    return matchday

            # Try to get from specific title elements as fallback
            elements = await self.page.query_selector_all('h1, h2, h3, .title, .header')
            for el in elements:
                text = await el.inner_text()
                match = re.search(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', text)
                if match:
                    matchday = int(match.group(1))
                    if 1 <= matchday <= 38:
                        logger.info(f"Detected matchday J{matchday} from element: '{text[:30]}...'")
                        return matchday
            
            # Fallback: try page title
            title = await self.page.title()
            matches = re.findall(r'(?i)(?:journ[eé]e|jornada)\s*(\d+)', title)
            if matches:
                matchdays = [int(m) for m in matches if 1 <= int(m) <= 38]
                if matchdays:
                    logger.info(f"Detected matchday J{max(matchdays)} from title: '{title}'")
                    return max(matchdays)
                        
        except Exception as e:
            logger.warning(f"Could not detect matchday: {e}")
        
        return 0
    
    async def _count_result_groups_on_page(self) -> int:
        """Count how many groups of 10 matches (matchdays) are visible on the results page."""
        try:
            text = await self.page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            score_count = 0
            for i, line in enumerate(lines):
                if ':' in line and not line.startswith('MT') and not line.startswith('MT:'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        p1 = parts[0].strip().replace('\xa0', '')
                        p2 = parts[1].strip().replace('\xa0', '')
                        if p1.isdigit() and p2.isdigit():
                            score_count += 1
            groups = max(1, score_count // self.MATCHES_PER_MATCHDAY)
            logger.info(f"Detected ~{score_count} scores = ~{groups} matchday groups on page")
            return groups
        except Exception as e:
            logger.warning(f"Could not count result groups: {e}")
            return 1

    async def _get_completed_matchdays_in_db(self) -> set:
        """Return set of matchday numbers that are fully completed in DB (10 matches each)."""
        try:
            db = SessionLocal()
            active_season = db.query(Season).filter(Season.is_active == True).first()
            if not active_season:
                db.close()
                return set()
            from sqlalchemy import func
            rows = db.query(
                Match.matchday, func.count(Match.id).label('cnt')
            ).filter(
                Match.season_id == active_season.id,
                Match.is_completed == True
            ).group_by(Match.matchday).all()
            db.close()
            complete = {row.matchday for row in rows if row.cnt >= self.MATCHES_PER_MATCHDAY}
            partial = {row.matchday for row in rows if row.cnt < self.MATCHES_PER_MATCHDAY}
            if partial:
                logger.info(f"Partial matchdays in DB (< 10 matches): {sorted(partial)}")
            if complete:
                logger.info(f"Complete matchdays in DB: {sorted(complete)}")
            return complete
        except Exception as e:
            logger.error(f"Error getting completed matchdays: {e}")
            return set()

    async def _scrape_results(self):
        """Scrape results page - detect and backfill all missing matchdays.
        
        NEVER BLOCKS: Always fetches new results regardless of DB state.
        Uses the actual site matchday as source of truth, not DB last matchday.
        """
        logger.info("Scraping RESULTS...")
        
        # Get last completed matchday from DB (for reference only)
        db_last = self._get_db_last_matchday()
        # Get fully complete matchdays (10 matches each)
        complete_matchdays = await self._get_completed_matchdays_in_db()
        
        if db_last > 0:
            logger.info(f"DB last completed matchday: J{db_last} ({len(complete_matchdays)} full matchdays)")
        
        try:
            if not await self._safe_goto(self.RESULTS_URL):
                logger.error("Could not reach results page")
                return
            
            # Try to detect matchday from page title/headers
            detected_matchday = await self._detect_matchday_from_page()
            
            # If page detection failed, estimate from counting score groups
            if detected_matchday == 0:
                await self._load_more_results(max_clicks=3)
                groups_visible = await self._count_result_groups_on_page()
                detected_matchday = groups_visible
                logger.info(f"Estimated current matchday from page: J{detected_matchday}")
            
            logger.info(f"Site matchday: J{detected_matchday}, DB last: J{db_last}, Complete count: {len(complete_matchdays)}")
            
            # Determine how many 'Afficher plus' clicks needed
            if db_last == 0:
                # DB empty - load everything
                clicks_needed = 15
            else:
                # Calculate missing matchdays
                if detected_matchday > 0:
                    missing_matchdays = [md for md in range(1, detected_matchday + 1) if md not in complete_matchdays]
                    # Need at least 1 click to ensure we see the latest matchday
                    clicks_needed = max(1, min(len(missing_matchdays) + 1, 5))
                else:
                    clicks_needed = 2
            
            await self._load_more_results(max_clicks=clicks_needed)
            
            # Extract results from page
            results = await self._extract_results(detected_matchday if detected_matchday > 0 else 1)
            
            if not results:
                logger.info("No results extracted from page")
                return
            
            logger.info(f"Extracted {len(results)} results from page, processing new ones...")
            
            # Process ALL results that are not yet in DB
            # Do NOT use strict db_last filter - check each result individually
            new_results_count = 0
            for result in results:
                md = result['matchday']
                key = f"{md}_{result['home_team']}_{result['away_team']}"
                
                # Skip only if we already processed this exact match this session
                if key in self.known_results:
                    continue
                
                # Always register the key to avoid duplicate callbacks
                self.known_results.add(key)
                
                # Only trigger callback if matchday not fully complete in DB
                if md not in complete_matchdays:
                    logger.info(f"NEW RESULT: {result['home_team']} {result['score_home']}-{result['score_away']} {result['away_team']} (J{md})")
                    new_results_count += 1
                    if self.on_results:
                        await self.on_results(result)
            
            if new_results_count == 0:
                logger.info(f"No new results found (site J{detected_matchday}, DB J{db_last})")
            else:
                logger.info(f"Processed {new_results_count} new result(s)")
                
        except Exception as e:
            logger.error(f"Error scraping results: {e}")

    async def _extract_results_multi(self, detected_matchday: int, max_days: int) -> List[Dict]:
        """Extract multiple matchdays from results page.

        Bet261 shows results as a continuous list. We chunk matches by groups of 10
        and assign matchday numbers from newest (detected_matchday) backwards.
        """
        flat = await self._extract_results(detected_matchday)
        if not flat:
            return []

    async def _extract_results_by_chunking(self, detected_matchday: int, max_days: int) -> List[Dict]:
        """Extract results by grouping matches into chunks of 10.

        This is resilient when the page does not include reliable 'Journée X' headers.
        The page is assumed to list matches from newest -> oldest.
        """
        try:
            text = await self.page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            extracted: List[Dict] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if ':' in line and not line.startswith('MT') and not line.startswith('MT:'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            p1 = parts[0].strip().replace('\xa0', '')
                            p2 = parts[1].strip().replace('\xa0', '')
                            if p1.isdigit() and p2.isdigit():
                                score_home = int(p1)
                                score_away = int(p2)

                                home_team = None
                                for j in range(i-1, max(i-5, 0), -1):
                                    prev_line = lines[j]
                                    if "'" in prev_line or 'MT:' in prev_line or ':' in prev_line:
                                        continue
                                    if prev_line and not prev_line.isdigit() and "Journ" not in prev_line:
                                        home_team = prev_line
                                        break

                                away_team = None
                                for j in range(i+1, min(i+6, len(lines))):
                                    next_line = lines[j]
                                    if 'MT:' in next_line or "'" in next_line:
                                        continue
                                    if ':' in next_line or "Journ" in next_line:
                                        break
                                    if next_line and not next_line.isdigit():
                                        away_team = next_line
                                        break

                                if home_team and away_team and not home_team.isdigit() and not away_team.isdigit():
                                    if any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                                        extracted.append({
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'score_home': score_home,
                                            'score_away': score_away,
                                        })
                        except Exception:
                            pass
                i += 1

            if not extracted:
                return []

            max_days = max(1, int(max_days))
            detected_matchday = max(1, int(detected_matchday))

            # Keep only the newest N matchdays worth of matches
            needed_matches = min(len(extracted), max_days * self.MATCHES_PER_MATCHDAY)
            newest_window = extracted[:needed_matches]

            results: List[Dict] = []
            for idx, m in enumerate(newest_window):
                md_offset = idx // self.MATCHES_PER_MATCHDAY
                md = detected_matchday - md_offset
                if md < 1:
                    break
                line_position = (idx % self.MATCHES_PER_MATCHDAY) + 1
                results.append({
                    'matchday': md,
                    'line_position': line_position,
                    'home_team': m['home_team'],
                    'away_team': m['away_team'],
                    'score_home': m['score_home'],
                    'score_away': m['score_away'],
                    'odd_home': None,
                    'odd_draw': None,
                    'odd_away': None,
                    'result': self._get_result(m['score_home'], m['score_away']),
                    'is_completed': True,
                    'timestamp': datetime.utcnow()
                })

            return results
        except Exception as e:
            logger.error(f"Error extracting results by chunking: {e}")
            return []

        # The current _extract_results already limits to 10. For multi-day we need to parse
        # the full page again but without limiting.
        results: List[Dict] = []
        try:
            text = await self.page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            tmp: List[Dict] = []
            current_matchday_in_loop = detected_matchday

            # We only want a bounded window of matchdays (newest -> older)
            max_days = max(1, int(max_days))
            min_md = max(1, detected_matchday - max_days + 1)
            max_md = min(self.SEASON_MATCHDAYS, detected_matchday)
            per_md_counts: Dict[int, int] = {}

            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Check for Matchday header in the line like "Journée 12"
                import re
                md_match = re.search(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', line)
                if md_match:
                    parsed_md = int(md_match.group(1))
                    if 1 <= parsed_md <= 38:
                        current_matchday_in_loop = parsed_md
                        
                if ':' in line and not line.startswith('MT') and not line.startswith('MT:'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            p1 = parts[0].strip().replace('\xa0', '')
                            p2 = parts[1].strip().replace('\xa0', '')
                            if p1.isdigit() and p2.isdigit():
                                score_home = int(p1)
                                score_away = int(p2)

                                home_team = None
                                for j in range(i-1, max(i-5, -1), -1):
                                    prev_line = lines[j]
                                    if "'" in prev_line or 'MT:' in prev_line or ':' in prev_line:
                                        continue
                                    if prev_line and not prev_line.isdigit() and "Journ" not in prev_line:
                                        home_team = prev_line
                                        break

                                away_team = None
                                for j in range(i+1, min(i+6, len(lines))):
                                    next_line = lines[j]
                                    if 'MT:' in next_line or "'" in next_line:
                                        continue
                                    if ':' in next_line or "Journ" in next_line:
                                        break
                                    if next_line and not next_line.isdigit():
                                        away_team = next_line
                                        break

                                if home_team and away_team and not home_team.isdigit() and not away_team.isdigit():
                                    if any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                                        md = int(current_matchday_in_loop)
                                        if min_md <= md <= max_md:
                                            cnt = per_md_counts.get(md, 0)
                                            if cnt < self.MATCHES_PER_MATCHDAY:
                                                per_md_counts[md] = cnt + 1
                                                tmp.append({
                                                    'matchday': md,
                                                    'line_position': 0, # assigned below
                                                    'home_team': home_team,
                                                    'away_team': away_team,
                                                    'score_home': score_home,
                                                    'score_away': score_away,
                                                    'odd_home': None,
                                                    'odd_draw': None,
                                                    'odd_away': None,
                                                    'result': self._get_result(score_home, score_away),
                                                    'is_completed': True,
                                                    'timestamp': datetime.utcnow()
                                                })

                                            # Early exit if we have 10 matches for each requested matchday
                                            if per_md_counts and all(per_md_counts.get(md_check, 0) >= self.MATCHES_PER_MATCHDAY for md_check in range(min_md, max_md + 1)):
                                                break
                        except Exception:
                            pass
                i += 1

                if per_md_counts and all(per_md_counts.get(md_check, 0) >= self.MATCHES_PER_MATCHDAY for md_check in range(min_md, max_md + 1)):
                    break

            # Make sure we don't return malformed/over-processed data
            # Calculate actual line_positions per matchday
            counts_per_md = {}
            for r in tmp:
                md = r['matchday']
                counts_per_md[md] = counts_per_md.get(md, 0) + 1
                r['line_position'] = counts_per_md[md]
                results.append(r)

            logger.info(f"Extracted {len(results)} individual results")
            return results
        except Exception as e:
            logger.error(f"Error extracting multi results: {e}")
            return []

    async def get_season_first_match_snapshot(self) -> Optional[Dict]:
        """Return the match #1 of matchday 1 from the results page.

        This is used to verify if the website is still on the same season as the DB.
        """
        try:
            if not await self._safe_goto(self.RESULTS_URL):
                return None

            # Ensure older matchdays (including J1) are loaded
            await self._load_more_results(max_clicks=25)

            # Allow DOM to settle after pagination
            await asyncio.sleep(2)

            # Parse full page text and extract ALL matches (no 10-limit)
            text = await self.page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            extracted: List[Dict] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if ':' in line and not line.startswith('MT') and not line.startswith('MT:'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            p1 = parts[0].strip().replace('\xa0', '')
                            p2 = parts[1].strip().replace('\xa0', '')
                            if p1.isdigit() and p2.isdigit():
                                score_home = int(p1)
                                score_away = int(p2)

                                home_team = None
                                for j in range(i-1, max(i-5, 0), -1):
                                    prev_line = lines[j]
                                    if "'" in prev_line or 'MT:' in prev_line or ':' in prev_line:
                                        continue
                                    if prev_line and not prev_line.isdigit() and "Journ" not in prev_line:
                                        home_team = prev_line
                                        break

                                away_team = None
                                for j in range(i+1, min(i+6, len(lines))):
                                    next_line = lines[j]
                                    if 'MT:' in next_line or "'" in next_line:
                                        continue
                                    if ':' in next_line or "Journ" in next_line:
                                        break
                                    if next_line and not next_line.isdigit():
                                        away_team = next_line
                                        break

                                if home_team and away_team and not home_team.isdigit() and not away_team.isdigit():
                                    if any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                                        extracted.append({
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'score_home': score_home,
                                            'score_away': score_away,
                                        })
                        except Exception:
                            pass
                i += 1

            if len(extracted) < self.MATCHES_PER_MATCHDAY:
                return None

            # Bet261 results list is newest -> oldest. So oldest matchday (J1) is at the END.
            # Take the last 10 matches (preserve order as displayed) and treat them as matchday 1.
            oldest_chunk = extracted[-self.MATCHES_PER_MATCHDAY:]
            if len(oldest_chunk) < self.MATCHES_PER_MATCHDAY:
                return None

            first = oldest_chunk[0]
            return {
                'matchday': 1,
                'line_position': 1,
                'home_team': first['home_team'],
                'away_team': first['away_team'],
                'score_home': first['score_home'],
                'score_away': first['score_away'],
            }
        except Exception as e:
            logger.error(f"Error getting season first match snapshot: {e}")
            return None
     
    async def _scrape_matches(self) -> str:
        """Scrape upcoming matches page - wait for upcoming matches to appear."""
        logger.info("Scraping MATCHES...")
        
        # Get all FULLY complete matchdays (10 matches) from DB
        complete_matchdays = await self._get_completed_matchdays_in_db()
        db_last = self._get_db_last_matchday()
        
        try:
            if not await self._safe_goto(self.MATCHES_URL):
                logger.error("Could not reach matches page")
                return "ERROR"

            # If the site is currently showing a LIVE match phase, ignore it.
            # Upcoming matches appear after the live phase ends.
            if await self._is_live_phase():
                self._matches_was_live = True
                logger.info("Matches page is in LIVE phase - ignoring and waiting for results/upcoming fixtures.")
                return "LIVE"

            # Transition: LIVE -> NOT LIVE
            if self._matches_was_live:
                self._matches_was_live = False
                logger.info("LIVE phase ended on Matches page. Waiting 10s then fetching Results before scraping upcoming...")
                await asyncio.sleep(10)
                return "NEED_RESULTS_AFTER_LIVE"
            
            # Detect matchday from page
            detected_matchday = await self._detect_matchday_from_page()
            if detected_matchday > 0:
                logger.info(f"Matches page shows matchday: J{detected_matchday}")

            # If matchday cannot be detected from the page, fall back to DB state.
            # Upcoming fixtures are expected to be the next matchday after the last completed.
            fallback_matchday = detected_matchday if detected_matchday > 0 else max(1, db_last + 1)
            
            # Wait for upcoming matches to appear
            max_wait = 15  # Réduit de 30s à 15s pour rester dans le cycle de 30s
            waited = 0
            matches = []
            
            # Check countdown first - ne pas attendre si countdown trop long
            countdown = await self._get_countdown()
            if countdown and countdown > 0:
                if countdown < 20:  # Seulement si très imminent
                    logger.info(f"Upcoming match in {countdown}s - courte attente...")
                    await asyncio.sleep(min(countdown + 2, 10))  # Max 10s
                    await self.page.reload()
                    await asyncio.sleep(1)
                else:
                    logger.info(f"Next match in {countdown}s, will check again in next cycle.")
            
            while waited < max_wait:
                matches = await self._extract_matches(fallback_matchday=fallback_matchday)
                # Accepter >= 9 matchs (pas forcément 10 - bug connu du site)
                if matches and len(matches) >= 9:
                    # Check if odds are present in matches
                    if all(m.get('has_odds') for m in matches):
                        break
                
                logger.debug(f"Waiting for matches with odds... ({waited}s)")
                await asyncio.sleep(2)  # Réduit de 3s à 2s
                waited += 2
                if waited % 10 == 0:
                    await self.page.reload()
                    await asyncio.sleep(1)
            
            if matches:
                match_matchday = matches[0]['matchday']
                logger.info(f"Found {len(matches)} upcoming matches for J{match_matchday}")
                
                # Send upcoming matches if their matchday is NOT fully complete in DB
                if match_matchday not in complete_matchdays:
                    logger.info(f"{len(matches)} upcoming matches for J{match_matchday} (not yet complete in DB)")
                    if self.on_upcoming_matches:
                        await self.on_upcoming_matches(matches)
                    return "UPCOMING"
                elif match_matchday > db_last:
                    logger.info(f"{len(matches)} matches to predict for J{match_matchday} (DB at J{db_last})")
                    if self.on_upcoming_matches:
                        await self.on_upcoming_matches(matches)
                    return "UPCOMING"
                if match_matchday in complete_matchdays:
                    logger.info(f"Matchday J{match_matchday} already complete in DB ({len(complete_matchdays)} full matchdays stored)")
                    return "SKIP_ALREADY_COMPLETE"
            
            return "NO_UPCOMING"
        except Exception as e:
            logger.error(f"Error scraping matches: {e}")
            return "ERROR"
    
    async def _extract_results(self, detected_matchday: int = 0) -> List[Dict]:
        """Extract ALL results from page - Bet261 specific format.
        
        Returns results with matchday detected from page text.
        """
        results = []
        
        try:
            # Bet261 format in one big element:
            # HomeTeam\nGoalMinutes\nScore\nMT: Score\nAwayTeam\nGoalMinutes
            # Example: "Girona\n56' 82' 87'\n3:1\nMT: 0:1\nVigo\n30'"
            
            # Get all text from page
            text = await self.page.inner_text('body')
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            
            logger.info(f"_extract_results: Page has {len(lines)} lines of text, {len(text)} chars")
            
            # Log first 20 lines for debugging
            for i, line in enumerate(lines[:20]):
                logger.debug(f"  Line {i}: '{line}'")
            
            # Detect matchdays from page text
            matchday_map = {}  # line_index -> matchday
            current_matchday = detected_matchday if detected_matchday > 0 else 1
            
            # Find all matchday markers in the text
            for i, line in enumerate(lines):
                match = re.search(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', line)
                if match:
                    md = int(match.group(1))
                    if 1 <= md <= 38:
                        current_matchday = md
                        logger.debug(f"Found matchday J{md} at line {i}")
                matchday_map[i] = current_matchday
            
            # Parse matches - look for score pattern "X:Y" (not MT:)
            i = 0
            last_matchday_line = -10  # Track last matchday marker
            
            while i < len(lines):
                line = lines[i]
                
                # Update current matchday if we see a matchday marker
                match = re.search(r'(?i)(?:journ[eé]e|jornada|matchday)\s*(\d{1,2})', line)
                if match:
                    md = int(match.group(1))
                    if 1 <= md <= 38:
                        current_matchday = md
                        last_matchday_line = i
                        logger.debug(f"Current matchday updated to J{current_matchday}")
                
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
                                    # Validate team names (should be alphabetic or have common patterns)
                                    if any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                                        # Determine matchday - use the most recent matchday marker
                                        md = current_matchday
                                        
                                        results.append({
                                            'matchday': md,
                                            'line_position': len([r for r in results if r['matchday'] == md]) + 1,
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'score_home': score_home,
                                            'score_away': score_away,
                                            'odd_home': None,
                                            'odd_draw': None,
                                            'odd_away': None,
                                            'result': self._get_result(score_home, score_away),
                                            'is_completed': True,
                                            'timestamp': datetime.utcnow()
                                        })
                                        logger.debug(f"Found: J{md} {home_team} {score_home}-{score_away} {away_team}")
                        except Exception as e:
                            pass
                
                i += 1
            
            # DO NOT limit to 10 - we want ALL results from the page
            # Sort by matchday descending (most recent first)
            results.sort(key=lambda x: (-x['matchday'], x['line_position']))
            
            if results:
                matchdays_found = sorted(set(r['matchday'] for r in results))
                logger.info(f"Extracted {len(results)} results from matchdays: {matchdays_found}")
                for r in results[:5]:
                    logger.info(f"  J{r['matchday']}: {r['home_team']} {r['score_home']}-{r['score_away']} {r['away_team']}")
                if len(results) > 5:
                    logger.info(f"  ... and {len(results) - 5} more")
                    
        except Exception as e:
            logger.error(f"Error extracting results: {e}")
            
        return results
    
    async def _extract_matches(self, fallback_matchday: int = 1) -> List[Dict]:
        """Extract upcoming matches from page.
        
        Bet261 match page structure (when not LIVE):
        - Each match is in a container with team names and 3 odds buttons (1, X, 2)
        - Odds are displayed as decimal numbers
        """
        matches = []
        
        try:
            # Detect matchday from page first
            detected_matchday = await self._detect_matchday_from_page()
            
            # Method 1: Try to extract from the full page text (most reliable)
            page_text = await self.page.inner_text('body')
            
            # Debug: log more lines when extraction fails
            lines = [l.strip() for l in page_text.split('\n') if l.strip()]
            logger.info(f"Page has {len(lines)} lines of text")
            if len(lines) < 50:
                logger.warning(f"Page content seems incomplete. Lines: {lines[:50]}")
            
            matches = await self._extract_matches_from_text(page_text, detected_matchday, fallback_matchday)
            
            if matches and len(matches) >= 10:
                logger.info(f"Extracted {len(matches)} matches from page text")
                return matches
            
            # Method 2: Try specific selectors for match elements with odds
            # Priority: find elements that contain BOTH team names AND odds
            selectors = [
                '[class*="match-event"]',
                '[class*="match-item"]',
                '[class*="match-row"]',
                '[class*="event-match"]',
                '.match-container',
                '[class*="fixture"]',
                'tr[class*="row"]',  # Table rows
                'li[class*="item"]',  # List items
                '.event',
                '.fixture',
            ]
            
            elements = []
            for sel in selectors:
                els = await self.page.query_selector_all(sel)
                if els and len(els) >= 5:
                    elements = els
                    logger.info(f"Found {len(elements)} match elements with selector: {sel}")
                    break
            
            # Fallback: get all elements that have both text and odds
            if not elements:
                # Try getting all clickable elements with odds
                elements = await self.page.query_selector_all('[class*="match"], [class*="event"], [class*="row"]')
                logger.info(f"Found {len(elements)} elements with generic selectors")
            
            # Limit to first 20 elements to avoid noise
            elements = elements[:20]
            
            for i, el in enumerate(elements):
                try:
                    text = await el.inner_text()
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    
                    # Log first few elements for debugging
                    if i < 3:
                        logger.info(f"Element {i+1}: {len(lines)} lines - {text[:100]}...")
                    
                    # Format from screenshot: 2 lines per match
                    # Line 1: Team1 + odd_home
                    # Line 2: Team2 + odd_draw + odd_away
                    
                    # Try 2-line format first
                    if len(lines) >= 2:
                        # Extract odds from both lines
                        odds1 = re.findall(r"\d+(?:[\.,]\d+)?", lines[0])
                        odds2 = re.findall(r"\d+(?:[\.,]\d+)?", lines[1])
                        
                        # Format: line1 has 1 odd, line2 has 2 odds
                        if len(odds1) >= 1 and len(odds2) >= 2:
                            try:
                                odd_home = float(odds1[0].replace(',', '.'))
                                odd_draw = float(odds2[0].replace(',', '.'))
                                odd_away = float(odds2[1].replace(',', '.'))
                                
                                # Validate odds
                                if not (1.0 <= odd_home <= 50.0 and 1.0 <= odd_draw <= 50.0 and 1.0 <= odd_away <= 50.0):
                                    continue
                                
                                # Extract team names by removing odds
                                home = re.sub(r'\d+(?:[\.,]\d+)?', '', lines[0]).strip()
                                away = re.sub(r'\d+(?:[\.,]\d+)?', '', lines[1]).strip()
                                
                                # Validate team names
                                if not home or not away:
                                    continue
                                if not any(c.isalpha() for c in home) or not any(c.isalpha() for c in away):
                                    continue
                                
                                matchday = detected_matchday if detected_matchday > 0 else max(1, int(fallback_matchday))
                                line_position = len(matches) + 1
                                
                                matches.append({
                                    'matchday': matchday,
                                    'line_position': line_position,
                                    'home_team': home,
                                    'away_team': away,
                                    'odd_home': odd_home,
                                    'odd_draw': odd_draw,
                                    'odd_away': odd_away,
                                    'is_upcoming': True,
                                    'has_odds': True
                                })
                                
                                logger.debug(f"Match {line_position}: {home} vs {away} - {odd_home}/{odd_draw}/{odd_away}")
                                continue
                            except:
                                pass
                    
                    # Fallback: old 5-line format
                    if len(lines) >= 5:
                        home = lines[0]
                        away = lines[1]

                        if not home or not away:
                            continue
                        if home.isdigit() or away.isdigit():
                            continue
                        if not any(c.isalpha() for c in home) or not any(c.isalpha() for c in away):
                            continue
                        
                        odd_home = odd_draw = odd_away = None
                        has_odds = False
                        
                        try:
                            rest = "\n".join(lines[2:])
                            nums = re.findall(r"\d+(?:[\.,]\d+)?", rest)
                            if len(nums) >= 3:
                                odd_home = float(nums[0].replace(',', '.'))
                                odd_draw = float(nums[1].replace(',', '.'))
                                odd_away = float(nums[2].replace(',', '.'))

                                if (1.0 <= odd_home <= 50.0 and 1.0 <= odd_draw <= 50.0 and 1.0 <= odd_away <= 50.0):
                                    has_odds = True
                        except:
                            pass
                        
                        matchday = detected_matchday if detected_matchday > 0 else max(1, int(fallback_matchday))
                        line_position = len(matches) + 1
                        
                        matches.append({
                            'matchday': matchday,
                            'line_position': line_position,
                            'home_team': home,
                            'away_team': away,
                            'odd_home': odd_home,
                            'odd_draw': odd_draw,
                            'odd_away': odd_away,
                            'is_upcoming': True,
                            'has_odds': has_odds
                        })
                        
                        if has_odds:
                            logger.debug(f"Match {line_position}: {home} vs {away} - {odd_home}/{odd_draw}/{odd_away}")
                        else:
                            logger.info(f"Match {line_position}: {home} vs {away} - (Waiting for odds)")
                        
                        # Stop at 10 matches
                        if len(matches) >= 10:
                            break
                    else:
                        logger.debug(f"Element {i+1}: not enough lines ({len(lines)})")
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting matches: {e}")
            
        return matches
    
    async def _extract_matches_from_text(self, page_text: str, detected_matchday: int, fallback_matchday: int) -> List[Dict]:
        """Extract matches from full page text using pattern matching.
        
        Bet261 format for upcoming matches:
        Team1    cote1
        Team2    coteX    cote2
        
        Each match takes 2 consecutive lines in the page text.
        Correction bug : scan toutes les paires de lignes pour ne pas manquer le 10ème match.
        """
        matches = []
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]
        
        logger.info(f"Extracting matches from {len(lines)} lines of text")
        
        # Log first 30 lines for debugging
        for i, line in enumerate(lines[:30]):
            logger.info(f"Line {i}: {line[:80]}")
        
        # Pattern pour les cotes décimales (ex: 2.50, 3,20)
        odds_pattern = re.compile(r'(\d+[,.]\d+)')
        
        # Mots-clés à exclure
        EXCLUDE_KEYWORDS = re.compile(
            r'\b(LIVE|live|Result|Score|EN DIRECT|Mot de passe|Se connecter|S\'inscrire|Connexion|Inscription|Paris|Bonus)\b',
            re.IGNORECASE
        )
        
        i = 0
        # On cherche jusqu'à 10 matchs (MATCHES_PER_MATCHDAY)
        while i < len(lines) - 1 and len(matches) < self.MATCHES_PER_MATCHDAY:
            line1 = lines[i]
            line2 = lines[i + 1] if i + 1 < len(lines) else ""
            
            # Extraire toutes les cotes décimales des deux lignes
            odds1 = odds_pattern.findall(line1)
            odds2 = odds_pattern.findall(line2)
            
            # Format Bet261 : line1 a 1 cote (domicile), line2 a 2 cotes (nul, extérieur)
            if len(odds1) >= 1 and len(odds2) >= 2:
                try:
                    odd_home = float(odds1[0].replace(',', '.'))
                    odd_draw = float(odds2[0].replace(',', '.'))
                    odd_away = float(odds2[1].replace(',', '.'))
                    
                    # Valider la plage des cotes
                    if not (1.0 <= odd_home <= 50.0 and 1.0 <= odd_draw <= 50.0 and 1.0 <= odd_away <= 50.0):
                        i += 1
                        continue
                    
                    # Extraire les noms d'équipes en retirant les cotes
                    home_team = odds_pattern.sub('', line1).strip()
                    away_team = odds_pattern.sub('', line2).strip()
                    
                    # Nettoyer les noms
                    home_team = re.sub(r'\s+', ' ', home_team).strip()
                    away_team = re.sub(r'\s+', ' ', away_team).strip()
                    
                    # Valider les noms d'équipes
                    if not home_team or not away_team:
                        i += 1
                        continue
                    if home_team.isdigit() or away_team.isdigit():
                        i += 1
                        continue
                    if not any(c.isalpha() for c in home_team) or not any(c.isalpha() for c in away_team):
                        i += 1
                        continue
                    
                    # Exclure les mots-clés non-match
                    if EXCLUDE_KEYWORDS.search(home_team + ' ' + away_team):
                        i += 1
                        continue
                    
                    # Vérifier que les noms ne sont pas identiques (doublon)
                    if home_team.lower() == away_team.lower():
                        i += 1
                        continue
                    
                    # Éviter les doublons de matchs déjà extraits
                    already_added = any(
                        m['home_team'].lower() == home_team.lower() and
                        m['away_team'].lower() == away_team.lower()
                        for m in matches
                    )
                    if already_added:
                        i += 2
                        continue
                    
                    matchday = detected_matchday if detected_matchday > 0 else fallback_matchday
                    
                    matches.append({
                        'matchday': matchday,
                        'line_position': len(matches) + 1,
                        'home_team': home_team,
                        'away_team': away_team,
                        'odd_home': odd_home,
                        'odd_draw': odd_draw,
                        'odd_away': odd_away,
                        'is_upcoming': True,
                        'has_odds': True
                    })
                    
                    logger.info(f"Extracted match {len(matches)}: {home_team} vs {away_team} [{odd_home}/{odd_draw}/{odd_away}]")
                    
                    # Avancer de 2 lignes (on a consommé line1 + line2)
                    i += 2
                    
                except Exception as e:
                    logger.debug(f"Error parsing match at line {i}: {e}")
                    i += 1
            else:
                i += 1
        
        if matches:
            logger.info(f"Extracted {len(matches)}/{self.MATCHES_PER_MATCHDAY} matches with odds from text")
        else:
            logger.warning("No matches extracted from text")
        
        return matches
    
    async def _extract_matches_fallback(self, page_text: str, detected_matchday: int, fallback_matchday: int) -> List[Dict]:
        """Fallback extraction for different odds format."""
        matches = []
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]
        
        # Pattern: 3 odds on same line
        odds_pattern = re.compile(r'(\d+[,.]\d+)\s+(\d+[,.]\d+)\s+(\d+[,.]\d+)')
        
        i = 0
        while i < len(lines) and len(matches) < 15:
            line = lines[i]
            
            odds_match = odds_pattern.search(line)
            if odds_match:
                odd_home = float(odds_match.group(1).replace(',', '.'))
                odd_draw = float(odds_match.group(2).replace(',', '.'))
                odd_away = float(odds_match.group(3).replace(',', '.'))
                
                if not (1.0 <= odd_home <= 50.0 and 1.0 <= odd_draw <= 50.0 and 1.0 <= odd_away <= 50.0):
                    i += 1
                    continue
                
                # Look backwards for team names
                home_team = None
                away_team = None
                
                for j in range(i - 1, max(i - 6, 0), -1):
                    prev_line = lines[j]
                    if odds_pattern.search(prev_line):
                        continue
                    if prev_line.isdigit():
                        continue
                    if re.search(r'\b(journ|jornada|matchday|live|result|score)\b', prev_line, re.I):
                        continue
                    
                    vs_match = re.search(r'(.+)\s+(?:vs|VS|v\.?)\s+(.+)', prev_line)
                    if vs_match:
                        home_team = vs_match.group(1).strip()
                        away_team = vs_match.group(2).strip()
                        break
                    elif any(c.isalpha() for c in prev_line):
                        if not away_team:
                            away_team = prev_line
                        else:
                            home_team = prev_line
                            break
                
                if home_team and away_team and any(c.isalpha() for c in home_team) and any(c.isalpha() for c in away_team):
                    matchday = detected_matchday if detected_matchday > 0 else fallback_matchday
                    matches.append({
                        'matchday': matchday,
                        'line_position': len(matches) + 1,
                        'home_team': home_team,
                        'away_team': away_team,
                        'odd_home': odd_home,
                        'odd_draw': odd_draw,
                        'odd_away': odd_away,
                        'is_upcoming': True,
                        'has_odds': True
                    })
            
            i += 1
        
        return matches
    
    async def _get_countdown(self) -> Optional[int]:
        """Get countdown timer in seconds from LIVE match.
        
        Virtual match: 90 virtual minutes = 50 real seconds.
        Each virtual minute = 50/90 ≈ 0.56 real seconds.
        """
        try:
            page_text = await self.page.inner_text('body')
            
            # Look for elapsed time with apostrophe (e.g., "45'", "90'")
            elapsed_match = re.search(r"(\d{1,2})'", page_text)
            if elapsed_match:
                elapsed_mins = int(elapsed_match.group(1))
                if elapsed_mins <= 90:
                    # Calculate remaining real seconds
                    remaining_virtual = 90 - elapsed_mins
                    remaining_real = int(remaining_virtual * (50 / 90))
                    logger.info(f"Found elapsed time: {elapsed_mins}' ({remaining_real}s remaining)")
                    return remaining_real
            
            # Look for halftime indicator
            if re.search(r"\bMT\b|\bHT\b|\bMI-TEMPS\b", page_text, re.I):
                logger.info("Match at halftime - 5s break")
                return 5  # Short halftime for virtual matches
            
        except Exception as e:
            logger.debug(f"Error getting countdown: {e}")
        
        return None

    async def _is_live_phase(self) -> bool:
        """Return True if the matches page is showing a LIVE match in progress.
        
        IMPORTANT: We must distinguish between:
        - LIVE match: has timer (e.g., "45'"), scores, "EN DIRECT" badge
        - UPCOMING matches: has odds (e.g., "2.50 3.20 2.80"), no timer, no scores
        
        We check for UPCOMING indicators FIRST - if found, it's NOT live.
        """
        try:
            if not self.page:
                return False

            # Get page text for analysis
            body_text = await self.page.inner_text('body')
            
            # FIRST: Check for UPCOMING indicators (odds pattern)
            # Find all decimal numbers that could be odds
            odds_pattern = re.compile(r'\b\d+[\.,]\d{1,2}\b')
            odds_matches = odds_pattern.findall(body_text)
            
            valid_odds_count = 0
            for odd_str in odds_matches:
                try:
                    val = float(odd_str.replace(',', '.'))
                    if 1.0 <= val <= 50.0:
                        valid_odds_count += 1
                except:
                    pass
            
            if valid_odds_count >= 10:  # At least 10 valid odds found
                logger.info(f"UPCOMING phase detected via odds count ({valid_odds_count}) - NOT live")
                return False
            
            # SECOND: Check for LIVE indicators
            # Look for actual match timer with apostrophe (e.g., "45'", "90'")
            if re.search(r"\d{1,2}'", body_text):  # Timer like "45'" with apostrophe
                logger.info("LIVE phase detected via timer pattern (minutes with ')")
                return True
            
            # Look for halftime indicator
            if re.search(r"\bMT\b|\bHT\b|\bMI-TEMPS\b", body_text, re.I):
                logger.info("LIVE phase detected via halftime indicator")
                return True
            
            # Look for "EN DIRECT" badge that's actually visible and prominent
            # (not just in navigation or footer)
            live_el = await self.page.query_selector('[class*="live-badge"], [class*="live-indicator"], .live')
            if live_el and await live_el.is_visible():
                txt = (await live_el.inner_text()).strip()
                # Must be just "LIVE" or "EN DIRECT", not a longer text
                if txt.upper() in ['LIVE', 'EN DIRECT', 'DIRECT']:
                    logger.info(f"LIVE phase detected via badge: {txt}")
                    return True
            
            # Check for actual match score with timer (format: "Team1 2 - 1 Team2 45'")
            # This indicates a live match
            if re.search(r'\d\s*-\s*\d.*\d{1,2}\'', body_text):
                logger.info("LIVE phase detected via score+timer pattern")
                return True
            
            # Default: NOT live
            logger.debug("No LIVE indicators found - assuming UPCOMING")
            return False
            
        except Exception as e:
            logger.debug(f"Error in _is_live_phase: {e}")
            return False
    
    def _get_result(self, score_home: int, score_away: int) -> str:
        """Get result code: V (home win), N (draw), D (away win)."""
        if score_home > score_away:
            return 'V'
        elif score_home < score_away:
            return 'D'
        else:
            return 'N'
    
    async def detect_current_matchday(self) -> int:
        """Detect current matchday from site (results or matches page)."""
        import re
        try:
            # Check results page first
            if not await self._safe_goto(self.RESULTS_URL):
                logger.warning("Could not reach results page for detection")
            else:
                # Wait for content to load
                await asyncio.sleep(2)
                
                # Get page text and look for "Journée X" pattern
                text = await self.page.inner_text('body')
                
                # Look for "Journée X" pattern - take the FIRST one (most recent on page)
                matches = re.findall(r'Journ[eé]e\s*(\d{1,2})', text, re.IGNORECASE)
                if matches:
                    # Filter for valid range
                    md_list = [int(m) for m in matches if 1 <= int(m) <= 38]
                    if md_list:
                        # Take the FIRST matchday found (most recent on results page)
                        # Results page shows most recent first
                        matchday = md_list[0]
                        logger.info(f"Detected current matchday J{matchday} from results page (found: {sorted(set(md_list))})")
                        return matchday
                
                # If no "Journée" found, try to extract from results structure
                results = await self._extract_results()
                if results:
                    # Get the max matchday from extracted results
                    max_md = max(r.get('matchday', 0) for r in results)
                    if max_md > 0:
                        logger.info(f"Detected current matchday J{max_md} from extracted results")
                        return max_md
            
            # Fallback to _detect_matchday_from_page
            matchday = await self._detect_matchday_from_page()
            if matchday > 0:
                return matchday

            # Enforce strict order at startup: do not navigate to MATCHES page
            # until orchestrator explicitly enables it.
            if not self.matches_enabled:
                return 0
            
            # Fallback: check matches page
            if not await self._safe_goto(self.MATCHES_URL):
                logger.warning("Could not reach matches page for detection")
            else:
                text = await self.page.inner_text('body')
                matches = re.findall(r'Journ[eé]e\s*(\d{1,2})', text, re.IGNORECASE)
                if matches:
                    matchday = int(matches[0])
                    if 1 <= matchday <= 38:
                        logger.info(f"Detected current matchday J{matchday} from matches page")
                        return matchday
            
            return 0
            
        except Exception as e:
            logger.error(f"Error detecting current matchday: {e}")
            return 0

    async def get_latest_results_snapshot(self) -> List[Dict]:
        """Return the latest 10 results from the results page (current matchday only).

        Used for season verification at startup.
        """
        try:
            if not await self._safe_goto(self.RESULTS_URL):
                return []

            detected_matchday = await self._detect_matchday_from_page()
            if detected_matchday == 0:
                # Try a small load-more and re-detect
                await self._load_more_results(max_clicks=1)
                detected_matchday = await self._detect_matchday_from_page()

            # Extract only the latest matchday results (10 matches max)
            return await self._extract_results(detected_matchday if detected_matchday > 0 else 1)
        except Exception as e:
            logger.error(f"Error getting latest results snapshot: {e}")
            return []
    
    async def backfill_all_results(self, fast_mode=True):
        """Backfill all results from site for new season."""
        try:
            if not await self._safe_goto(self.RESULTS_URL):
                logger.error("Backfill failed: navigation error")
                return
            
            # Click "Afficher plus" until no more button (load ALL results)
            click_count = 0
            
            while True:
                # Use SPAN selector for "Afficher plus" button
                button = await self.page.query_selector('span:has-text("Afficher plus")')
                if button:
                    is_visible = await button.is_visible()
                    if is_visible:
                        await button.click()
                        await asyncio.sleep(2)
                        click_count += 1
                        logger.info(f"Clicked Afficher plus ({click_count})")
                        continue
                # No more button - we have all results
                break
            
            logger.info(f"Total clicks: {click_count}")
            
            # Detect current matchday from page
            detected_matchday = await self._detect_matchday_from_page()
            if detected_matchday == 0:
                detected_matchday = await self._count_result_groups_on_page()
                detected_matchday = max(1, int(detected_matchday))
            
            # Extract results using chunking (reliable even without matchday headers).
            # We cannot necessarily load a full 38 matchdays due to site pagination;
            # use a safe high value and let chunking cap based on available items.
            results = await self._extract_results_by_chunking(detected_matchday, max_days=60)
            logger.info(f"Extracted {len(results)} results from page")
            
            if results and self.on_results:
                # Use fast callback for backfill if available
                callback = self.on_results_fast if hasattr(self, 'on_results_fast') and fast_mode else self.on_results
                
                # Process in batches by matchday
                matchdays = {}
                for r in results:
                    md = r.get('matchday', 1)
                    # Ensure md is an integer
                    if isinstance(md, list):
                        md = md[0] if md else 1
                    try:
                        md = int(md) if md else 1
                    except (ValueError, TypeError):
                        md = 1
                    # Also fix in the result dict
                    r['matchday'] = md
                    if md not in matchdays:
                        matchdays[md] = []
                    matchdays[md].append(r)
                
                total_processed = 0
                for md in sorted(matchdays.keys()):
                    logger.info(f"Processing J{md}: {len(matchdays[md])} matches")
                    for r in matchdays[md]:
                        await callback(r)
                        total_processed += 1
                
                logger.info(f"Backfilled {total_processed} results")
            
        except Exception as e:
            import traceback
            logger.error(f"Error backfilling results: {e}")
            traceback.print_exc()
    
    async def close(self):
        """Close browser."""
        self.is_running = False
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Bet261 scraper closed")
