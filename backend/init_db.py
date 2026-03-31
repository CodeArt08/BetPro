"""Initialize database with teams and initial season."""
from app.core.database import SessionLocal, init_db
from app.models import Team, Season
from app.core.config import settings

def seed_database():
    """Seed database with initial data."""
    init_db()
    
    db = SessionLocal()
    try:
        # Check if teams already exist
        if db.query(Team).count() == 0:
            print("Creating teams...")
            for team_name in settings.TEAMS:
                team = Team(name=team_name)
                db.add(team)
            db.commit()
            print(f"Created {len(settings.TEAMS)} teams")
        else:
            print(f"Teams already exist: {db.query(Team).count()} teams")
        
        # Check if season exists
        if db.query(Season).count() == 0:
            print("Creating initial season...")
            season = Season(
                season_number=1,
                is_active=True,
                is_completed=False
            )
            db.add(season)
            db.commit()
            print("Created season 1")
        else:
            print(f"Seasons exist: {db.query(Season).count()} seasons")
        
        print("\nDatabase initialized successfully!")
        print(f"Teams: {db.query(Team).count()}")
        print(f"Seasons: {db.query(Season).count()}")
        
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
