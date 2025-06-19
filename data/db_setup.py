import os

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
)
from sqlalchemy.orm import sessionmaker

from models.base import Base

# Load environment variables
load_dotenv()

# Setup SQLAlchemy
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True)  # echo=True logs SQL queries


# Create extension + table + add vector column if missing
def setup_database():
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create table if it doesn't exist
        Base.metadata.create_all(bind=engine)

        # Check if 'description_embedding' column exists
        result = conn.execute(
            text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'hotels' AND column_name = 'description_embedding';
        """)
        )

        if not result.first():
            print("[INFO] 'description_embedding' column does not exist. Creating...")
            conn.execute(
                text("""
                ALTER TABLE hotels
                ADD COLUMN description_embedding vector(384);
            """)
            )
        else:
            print("[INFO] 'description_embedding' column already exists.")

        conn.commit()


def reset_hotels_table():
    with engine.connect() as conn:
        # Drop the hotels table if it exists
        conn.execute(text("DROP TABLE IF EXISTS hotels CASCADE;"))
        conn.commit()
        print("[INFO] Dropped existing 'hotels' table.")


# Initialize DB
if __name__ == "__main__":
    setup_database()

    # Optional: test session creation
    Session = sessionmaker(bind=engine)
    session = Session()
    print("[INFO] Database setup complete.")
