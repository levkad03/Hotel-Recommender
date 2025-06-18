import os

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables
load_dotenv()

# Setup SQLAlchemy
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True)  # echo=True logs SQL queries
Base = declarative_base()


# Define the hotels table
class Hotel(Base):
    __tablename__ = "hotels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    price_per_night = Column(Float, nullable=False)
    star_rating = Column(Float, nullable=False)
    review_score = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    description_embedding = Column(Vector(384))


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
