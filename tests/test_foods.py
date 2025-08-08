import os

# Set the database URL to use SQLite in-memory before importing the app
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.db.models import Food
from app.schemas import FoodOut

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_autocomplete_foods():
    db = TestingSessionLocal()

    # Add some test data
    food1 = Food(name="Apple", calories=52, protein=0.3, carbs=14.0, fat=0.2)
    food2 = Food(name="Banana", calories=89, protein=1.1, carbs=23.0, fat=0.3)
    food3 = Food(name="Orange", calories=47, protein=0.9, carbs=12.0, fat=0.1)
    food4 = Food(name="Pineapple", calories=50, protein=0.5, carbs=13.0, fat=0.1)
    food5 = Food(name="Applesauce", calories=42, protein=0.2, carbs=11.0, fat=0.1)
    food6 = Food(name="Avocado", calories=160, protein=2.0, carbs=9.0, fat=15.0)

    db.add_all([food1, food2, food3, food4, food5, food6])
    db.commit()
    db.refresh(food1)
    db.refresh(food2)
    db.refresh(food3)
    db.refresh(food4)
    db.refresh(food5)
    db.refresh(food6)

    # Test with a partial query
    response = client.get("/api/v1/foods/autocomplete?q=app")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert sorted([f["name"] for f in data]) == sorted(["Apple", "Applesauce"])

    # Test with a different partial query
    response = client.get("/api/v1/foods/autocomplete?q=ora")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Orange"

    # Test with no matching results
    response = client.get("/api/v1/foods/autocomplete?q=xyz")
    assert response.status_code == 200
    assert len(response.json()) == 0

    # Test with query less than min_length (now min_length is 1)
    response = client.get("/api/v1/foods/autocomplete?q=")
    assert response.status_code == 422  # Unprocessable Entity

    # Test with query of min_length
    response = client.get("/api/v1/foods/autocomplete?q=a")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert sorted([f["name"] for f in data]) == sorted(
        ["Apple", "Applesauce", "Avocado"]
    )

    db.close()
