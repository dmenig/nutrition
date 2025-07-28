import uuid
from sqlalchemy import Column, String, Float, Integer, ForeignKey, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)


class FoodLog(Base):
    __tablename__ = "food_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    food_name = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    calories = Column(Float, nullable=False)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)
    logged_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )

    user = relationship("User", back_populates="food_logs")


class SportActivity(Base):
    __tablename__ = "sport_activities"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    activity_name = Column(String, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    calories_expended = Column(Float)
    logged_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )

    user = relationship("User", back_populates="sport_activities")


class CustomFood(Base):
    __tablename__ = "custom_foods"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    food_name = Column(String, nullable=False)
    calories_per_100g = Column(Float, nullable=False)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fat_per_100g = Column(Float)

    user = relationship("User", back_populates="custom_foods")
