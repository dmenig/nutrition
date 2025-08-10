import uuid
from sqlalchemy import Column, String, Float, Integer, ForeignKey, TIMESTAMP, Date, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


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
    logged_date = Column(Date, nullable=True)

    user = relationship("User")


class SportActivity(Base):
    __tablename__ = "sport_activities"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    activity_name = Column(String, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    carried_weight_kg = Column(Float)
    distance_m = Column(Float)
    calories_expended = Column(Float)
    logged_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    logged_date = Column(Date, nullable=True)

    user = relationship("User")


class CustomFood(Base):
    __tablename__ = "custom_foods"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    food_name = Column(String, nullable=False)
    calories_per_100g = Column(Float, nullable=False)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fat_per_100g = Column(Float)

    user = relationship("User")


class Food(Base):
    __tablename__ = "foods"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True, nullable=False)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)


class DailySummary(Base):
    __tablename__ = "daily_summaries"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    date = Column(Date, nullable=False, index=True)
    calories_total = Column(Float, nullable=False, default=0.0)
    protein_g_total = Column(Float, nullable=False, default=0.0)
    carbs_g_total = Column(Float, nullable=False, default=0.0)
    fat_g_total = Column(Float, nullable=False, default=0.0)
    sport_calories_total = Column(Float, nullable=False, default=0.0)

    user = relationship("User")

    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_daily_summaries_user_date"),
    )
