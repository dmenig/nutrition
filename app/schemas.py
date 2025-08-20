from pydantic import BaseModel, EmailStr
from typing import Optional, Union
from uuid import UUID


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    hashed_password: str

    class Config:
        from_attributes = True


class UserOut(UserBase):
    id: UUID

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class CustomFoodBase(BaseModel):
    food_name: str
    calories_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float


class CustomFoodCreate(CustomFoodBase):
    pass


class CustomFoodUpdate(CustomFoodBase):
    pass


class CustomFoodOut(CustomFoodBase):
    id: UUID
    user_id: UUID

    class Config:
        from_attributes = True


from datetime import datetime


class SportActivityBase(BaseModel):
    activity_name: str
    logged_at: datetime
    duration_minutes: int
    carried_weight_kg: Optional[float] = None
    distance_m: Optional[float] = None


class SportActivityCreate(SportActivityBase):
    pass


class SportActivityUpdate(SportActivityBase):
    pass


class SportActivityOut(SportActivityBase):
    id: UUID
    user_id: UUID
    calories_expended: float | None = None

    class Config:
        from_attributes = True


class FoodLogBase(BaseModel):
    food_name: str
    quantity: float
    unit: Optional[str] = "g"
    logged_at: datetime
    calories: float
    protein: float
    carbs: float
    fat: float


class FoodLogCreate(FoodLogBase):
    pass


class FoodLogUpdate(FoodLogBase):
    pass


class FoodLogOut(FoodLogBase):
    id: UUID
    user_id: UUID

    class Config:
        from_attributes = True


class DailyFoodLogSummary(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class PlotPoint(BaseModel):
    time_index: int
    value: float


class WeightPlotResponse(BaseModel):
    W_obs: list[PlotPoint]
    W_adj_pred: list[PlotPoint]


class MetabolismPlotResponse(BaseModel):
    M_base: list[PlotPoint]


class EnergyBalancePlotResponse(BaseModel):
    calories_unnorm: list[PlotPoint]
    C_exp_t: list[PlotPoint]


from datetime import date as _date


class WeightLogBase(BaseModel):
    weight_kg: float
    logged_at: datetime


class WeightLogCreate(WeightLogBase):
    pass


class WeightLogOut(WeightLogBase):
    id: UUID
    user_id: UUID

    class Config:
        from_attributes = True


class FoodOut(BaseModel):
    id: UUID
    name: str
    calories: float
    protein: Optional[float]
    carbs: Optional[float]
    fat: Optional[float]
    # Optional additional nutrients (may not be used by all clients)
    sugar: Optional[float] = None
    sfat: Optional[float] = None
    free_sugar: Optional[float] = None
    fibres: Optional[float] = None
    sel: Optional[float] = None
    alcool: Optional[float] = None

    class Config:
        from_attributes = True


class FoodCreate(BaseModel):
    name: str
    calories: float
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None
    # Extra nutrients (per 100g) supported by the DB
    sugar: Optional[float] = None
    sfat: Optional[float] = None
    free_sugar: Optional[float] = None
    fibres: Optional[float] = None
    sel: Optional[float] = None
    alcool: Optional[float] = None
