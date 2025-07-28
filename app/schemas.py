from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    hashed_password: str

    class Config:
        from_attributes = True


class UserOut(UserBase):
    id: int

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str | None = None


from uuid import UUID


class CustomFoodBase(BaseModel):
    name: str
    calories: float
    protein: float
    carbohydrates: float
    fat: float


class CustomFoodCreate(CustomFoodBase):
    pass


class CustomFoodUpdate(CustomFoodBase):
    pass


class CustomFoodOut(CustomFoodBase):
    id: UUID
    owner_id: int

    class Config:
        from_attributes = True


from datetime import datetime


class SportActivityBase(BaseModel):
    name: str
    date: datetime
    duration_minutes: int
    calories_burned: float


class SportActivityCreate(SportActivityBase):
    pass


class SportActivityUpdate(SportActivityBase):
    pass


class SportActivityOut(SportActivityBase):
    id: UUID
    owner_id: int

    class Config:
        from_attributes = True


class FoodLogBase(BaseModel):
    food_item: str
    quantity: float
    unit: str
    log_date: datetime


class FoodLogCreate(FoodLogBase):
    pass


class FoodLogUpdate(FoodLogBase):
    pass


class FoodLogOut(FoodLogBase):
    id: UUID
    owner_id: int

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
