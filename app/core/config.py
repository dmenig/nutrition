from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key"  # Replace with a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    ADMIN_API_KEY: str = "your-admin-api-key"

    class Config:
        env_file = [".env", ".env.local"]
        env_prefix = ""
        extra = "ignore"


settings = Settings()
