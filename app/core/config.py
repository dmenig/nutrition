from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key"  # Replace with a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str = "postgresql://user:password@localhost/db"
    ADMIN_API_KEY: str = "your-admin-api-key"

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
