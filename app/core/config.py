from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key"  # Replace with a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str = "postgresql://neondb_owner:npg_mqfC7hVBTw0D@ep-nameless-morning-a2z4vx7f-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    ADMIN_API_KEY: str = "your-admin-api-key"

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
