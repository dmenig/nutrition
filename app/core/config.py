from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "your-secret-key"  # Replace with a strong secret key
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ADMIN_API_KEY: str = "your-admin-api-key"

    DATABASE_URL: str = "postgresql://neondb_owner:npg_mqfC7hVBTw0D@ep-nameless-morning-a2z4vx7f-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require"

    class Config:
        env_file = [".env", ".env.local"]
        env_prefix = ""
        extra = "ignore"


settings = Settings()
