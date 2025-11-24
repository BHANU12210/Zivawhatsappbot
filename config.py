from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = None
    TWILIO_ACCOUNT_SID: str = None
    TWILIO_AUTH_TOKEN: str = None
    TWILIO_PHONE_NUMBER: str = None
    PORT: int = 3000

    class Config:
        env_file = ".env"

settings = Settings()
