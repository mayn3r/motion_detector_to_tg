from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    bot_token: SecretStr = SecretStr("*")
    chat_id: int = 0
    proxy_url: str = ""
    
    cooldown: int = 10
    area_threshold: int = 500
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )