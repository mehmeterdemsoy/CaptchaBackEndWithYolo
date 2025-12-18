from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/yolov8n.pt"
    ENABLE_SPOOF_CHECKS: bool = True

settings = Settings()
