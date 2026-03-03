import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Configurações da aplicação carregadas de variáveis de ambiente"""

    # Storage R2/S3
    R2_ENDPOINT: str = os.getenv("R2_ENDPOINT")
    R2_ACCESS_KEY: str = os.getenv("R2_ACCESS_KEY")
    R2_SECRET_KEY: str = os.getenv("R2_SECRET_KEY")
    R2_REGION: str = os.getenv("R2_REGION", "auto")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50051"))

    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    DEFAULT_TEMPLATE: str = os.getenv("DEFAULT_TEMPLATE", "generico")

settings = Settings()