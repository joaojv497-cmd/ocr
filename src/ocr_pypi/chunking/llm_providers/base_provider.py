from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Interface abstrata para provedores de LLM"""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 8192
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Envia prompt para a LLM e retorna a resposta como string.
        A resposta deve ser JSON válido.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome do provedor (openai, gemini, anthropic)"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"