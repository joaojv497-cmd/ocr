import logging
from typing import Dict, Type, Optional
from ocr_pypi.chunking.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:

    _registry: Dict[str, Type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Registra um novo provedor de LLM"""
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(
                f"{provider_class.__name__} deve herdar de LLMProvider"
            )
        cls._registry[name.lower()] = provider_class
        logger.info(f"LLM Provider registrado: {name}")

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192
    ) -> LLMProvider:
        """
        Cria uma instância do provedor de LLM.

        Args:
            provider_name: Nome do provedor (openai, gemini, anthropic)
            api_key: Chave de API
            model: Nome do modelo (usa default do provider se não informado)
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta

        Returns:
            Instância de LLMProvider

        Raises:
            ValueError: Se o provedor não estiver registrado
        """
        cls._ensure_defaults_registered()

        name = provider_name.lower()
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Provedor '{provider_name}' não registrado. "
                f"Disponíveis: {available}"
            )

        provider_class = cls._registry[name]

        kwargs = {
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            kwargs["model"] = model

        provider = provider_class(**kwargs)
        logger.info(f"LLM Provider criado: {provider}")
        return provider

    @classmethod
    def list_providers(cls) -> list:
        """Lista todos os provedores registrados"""
        cls._ensure_defaults_registered()
        return list(cls._registry.keys())

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Registra os provedores padrão se ainda não foram registrados"""
        if cls._registry:
            return

        from ocr_pypi.chunking.llm_providers.openai_provider import OpenAIProvider
        from ocr_pypi.chunking.llm_providers.anthropic_provider import AnthropicProvider

        cls._registry["openai"] = OpenAIProvider
        cls._registry["anthropic"] = AnthropicProvider