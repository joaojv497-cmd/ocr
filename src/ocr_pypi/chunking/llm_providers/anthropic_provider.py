import logging
from anthropic import Anthropic
from ocr_pypi.chunking.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é um assistente especializado em análise de documentos jurídicos brasileiros. "
    "Sempre responda EXCLUSIVAMENTE com JSON válido, sem texto adicional."
)


class AnthropicProvider(LLMProvider):
    """Provedor LLM usando Anthropic Claude"""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 8192
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(self, prompt: str) -> str:
        logger.info(f"[Anthropic] Enviando para {self.model}...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        result = response.content[0].text
        logger.info(f"[Anthropic] Resposta recebida ({len(result)} chars)")
        return result