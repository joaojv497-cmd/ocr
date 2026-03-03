import logging
from openai import OpenAI
from ocr_pypi.chunking.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é um assistente especializado em análise de documentos jurídicos brasileiros. "
    "Sempre responda EXCLUSIVAMENTE com JSON válido, sem texto adicional."
)


class OpenAIProvider(LLMProvider):
    """Provedor LLM usando OpenAI (GPT-4o, GPT-4o-mini, etc.)"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 8192
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(self, prompt: str) -> str:
        logger.info(f"[OpenAI] Enviando para {self.model}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content
        logger.info(f"[OpenAI] Resposta recebida ({len(result)} chars)")
        return result