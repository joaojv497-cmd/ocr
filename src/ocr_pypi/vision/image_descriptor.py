"""Geração de descrições de imagens usando LLM com suporte a visão."""
import base64
import logging
from typing import List, Optional, Generator

from commons_pypi.llm_providers.base_provider import LLMProvider
from commons_pypi.llm_providers.factory import LLMProviderFactory

from ocr_pypi.config import settings
from ocr_pypi.models.document import ImageDescription, ImageInfo

logger = logging.getLogger(__name__)

JURIDIC_SYSTEM_PROMPT = (
    "Você é um assistente especializado em análise de documentos jurídicos brasileiros. "
    "Sempre responda EXCLUSIVAMENTE com JSON válido, sem texto adicional."
)

JURIDIC_VISION_PROMPT = (
    "Você é um especialista em análise de documentos jurídicos brasileiros. "
    "Analise detalhadamente esta imagem extraída de um documento PDF jurídico. "
    "Foque em: "
    "- Tipo de imagem (gráfico, tabela, foto, diagrama, assinatura, carimbo, etc.) "
    "- Conteúdo principal e informações jurídicas relevantes "
    "- Dados, números, valores ou datas visíveis "
    "- Texto presente na imagem (especialmente nomes, cargos, órgãos) "
    "- Elementos jurídicos específicos (carimbos de tribunal, assinaturas de autoridades, etc.) "
    "Forneça uma descrição objetiva e completa em português, priorizando aspectos juridicamente relevantes."
)


class ImageDescriptor:
    """Gera descrições de imagens usando LLM com suporte a visão."""

    def __init__(
        self,
        provider: LLMProvider = None,
        provider_name: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None,
        vision_system_prompt: str = None,
    ):
        """
        Inicializa o ImageDescriptor.

        Pode receber um provider já instanciado OU parâmetros para criar via factory.
        Se nada for informado, usa as configs do settings.

        Args:
            provider: Provider LLM já instanciado (opcional).
            provider_name: Nome do provider (ex: 'openai', 'anthropic').
            api_key: Chave de API do provider.
            model: Nome do modelo a usar.
            temperature: Temperatura para geração de texto.
            max_tokens: Número máximo de tokens na resposta.
            system_prompt: Prompt de sistema para chamadas de texto. Se não
                informado (ou ``None``), usa ``settings.LLM_SYSTEM_PROMPT``
                (variável de ambiente ``LLM_SYSTEM_PROMPT``) quando definida, ou
                o padrão jurídico ``JURIDIC_SYSTEM_PROMPT`` como fallback final.
            vision_system_prompt: Prompt de sistema para chamadas de visão. Se
                não informado (ou ``None``), usa ``settings.LLM_VISION_PROMPT``
                (variável de ambiente ``LLM_VISION_PROMPT``) quando definida, ou
                o padrão jurídico ``JURIDIC_VISION_PROMPT`` como fallback final.
        """
        resolved_system_prompt = (
            system_prompt
            or settings.LLM_SYSTEM_PROMPT
            or JURIDIC_SYSTEM_PROMPT
        )
        resolved_vision_prompt = (
            vision_system_prompt
            or settings.LLM_VISION_PROMPT
            or JURIDIC_VISION_PROMPT
        )

        if provider:
            self.provider = provider
        else:
            self.provider = LLMProviderFactory.create(
                provider_name=provider_name or settings.LLM_PROVIDER,
                api_key=api_key or settings.LLM_API_KEY,
                model=model or settings.LLM_MODEL,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                system_prompt=resolved_system_prompt,
                vision_system_prompt=resolved_vision_prompt,
            )

        self._vision_prompt = resolved_vision_prompt
        logger.info(f"ImageDescriptor inicializado com {self.provider}")

    def describe_images(
        self, images: List[ImageInfo]
    ) -> List[ImageDescription]:
        """
        Gera descrições para uma lista de imagens.

        Args:
            images: Lista de ImageInfo com dados das imagens.

        Returns:
            Lista de ImageDescription com descrições geradas.
        """
        return list(self.describe_images_iter(images))

    def describe_images_iter(
        self, images: List[ImageInfo]
    ) -> Generator[ImageDescription, None, None]:
        """
        Gera descrições para uma lista de imagens, uma por vez.

        Args:
            images: Lista de ImageInfo com dados das imagens.

        Yields:
            ImageDescription para cada imagem processada com sucesso.
        """
        for image_info in images:
            description = self._describe_single(image_info)
            if description is not None:
                yield description

    def _describe_single(self, image_info: ImageInfo) -> Optional[ImageDescription]:
        """Gera descrição para uma única imagem."""
        try:
            description_text = self._call_vision_llm(image_info)
            return ImageDescription(
                image_info=image_info,
                description=description_text,
                page_number=image_info.page_number,
                metadata={
                    "image_index": image_info.image_index,
                    "width": image_info.width,
                    "height": image_info.height,
                    "format": image_info.format,
                },
            )
        except Exception as e:
            logger.warning(
                f"Erro ao descrever imagem {image_info.image_index} "
                f"da página {image_info.page_number}: {e}"
            )
            return None

    def _call_vision_llm(self, image_info: ImageInfo) -> str:
        """
        Chama a LLM com a imagem em base64.

        Tenta usar o método `generate_with_image` se disponível no provider;
        caso contrário, tenta `generate` com prompt de texto apenas como fallback.
        """
        image_b64 = base64.b64encode(image_info.image_data).decode("utf-8")

        # Try vision-capable method first
        if hasattr(self.provider, "generate_with_image"):
            return self.provider.generate_with_image(
                prompt=self._vision_prompt,
                image_base64=image_b64,
                image_mime_type="image/png",
            )

        # Fallback: describe without actual image (text-only context)
        logger.debug(
            "Provider não suporta visão; usando fallback de descrição por posição."
        )
        fallback_prompt = (
            f"Uma imagem foi encontrada na página {image_info.page_number} "
            f"do documento (largura={image_info.width}px, altura={image_info.height}px). "
            "Não é possível processar a imagem diretamente. "
            "Registre que há uma imagem nesta posição do documento."
        )
        return self.provider.generate(fallback_prompt)
