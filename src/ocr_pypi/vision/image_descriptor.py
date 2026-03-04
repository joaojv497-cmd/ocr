"""Geração de descrições de imagens usando LLM com suporte a visão."""
import base64
import logging
from typing import List, Optional

from commons_pypi.llm_providers.base_provider import LLMProvider
from commons_pypi.llm_providers.factory import LLMProviderFactory

from ocr_pypi.config import settings
from ocr_pypi.models.document import ImageDescription, ImageInfo

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = """Você é um especialista em análise de documentos.
Descreva detalhadamente a imagem a seguir, que foi extraída de um documento PDF.
Foque em:
- Tipo de imagem (gráfico, tabela, foto, diagrama, assinatura, etc.)
- Conteúdo principal e informações relevantes
- Dados ou números visíveis
- Texto presente na imagem
Forneça uma descrição objetiva e completa em português."""


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
    ):
        """
        Inicializa o ImageDescriptor.

        Pode receber um provider já instanciado OU parâmetros para criar via factory.
        Se nada for informado, usa as configs do settings.
        """
        if provider:
            self.provider = provider
        else:
            self.provider = LLMProviderFactory.create(
                provider_name=provider_name or settings.LLM_PROVIDER,
                api_key=api_key or settings.LLM_API_KEY,
                model=model or settings.LLM_MODEL,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            )

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
        descriptions: List[ImageDescription] = []
        for image_info in images:
            description = self._describe_single(image_info)
            if description is not None:
                descriptions.append(description)
        return descriptions

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
                prompt=DESCRIPTION_PROMPT,
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
