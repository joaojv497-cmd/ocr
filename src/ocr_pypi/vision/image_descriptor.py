"""Geração de descrições de imagens usando LLM com suporte a visão."""
import base64
import io
import logging
import time
from typing import List, Optional, Generator

from PIL import Image

from commons_pypi.llm_providers.base_provider import LLMProvider
from commons_pypi.llm_providers.factory import LLMProviderFactory

from ocr_pypi.config import settings
from ocr_pypi.models.document import ImageDescription, ImageInfo
from ocr_pypi.preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Maximum number of retries for transient vision API errors
_MAX_RETRIES = 1
_RETRY_DELAY = 1.0


class NoVisionSupportError(Exception):
    """Raised when the configured LLM provider does not support image analysis."""


JURIDIC_SYSTEM_PROMPT = (
    "Você é um assistente especializado em análise de documentos. "
    "Forneça análises claras e objetivas."
)

JURIDIC_VISION_PROMPT = (
    "Analise esta imagem e descreva objetivamente o que você vê. "
    "Foque em: "
    "- Tipo de conteúdo (texto, gráfico, tabela, foto, diagrama, etc.) "
    "- Elementos visuais principais "
    "- Texto presente na imagem "
    "- Estrutura e layout "
    "Forneça uma descrição clara e direta em português."
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
        vision_max_width: int = None,
        vision_max_height: int = None,
        vision_jpeg_quality: int = None,
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
            vision_max_width: Maximum width in pixels for vision preprocessing.
                Defaults to ``settings.VISION_MAX_WIDTH``.
            vision_max_height: Maximum height in pixels for vision preprocessing.
                Defaults to ``settings.VISION_MAX_HEIGHT``.
            vision_jpeg_quality: JPEG quality (1–95) for vision preprocessing.
                Values above 95 disable JPEG compression optimisations and
                produce disproportionately large files.
                Defaults to ``settings.VISION_JPEG_QUALITY``.
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
        self._preprocessor = ImagePreprocessor()
        self._vision_max_width = vision_max_width if vision_max_width is not None else settings.VISION_MAX_WIDTH
        self._vision_max_height = vision_max_height if vision_max_height is not None else settings.VISION_MAX_HEIGHT
        self._vision_jpeg_quality = vision_jpeg_quality if vision_jpeg_quality is not None else settings.VISION_JPEG_QUALITY
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
                success=True,
                metadata={
                    "image_index": image_info.image_index,
                    "width": image_info.width,
                    "height": image_info.height,
                    "format": image_info.format,
                    "processing_status": "success",
                },
            )
        except NoVisionSupportError:
            logger.warning(
                f"Provider não suporta visão para imagem {image_info.image_index} "
                f"da página {image_info.page_number}."
            )
            return ImageDescription(
                image_info=image_info,
                description=(
                    f"❌ FALHA NO PROCESSAMENTO - Provider não suporta análise de imagens. "
                    f"Imagem detectada mas não analisada "
                    f"({image_info.width}x{image_info.height}px)."
                ),
                page_number=image_info.page_number,
                success=False,
                error_type="no_vision_support",
                metadata={
                    "image_index": image_info.image_index,
                    "width": image_info.width,
                    "height": image_info.height,
                    "format": image_info.format,
                    "processing_status": "failed",
                },
            )
        except Exception as e:
            logger.warning(
                f"Erro ao descrever imagem {image_info.image_index} "
                f"da página {image_info.page_number}: {e}"
            )
            return ImageDescription(
                image_info=image_info,
                description=(
                    f"❌ FALHA NO PROCESSAMENTO - Erro ao processar imagem "
                    f"({image_info.width}x{image_info.height}px)."
                ),
                page_number=image_info.page_number,
                success=False,
                error_type="processing_error",
                metadata={
                    "image_index": image_info.image_index,
                    "width": image_info.width,
                    "height": image_info.height,
                    "format": image_info.format,
                    "processing_status": "failed",
                },
            )

    def _call_vision_llm(self, image_info: ImageInfo) -> str:
        """
        Chama a LLM com a imagem em base64.

        Preprocessa a imagem (redimensionamento + compressão JPEG) antes do envio
        para reduzir custos de token.  Usa o método ``generate_with_image`` se
        disponível no provider; caso contrário, levanta ``NoVisionSupportError``.
        Erros transientes são retentados até ``_MAX_RETRIES`` vezes antes de
        propagar a exceção.
        """
        if not hasattr(self.provider, "generate_with_image"):
            logger.warning(
                "Provider não suporta visão; não é possível analisar a imagem."
            )
            raise NoVisionSupportError(
                f"Provider '{self.provider}' does not support image analysis."
            )

        try:
            pil_img = Image.open(io.BytesIO(image_info.image_data))
            jpeg_bytes = self._preprocessor.preprocess_for_vision(
                pil_img,
                max_width=self._vision_max_width,
                max_height=self._vision_max_height,
                jpeg_quality=self._vision_jpeg_quality,
            )
        except Exception as e:
            logger.warning(
                f"Falha ao preprocessar imagem {image_info.image_index} "
                f"para visão: {e}. Usando dados originais."
            )
            jpeg_bytes = image_info.image_data

        image_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self.provider.generate_with_image(
                    prompt=self._vision_prompt,
                    image_base64=image_b64,
                    image_mime_type="image/jpeg",
                )
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.debug(
                        f"Tentativa {attempt + 1} de visão falhou "
                        f"(imagem {image_info.image_index}, "
                        f"página {image_info.page_number}): {exc}. Retentando..."
                    )
                    time.sleep(_RETRY_DELAY)
                else:
                    logger.debug(
                        f"Todas as tentativas falharam para imagem "
                        f"{image_info.image_index}, página {image_info.page_number}."
                    )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Nenhuma tentativa de visão foi executada.")
