import os
import fitz
import pytesseract
from PIL import Image
import io
from typing import Generator, List, Dict, Any

from commons_pypi.storage import get_temp_file
from ocr_pypi.chunking.llm_chunker import LLMChunker
from ocr_pypi.storage import get_storage
from ocr_pypi.config import settings

import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processa documentos PDF: OCR → LLM Chunking estruturado"""

    def __init__(self, language: str = "por"):
        self.language = language

    def process(
        self,
        bucket: str,
        file_key: str,
        chunk_options: Dict[str, Any] = None,
    ) -> Generator[Dict, None, None]:
        """
        Processa documento e retorna chunks via generator.

        chunk_options pode conter (todos opcionais, fallback para env):
            - template: nome do template
            - llm_provider: nome do provider
            - llm_model: nome do modelo
            - llm_api_key: chave da API
            - llm_temperature: temperatura
            - llm_max_tokens: max tokens

        Yields:
            Dict com 'type' ('progress', 'chunk', 'complete', 'error')
        """
        temp_path = None
        chunk_options = chunk_options or {}

        try:
            # 1. Download do arquivo
            storage = get_storage(bucket)
            temp_path = get_temp_file(".pdf")
            storage.download_file(file_key, temp_path)

            yield {
                "type": "progress",
                "stage": "download_complete",
            }

            # 2. Extração de texto (OCR)
            pages_data = list(self._extract_text(temp_path))

            yield {
                "type": "progress",
                "stage": "text_extraction_complete",
                "total_pages": len(pages_data),
            }

            # 3. Chunking via LLM (provider + template dinâmicos)
            yield from self._chunk_with_llm(pages_data, chunk_options)

        except Exception as e:
            logger.error(f"Erro ao processar documento: {e}")
            yield {
                "type": "error",
                "message": str(e),
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _chunk_with_llm(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """
        Realiza chunking estruturado via LLM.

        O provider e template vêm do request;
        se não vierem, o LLMChunker usa os defaults do settings.
        """

        # Filtra None para que o LLMChunker use os defaults internos
        provider_kwargs = {
            k: v for k, v in {
                "provider_name": chunk_options.get("llm_provider"),
                "api_key": chunk_options.get("llm_api_key"),
                "model": chunk_options.get("llm_model"),
                "temperature": chunk_options.get("llm_temperature"),
                "max_tokens": chunk_options.get("llm_max_tokens"),
            }.items() if v is not None
        }

        chunker = LLMChunker(**provider_kwargs)

        template_name = chunk_options.get("template") or settings.DEFAULT_TEMPLATE

        yield {
            "type": "progress",
            "stage": "llm_chunking_starting",
            "template": template_name,
            "provider": chunker.provider.provider_name,
            "model": chunker.provider.model,
        }

        chunks = chunker.chunk(
            pages=pages_data,
            template_name=template_name,
        )

        yield {
            "type": "progress",
            "stage": "chunking_complete",
            "total_chunks": len(chunks),
        }

        for chunk in chunks:
            yield {
                "type": "chunk",
                "data": chunk,
            }

        yield {
            "type": "complete",
            "total_chunks": len(chunks),
        }

    def _extract_text(self, pdf_path: str) -> Generator[Dict, None, None]:
        """
        Extrai texto do PDF.
        Tenta extração nativa primeiro; se vazio, faz OCR com Tesseract.
        """
        pdf = fitz.open(pdf_path)

        for i in range(len(pdf)):
            page = pdf.load_page(i)
            text = page.get_text()

            if not text.strip():
                pix = page.get_pixmap(dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(image, lang=self.language)

            yield {
                "page_number": i + 1,
                "text": text,
            }

        pdf.close()