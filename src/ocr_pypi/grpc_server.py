import grpc
import json
import os
from typing import Any, Dict

import pytesseract
from Ocr import ocr_pb2_grpc, ocr_pb2
from Util import types_pb2
from commons_pypi.llm_providers.factory import LLMProviderFactory


from ocr_pypi.services.document_processor import DocumentProcessor
from ocr_pypi.storage import get_storage

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enum conversion maps
# ---------------------------------------------------------------------------

PROTO_TO_PROVIDER = {
    types_pb2.LLM_PROVIDER_OPENAI: "openai",
    types_pb2.LLM_PROVIDER_ANTHROPIC: "anthropic",
}

PROVIDER_TO_PROTO = {v: k for k, v in PROTO_TO_PROVIDER.items()}

PROTO_TO_LANGUAGE = {
    types_pb2.OCR_LANGUAGE_PORTUGUESE: "por",
    types_pb2.OCR_LANGUAGE_ENGLISH: "eng",
    types_pb2.OCR_LANGUAGE_SPANISH: "spa",
    types_pb2.OCR_LANGUAGE_FRENCH: "fra",
    types_pb2.OCR_LANGUAGE_GERMAN: "deu",
    types_pb2.OCR_LANGUAGE_ITALIAN: "ita",
}

LANGUAGE_TO_PROTO = {v: k for k, v in PROTO_TO_LANGUAGE.items()}

PROTO_TO_CHUNKING = {
    types_pb2.CHUNKING_METHOD_UNSPECIFIED: "page",
    types_pb2.CHUNKING_METHOD_SEMANTIC: "semantic",
    types_pb2.CHUNKING_METHOD_PARAGRAPH: "paragraph",
    types_pb2.CHUNKING_METHOD_HYBRID: "hybrid",
}

CHUNKING_TO_PROTO = {
    "page": types_pb2.CHUNKING_METHOD_UNSPECIFIED,
    "semantic": types_pb2.CHUNKING_METHOD_SEMANTIC,
    "paragraph": types_pb2.CHUNKING_METHOD_PARAGRAPH,
    "hybrid": types_pb2.CHUNKING_METHOD_HYBRID,
    "image_description": types_pb2.CHUNKING_METHOD_UNSPECIFIED,
}

tesseract_env = os.getenv("TESSERACT_CMD")
if tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = tesseract_env


class OCRGrpcServer(ocr_pb2_grpc.OCRServiceServicer):
    """Implementação do serviço gRPC de OCR"""

    def ProcessDocument(self, request, context):
        """Processa documento e retorna chunks via streaming"""

        language = PROTO_TO_LANGUAGE.get(request.language, "por")
        processor = DocumentProcessor(language=language)

        # Resolve chunking method
        chunking_method_proto = request.chunking_method
        chunk_strategy = PROTO_TO_CHUNKING.get(chunking_method_proto, "page")

        # Extract LLM config (API key ONLY from environment)
        llm_config = request.llm_config
        llm_api_key = os.environ.get("LLM_API_KEY") or None

        # Monta chunk_options a partir do request
        chunk_options = {
            "enable_chunking": True,
            "chunk_strategy": chunk_strategy,

            # LLM Provider (used for image description)
            "llm_provider": PROTO_TO_PROVIDER.get(llm_config.provider) if llm_config.provider != types_pb2.LLM_PROVIDER_UNSPECIFIED else None,
            "llm_model": llm_config.model or None,
            "llm_api_key": llm_api_key,
            "llm_temperature": llm_config.temperature if llm_config.temperature > 0 else None,
            "llm_max_tokens": llm_config.max_tokens if llm_config.max_tokens > 0 else None,
        }

        logger.info(
            f"ProcessDocument: strategy={chunk_strategy}, "
            f"provider={chunk_options['llm_provider'] or 'default'}, "
            f"model={chunk_options['llm_model'] or 'default'}"
        )

        template_used = ""

        for result in processor.process(request.bucket, request.file_key, chunk_options):
            if not context.is_active():
                return

            if result["type"] == "chunk":
                chunk = result["data"]

                yield ocr_pb2.ProcessDocumentResponse(
                    status=types_pb2.OCRStatus.PROCESSING,
                    chunk_index=chunk.chunk_index,
                    page_numbers=chunk.page_numbers,
                    text=chunk.content,
                    chunk_metadata=json.dumps(chunk.metadata, ensure_ascii=False),
                    chunking_method=CHUNKING_TO_PROTO.get(
                        chunk.metadata.get("chunking_method", chunk_strategy),
                        types_pb2.CHUNKING_METHOD_UNSPECIFIED,
                    ),
                    template_used=template_used,
                )

            elif result["type"] == "complete":
                yield ocr_pb2.ProcessDocumentResponse(
                    status=types_pb2.OCRStatus.COMPLETED,
                    stage=types_pb2.OCRStage.FINISHED,
                    total_chunks=result["total_chunks"],
                    chunking_method=CHUNKING_TO_PROTO.get(chunk_strategy, types_pb2.CHUNKING_METHOD_UNSPECIFIED),
                    template_used=template_used,
                )

            elif result["type"] == "error":
                yield ocr_pb2.ProcessDocumentResponse(
                    status=types_pb2.OCRStatus.FAILED,
                    error_message=result["message"],
                )

    def ListProviders(self, request, context):
        """Lista todos os LLM providers disponíveis"""
        providers = LLMProviderFactory.list_providers()
        return ocr_pb2.ListProvidersResponse(providers=providers)

    def ValidateDocument(self, request, context):
        """Valida se documento existe e é suportado"""
        storage = get_storage(request.bucket)

        exists = storage.file_exists(request.file_key)
        supported = request.file_key.lower().endswith(".pdf")

        return ocr_pb2.ValidateDocumentResponse(
            exists=exists,
            supported=supported,
            message="OK" if exists and supported else "Documento inválido",
        )

    def HealthCheck(self, request, context):
        """Health check do serviço"""
        return ocr_pb2.HealthCheckResponse(
            status=types_pb2.HealthStatus.HEALTH_SERVING,
            message="OCR Service running",
        )

    def _map_stage(self, stage: str) -> int:
        """Mapeia stage string para enum do proto"""
        stage_map = {
            # Stages existentes
            "validating": types_pb2.OCRStage.VALIDATING,
            "downloading": types_pb2.OCRStage.DOWNLOADING,
            "extracting": types_pb2.OCRStage.EXTRACTING,
            "chunking": types_pb2.OCRStage.CHUNKING,
            "finished": types_pb2.OCRStage.FINISHED,

            "download_complete": types_pb2.OCRStage.DOWNLOADING,
            "type_detection_complete": types_pb2.OCRStage.EXTRACTING,
            "page_extracted": types_pb2.OCRStage.EXTRACTING,
            "text_extraction_complete": types_pb2.OCRStage.EXTRACTING,
            "layout_analysis_complete": types_pb2.OCRStage.EXTRACTING,
            "noise_removal_complete": types_pb2.OCRStage.EXTRACTING,
            "section_classification_complete": types_pb2.OCRStage.EXTRACTING,
            "image_detection_complete": types_pb2.OCRStage.EXTRACTING,
            "llm_chunking_starting": types_pb2.OCRStage.CHUNKING,
            "semantic_chunking_starting": types_pb2.OCRStage.CHUNKING,
            "paragraph_chunking_starting": types_pb2.OCRStage.CHUNKING,
            "chunking_complete": types_pb2.OCRStage.CHUNKING,
        }
        return stage_map.get(stage.lower(), types_pb2.OCRStage.IDLE)