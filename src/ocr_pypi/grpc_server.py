import json
import os
from Ocr import ocr_pb2_grpc, ocr_pb2
import Util.types_pb2 as types_pb2

from ocr_pypi.services.document_processor import DocumentProcessor
from ocr_pypi.storage import get_storage
from ocr_pypi.chunking.templates.registry import TemplateRegistry
from ocr_pypi.chunking.llm_providers.factory import LLMProviderFactory

import logging

logger = logging.getLogger(__name__)


class OCRGrpcServer(ocr_pb2_grpc.OCRServiceServicer):
    """Implementação do serviço gRPC de OCR"""

    def ProcessDocument(self, request, context):
        """Processa documento e retorna chunks via streaming"""

        processor = DocumentProcessor(language=request.language or "por")

        # API key priority: environment > request (security)
        api_key_from_env = os.environ.get("LLM_API_KEY", "")
        api_key_from_request = request.llm_api_key or None
        if api_key_from_request and api_key_from_env:
            logger.warning(
                "API key provided in request, but using environment configuration "
                "for authentication."
            )
        llm_api_key = api_key_from_env or api_key_from_request or None

        # Resolve dynamic template if template_definition is present in the request
        template_instance = None
        template_definition_raw = getattr(request, "template_definition", "") or ""
        if template_definition_raw:
            try:
                definition = json.loads(template_definition_raw)
                template_instance = TemplateRegistry.register_dynamic(definition)
                logger.info(f"Dynamic template registered: {template_instance.name}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid template_definition in request: {e}")

        # Monta chunk_options a partir do request
        chunk_options = {
            # Chunking strategy
            "chunk_strategy": request.chunk_strategy or None,
            "enable_chunking": request.enable_chunking if hasattr(request, "enable_chunking") else True,
            "chunk_size": request.chunk_size if request.chunk_size > 0 else None,
            "chunk_overlap": request.chunk_overlap if request.chunk_overlap > 0 else None,
            "min_chunk_size": request.min_chunk_size if request.min_chunk_size > 0 else None,

            # Template
            "template": request.template_name or None,
            "template_instance": template_instance,

            # LLM Provider (dinâmico por request)
            "llm_provider": request.llm_provider or None,
            "llm_model": request.llm_model or None,
            "llm_api_key": llm_api_key,
            "llm_temperature": request.llm_temperature if request.llm_temperature > 0 else None,
            "llm_max_tokens": request.llm_max_tokens if request.llm_max_tokens > 0 else None,
        }

        logger.info(
            f"ProcessDocument: strategy={chunk_options['chunk_strategy'] or 'default'}, "
            f"provider={chunk_options['llm_provider'] or 'default'}, "
            f"template={chunk_options['template'] or 'default'}, "
            f"model={chunk_options['llm_model'] or 'default'}"
        )

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
                    detected_areas=[],
                )

            elif result["type"] == "progress":
                yield ocr_pb2.ProcessDocumentResponse(
                    status=types_pb2.OCRStatus.PROCESSING,
                    stage=self._map_stage(result.get("stage", "")),
                    progress=result.get("progress", 0),
                )

            elif result["type"] == "complete":
                yield ocr_pb2.ProcessDocumentResponse(
                    status=types_pb2.OCRStatus.COMPLETED,
                    stage=types_pb2.OCRStage.FINISHED,
                    total_chunks=result["total_chunks"],
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

    def ListTemplates(self, request, context):
        """Lista todos os templates de documentos disponíveis"""
        templates = []
        for name in TemplateRegistry.list_templates():
            template = TemplateRegistry.get(name)
            templates.append(ocr_pb2.TemplateInfo(
                name=template.name,
                description=template.description,
            ))
        return ocr_pb2.ListTemplatesResponse(templates=templates)

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
            "validating": types_pb2.OCRStage.VALIDATING,
            "downloading": types_pb2.OCRStage.DOWNLOADING,
            "extracting": types_pb2.OCRStage.EXTRACTING,
            "chunking": types_pb2.OCRStage.CHUNKING,
            "finished": types_pb2.OCRStage.FINISHED,
        }
        return stage_map.get(stage.lower(), types_pb2.OCRStage.IDLE)