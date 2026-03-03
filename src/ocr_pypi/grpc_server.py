import json
import os
from typing import Any, Dict
from ocr_pypi.proto import ocr_pb2_grpc, ocr_pb2
import ocr_pypi.proto.types_pb2 as types_pb2

from ocr_pypi.services.document_processor import DocumentProcessor
from ocr_pypi.storage import get_storage
from ocr_pypi.chunking.templates.registry import TemplateRegistry
from ocr_pypi.chunking.llm_providers.factory import LLMProviderFactory
from ocr_pypi.chunking.templates.dynamic_template import DynamicTemplate

import logging

logger = logging.getLogger(__name__)


def _proto_section_to_dict(section_proto) -> Dict[str, Any]:
    """Converts a TemplateSection proto message to a dict compatible with DynamicTemplate."""
    return {
        "name": section_proto.name,
        "description": section_proto.description,
        "required": section_proto.required,
        "subsections": [_proto_section_to_dict(s) for s in section_proto.subsections],
    }


class OCRGrpcServer(ocr_pb2_grpc.OCRServiceServicer):
    """Implementação do serviço gRPC de OCR"""

    def ProcessDocument(self, request, context):
        """Processa documento e retorna chunks via streaming"""

        processor = DocumentProcessor(language=request.language or "por")

        # Resolve template from oneof template_option
        template_instance = None
        template_name = None
        which_template = request.WhichOneof("template_option")
        if which_template == "template":
            template_proto = request.template
            definition = {
                "template_name": template_proto.name,
                "description": template_proto.description,
                "sections": [_proto_section_to_dict(s) for s in template_proto.sections],
            }
            try:
                template_instance = TemplateRegistry.register_dynamic(definition)
                logger.info(f"Dynamic template registered: {template_instance.name}")
            except ValueError as e:
                logger.warning(f"Invalid template in request: {e}")
        elif which_template == "template_name":
            template_name = request.template_name or None

        # Extract LLM config
        llm_config = request.llm_config
        api_key_from_env = os.environ.get("LLM_API_KEY", "")
        api_key_from_request = llm_config.api_key or None
        if api_key_from_request and api_key_from_env:
            logger.warning(
                "API key provided in request, but using environment configuration "
                "for authentication."
            )
        llm_api_key = api_key_from_env or api_key_from_request or None

        # Monta chunk_options a partir do request
        chunk_options = {
            "enable_chunking": request.enable_chunking,

            # Template
            "template": template_name,
            "template_instance": template_instance,

            # LLM Provider (dinâmico por request)
            "llm_provider": llm_config.provider or None,
            "llm_model": llm_config.model or None,
            "llm_api_key": llm_api_key,
            "llm_temperature": llm_config.temperature if llm_config.temperature > 0 else None,
            "llm_max_tokens": llm_config.max_tokens if llm_config.max_tokens > 0 else None,
        }

        logger.info(
            f"ProcessDocument: provider={chunk_options['llm_provider'] or 'default'}, "
            f"template={chunk_options['template'] or (template_instance.name if template_instance else 'default')}, "
            f"model={chunk_options['llm_model'] or 'default'}"
        )

        template_used = (
            template_instance.name if template_instance
            else (template_name or "")
        )
        chunking_method = "llm"

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
                    chunking_method=chunk.metadata.get("chunking_method", chunking_method),
                    template_used=template_used,
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
                    chunking_method=chunking_method,
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

    def ListTemplates(self, request, context):
        """Lista todos os templates de documentos disponíveis"""
        templates = []
        for name in TemplateRegistry.list_templates():
            template = TemplateRegistry.get(name)
            templates.append(ocr_pb2.TemplateInfo(
                name=template.name,
                description=template.description,
                is_custom=template.is_custom,
                document_types=template.document_types,
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

    def ValidateTemplate(self, request, context):
        """Valida a definição de um template customizado"""
        template_proto = request.template
        missing_fields = []
        invalid_fields = []

        if not template_proto.name:
            return ocr_pb2.ValidateTemplateResponse(
                is_valid=False,
                validation_message="template.name is required",
                missing_fields=["name"],
                invalid_fields=[],
            )

        definition = {
            "template_name": template_proto.name,
            "description": template_proto.description,
            "sections": [_proto_section_to_dict(s) for s in template_proto.sections],
        }

        try:
            DynamicTemplate(definition)
        except ValueError as e:
            error_msg = str(e)
            if not definition.get("template_name"):
                missing_fields.append("template_name")
            elif not isinstance(definition.get("sections"), list):
                invalid_fields.append("sections")
            return ocr_pb2.ValidateTemplateResponse(
                is_valid=False,
                validation_message=error_msg,
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
            )

        return ocr_pb2.ValidateTemplateResponse(
            is_valid=True,
            validation_message="Template is valid",
            missing_fields=[],
            invalid_fields=[],
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