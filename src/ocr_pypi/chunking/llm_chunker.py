import json
import logging
from typing import List, Dict, Any, Optional

from commons_pypi.llm_providers.base_provider import LLMProvider
from commons_pypi.llm_providers.factory import LLMProviderFactory

from ocr_pypi.chunking.templates.base_template import DocumentTemplate
from ocr_pypi.chunking.templates.registry import TemplateRegistry
from ocr_pypi.models.document import Chunk, StructuredDocument, StructuredSection
from ocr_pypi.config import settings

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 4
# Default maximum input characters before splitting the document
DEFAULT_MAX_INPUT_CHARS = 80_000
# Default maximum characters per chunk sent to the LLM in incremental mode
DEFAULT_MAX_CHUNK_CHARS = 4_000
# Default maximum tokens per chunk (used together with MAX_CHUNK_CHARS)
DEFAULT_MAX_CHUNK_TOKENS = 1_000


class LLMChunker:
    """
    Chunking estruturado usando LLM + templates de documentos jurídicos.

    Substitui LayoutLMAnalyzer + SemanticChunker.
    Recebe texto já extraído pelo OCR e usa a LLM para separar
    em chunks estruturados conforme o template fornecido.

    Para documentos grandes, divide automaticamente em seções menores,
    processa cada seção separadamente e combina os resultados.
    """

    def __init__(
        self,
        provider: LLMProvider = None,
        provider_name: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    ):
        """
        Inicializa o LLMChunker.

        Pode receber um provider já instanciado OU os parâmetros para
        criar um via factory. Se nada for informado, usa as configs do settings.

        Args:
            max_input_chars: Limite absoluto de caracteres do documento antes de
                             acionar o processamento incremental. Funciona como
                             barreira de segurança máxima.
            max_chunk_chars: Máximo de caracteres por grupo de páginas enviado à LLM
                             no modo incremental. Documentos maiores que este valor são
                             automaticamente divididos para que cada chamada à LLM receba
                             no máximo ``max_chunk_chars`` caracteres, mantendo os chunks
                             focados e otimizados para embedding.
            max_chunk_tokens: Máximo de tokens por chunk (informativo; usado em conjunto
                              com ``max_chunk_chars`` — o menor valor prevalece).
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

        self.max_input_chars = max_input_chars
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_tokens = max_chunk_tokens
        # Effective per-section limit: honour both char and token constraints
        self._effective_chunk_limit = min(
            max_chunk_chars,
            max_chunk_tokens * CHARS_PER_TOKEN,
        )
        logger.info(f"LLMChunker inicializado com {self.provider}")

    def chunk(
        self,
        pages: List[Dict[str, Any]],
        template_name: str = None,
        template: DocumentTemplate = None,
    ) -> List[Chunk]:
        """
        Processa texto extraído e retorna chunks estruturados via LLM.

        Args:
            pages: Lista de dicts com 'page_number' e 'text'
            template_name: Nome do template registrado (ex: 'peticao_inicial')
            template: Instância de DocumentTemplate (tem prioridade sobre template_name)

        Returns:
            Lista de Chunks estruturados
        """
        # Resolve template
        if template is None:
            name = template_name or None
            template = TemplateRegistry.get(name)

        # 1. Combina texto de todas as páginas
        full_text = self._combine_pages(pages)

        if not full_text.strip():
            logger.warning("Texto vazio recebido para chunking")
            return []

        logger.info(
            f"Processando {len(pages)} páginas ({len(full_text)} chars) "
            f"com template '{template.name}' via {self.provider}"
        )

        # 2. Decide se processa direto ou de forma incremental.
        # Usa o menor entre max_input_chars e _effective_chunk_limit para respeitar
        # tanto o tamanho por chunk quanto o limite absoluto máximo do documento.
        split_threshold = min(self.max_input_chars, self._effective_chunk_limit)
        if len(full_text) > split_threshold:
            return self._chunk_large_document(pages, template)

        # 3. Constrói prompt com o template
        prompt = template.build_prompt(full_text)

        # 4. Envia para LLM
        try:
            llm_response = self.provider.generate(prompt)
            structured_data = self._parse_llm_response(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao parsear JSON da LLM: {e}")
            return self._fallback_chunks(pages)
        except Exception as e:
            logger.error(f"Erro na comunicação com LLM: {e}")
            return self._fallback_chunks(pages)

        # 5. Converte resposta estruturada em Chunks
        chunks = self._build_chunks(structured_data, pages)
        logger.info(f"Criados {len(chunks)} chunks estruturados")

        return chunks

    def chunk_to_structured(
        self,
        pages: List[Dict[str, Any]],
        template_name: str = None,
        template: DocumentTemplate = None,
    ) -> StructuredDocument:
        """
        Variante que retorna StructuredDocument em vez de lista de Chunks.
        Útil quando se quer acessar a estrutura hierárquica completa.
        """
        if template is None:
            name = template_name or None
            template = TemplateRegistry.get(name)

        full_text = self._combine_pages(pages)
        if not full_text.strip():
            return StructuredDocument(document_type=template.name)

        split_threshold = min(self.max_input_chars, self._effective_chunk_limit)
        if len(full_text) > split_threshold:
            chunks = self._chunk_large_document(pages, template)
            sections = [
                StructuredSection(
                    section_name=c.metadata.get("section_name", ""),
                    title=c.metadata.get("section_title", ""),
                    content=c.content,
                    page_numbers=c.page_numbers,
                    subsections=[
                        StructuredSection(
                            section_name=sub.get("section_name", ""),
                            title=sub.get("title", ""),
                            content=sub.get("content", ""),
                        )
                        for sub in c.metadata.get("subsections", [])
                    ],
                    metadata=c.metadata,
                )
                for c in chunks
            ]
            return StructuredDocument(
                document_type=template.name,
                sections=sections,
            )

        prompt = template.build_prompt(full_text)

        try:
            llm_response = self.provider.generate(prompt)
            data = self._parse_llm_response(llm_response)
        except Exception as e:
            logger.error(f"Erro ao obter documento estruturado: {e}")
            return StructuredDocument(document_type=template.name)

        return self._build_structured_document(data)

    # ── Métodos internos ────────────────────────────────────────────────

    def _combine_pages(self, pages: List[Dict[str, Any]]) -> str:
        """Combina texto de todas as páginas com marcadores de página"""
        parts = []
        for page in pages:
            page_num = page["page_number"]
            text = page["text"].strip()
            if text:
                parts.append(f"[PÁGINA {page_num}]\n{text}")
        return "\n\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estima o número de tokens com base no comprimento do texto."""
        return len(text) // CHARS_PER_TOKEN

    def _split_pages_into_sections(
        self, pages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Divide a lista de páginas em seções menores respeitando o limite de caracteres.

        Cada seção contém no máximo ``_effective_chunk_limit`` caracteres para que
        cada chamada à LLM produza chunks focados e otimizados para embedding.

        Mantém contexto entre seções ao incluir a última página da seção anterior
        no início da próxima seção (overlap de 1 página).
        """
        sections: List[List[Dict[str, Any]]] = []
        current_section: List[Dict[str, Any]] = []
        current_chars = 0

        for page in pages:
            page_text = page.get("text", "")
            page_chars = len(page_text)

            if current_chars + page_chars > self._effective_chunk_limit and current_section:
                sections.append(current_section)
                # Overlap: inclui a última página da seção anterior para contexto
                current_section = [current_section[-1], page]
                current_chars = len(current_section[-2].get("text", "")) + page_chars
            else:
                current_section.append(page)
                current_chars += page_chars

        if current_section:
            sections.append(current_section)

        return sections

    def _chunk_large_document(
        self,
        pages: List[Dict[str, Any]],
        template: DocumentTemplate,
    ) -> List[Chunk]:
        """
        Processa documentos grandes dividindo-os em seções menores.

        Divide as páginas em grupos que respeitam o limite de caracteres,
        processa cada grupo separadamente e combina os resultados preservando
        a estrutura hierárquica e reindexando os chunks.
        """
        page_sections = self._split_pages_into_sections(pages)
        logger.info(
            f"Documento grande dividido em {len(page_sections)} seções para processamento"
        )

        all_chunks: List[Chunk] = []
        chunk_offset = 0

        for section_idx, section_pages in enumerate(page_sections):
            section_text = self._combine_pages(section_pages)
            logger.info(
                f"Processando seção {section_idx + 1}/{len(page_sections)} "
                f"({len(section_pages)} páginas, {len(section_text)} chars)"
            )

            prompt = template.build_prompt(section_text)

            try:
                llm_response = self.provider.generate(prompt)
                structured_data = self._parse_llm_response(llm_response)
                section_chunks = self._build_chunks(structured_data, section_pages)
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao parsear JSON na seção {section_idx + 1}: {e}")
                section_chunks = self._fallback_chunks(section_pages)
            except Exception as e:
                logger.error(f"Erro na LLM na seção {section_idx + 1}: {e}")
                section_chunks = self._fallback_chunks(section_pages)

            # Reindex chunks to maintain global ordering
            for chunk in section_chunks:
                chunk.chunk_index = chunk_offset + chunk.chunk_index
                chunk.metadata["section_index"] = section_idx
                chunk.metadata["total_sections"] = len(page_sections)
            chunk_offset += len(section_chunks)

            all_chunks.extend(section_chunks)

        logger.info(f"Criados {len(all_chunks)} chunks totais de {len(page_sections)} seções")
        return all_chunks

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse da resposta JSON da LLM, limpando marcadores de código"""
        cleaned = response.strip()

        # Remove marcadores ```json ... ``` se presentes
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return json.loads(cleaned.strip())

    def _build_chunks(
        self,
        structured_data: Dict,
        pages: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """Converte dados estruturados da LLM em objetos Chunk"""
        chunks = []
        sections = structured_data.get("sections", [])
        doc_metadata = structured_data.get("metadata", {})
        doc_type = structured_data.get("document_type", "")

        for idx, section in enumerate(sections):
            content = section.get("content")
            if not content:
                continue

            page_numbers = section.get("page_numbers", [])
            if not page_numbers:
                page_numbers = self._infer_pages(content, pages)

            # Processa subseções como chunks filhos
            subsections_data = []
            for sub in section.get("subsections", []):
                if sub.get("content"):
                    subsections_data.append({
                        "section_name": sub.get("section_name", ""),
                        "title": sub.get("title", ""),
                        "content": sub.get("content", ""),
                    })

            chunks.append(Chunk(
                content=content,
                page_numbers=page_numbers,
                chunk_index=idx,
                metadata={
                    "section_name": section.get("section_name", ""),
                    "section_title": section.get("title", ""),
                    "document_type": doc_type,
                    "document_metadata": doc_metadata,
                    "subsections": subsections_data,
                    "subsections_count": len(subsections_data),
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "chunking_method": "llm_template",
                },
                detected_areas=[],
            ))

        return chunks

    def _build_structured_document(self, data: Dict) -> StructuredDocument:
        """Converte dict da LLM em StructuredDocument"""

        def _parse_section(s: Dict) -> StructuredSection:
            return StructuredSection(
                section_name=s.get("section_name", ""),
                title=s.get("title", ""),
                content=s.get("content", ""),
                page_numbers=s.get("page_numbers", []),
                subsections=[
                    _parse_section(sub) for sub in s.get("subsections", [])
                ],
                metadata=s.get("metadata", {}),
            )

        return StructuredDocument(
            document_type=data.get("document_type", ""),
            sections=[_parse_section(s) for s in data.get("sections", [])],
            metadata=data.get("metadata", {}),
        )

    def _infer_pages(
        self, content: str, pages: List[Dict[str, Any]]
    ) -> List[int]:
        """Infere em quais páginas o conteúdo aparece"""
        matched = []
        content_words = set(content.lower().split()[:20])

        for page in pages:
            page_words = set(page["text"].lower().split())
            overlap = len(content_words & page_words) / max(len(content_words), 1)
            if overlap > 0.3:
                matched.append(page["page_number"])

        return matched or [1]

    def _fallback_chunks(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Fallback: retorna um chunk por página quando a LLM falhar"""
        logger.warning("Usando fallback: um chunk por página")
        return [
            Chunk(
                content=page["text"],
                page_numbers=[page["page_number"]],
                chunk_index=i,
                metadata={"chunking_method": "fallback_per_page"},
                detected_areas=[],
            )
            for i, page in enumerate(pages)
            if page["text"].strip()
        ]