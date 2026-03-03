"""Standardized JSON output formatter for document processing pipeline."""
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from ocr_pypi.models.document import Chunk, StructuredSection

logger = logging.getLogger(__name__)


class JSONFormatter:
    """Formats pipeline output into a standardized JSON schema.

    Produces structured output ready for embedding and RAG systems.
    """

    def format(
        self,
        chunks: List[Chunk],
        metadata: Dict[str, Any],
        sections: Optional[List[Dict[str, Any]]] = None,
        statistics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format chunks and metadata into standardized JSON.

        Args:
            chunks: List of Chunk objects from the pipeline.
            metadata: Document metadata (filename, type, pdf_type, etc.).
            sections: Optional list of classified sections.
            statistics: Optional processing statistics.

        Returns:
            Dict conforming to the standardized document schema.
        """
        document_id = metadata.get("document_id") or str(uuid.uuid4())
        processing_date = datetime.now(timezone.utc).isoformat()

        formatted_chunks = [self._format_chunk(c) for c in chunks]
        formatted_sections = [
            self._format_section(s, i) for i, s in enumerate(sections or [])
        ]

        entities = self._extract_entities(chunks)

        total_chunks = len(formatted_chunks)
        total_sections = len(formatted_sections)
        avg_confidence = (
            sum(c.get("metadata", {}).get("confidence", 0.9) for c in formatted_chunks)
            / max(total_chunks, 1)
        )

        output = {
            "document_id": document_id,
            "metadata": {
                "filename": metadata.get("filename", ""),
                "document_type": metadata.get("document_type", "generico"),
                "pdf_type": metadata.get("pdf_type", "digital"),
                "total_pages": metadata.get("total_pages", 0),
                "processing_date": metadata.get("processing_date", processing_date),
                "language": metadata.get("language", "por"),
                "ocr_engine": metadata.get("ocr_engine", "tesseract"),
                "llm_provider": metadata.get("llm_provider", ""),
                "llm_model": metadata.get("llm_model", ""),
                "template_used": metadata.get("template_used", "generico"),
            },
            "sections": formatted_sections,
            "chunks": formatted_chunks,
            "entities": entities,
            "statistics": {
                "total_chunks": total_chunks,
                "total_sections": total_sections,
                "avg_confidence": round(avg_confidence, 4),
                "processing_time_seconds": (
                    statistics.get("processing_time_seconds", 0)
                    if statistics else 0
                ),
                **(statistics or {}),
            },
        }

        return output

    def _format_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """Format a single Chunk into the output schema."""
        meta = chunk.metadata or {}
        return {
            "chunk_id": f"chunk_{chunk.chunk_index:03d}",
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "pages": chunk.page_numbers,
            "section_refs": meta.get("section_refs", []),
            "metadata": {
                "token_count": meta.get("token_count", len(chunk.content.split())),
                "char_count": meta.get("char_count", len(chunk.content)),
                "semantic_group": meta.get("section_name", ""),
                "entities": meta.get("entities", []),
                "citations": meta.get("citations", []),
                "chunking_method": meta.get("chunking_method", ""),
                "confidence": meta.get("confidence", 0.9),
            },
            "embedding_ready": bool(chunk.content.strip()),
        }

    def _format_section(
        self, section: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Format a section dict into the output schema."""
        return {
            "section_id": f"sec_{idx + 1:03d}",
            "section_type": section.get("section_type", "custom"),
            "section_name": section.get("section_name", section.get("title", "")),
            "confidence": section.get("section_confidence", 0.8),
            "pages": section.get("page_numbers", section.get("pages", [])),
            "bbox": section.get("bbox"),
            "content": section.get("content", section.get("text", "")),
            "subsections": [
                self._format_section(sub, i)
                for i, sub in enumerate(section.get("subsections", []))
            ],
            "metadata": section.get("metadata", {}),
        }

    def _extract_entities(self, chunks: List[Chunk]) -> Dict[str, List[str]]:
        """Extract entities from chunk metadata."""
        pessoas: List[str] = []
        organizacoes: List[str] = []
        legislacao: List[str] = []
        valores: List[str] = []
        datas: List[str] = []

        for chunk in chunks:
            meta = chunk.metadata or {}
            entities = meta.get("entities", [])
            if isinstance(entities, list):
                for e in entities:
                    if not isinstance(e, str):
                        continue
                    if "Autor" in e or "Réu" in e:
                        pessoas.append(e)
                    elif "Lei" in e or "art." in e or "CF" in e:
                        legislacao.append(e)
                    elif "R$" in e or "reais" in e.lower():
                        valores.append(e)

        return {
            "pessoas": list(set(pessoas)),
            "organizacoes": list(set(organizacoes)),
            "legislacao": list(set(legislacao)),
            "valores": list(set(valores)),
            "datas": list(set(datas)),
        }
