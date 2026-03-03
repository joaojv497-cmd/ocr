"""Factory pattern for document chunking strategies."""
import logging
from typing import List, Dict, Any, Generator

from ocr_pypi.models.document import Chunk

logger = logging.getLogger(__name__)

STRATEGY_LLM = "llm"
STRATEGY_SEMANTIC = "semantic"
STRATEGY_PARAGRAPH = "paragraph"
STRATEGY_HYBRID = "hybrid"

_VALID_STRATEGIES = {STRATEGY_LLM, STRATEGY_SEMANTIC, STRATEGY_PARAGRAPH, STRATEGY_HYBRID}


class ChunkingStrategy:
    """
    Factory and dispatcher for chunking strategies.

    Supported strategies:
        "llm"       - Uses LLMChunker (default)
        "semantic"  - Uses SemanticChunker (sentence embeddings)
        "paragraph" - Uses ParagraphChunker (simple paragraph splitting)
        "hybrid"    - Runs LLMChunker first; falls back to SemanticChunker on error
    """

    @staticmethod
    def chunk(
        pages: List[Dict[str, Any]],
        strategy: str,
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """
        Dispatch chunking to the appropriate strategy implementation.

        Yields progress/chunk/complete dicts compatible with DocumentProcessor.

        Args:
            pages: Extracted pages list.
            strategy: One of 'llm', 'semantic', 'paragraph', 'hybrid'.
            chunk_options: Options dict (see DocumentProcessor.process docs).
        """
        strategy = (strategy or STRATEGY_LLM).lower()
        if strategy not in _VALID_STRATEGIES:
            logger.warning(
                f"Unknown chunking strategy '{strategy}'; falling back to 'llm'."
            )
            strategy = STRATEGY_LLM

        if strategy == STRATEGY_LLM:
            yield from ChunkingStrategy._chunk_llm(pages, chunk_options)
        elif strategy == STRATEGY_SEMANTIC:
            yield from ChunkingStrategy._chunk_semantic(pages, chunk_options)
        elif strategy == STRATEGY_PARAGRAPH:
            yield from ChunkingStrategy._chunk_paragraph(pages, chunk_options)
        elif strategy == STRATEGY_HYBRID:
            yield from ChunkingStrategy._chunk_hybrid(pages, chunk_options)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_llm(
        pages: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        from ocr_pypi.chunking.llm_chunker import LLMChunker
        from ocr_pypi.config import settings

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

        template_name = chunk_options.get("template") or None
        template_instance = chunk_options.get("template_instance")

        yield {
            "type": "progress",
            "stage": "llm_chunking_starting",
            "template": template_name,
            "provider": chunker.provider.provider_name,
            "model": chunker.provider.model,
        }

        chunks = chunker.chunk(
            pages=pages,
            template_name=template_name if template_instance is None else None,
            template=template_instance,
        )

        yield from _emit_chunks(chunks)

    @staticmethod
    def _chunk_semantic(
        pages: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        from ocr_pypi.chunking.semantic_chunker import SemanticChunker
        from ocr_pypi.config import settings

        model_name = chunk_options.get("embedding_model") or settings.EMBEDDING_MODEL
        chunk_size = chunk_options.get("chunk_size") or settings.CHUNK_SIZE
        min_chunk_size = chunk_options.get("min_chunk_size") or settings.MIN_CHUNK_SIZE

        chunker = SemanticChunker(
            model_name=model_name,
            max_chunk_chars=chunk_size,
            min_chunk_chars=min_chunk_size,
        )

        similarity_threshold = chunk_options.get(
            "similarity_threshold", settings.SIMILARITY_THRESHOLD
        )

        yield {"type": "progress", "stage": "semantic_chunking_starting"}

        chunks = chunker.chunk(pages=pages, similarity_threshold=similarity_threshold)

        yield from _emit_chunks(chunks)

    @staticmethod
    def _chunk_paragraph(
        pages: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        from ocr_pypi.chunking.paragraph_chunker import ParagraphChunker
        from ocr_pypi.config import settings

        chunk_size = chunk_options.get("chunk_size") or settings.CHUNK_SIZE
        min_chunk_size = chunk_options.get("min_chunk_size") or settings.MIN_CHUNK_SIZE
        chunk_overlap = chunk_options.get("chunk_overlap") or settings.CHUNK_OVERLAP

        chunker = ParagraphChunker(
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
        )

        yield {"type": "progress", "stage": "paragraph_chunking_starting"}

        chunks = chunker.chunk(pages=pages)

        yield from _emit_chunks(chunks)

    @staticmethod
    def _chunk_hybrid(
        pages: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """LLM chunking with SemanticChunker as fallback."""
        try:
            results = list(ChunkingStrategy._chunk_llm(pages, chunk_options))
            # Check if any chunk was produced
            has_chunks = any(r.get("type") == "chunk" for r in results)
            if has_chunks:
                yield from results
                return
        except Exception as e:
            logger.warning(f"LLM chunking failed in hybrid mode: {e}; falling back to semantic.")

        yield {"type": "progress", "stage": "hybrid_fallback_to_semantic"}
        yield from ChunkingStrategy._chunk_semantic(pages, chunk_options)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _emit_chunks(chunks: List[Chunk]) -> Generator[Dict, None, None]:
    """Emit progress, chunk, and complete events for a list of Chunk objects."""
    yield {
        "type": "progress",
        "stage": "chunking_complete",
        "total_chunks": len(chunks),
    }
    for chunk in chunks:
        yield {"type": "chunk", "data": chunk}
    yield {"type": "complete", "total_chunks": len(chunks)}
