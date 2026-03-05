"""Factory pattern for document chunking strategies."""
import logging
from typing import List, Dict, Any, Generator

from ocr_pypi.models.document import Chunk

logger = logging.getLogger(__name__)

STRATEGY_PAGE = "page"
STRATEGY_SEMANTIC = "semantic"
STRATEGY_PARAGRAPH = "paragraph"

_VALID_STRATEGIES = {STRATEGY_PAGE, STRATEGY_SEMANTIC, STRATEGY_PARAGRAPH}


class ChunkingStrategy:
    """
    Factory and dispatcher for chunking strategies.

    Supported strategies:
        "page"      - Uses PageChunker (one chunk per page)
        "semantic"  - Uses SemanticChunker (sentence embeddings, cross-page)
        "paragraph" - Uses ParagraphChunker (paragraph splitting, cross-page)
    """

    @staticmethod
    def chunk(
        pages: List[Dict[str, Any]],
        strategy: str,
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """
        Dispatch chunking to the appropriate strategy implementation.

        Yields chunk/complete dicts compatible with DocumentProcessor.

        Args:
            pages: Extracted pages list.
            strategy: One of 'page', 'semantic', 'paragraph'.
            chunk_options: Options dict (see DocumentProcessor.process docs).
        """
        strategy = (strategy or STRATEGY_PAGE).lower()
        if strategy not in _VALID_STRATEGIES:
            logger.warning(
                f"Unknown chunking strategy '{strategy}'; falling back to 'page'."
            )
            strategy = STRATEGY_PAGE

        if strategy == STRATEGY_PAGE:
            yield from ChunkingStrategy._chunk_page(pages, chunk_options)
        elif strategy == STRATEGY_SEMANTIC:
            yield from ChunkingStrategy._chunk_semantic(pages, chunk_options)
        elif strategy == STRATEGY_PARAGRAPH:
            yield from ChunkingStrategy._chunk_paragraph(pages, chunk_options)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_page(
        pages: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        from ocr_pypi.chunking.page_chunker import PageChunker

        chunker = PageChunker()
        chunks = chunker.chunk(pages=pages)
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

        chunks = chunker.chunk(pages=pages)

        yield from _emit_chunks(chunks)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _emit_chunks(chunks: List[Chunk]) -> Generator[Dict, None, None]:
    """Emit chunk and complete events for a list of Chunk objects."""
    for chunk in chunks:
        yield {"type": "chunk", "data": chunk}
    yield {"type": "complete", "total_chunks": len(chunks)}
