"""Page-based chunker: one chunk per page."""
import logging
from typing import List, Dict, Any

from ocr_pypi.models.document import Chunk

logger = logging.getLogger(__name__)


class PageChunker:
    """
    Chunks document text by creating one chunk per page.

    The simplest chunking strategy: each page becomes a single chunk.
    Useful when page-level granularity is required or as a fast fallback.
    """

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk pages into one chunk per page.

        Args:
            pages: List of page dicts with 'page_number' and 'text'.

        Returns:
            List of Chunk objects.
        """
        chunks: List[Chunk] = []
        chunk_idx = 0
        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            chunks.append(Chunk(
                content=text,
                page_numbers=[page["page_number"]],
                chunk_index=chunk_idx,
                metadata={
                    "chunking_method": "page",
                    "char_count": len(text),
                    "word_count": len(text.split()),
                },
                detected_areas=[],
            ))
            chunk_idx += 1
        return chunks
