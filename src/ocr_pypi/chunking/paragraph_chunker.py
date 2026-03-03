"""Paragraph-based chunker using simple text structure (no LLM or embeddings)."""
import re
import logging
from typing import List, Dict, Any

from ocr_pypi.models.document import Chunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 4000
DEFAULT_MIN_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0


class ParagraphChunker:
    """
    Chunks document text based on paragraph breaks and section markers.

    Splits on double newlines and section headings (numbered or titled),
    then merges small paragraphs until reaching chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk pages into paragraph-based chunks.

        Args:
            pages: List of page dicts with 'page_number' and 'text'.

        Returns:
            List of Chunk objects.
        """
        paragraphs = self._extract_paragraphs(pages)
        if not paragraphs:
            return []

        return self._merge_paragraphs(paragraphs)

    def _extract_paragraphs(
        self, pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split pages into individual paragraphs with page metadata."""
        paragraphs = []
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page_number", 1)
            for para in re.split(r"\n{2,}", text):
                para = para.strip()
                if len(para) >= max(1, self.min_chunk_size // 4):
                    paragraphs.append({"text": para, "page_number": page_num})
        return paragraphs

    def _merge_paragraphs(
        self, paragraphs: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """Merge paragraphs into chunks respecting chunk_size."""
        chunks: List[Chunk] = []
        current_texts: List[str] = []
        current_pages: List[int] = []
        current_chars = 0
        chunk_idx = 0

        for para in paragraphs:
            text = para["text"]
            page_num = para["page_number"]

            if current_chars + len(text) > self.chunk_size and current_texts:
                chunks.append(
                    self._make_chunk(current_texts, current_pages, chunk_idx)
                )
                chunk_idx += 1
                # Apply overlap: carry over last paragraph(s) to next chunk
                if self.chunk_overlap > 0:
                    overlap_texts: List[str] = []
                    overlap_pages: List[int] = []
                    overlap_chars = 0
                    for t, p in zip(
                        reversed(current_texts), reversed(current_pages)
                    ):
                        if overlap_chars + len(t) <= self.chunk_overlap:
                            overlap_texts.insert(0, t)
                            overlap_pages.insert(0, p)
                            overlap_chars += len(t)
                        else:
                            break
                    current_texts = overlap_texts
                    current_pages = overlap_pages
                    current_chars = overlap_chars
                else:
                    current_texts = []
                    current_pages = []
                    current_chars = 0

            current_texts.append(text)
            current_pages.append(page_num)
            current_chars += len(text)

        if current_texts:
            chunks.append(
                self._make_chunk(current_texts, current_pages, chunk_idx)
            )

        return chunks

    def _make_chunk(
        self,
        texts: List[str],
        pages: List[int],
        idx: int,
    ) -> Chunk:
        """Create a Chunk from collected paragraph texts."""
        content = "\n\n".join(texts)
        unique_pages = sorted(set(pages))
        return Chunk(
            content=content,
            page_numbers=unique_pages,
            chunk_index=idx,
            metadata={
                "chunking_method": "paragraph",
                "char_count": len(content),
                "word_count": len(content.split()),
                "paragraph_count": len(texts),
            },
            detected_areas=[],
        )
