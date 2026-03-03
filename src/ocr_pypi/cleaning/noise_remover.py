"""Noise removal for document text blocks."""
import logging
import re
from typing import List, Dict, Any, Set
from ocr_pypi.cleaning.header_footer_detector import HeaderFooterDetector

logger = logging.getLogger(__name__)

MIN_BLOCK_LENGTH = 3  # Characters


class NoiseRemover:
    """Removes noise from document text blocks.

    Handles repeated headers/footers, page numbers, watermarks, and
    very short non-meaningful blocks.
    """

    def __init__(self):
        self._detector = HeaderFooterDetector()

    def remove_noise(
        self,
        pages_data: List[Dict[str, Any]],
        page_height: float = 842.0,
    ) -> List[Dict[str, Any]]:
        """
        Remove noise from all pages of a document.

        Args:
            pages_data: List of page dicts with 'page_number', 'text', 'blocks'.
            page_height: Page height for zone detection.

        Returns:
            Cleaned pages_data with noise blocks removed.
        """
        pages_blocks = [p.get("blocks", []) for p in pages_data]
        noise_patterns = self._detector.detect(pages_blocks, page_height)
        noise_texts = noise_patterns["headers"] | noise_patterns["footers"]

        cleaned_pages = []
        for page in pages_data:
            cleaned_blocks = self._filter_blocks(
                page.get("blocks", []), noise_texts
            )
            cleaned_text = "\n\n".join(
                b["text"] for b in cleaned_blocks if b.get("text", "").strip()
            )
            cleaned_pages.append({
                **page,
                "text": cleaned_text,
                "blocks": cleaned_blocks,
            })

        return cleaned_pages

    def _filter_blocks(
        self,
        blocks: List[Dict[str, Any]],
        noise_texts: Set[str],
    ) -> List[Dict[str, Any]]:
        """Filter out noise blocks."""
        result = []
        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            if len(text) < MIN_BLOCK_LENGTH:
                continue
            if text in noise_texts:
                continue
            result.append(block)
        return result
