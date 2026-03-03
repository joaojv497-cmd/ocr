"""Header and footer detection for multi-page documents."""
import logging
import re
from typing import List, Dict, Any, Set
from collections import Counter

logger = logging.getLogger(__name__)

REPETITION_THRESHOLD = 0.6  # Fraction of pages a text must appear in to be a header/footer
PAGE_NUMBER_PATTERNS = [
    r"^\s*\d+\s*$",
    r"^\s*[Pp]ágina?\s+\d+\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"^\s*-\s*\d+\s*-\s*$",
]


class HeaderFooterDetector:
    """Detects repeated headers and footers across document pages."""

    def __init__(self):
        self._page_number_re = [re.compile(p) for p in PAGE_NUMBER_PATTERNS]

    def detect(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        page_height: float = 842.0,
    ) -> Dict[str, Set[str]]:
        """
        Detect header and footer text patterns across pages.

        Args:
            pages_blocks: List of pages, each page being a list of blocks.
            page_height: Page height in points for zone detection.

        Returns:
            Dict with 'headers' and 'footers' sets of repeated text patterns.
        """
        total_pages = len(pages_blocks)
        if total_pages < 2:
            return {"headers": set(), "footers": set()}

        header_texts: List[str] = []
        footer_texts: List[str] = []

        for page_blocks in pages_blocks:
            for block in page_blocks:
                bbox = block.get("bbox", {})
                y_center = (bbox.get("y0", 0) + bbox.get("y1", page_height)) / 2
                text = block.get("text", "").strip()
                if not text:
                    continue
                if y_center < page_height * 0.15:
                    header_texts.append(text)
                elif y_center > page_height * 0.85:
                    footer_texts.append(text)

        headers = self._find_repeated(header_texts, total_pages)
        footers = self._find_repeated(footer_texts, total_pages)

        # Always mark page number patterns as footers
        for page_blocks in pages_blocks:
            for block in page_blocks:
                text = block.get("text", "").strip()
                if self._is_page_number(text):
                    footers.add(text)

        logger.info(f"Detected {len(headers)} headers, {len(footers)} footers")
        return {"headers": headers, "footers": footers}

    def _find_repeated(self, texts: List[str], total_pages: int) -> Set[str]:
        """Find texts that appear on at least REPETITION_THRESHOLD fraction of pages."""
        counter = Counter(texts)
        threshold = max(2, int(total_pages * REPETITION_THRESHOLD))
        return {text for text, count in counter.items() if count >= threshold}

    def _is_page_number(self, text: str) -> bool:
        """Check if text matches a page number pattern."""
        return any(p.match(text) for p in self._page_number_re)
