"""Layout analysis for document structure detection."""
import logging
from typing import List, Dict, Any, Optional
from ocr_pypi.models.document import AreaType

logger = logging.getLogger(__name__)

TITLE_FONT_SIZE_MULTIPLIER = 1.3  # Font size ratio to consider a block as title
HEADER_ZONE_FRACTION = 0.1  # Top 10% of page
FOOTER_ZONE_FRACTION = 0.1  # Bottom 10% of page


class LayoutAnalyzer:
    """Analyzes document layout to identify structural elements.

    Detects titles, paragraphs, headers, footers, lists, and tables.
    """

    def analyze(
        self,
        blocks: List[Dict[str, Any]],
        page_height: float = 842.0,
        page_width: float = 595.0,
    ) -> List[Dict[str, Any]]:
        """
        Analyze layout of page blocks and annotate with area types.

        Args:
            blocks: List of block dicts with 'text', 'bbox', 'font_size'.
            page_height: Height of the page in points.
            page_width: Width of the page in points.

        Returns:
            Blocks with 'area_type' field added.
        """
        if not blocks:
            return []

        font_sizes = [b.get("font_size", 12) for b in blocks if b.get("font_size")]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

        annotated = []
        for block in blocks:
            area_type = self._classify_block(
                block, avg_font_size, page_height, page_width
            )
            annotated_block = dict(block)
            annotated_block["area_type"] = area_type.value
            annotated.append(annotated_block)

        return annotated

    def _classify_block(
        self,
        block: Dict[str, Any],
        avg_font_size: float,
        page_height: float,
        page_width: float,
    ) -> AreaType:
        """Classify a single block into an AreaType."""
        bbox = block.get("bbox", {})
        y0 = bbox.get("y0", 0)
        y1 = bbox.get("y1", page_height)
        font_size = block.get("font_size", avg_font_size)
        text = block.get("text", "")
        is_bold = block.get("is_bold", False)

        # Header zone (top of page)
        if y1 < page_height * HEADER_ZONE_FRACTION:
            return AreaType.HEADER

        # Footer zone (bottom of page)
        if y0 > page_height * (1 - FOOTER_ZONE_FRACTION):
            return AreaType.FOOTER

        # Title detection (large font, bold, or short text)
        if font_size >= avg_font_size * TITLE_FONT_SIZE_MULTIPLIER:
            if is_bold or len(text.strip()) < 100:
                return AreaType.TITLE

        # List detection (starts with bullet, number, or dash)
        stripped = text.strip()
        if stripped and (
            stripped[0] in "-•*" or
            (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
        ):
            return AreaType.LIST

        return AreaType.PARAGRAPH

    def detect_titles(
        self, blocks: List[Dict[str, Any]], avg_font_size: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Return only blocks classified as titles or subtitles."""
        if avg_font_size is None:
            font_sizes = [b.get("font_size", 12) for b in blocks]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        return [
            b for b in blocks
            if b.get("font_size", 12) >= avg_font_size * TITLE_FONT_SIZE_MULTIPLIER
        ]
