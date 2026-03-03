"""Section classifier for Brazilian legal documents."""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from ocr_pypi.models.document import SectionType
from ocr_pypi.semantic.legal_patterns import match_section_type

logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 150  # Characters - longer texts are unlikely to be titles
TEXT_SAMPLE_LENGTH = TITLE_MAX_LENGTH + 50  # Characters to sample from longer blocks


class SectionClassifier:
    """
    Classifies sections of legal documents using a hybrid strategy:
    regex patterns + layout analysis (font size, position).

    Falls back to CUSTOM when no pattern matches.
    """

    def classify_blocks(
        self,
        blocks: List[Dict[str, Any]],
        avg_font_size: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify each block with a section type.

        Args:
            blocks: List of block dicts with 'text', 'bbox', 'font_size', 'is_bold'.
            avg_font_size: Average font size for title detection heuristic.

        Returns:
            Blocks with 'section_type' and 'section_confidence' fields added.
        """
        if avg_font_size is None:
            font_sizes = [b.get("font_size", 12) for b in blocks if b.get("font_size")]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

        classified = []
        for block in blocks:
            section_type, confidence = self._classify_block(block, avg_font_size)
            annotated = dict(block)
            annotated["section_type"] = section_type.value
            annotated["section_confidence"] = confidence
            classified.append(annotated)

        return classified

    def _classify_block(
        self,
        block: Dict[str, Any],
        avg_font_size: float,
    ) -> Tuple[SectionType, float]:
        """Classify a single block, returning (SectionType, confidence)."""
        text = block.get("text", "").strip()
        if not text:
            return SectionType.CUSTOM, 0.0

        font_size = block.get("font_size", avg_font_size)
        is_bold = block.get("is_bold", False)
        is_title_candidate = (
            len(text) <= TITLE_MAX_LENGTH and
            (font_size >= avg_font_size * 1.1 or is_bold or text.isupper())
        )

        if is_title_candidate:
            matches = match_section_type(text)
            if matches:
                section_name = matches[0]
                try:
                    section_type = SectionType(section_name)
                    return section_type, 0.9
                except ValueError:
                    pass

        # Try matching longer text (first TEXT_SAMPLE_LENGTH chars)
        matches = match_section_type(text[:TEXT_SAMPLE_LENGTH])
        if matches:
            section_name = matches[0]
            try:
                section_type = SectionType(section_name)
                return section_type, 0.7
            except ValueError:
                pass

        return SectionType.CUSTOM, 0.3
