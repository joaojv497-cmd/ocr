"""Abstract text extractor interface."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class TextExtractor(ABC):
    """Abstract base class for PDF text extractors."""

    @abstractmethod
    def extract_with_layout(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text with layout information from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dicts with keys: page_number, text, blocks.
            Each block has: text, bbox, font_size, is_bold.
        """
