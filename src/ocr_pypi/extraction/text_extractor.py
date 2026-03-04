"""Abstract text extractor interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator


class TextExtractor(ABC):
    """Abstract base class for PDF text extractors."""

    @abstractmethod
    def extract_with_layout(self, pdf_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Extract text with layout information from a PDF, yielding one page at a time.

        Args:
            pdf_path: Path to the PDF file.

        Yields:
            Dicts with keys: page_number, text, blocks.
            Each block has: text, bbox, font_size, is_bold.
        """
