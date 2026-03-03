"""Abstract OCR engine interface."""
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from ocr_pypi.models.ocr_types import OCRBlock


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_with_layout(
        self,
        image: Image.Image,
        lang: str = "por",
    ) -> List[OCRBlock]:
        """
        Extract text with layout information from an image.

        Args:
            image: PIL Image to process.
            lang: Language code (e.g. 'por' for Portuguese).

        Returns:
            List of OCRBlock with text, bounding boxes, and confidence.
        """
