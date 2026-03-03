"""Tesseract OCR engine implementation."""
import logging
from typing import List
import pytesseract
from PIL import Image
from ocr_pypi.models.ocr_types import OCRBlock, OCRLevel
from ocr_pypi.models.document import BoundingBox
from ocr_pypi.ocr.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """OCR engine backed by Tesseract via pytesseract."""

    def extract_with_layout(
        self,
        image: Image.Image,
        lang: str = "por",
    ) -> List[OCRBlock]:
        """
        Extract text with bounding-box layout from an image using Tesseract.

        Args:
            image: PIL Image to process.
            lang: Tesseract language string (default 'por').

        Returns:
            List of OCRBlock instances with coordinates and confidence.
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            logger.error(f"Tesseract failed: {e}")
            return []

        blocks: List[OCRBlock] = []
        n = len(data["level"])
        for i in range(n):
            text = data["text"][i]
            if not str(text).strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < 0:
                continue

            level_int = int(data["level"][i])
            level = OCRLevel(min(level_int, 5))

            bbox = BoundingBox(
                x0=float(data["left"][i]),
                y0=float(data["top"][i]),
                x1=float(data["left"][i]) + float(data["width"][i]),
                y1=float(data["top"][i]) + float(data["height"][i]),
            )

            blocks.append(OCRBlock(
                level=level,
                text=str(text),
                confidence=conf / 100.0,
                bbox=bbox,
                page_number=int(data["page_num"][i]),
                block_number=int(data["block_num"][i]),
                paragraph_number=int(data["par_num"][i]),
                line_number=int(data["line_num"][i]),
                word_number=int(data["word_num"][i]),
            ))

        logger.debug(f"Tesseract extracted {len(blocks)} blocks")
        return blocks
