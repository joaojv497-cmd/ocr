"""PDF type detection module."""
import logging
from typing import List
import fitz  # PyMuPDF
from ocr_pypi.models.pdf_types import PDFType, PDFTypeResult

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS_PER_PAGE = 50
TEXT_PAGE_THRESHOLD = 0.5  # fraction of pages that must have text to be "digital"


class PDFTypeDetector:
    """Detects whether a PDF is digital (with text), scanned (images only), or hybrid."""

    def detect_type(self, pdf_path: str) -> PDFTypeResult:
        """
        Detect the type of a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            PDFTypeResult with type, page lists, and confidence.
        """
        pdf = fitz.open(pdf_path)
        total_pages = len(pdf)
        digital_pages: List[int] = []
        scanned_pages: List[int] = []
        total_text_chars = 0
        total_image_area = 0.0
        total_page_area = 0.0

        for i in range(total_pages):
            page = pdf.load_page(i)
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            text = page.get_text().strip()
            text_chars = len(text)
            total_text_chars += text_chars

            # Image area on this page
            image_list = page.get_images(full=True)
            image_area = 0.0
            for img in image_list:
                # Use clip rect if available; otherwise estimate from page area
                rects = page.get_image_rects(img[0])
                for rect in rects:
                    image_area += rect.width * rect.height

            total_image_area += image_area
            total_page_area += page_area

            if text_chars >= MIN_TEXT_CHARS_PER_PAGE:
                digital_pages.append(i + 1)
            else:
                scanned_pages.append(i + 1)

        pdf.close()

        text_density = total_text_chars / max(total_pages, 1)
        image_density = total_image_area / max(total_page_area, 1)
        digital_ratio = len(digital_pages) / max(total_pages, 1)

        if digital_ratio >= TEXT_PAGE_THRESHOLD and len(scanned_pages) == 0:
            pdf_type = PDFType.DIGITAL
            confidence = min(1.0, digital_ratio)
        elif len(digital_pages) == 0:
            pdf_type = PDFType.SCANNED
            confidence = min(1.0, 1.0 - digital_ratio)
        else:
            pdf_type = PDFType.HYBRID
            confidence = 0.8

        logger.info(
            f"PDF type detected: {pdf_type.value} "
            f"(digital={len(digital_pages)}, scanned={len(scanned_pages)}, "
            f"total={total_pages}, confidence={confidence:.2f})"
        )

        return PDFTypeResult(
            pdf_type=pdf_type,
            digital_pages=digital_pages,
            scanned_pages=scanned_pages,
            total_pages=total_pages,
            text_density=text_density,
            image_density=image_density,
            confidence=confidence,
        )
