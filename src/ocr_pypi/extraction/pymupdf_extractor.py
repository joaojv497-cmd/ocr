"""PyMuPDF-based text extractor for digital PDFs."""
import logging
from typing import List, Dict, Any
import fitz
from ocr_pypi.extraction.text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class PyMuPDFExtractor(TextExtractor):
    """Extract text with layout information from digital PDFs using PyMuPDF."""

    def extract_with_layout(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract structured text from a digital PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of page dicts with text and block-level layout info.
        """
        pages_data: List[Dict[str, Any]] = []
        pdf = fitz.open(pdf_path)

        for page_idx in range(len(pdf)):
            page = pdf.load_page(page_idx)
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            page_text_parts = []
            blocks_info: List[Dict[str, Any]] = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue

                block_text_parts = []
                font_sizes = []
                is_bold_flags = []

                for line in block.get("lines", []):
                    line_parts = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            line_parts.append(span_text)
                            font_sizes.append(span.get("size", 12))
                            flags = span.get("flags", 0)
                            is_bold_flags.append(bool(flags & 2**4))  # bold flag
                    if line_parts:
                        block_text_parts.append(" ".join(line_parts))

                block_text = "\n".join(block_text_parts)
                if not block_text.strip():
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
                is_bold = any(is_bold_flags)

                blocks_info.append({
                    "text": block_text,
                    "bbox": {
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                    },
                    "font_size": avg_font_size,
                    "is_bold": is_bold,
                })
                page_text_parts.append(block_text)

            full_text = "\n\n".join(page_text_parts)
            pages_data.append({
                "page_number": page_idx + 1,
                "text": full_text,
                "blocks": blocks_info,
            })

        pdf.close()
        logger.info(f"PyMuPDF extracted {len(pages_data)} pages from {pdf_path}")
        return pages_data
