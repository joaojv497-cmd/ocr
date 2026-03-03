"""Unit tests for PDF type detector."""
import pytest
from unittest.mock import MagicMock, patch
from ocr_pypi.detection.pdf_type_detector import PDFTypeDetector
from ocr_pypi.models.pdf_types import PDFType


class TestPDFTypeDetector:
    def test_detect_digital_pdf(self, tmp_path):
        """Test detection of a digital PDF (with text)."""
        import fitz
        pdf_path = str(tmp_path / "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This is a test document with enough text to be digital. " * 5)
        doc.save(pdf_path)
        doc.close()

        detector = PDFTypeDetector()
        result = detector.detect_type(pdf_path)
        assert result.pdf_type == PDFType.DIGITAL
        assert result.total_pages == 1
        assert len(result.digital_pages) == 1

    def test_detect_empty_pdf_as_scanned(self, tmp_path):
        """Test that a PDF with no text is detected as scanned."""
        import fitz
        pdf_path = str(tmp_path / "empty.pdf")
        doc = fitz.open()
        doc.new_page()  # blank page
        doc.save(pdf_path)
        doc.close()

        detector = PDFTypeDetector()
        result = detector.detect_type(pdf_path)
        assert result.pdf_type == PDFType.SCANNED
        assert result.total_pages == 1
        assert len(result.scanned_pages) == 1

    def test_result_fields(self, tmp_path):
        """Test that result has all required fields."""
        import fitz
        pdf_path = str(tmp_path / "test.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(pdf_path)
        doc.close()

        detector = PDFTypeDetector()
        result = detector.detect_type(pdf_path)
        assert hasattr(result, "pdf_type")
        assert hasattr(result, "digital_pages")
        assert hasattr(result, "scanned_pages")
        assert hasattr(result, "total_pages")
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0
