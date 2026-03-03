"""Unit tests for Tesseract OCR engine."""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from ocr_pypi.ocr.tesseract_engine import TesseractEngine
from ocr_pypi.models.ocr_types import OCRBlock, OCRLevel


class TestTesseractEngine:
    def test_returns_list(self):
        """Test that extract_with_layout returns a list."""
        engine = TesseractEngine()
        image = Image.new("RGB", (100, 100), color="white")

        mock_data = {
            "level": [5],
            "text": ["Hello"],
            "conf": ["90"],
            "left": [10],
            "top": [20],
            "width": [50],
            "height": [15],
            "page_num": [1],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            result = engine.extract_with_layout(image)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], OCRBlock)
        assert result[0].text == "Hello"
        assert result[0].confidence == 0.9

    def test_filters_empty_text(self):
        """Test that blocks with empty text are filtered out."""
        engine = TesseractEngine()
        image = Image.new("RGB", (100, 100), color="white")

        mock_data = {
            "level": [5, 5],
            "text": ["", "World"],
            "conf": ["90", "85"],
            "left": [0, 10],
            "top": [0, 20],
            "width": [50, 50],
            "height": [15, 15],
            "page_num": [1, 1],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            result = engine.extract_with_layout(image)

        assert len(result) == 1
        assert result[0].text == "World"

    def test_handles_tesseract_error(self):
        """Test graceful handling of Tesseract errors."""
        engine = TesseractEngine()
        image = Image.new("RGB", (100, 100), color="white")

        with patch("pytesseract.image_to_data", side_effect=Exception("Tesseract error")):
            result = engine.extract_with_layout(image)

        assert result == []
