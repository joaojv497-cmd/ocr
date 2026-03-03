"""Unit tests for section classifier."""
import pytest
from ocr_pypi.semantic.section_classifier import SectionClassifier
from ocr_pypi.models.document import SectionType


class TestSectionClassifier:
    def test_classify_fatos_section(self):
        """Test classification of 'DOS FATOS' section title."""
        classifier = SectionClassifier()
        blocks = [
            {"text": "DOS FATOS", "font_size": 14, "is_bold": True, "bbox": {"x0": 50, "y0": 100, "x1": 200, "y1": 120}},
        ]
        result = classifier.classify_blocks(blocks)
        assert result[0]["section_type"] == SectionType.FATOS.value
        assert result[0]["section_confidence"] > 0.5

    def test_classify_pedidos_section(self):
        """Test classification of 'DOS PEDIDOS' section title."""
        classifier = SectionClassifier()
        blocks = [
            {"text": "DOS PEDIDOS", "font_size": 14, "is_bold": True, "bbox": {"x0": 50, "y0": 100, "x1": 200, "y1": 120}},
        ]
        result = classifier.classify_blocks(blocks)
        assert result[0]["section_type"] == SectionType.PEDIDOS.value

    def test_classify_unknown_returns_custom(self):
        """Test that unknown text returns CUSTOM section type."""
        classifier = SectionClassifier()
        blocks = [
            {"text": "Texto aleatório sem correspondência jurídica conhecida xyz.", "font_size": 12, "is_bold": False, "bbox": {"x0": 50, "y0": 100, "x1": 500, "y1": 120}},
        ]
        result = classifier.classify_blocks(blocks)
        assert result[0]["section_type"] == SectionType.CUSTOM.value

    def test_empty_blocks(self):
        """Test handling of empty blocks list."""
        classifier = SectionClassifier()
        result = classifier.classify_blocks([])
        assert result == []

    def test_adds_required_fields(self):
        """Test that section_type and section_confidence are added."""
        classifier = SectionClassifier()
        blocks = [{"text": "Some text", "font_size": 12, "is_bold": False, "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20}}]
        result = classifier.classify_blocks(blocks)
        assert "section_type" in result[0]
        assert "section_confidence" in result[0]
