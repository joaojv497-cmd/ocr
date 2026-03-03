"""Integration tests for the full document processing pipeline."""
import pytest
from unittest.mock import patch, MagicMock
from ocr_pypi.detection.pdf_type_detector import PDFTypeDetector
from ocr_pypi.extraction.pymupdf_extractor import PyMuPDFExtractor
from ocr_pypi.layout.layout_analyzer import LayoutAnalyzer
from ocr_pypi.layout.reading_order_reconstructor import ReadingOrderReconstructor
from ocr_pypi.cleaning.noise_remover import NoiseRemover
from ocr_pypi.semantic.section_classifier import SectionClassifier
from ocr_pypi.output.json_formatter import JSONFormatter
from ocr_pypi.models.document import Chunk


class TestFullPipeline:
    """Integration tests for the complete pipeline components working together."""

    def test_layout_then_noise_removal(self):
        """Test layout analysis followed by noise removal."""
        blocks = [
            {"text": "Page Header", "bbox": {"x0": 50, "y0": 5, "x1": 500, "y1": 30}, "font_size": 10, "is_bold": False},
            {"text": "Main content paragraph with substantial text.", "bbox": {"x0": 50, "y0": 200, "x1": 500, "y1": 220}, "font_size": 12, "is_bold": False},
        ]
        pages_data = [
            {"page_number": i, "text": "", "blocks": list(blocks)} for i in range(1, 4)
        ]

        analyzer = LayoutAnalyzer()
        reconstructor = ReadingOrderReconstructor()
        remover = NoiseRemover()

        analyzed = []
        for page in pages_data:
            annotated_blocks = analyzer.analyze(page["blocks"])
            sorted_blocks = reconstructor.reconstruct(annotated_blocks)
            analyzed.append({**page, "blocks": sorted_blocks})

        cleaned = remover.remove_noise(analyzed)
        assert len(cleaned) == 3

    def test_classifier_then_formatter(self):
        """Test section classification followed by JSON formatting."""
        classifier = SectionClassifier()
        formatter = JSONFormatter()

        blocks = [
            {"text": "DOS FATOS", "font_size": 14, "is_bold": True, "bbox": {"x0": 50, "y0": 100, "x1": 200, "y1": 120}},
            {"text": "O autor narra que...", "font_size": 12, "is_bold": False, "bbox": {"x0": 50, "y0": 140, "x1": 500, "y1": 300}},
        ]
        classified = classifier.classify_blocks(blocks)

        chunks = [
            Chunk(content="DOS FATOS\nO autor narra que...", page_numbers=[1], chunk_index=0, metadata={"section_name": "fatos"})
        ]

        output = formatter.format(
            chunks=chunks,
            metadata={"filename": "test.pdf", "document_type": "peticao_inicial"},
            sections=classified,
        )

        assert "document_id" in output
        assert "chunks" in output
        assert "metadata" in output
        assert output["statistics"]["total_chunks"] == 1
        assert output["chunks"][0]["embedding_ready"] is True

    def test_digital_pdf_extraction(self, tmp_path):
        """Test extraction of a digital PDF."""
        import fitz
        pdf_path = str(tmp_path / "test_digital.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "DOS FATOS\nEste é um documento de teste jurídico.")
        doc.save(pdf_path)
        doc.close()

        detector = PDFTypeDetector()
        extractor = PyMuPDFExtractor()

        type_result = detector.detect_type(pdf_path)
        pages = extractor.extract_with_layout(pdf_path)

        assert len(pages) == 1
        assert pages[0]["page_number"] == 1
        assert "DOS FATOS" in pages[0]["text"] or pages[0]["text"].strip()
