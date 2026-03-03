"""Document processor orchestrating the full OCR pipeline."""
import os
import uuid
import time
import logging
from typing import Generator, List, Dict, Any, Optional

import fitz
import pytesseract
from PIL import Image
import io

from commons_pypi.storage import get_temp_file
from ocr_pypi.chunking.llm_chunker import LLMChunker
from ocr_pypi.chunking.chunking_strategy import ChunkingStrategy
from ocr_pypi.storage import get_storage
from ocr_pypi.config import settings
from ocr_pypi.detection.pdf_type_detector import PDFTypeDetector
from ocr_pypi.models.pdf_types import PDFType
from ocr_pypi.preprocessing.image_preprocessor import ImagePreprocessor
from ocr_pypi.extraction.pymupdf_extractor import PyMuPDFExtractor
from ocr_pypi.layout.layout_analyzer import LayoutAnalyzer
from ocr_pypi.layout.reading_order_reconstructor import ReadingOrderReconstructor
from ocr_pypi.cleaning.noise_remover import NoiseRemover
from ocr_pypi.semantic.section_classifier import SectionClassifier
from ocr_pypi.output.json_formatter import JSONFormatter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates the full document processing pipeline.

    Pipeline:
    1. Detect PDF type (digital/scanned/hybrid)
    2a. Digital: extract text directly (PyMuPDF)
    2b. Scanned: preprocess + OCR (Tesseract)
    3. Analyze layout and reconstruct reading order
    4. Remove headers/footers/noise
    5. Classify sections semantically (hybrid)
    6. Apply LLM chunking with appropriate template
    7. Validate and enrich chunks
    8. Generate structured JSON output
    """

    def __init__(self, language: str = "por"):
        self.language = language
        self._detector = PDFTypeDetector()
        self._preprocessor = ImagePreprocessor()
        self._extractor = PyMuPDFExtractor()
        self._layout_analyzer = LayoutAnalyzer()
        self._order_reconstructor = ReadingOrderReconstructor()
        self._noise_remover = NoiseRemover()
        self._section_classifier = SectionClassifier()
        self._formatter = JSONFormatter()

    def process(
        self,
        bucket: str,
        file_key: str,
        chunk_options: Dict[str, Any] = None,
    ) -> Generator[Dict, None, None]:
        """
        Process a document and yield progress events, chunks, and a final result.

        chunk_options may contain (all optional, fallback to env/settings):
            - chunk_strategy: 'llm' (default) | 'semantic' | 'paragraph' | 'hybrid'
            - template: template name
            - template_instance: pre-built DocumentTemplate instance (overrides template)
            - llm_provider: provider name
            - llm_model: model name
            - llm_api_key: API key
            - llm_temperature: temperature
            - llm_max_tokens: max tokens
            - chunk_size: max chunk chars (semantic/paragraph strategies)
            - chunk_overlap: overlap chars (paragraph strategy)
            - min_chunk_size: minimum chunk chars (semantic/paragraph)
            - embedding_model: sentence-transformer model name (semantic strategy)
            - similarity_threshold: cosine threshold (semantic strategy)
            - preserve_structure: preserve document hierarchy (advanced)
            - max_chunks_per_section: max chunks per section (advanced)
            - metadata_fields: extra metadata fields to include (dict, from chunk_metadata_fields JSON)

        Yields:
            Dict with 'type' in ('progress', 'chunk', 'complete', 'error')
        """
        temp_path = None
        chunk_options = chunk_options or {}
        start_time = time.time()
        document_id = str(uuid.uuid4())

        try:
            # 1. Download
            storage = get_storage(bucket)
            temp_path = get_temp_file(".pdf")
            storage.download_file(file_key, temp_path)

            yield {"type": "progress", "stage": "download_complete"}

            # 2. Detect PDF type
            type_result = self._detector.detect_type(temp_path)
            yield {
                "type": "progress",
                "stage": "type_detection_complete",
                "pdf_type": type_result.pdf_type.value,
                "total_pages": type_result.total_pages,
            }

            # 3. Extract text (digital or OCR path)
            if type_result.pdf_type == PDFType.DIGITAL:
                pages_data = self._extractor.extract_with_layout(temp_path)
            elif type_result.pdf_type == PDFType.SCANNED:
                pages_data = list(self._ocr_extract(temp_path))
            else:  # HYBRID
                pages_data = self._hybrid_extract(temp_path, type_result)

            yield {
                "type": "progress",
                "stage": "text_extraction_complete",
                "total_pages": len(pages_data),
            }

            # 4. Layout analysis + reading order reconstruction
            pages_data = self._apply_layout_analysis(pages_data)
            yield {"type": "progress", "stage": "layout_analysis_complete"}

            # 5. Noise removal
            pages_data = self._noise_remover.remove_noise(pages_data)
            yield {"type": "progress", "stage": "noise_removal_complete"}

            # 6. Section classification
            pages_data = self._apply_section_classification(pages_data)
            yield {"type": "progress", "stage": "section_classification_complete"}

            # 7. Chunking (strategy-aware)
            yield from self._chunk_with_strategy(pages_data, chunk_options)

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            yield {"type": "error", "message": str(e)}
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _make_simple_block(text: str, width: int, height: int) -> Dict[str, Any]:
        """Create a single full-page block from raw OCR text."""
        return {
            "text": text,
            "bbox": {"x0": 0, "y0": 0, "x1": width, "y1": height},
            "font_size": 12,
            "is_bold": False,
        }

    def _ocr_extract(self, pdf_path: str) -> Generator[Dict, None, None]:
        """Extract text via Tesseract OCR with image preprocessing."""
        pdf = fitz.open(pdf_path)
        for i in range(len(pdf)):
            page = pdf.load_page(i)
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            image = self._preprocessor.preprocess(image, dpi=300)
            text = pytesseract.image_to_string(image, lang=self.language)
            yield {
                "page_number": i + 1,
                "text": text,
                "blocks": [self._make_simple_block(text, pix.width, pix.height)],
            }
        pdf.close()

    def _hybrid_extract(self, pdf_path: str, type_result) -> List[Dict]:
        """Extract text using digital or OCR path per page."""
        digital_set = set(type_result.digital_pages)

        # Get digital pages via PyMuPDF
        digital_pages = self._extractor.extract_with_layout(pdf_path)
        digital_map = {p["page_number"]: p for p in digital_pages}

        pdf = fitz.open(pdf_path)
        pages_data = []
        for i in range(len(pdf)):
            page_num = i + 1
            if page_num in digital_set and page_num in digital_map:
                pages_data.append(digital_map[page_num])
            else:
                page = pdf.load_page(i)
                pix = page.get_pixmap(dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                image = self._preprocessor.preprocess(image, dpi=300)
                text = pytesseract.image_to_string(image, lang=self.language)
                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "blocks": [self._make_simple_block(text, pix.width, pix.height)],
                })
        pdf.close()
        return pages_data

    def _apply_layout_analysis(self, pages_data: List[Dict]) -> List[Dict]:
        """Apply layout analysis and reading order reconstruction to all pages."""
        result = []
        for page in pages_data:
            blocks = page.get("blocks", [])
            if blocks:
                blocks = self._layout_analyzer.analyze(blocks)
                blocks = self._order_reconstructor.reconstruct(blocks)
                # Re-build text from sorted blocks
                text = "\n\n".join(
                    b["text"] for b in blocks if b.get("text", "").strip()
                )
                result.append({**page, "blocks": blocks, "text": text})
            else:
                result.append(page)
        return result

    def _apply_section_classification(self, pages_data: List[Dict]) -> List[Dict]:
        """Apply semantic section classification to all page blocks."""
        result = []
        for page in pages_data:
            blocks = page.get("blocks", [])
            if blocks:
                blocks = self._section_classifier.classify_blocks(blocks)
            result.append({**page, "blocks": blocks})
        return result

    def _chunk_with_strategy(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """Dispatch chunking to the configured strategy."""
        # Chunking está SEMPRE habilitado
        # Sempre usa estratégia LLM
        strategy = "llm"
        yield from ChunkingStrategy.chunk(pages_data, strategy, chunk_options)

    def _chunk_with_llm(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """Perform LLM-based chunking with dynamic provider and template."""
        provider_kwargs = {
            k: v for k, v in {
                "provider_name": chunk_options.get("llm_provider"),
                "api_key": chunk_options.get("llm_api_key"),
                "model": chunk_options.get("llm_model"),
                "temperature": chunk_options.get("llm_temperature"),
                "max_tokens": chunk_options.get("llm_max_tokens"),
            }.items() if v is not None
        }

        chunker = LLMChunker(**provider_kwargs)
        template_name = chunk_options.get("template") or None

        yield {
            "type": "progress",
            "stage": "llm_chunking_starting",
            "template": template_name,
            "provider": chunker.provider.provider_name,
            "model": chunker.provider.model,
        }

        chunks = chunker.chunk(pages=pages_data, template_name=template_name)

        yield {
            "type": "progress",
            "stage": "chunking_complete",
            "total_chunks": len(chunks),
        }

        for chunk in chunks:
            yield {"type": "chunk", "data": chunk}

        yield {"type": "complete", "total_chunks": len(chunks)}

    def _extract_text(self, pdf_path: str) -> Generator[Dict, None, None]:
        """
        Extract text from PDF.
        Tries native extraction first; falls back to Tesseract OCR.
        """
        pdf = fitz.open(pdf_path)
        for i in range(len(pdf)):
            page = pdf.load_page(i)
            text = page.get_text()
            if not text.strip():
                pix = page.get_pixmap(dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(image, lang=self.language)
            yield {"page_number": i + 1, "text": text}
        pdf.close()