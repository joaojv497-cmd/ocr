"""Document processor orchestrating the full OCR pipeline."""
import os
import logging
from typing import Generator, List, Dict, Any, Optional

import fitz
import pytesseract
from PIL import Image
import io

from commons_pypi.storage import get_temp_file
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
from ocr_pypi.vision.smart_image_detector import SmartImageDetector
from ocr_pypi.vision.image_descriptor import ImageDescriptor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates the full document processing pipeline.

    Pipeline:
    1. Detect PDF type (digital/scanned/hybrid)
    2a. Digital: extract text directly (PyMuPDF)
    2b. Scanned: preprocess + OCR (Tesseract)
    3. Analyze layout and reconstruct reading order
    4. Remove headers/footers/noise
    5. Classify sections semantically
    6. Detect and describe images with LLM (optional)
    7. Apply chunking with the configured strategy (page, semantic, or paragraph)
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
        self._image_detector = SmartImageDetector()
        self._image_descriptor: Optional[ImageDescriptor] = None

    def process(
        self,
        bucket: str,
        file_key: str,
        chunk_options: Dict[str, Any] = None,
    ) -> Generator[Dict, None, None]:
        """
        Process a document and yield chunks and a final result.

        chunk_options may contain (all optional, fallback to env/settings):
            - chunk_strategy: 'page' (default) | 'semantic' | 'paragraph'
            - llm_provider: provider name (used for image description)
            - llm_model: model name (used for image description)
            - llm_api_key: API key (used for image description)
            - llm_temperature: temperature (used for image description)
            - llm_max_tokens: max tokens (used for image description)
            - chunk_size: max chunk chars (semantic/paragraph strategies)
            - chunk_overlap: overlap chars (paragraph strategy)
            - min_chunk_size: minimum chunk chars (semantic/paragraph)
            - embedding_model: sentence-transformer model name (semantic strategy)
            - similarity_threshold: cosine threshold (semantic strategy)
            - detect_images: whether to detect and describe images (default: True)

        Yields:
            Dict with 'type' in ('chunk', 'complete', 'error')
        """
        temp_path = None
        chunk_options = chunk_options or {}

        try:
            # 1. Download
            storage = get_storage(bucket)
            temp_path = get_temp_file(".pdf")
            storage.download_file(file_key, temp_path)

            # 2. Detect PDF type
            type_result = self._detector.detect_type(temp_path)

            # 3. Extract text page by page, applying layout analysis per page
            if type_result.pdf_type == PDFType.DIGITAL:
                page_iter = self._extractor.extract_with_layout(temp_path)
            elif type_result.pdf_type == PDFType.SCANNED:
                page_iter = self._ocr_extract(temp_path)
            else:  # HYBRID
                page_iter = iter(self._hybrid_extract(temp_path, type_result))

            pages_data = []
            for page in page_iter:
                page = self._apply_layout_analysis_single(page)
                pages_data.append(page)

            # 4. Noise removal (requires all pages for header/footer detection)
            pages_data = self._noise_remover.remove_noise(pages_data)

            # 5. Section classification per page
            classified_pages = []
            for page in pages_data:
                page = self._apply_section_classification_single(page)
                classified_pages.append(page)
            pages_data = classified_pages

            # 6. Image detection and description (optional)
            # Image descriptions are emitted as chunk events and also appended
            # to each page's text so they are included in text chunking.
            detect_images = chunk_options.get("detect_images", True)
            if detect_images:
                pages_data = yield from self._apply_image_detection(
                    temp_path, pages_data, chunk_options
                )

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
        return [self._apply_layout_analysis_single(page) for page in pages_data]

    def _apply_layout_analysis_single(self, page: Dict) -> Dict:
        """Apply layout analysis and reading order reconstruction to a single page."""
        blocks = page.get("blocks", [])
        if blocks:
            blocks = self._layout_analyzer.analyze(blocks)
            blocks = self._order_reconstructor.reconstruct(blocks)
            # Re-build text from sorted blocks
            text = "\n\n".join(
                b["text"] for b in blocks if b.get("text", "").strip()
            )
            return {**page, "blocks": blocks, "text": text}
        return page

    def _apply_section_classification(self, pages_data: List[Dict]) -> List[Dict]:
        """Apply semantic section classification to all page blocks."""
        return [self._apply_section_classification_single(page) for page in pages_data]

    def _apply_section_classification_single(self, page: Dict) -> Dict:
        """Apply semantic section classification to a single page's blocks."""
        blocks = page.get("blocks", [])
        if blocks:
            blocks = self._section_classifier.classify_blocks(blocks)
        return {**page, "blocks": blocks}

    def _chunk_with_strategy(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, None]:
        """Dispatch chunking to the configured strategy."""
        strategy = chunk_options.get("chunk_strategy", "page")
        yield from ChunkingStrategy.chunk(pages_data, strategy, chunk_options)

    def _get_image_descriptor(self, chunk_options: Dict[str, Any]) -> ImageDescriptor:
        """Lazily creates or returns the shared ImageDescriptor instance."""
        if self._image_descriptor is None:
            self._image_descriptor = ImageDescriptor(
                provider_name=chunk_options.get("llm_provider"),
                api_key=chunk_options.get("llm_api_key"),
                model=chunk_options.get("llm_model"),
                temperature=chunk_options.get("llm_temperature"),
                max_tokens=chunk_options.get("llm_max_tokens"),
            )
        return self._image_descriptor

    def _apply_image_detection(
        self,
        pdf_path: str,
        pages_data: List[Dict],
        chunk_options: Dict[str, Any],
    ) -> Generator[Dict, None, List[Dict]]:
        """
        Detect images in the PDF and enrich each page with LLM-generated descriptions.

        Yields a chunk event for each image description so callers receive image
        descriptions as first-class chunks alongside text chunks.  Descriptions are
        also appended to the page text so they are included in downstream text chunking.

        Returns the enriched pages_data list (via generator return value).
        """
        from ocr_pypi.models.document import Chunk

        try:
            images = self._image_detector.detect_images(pdf_path)
        except Exception as e:
            logger.warning(f"Falha na detecção de imagens: {e}")
            return pages_data

        if not images:
            return pages_data

        logger.info(f"Detectadas {len(images)} imagens; gerando descrições...")

        descriptions_by_page: Dict[int, list] = {}
        image_chunk_idx = 0
        try:
            descriptor = self._get_image_descriptor(chunk_options)
            for desc in descriptor.describe_images_iter(images):
                descriptions_by_page.setdefault(desc.page_number, []).append(desc)
                yield {
                    "type": "chunk",
                    "data": Chunk(
                        content=(
                            f"[IMAGEM {desc.image_info.image_index + 1} - "
                            f"Página {desc.page_number}]: {desc.description}"
                        ),
                        page_numbers=[desc.page_number],
                        chunk_index=image_chunk_idx,
                        metadata={
                            "chunking_method": "image_description",
                            "image_index": desc.image_info.image_index,
                            "width": desc.image_info.width,
                            "height": desc.image_info.height,
                        },
                        detected_areas=[],
                    ),
                }
                image_chunk_idx += 1
        except Exception as e:
            logger.warning(f"Falha na descrição de imagens: {e}")
            return pages_data

        # Enrich pages_data with collected descriptions
        result = []
        for page in pages_data:
            page_num = page["page_number"]
            page_descs = descriptions_by_page.get(page_num, [])

            if not page_descs:
                result.append(page)
                continue

            # Build image description text blocks to append to page text
            desc_blocks = []
            for desc in page_descs:
                block = (
                    f"[IMAGEM {desc.image_info.image_index + 1} - "
                    f"Página {page_num}]: {desc.description}"
                )
                desc_blocks.append(block)

            enriched_text = page.get("text", "")
            if desc_blocks:
                enriched_text = enriched_text + "\n\n" + "\n".join(desc_blocks)

            result.append({
                **page,
                "text": enriched_text,
                "image_descriptions": [
                    {
                        "image_index": d.image_info.image_index,
                        "description": d.description,
                        "page_number": d.page_number,
                        "width": d.image_info.width,
                        "height": d.image_info.height,
                    }
                    for d in page_descs
                ],
            })

        return result

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