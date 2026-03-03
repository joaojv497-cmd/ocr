# System Architecture

## Overview
This system implements a complete document processing pipeline for Brazilian legal documents (petições, sentenças, contratos, recursos).

## Modules

### `detection/`
- `pdf_type_detector.py`: Detects whether PDF is digital, scanned, or hybrid.

### `preprocessing/`
- `image_preprocessor.py`: Improves image quality before OCR (binarization, deskew, noise removal, DPI normalization).

### `ocr/`
- `ocr_engine.py`: Abstract interface for OCR engines.
- `tesseract_engine.py`: Tesseract implementation with layout preservation.
- `ocr_block.py`: Re-export of OCR data models.

### `extraction/`
- `text_extractor.py`: Abstract interface for text extractors.
- `pymupdf_extractor.py`: PyMuPDF-based extractor for digital PDFs.

### `layout/`
- `reading_order_reconstructor.py`: Reconstructs correct reading order using spatial sorting.
- `layout_analyzer.py`: Classifies blocks as headers, footers, titles, paragraphs, lists.

### `cleaning/`
- `noise_remover.py`: Removes headers, footers, page numbers, and noise.
- `header_footer_detector.py`: Detects repeated patterns across pages.

### `semantic/`
- `section_classifier.py`: Hybrid regex + layout classifier for legal section types.
- `legal_patterns.py`: Brazilian legal document regex pattern library.

### `chunking/`
- `llm_chunker.py`: LLM-based structured chunking.
- `semantic_chunker.py`: Embedding-based chunking (no LLM required).

### `output/`
- `json_formatter.py`: Standardized JSON output formatter.

### `services/`
- `document_processor.py`: Full pipeline orchestrator.

## Data Flow
```
PDF → PDFTypeDetector → [Digital: PyMuPDFExtractor | Scanned: ImagePreprocessor + TesseractEngine]
    → LayoutAnalyzer + ReadingOrderReconstructor
    → NoiseRemover
    → SectionClassifier
    → LLMChunker / SemanticChunker
    → JSONFormatter → Output
```
