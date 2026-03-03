# Pipeline Description

## Stage 1: PDF Type Detection
`PDFTypeDetector.detect_type(pdf_path)` → `PDFTypeResult`
- Analyzes each page for text content
- Returns `PDFType.DIGITAL`, `PDFType.SCANNED`, or `PDFType.HYBRID`

## Stage 2: Text Extraction
### Digital PDFs
`PyMuPDFExtractor.extract_with_layout(pdf_path)` → `List[Dict]`
- Extracts text with font size, bold flags, and bounding boxes

### Scanned PDFs
`ImagePreprocessor.preprocess(image)` → preprocessed image
`TesseractEngine.extract_with_layout(image)` → `List[OCRBlock]`

## Stage 3: Layout Analysis
`LayoutAnalyzer.analyze(blocks)` → blocks with `area_type`
`ReadingOrderReconstructor.reconstruct(blocks)` → sorted blocks

## Stage 4: Noise Removal
`NoiseRemover.remove_noise(pages_data)` → cleaned pages
- Removes repeated headers/footers and page numbers

## Stage 5: Section Classification
`SectionClassifier.classify_blocks(blocks)` → blocks with `section_type`
- Hybrid: regex patterns + layout heuristics

## Stage 6: LLM Chunking
`LLMChunker.chunk(pages, template_name)` → `List[Chunk]`
- Falls back to per-page chunks if LLM fails

## Stage 7: JSON Formatting
`JSONFormatter.format(chunks, metadata)` → structured dict
- Ready for embedding and RAG systems
