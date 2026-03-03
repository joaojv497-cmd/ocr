# Common Errors and How to Avoid Them

1. **Loss of spatial context**: Always preserve bounding boxes (bbox) through every transformation.
2. **Incorrect reading order**: Use `ReadingOrderReconstructor` for multi-column layouts.
3. **Excessive LLM dependency**: Use `SemanticChunker` as fallback when LLM is unavailable.
4. **Chunks too large**: Respect token limits — `SemanticChunker` enforces `MAX_CHUNK_CHARS`.
5. **Lost metadata**: Propagate metadata through the pipeline using `**page` dict spread.
6. **OCR without preprocessing**: Always apply `ImagePreprocessor` before Tesseract.
7. **Ignoring hybrid PDFs**: Use `PDFTypeDetector` to route pages to the correct extractor.
8. **LLM output not validated**: `LLMChunker._parse_llm_response` cleans markdown and validates JSON.
9. **Headers/footers not removed**: Run `NoiseRemover` before chunking.
10. **Missing logging**: All modules use `logging.getLogger(__name__)`.
