# Future Improvements

1. **Pre-trained layout models**: Use LayoutLM/LayoutLMv3 for block classification.
2. **Fine-tuned OCR**: Train Tesseract on Brazilian legal documents for higher accuracy.
3. **Document clustering**: Group similar documents automatically.
4. **Named Entity Recognition (NER)**: Extract parties, laws, values using spaCy/transformers.
5. **Table extraction**: Dedicated parser for tables in documents.
6. **Multi-language support**: Spanish/English document support.
7. **Embedding cache**: Avoid reprocessing identical documents.
8. **Async pipeline**: Use Celery/RQ for batch processing.
9. **Quality metrics**: Calculate OCR/extraction quality scores.
10. **Web validation interface**: Allow human correction of chunks.
