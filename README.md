# README.md

Overview

This repository adds a modular, open-source OCR pipeline focused on Brazilian legal documents (petição inicial, contestação, sentença, recursos). The pipeline:

- Detects whether a PDF is digital or scanned.
- Extracts text directly from digital PDFs using pdfplumber while preserving coordinates.
- Converts scanned PDFs to images and runs OCR using Tesseract (pytesseract) with coordinated output.
- Uses OpenCV for preprocessing (binarization, deskew, denoising).
- Reconstructs layout (bounding boxes, blocks, pages) and reading order.
- Removes headers/footers via heuristics.
- Performs semantic section identification (rule-based + pluggable ML model).
- Produces structured JSON output ready for embeddings / RAG.

Usage

Install dependencies (recommend a virtualenv):

pip install -r requirements.txt

Install Tesseract on your system and ensure `tesseract` is on PATH.

Example:

python cli.py process input.pdf output.json
