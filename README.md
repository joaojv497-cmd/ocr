# OCR Service

Serviço de OCR com chunking inteligente usando IA.

## Features

- OCR de PDFs com Tesseract
- Chunking inteligente com IA (semantic_ai)
- Detecção automática de áreas do documento
- Streaming gRPC

## Instalação

```bash
# Instalar Tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-por  # Ubuntu/Debian
brew install tesseract tesseract-lang                  # macOS

# Instalar dependências
poetry install

# Configurar
cp .env.example .env
# Editar .env