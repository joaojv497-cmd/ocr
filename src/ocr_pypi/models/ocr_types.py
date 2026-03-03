"""OCR data models."""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from ocr_pypi.models.document import BoundingBox


class OCRLevel(Enum):
    PAGE = 1
    BLOCK = 2
    PARAGRAPH = 3
    LINE = 4
    WORD = 5


@dataclass
class OCRBlock:
    level: OCRLevel
    text: str
    confidence: float
    bbox: BoundingBox
    page_number: int
    block_number: int
    paragraph_number: int
    line_number: int
    word_number: int
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None
