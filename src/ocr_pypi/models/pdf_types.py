"""PDF type detection models."""
from enum import Enum
from dataclasses import dataclass
from typing import List


class PDFType(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"


@dataclass
class PDFTypeResult:
    pdf_type: PDFType
    digital_pages: List[int]
    scanned_pages: List[int]
    total_pages: int
    text_density: float
    image_density: float
    confidence: float
