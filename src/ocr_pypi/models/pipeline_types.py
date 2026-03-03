"""Pipeline control types."""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PipelineStage(Enum):
    DETECTION = "detection"
    EXTRACTION = "extraction"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    LAYOUT_ANALYSIS = "layout_analysis"
    NOISE_REMOVAL = "noise_removal"
    SECTION_CLASSIFICATION = "section_classification"
    CHUNKING = "chunking"
    OUTPUT_FORMATTING = "output_formatting"


@dataclass
class StageResult:
    stage: PipelineStage
    success: bool
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    document_id: str
    stages: List[StageResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
