from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AreaType(Enum):
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    SUBTITLE = "subtitle"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"


class SectionType(Enum):
    """Tipos de seção para documentos jurídicos"""
    ENDERECAMENTO = "enderecamento"
    QUALIFICACAO_PARTES = "qualificacao_partes"
    FATOS = "fatos"
    FUNDAMENTOS_JURIDICOS = "fundamentos_juridicos"
    PEDIDOS = "pedidos"
    VALOR_CAUSA = "valor_causa"
    PROVAS = "provas"
    ENCERRAMENTO = "encerramento"
    CLAUSULA = "clausula"
    PREAMBULO = "preambulo"
    OBJETO = "objeto"
    DISPOSITIVO = "dispositivo"
    RELATORIO = "relatorio"
    FUNDAMENTACAO = "fundamentacao"
    EMENTA = "ementa"
    CUSTOM = "custom"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class DocumentArea:
    area_type: AreaType
    text: str
    page_number: int
    confidence: float
    bbox: BoundingBox


@dataclass
class StructuredSection:
    """Seção estruturada retornada pela LLM"""
    section_name: str
    title: str
    content: str
    page_numbers: List[int] = field(default_factory=list)
    subsections: List['StructuredSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredDocument:
    """Documento completo estruturado pela LLM"""
    document_type: str
    sections: List[StructuredSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    page_numbers: List[int]
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_areas: List[DocumentArea] = field(default_factory=list)