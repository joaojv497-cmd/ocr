"""Legal pattern library for Brazilian legal documents."""
import re
from typing import Dict, List

# Section title patterns by section type
SECTION_PATTERNS: Dict[str, List[str]] = {
    "enderecamento": [
        r"exmo?\.?\s+sr\.?\s+dr\.?",
        r"excelent[ií]ssimo",
        r"meritíssimo",
        r"doutor[ao]?\s+juiz",
        r"juízo\s+d[ao]",
    ],
    "qualificacao_partes": [
        r"qualifica[çc][aã]o\s+das?\s+partes",
        r"das?\s+partes",
        r"requerente",
        r"requerido",
        r"autor[ao]?",
        r"r[eé]u",
    ],
    "fatos": [
        r"dos?\s+fatos",
        r"da\s+narrativa\s+dos?\s+fatos",
        r"relatório",
        r"dos?\s+fatos\s+e\s+fundamentos",
        r"fatos\s+relevantes",
        r"histórico",
    ],
    "fundamentos_juridicos": [
        r"do\s+direito",
        r"fundamentos?\s+jur[ií]dicos?",
        r"das?\s+razões",
        r"da\s+fundamenta[çc][aã]o",
        r"base\s+legal",
        r"dos?\s+fundamentos",
    ],
    "pedidos": [
        r"dos?\s+pedidos?",
        r"do\s+pedido",
        r"requer\s+a\s+vossa",
        r"pede\s+deferimento",
        r"requerimentos?",
        r"conclusão",
    ],
    "valor_causa": [
        r"valor\s+d[ao]\s+causa",
        r"valor\s+atribu[ií]do",
        r"da\s+causa",
    ],
    "provas": [
        r"das?\s+provas",
        r"meios?\s+de\s+prova",
        r"prova\s+a\s+produzir",
    ],
    "encerramento": [
        r"nestes?\s+termos",
        r"pede\s+deferimento",
        r"termos\s+em\s+que",
        r"deferimento",
    ],
    "clausula": [
        r"cl[aá]usula",
        r"§\s*\d+",
        r"artigo\s+\d+",
        r"art\.\s*\d+",
    ],
    "preambulo": [
        r"preâmbulo",
        r"considerando",
        r"pelo\s+presente\s+instrumento",
    ],
    "dispositivo": [
        r"dispositivo",
        r"diante\s+do\s+exposto",
        r"isto\s+posto",
        r"pelo\s+exposto",
    ],
    "relatorio": [
        r"relatório",
        r"trata-se\s+de",
        r"cuida-se\s+de",
    ],
    "fundamentacao": [
        r"fundamenta[çc][aã]o",
        r"fundamento",
        r"mérito",
    ],
    "ementa": [
        r"ementa",
        r"acórdão",
        r"súmula",
    ],
}

# Pre-compile all patterns for performance
_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    section: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    for section, patterns in SECTION_PATTERNS.items()
}


def match_section_type(text: str) -> List[str]:
    """
    Match text against legal section patterns.

    Args:
        text: Text to analyze (typically a title or heading).

    Returns:
        List of matching section type names, ordered by confidence.
    """
    matches = []
    normalized = text.strip().lower()
    for section_type, patterns in _COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized):
                matches.append(section_type)
                break  # One match per section type is enough
    return matches
