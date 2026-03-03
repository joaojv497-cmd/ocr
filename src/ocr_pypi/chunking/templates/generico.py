from typing import Dict, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate

class GenericoTemplate(DocumentTemplate):
    """Template genérico para documentos jurídicos não classificados"""

    @property
    def name(self) -> str:
        return "generico"

    @property
    def description(self) -> str:
        return (
            "Documento jurídico genérico. "
            "Identifique e separe as seções lógicas do documento, "
            "mesmo que não correspondam a um tipo específico."
        )

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "cabecalho",
                "description": "Cabeçalho do documento (tribunal, vara, número do processo)",
                "required": False
            },
            {
                "name": "identificacao",
                "description": "Identificação das partes envolvidas",
                "required": False
            },
            {
                "name": "corpo",
                "description": "Corpo principal do documento com todo o conteúdo substantivo",
                "required": True,
                "subsections": [
                    {
                        "name": "introducao",
                        "description": "Parte introdutória ou contextualização"
                    },
                    {
                        "name": "desenvolvimento",
                        "description": "Argumentação, fatos, fundamentação"
                    },
                    {
                        "name": "conclusao",
                        "description": "Conclusão, pedidos ou decisão"
                    }
                ]
            },
            {
                "name": "encerramento",
                "description": "Fecho, data, local, assinatura",
                "required": False
            }
        ]