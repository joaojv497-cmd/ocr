from typing import Dict, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate


class ContratoTemplate(DocumentTemplate):
    """Template para Contratos"""

    @property
    def name(self) -> str:
        return "contrato"

    @property
    def description(self) -> str:
        return "Contrato — instrumento particular ou público que formaliza acordo entre partes"

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "preambulo",
                "description": "Identificação do contrato, data, local e qualificação das partes contratantes",
                "required": True,
                "subsections": [
                    {
                        "name": "contratante",
                        "description": "Dados da parte contratante",
                        "required": True
                    },
                    {
                        "name": "contratado",
                        "description": "Dados da parte contratada",
                        "required": True
                    }
                ]
            },
            {
                "name": "objeto",
                "description": "Objeto do contrato — o que está sendo contratado",
                "required": True
            },
            {
                "name": "clausulas",
                "description": "Cláusulas do contrato (obrigações, prazos, valores, penalidades, rescisão, etc.)",
                "required": True,
                "subsections": [
                    {
                        "name": "obrigacoes",
                        "description": "Obrigações das partes"
                    },
                    {
                        "name": "valor_pagamento",
                        "description": "Valor, forma e condições de pagamento"
                    },
                    {
                        "name": "prazo_vigencia",
                        "description": "Prazo e vigência do contrato"
                    },
                    {
                        "name": "rescisao",
                        "description": "Condições de rescisão"
                    },
                    {
                        "name": "penalidades",
                        "description": "Multas e penalidades"
                    },
                    {
                        "name": "confidencialidade",
                        "description": "Cláusula de confidencialidade (se houver)"
                    }
                ]
            },
            {
                "name": "foro",
                "description": "Foro de eleição para resolução de conflitos",
                "required": True
            },
            {
                "name": "assinaturas",
                "description": "Assinaturas das partes e testemunhas",
                "required": True
            }
        ]