from typing import Dict, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate


class PeticaoInicialTemplate(DocumentTemplate):
    """Template para Petição Inicial"""

    @property
    def name(self) -> str:
        return "peticao_inicial"

    @property
    def description(self) -> str:
        return "Petição Inicial — documento que inicia um processo judicial"

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "enderecamento",
                "description": "Endereçamento ao juízo competente (Ex: 'Exmo. Sr. Dr. Juiz de Direito da X Vara...')",
                "required": True
            },
            {
                "name": "qualificacao_partes",
                "description": "Qualificação completa do autor e do réu",
                "required": True,
                "subsections": [
                    {
                        "name": "autor",
                        "description": "Nome, CPF/CNPJ, endereço, profissão, estado civil do autor/requerente",
                        "required": True
                    },
                    {
                        "name": "reu",
                        "description": "Nome, CPF/CNPJ, endereço do réu/requerido",
                        "required": True
                    }
                ]
            },
            {
                "name": "fatos",
                "description": "Narrativa dos fatos que fundamentam o pedido (Dos Fatos)",
                "required": True
            },
            {
                "name": "fundamentos_juridicos",
                "description": "Base legal, doutrina e jurisprudência citada (Do Direito)",
                "required": True
            },
            {
                "name": "pedidos",
                "description": "Pedidos formulados ao juízo (Do Pedido / Dos Pedidos)",
                "required": True
            },
            {
                "name": "valor_causa",
                "description": "Valor atribuído à causa",
                "required": True
            },
            {
                "name": "provas",
                "description": "Provas que pretende produzir (Das Provas)",
                "required": False
            },
            {
                "name": "encerramento",
                "description": "Fecho ('Nestes termos, pede deferimento'), local, data e assinatura do advogado com OAB",
                "required": True
            }
        ]