from typing import Dict, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate


class SentencaTemplate(DocumentTemplate):
    """Template para Sentenças Judiciais"""

    @property
    def name(self) -> str:
        return "sentenca"

    @property
    def description(self) -> str:
        return "Sentença Judicial — decisão proferida pelo juiz que resolve o mérito da causa"

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "cabecalho",
                "description": "Identificação do tribunal, vara, número do processo e partes",
                "required": True
            },
            {
                "name": "ementa",
                "description": "Ementa/resumo da decisão (quando presente, comum em acórdãos)",
                "required": False
            },
            {
                "name": "relatorio",
                "description": "Relatório — resumo do processo, pedidos e defesa",
                "required": True
            },
            {
                "name": "fundamentacao",
                "description": "Fundamentação/Motivação — análise dos fatos e do direito pelo juiz",
                "required": True
            },
            {
                "name": "dispositivo",
                "description": "Dispositivo — decisão final (procedente, improcedente, parcialmente procedente)",
                "required": True
            },
            {
                "name": "honorarios_custas",
                "description": "Fixação de honorários advocatícios e custas processuais",
                "required": False
            },
            {
                "name": "encerramento",
                "description": "Local, data e assinatura do juiz",
                "required": True
            }
        ]