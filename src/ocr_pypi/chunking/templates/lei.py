from typing import Dict, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate


class LeiTemplate(DocumentTemplate):
    """Template para Lei / Diploma Legal"""

    @property
    def name(self) -> str:
        return "lei"

    @property
    def description(self) -> str:
        return "Lei ou diploma normativo — ato normativo estruturado em artigos, parágrafos, incisos e alíneas"

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "identificacao",
                "description": "Número da lei/decreto, data de promulgação, órgão emissor e ementa",
                "required": True,
                "subsections": [
                    {
                        "name": "numero",
                        "description": "Número do ato normativo (ex: Decreto-Lei nº 4.657/1942)",
                        "required": True
                    },
                    {
                        "name": "data",
                        "description": "Data de promulgação/publicação",
                        "required": True
                    },
                    {
                        "name": "ementa",
                        "description": "Resumo oficial do conteúdo da norma",
                        "required": False
                    }
                ]
            },
            {
                "name": "preambulo",
                "description": "Autoridade que promulga a norma e fundamento constitucional",
                "required": False
            },
            {
                "name": "corpo_normativo",
                "description": "Disposições normativas organizadas em artigos",
                "required": True,
                "subsections": [
                    {
                        "name": "artigos",
                        "description": "Lista de artigos da lei",
                        "required": True,
                        "subsections": [
                            {
                                "name": "caput",
                                "description": "Texto principal do artigo"
                            },
                            {
                                "name": "paragrafos",
                                "description": "Parágrafos do artigo (§ 1º, § 2º, etc.)"
                            },
                            {
                                "name": "incisos",
                                "description": "Incisos do artigo (I, II, III, etc.)"
                            },
                            {
                                "name": "alineas",
                                "description": "Alíneas (a, b, c, etc.)"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "disposicoes_finais",
                "description": "Cláusulas de vigência, revogação e disposições transitórias",
                "required": False,
                "subsections": [
                    {
                        "name": "vigencia",
                        "description": "Dispositivo que trata da entrada em vigor"
                    },
                    {
                        "name": "revogacao",
                        "description": "Normas revogadas expressamente"
                    }
                ]
            },
            {
                "name": "assinatura",
                "description": "Autoridade que assina/promulga o ato",
                "required": False
            }
        ]