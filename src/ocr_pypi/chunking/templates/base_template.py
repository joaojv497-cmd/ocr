from abc import ABC, abstractmethod
from typing import Dict, List, Any


class DocumentTemplate(ABC):
    """Base para templates de documentos jurídicos"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador único do template (ex: 'peticao_inicial')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Descrição do tipo de documento"""
        pass

    @property
    @abstractmethod
    def sections(self) -> List[Dict[str, Any]]:
        """
        Define as seções esperadas no documento.
        Cada seção: {name, description, required, subsections?}
        """
        pass

    @property
    def is_custom(self) -> bool:
        """Indica se é um template customizado (dinâmico)."""
        return False

    @property
    def document_types(self) -> str:
        """Tipos de documento suportados pelo template (JSON string)."""
        return ""

    def build_prompt(self, extracted_text: str) -> str:
        """Constrói o prompt completo para a LLM"""
        sections_desc = self._format_sections()

        return f"""Você é um especialista em análise de documentos jurídicos brasileiros.

Analise o texto extraído por OCR abaixo e estruture-o de acordo com o template fornecido.

## TEMPLATE: {self.name}
{self.description}

## SEÇÕES ESPERADAS:
{sections_desc}

## REGRAS:
1. Extraia o conteúdo de cada seção definida no template acima
2. Se uma seção não existir no documento, retorne null no campo "content"
3. Mantenha o texto original, corrigindo apenas erros óbvios de OCR
4. Retorne APENAS JSON válido no formato especificado abaixo
5. Preserve a hierarquia do documento
6. Identifique os números das páginas onde cada seção aparece (marcadores [PÁGINA N])
7. Extraia metadados relevantes (partes envolvidas, datas, números de processo, tribunal)

## FORMATO DE SAÍDA (JSON):
{{
    "document_type": "{self.name}",
    "sections": [
        {{
            "section_name": "nome_da_secao",
            "title": "Título encontrado no documento",
            "content": "Conteúdo extraído da seção",
            "page_numbers": [1, 2],
            "subsections": [
                {{
                    "section_name": "nome_subsecao",
                    "title": "Título da subseção",
                    "content": "Conteúdo da subseção",
                    "page_numbers": [1],
                    "subsections": [],
                    "metadata": {{}}
                }}
            ],
            "metadata": {{}}
        }}
    ],
    "metadata": {{
        "parties": [],
        "dates": [],
        "case_number": null,
        "court": null,
        "judge": null,
        "lawyer": null,
        "oab_number": null
    }}
}}

## TEXTO EXTRAÍDO (OCR):
---
{extracted_text}
---

Retorne APENAS o JSON estruturado:"""

    def _format_sections(self) -> str:
        lines = []
        for i, section in enumerate(self.sections, 1):
            required = "✅ Obrigatória" if section.get("required", False) else "⬜ Opcional"
            lines.append(
                f"{i}. **{section['name']}** ({required}): {section['description']}"
            )
            for sub in section.get("subsections", []):
                sub_req = "✅" if sub.get("required", False) else "⬜"
                lines.append(
                    f"   {sub_req} {sub['name']}: {sub['description']}"
                )
        return "\n".join(lines)