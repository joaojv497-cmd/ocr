import logging
from typing import Dict, Type, Optional, List, Any
from ocr_pypi.chunking.templates.base_template import DocumentTemplate

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Registry de templates de documentos jurídicos.

    Uso:
        # Registra template customizado
        TemplateRegistry.register(MeuTemplate)

        # Obtém template pelo nome
        template = TemplateRegistry.get("peticao_inicial")

        # Lista disponíveis
        nomes = TemplateRegistry.list_templates()
    """

    _registry: Dict[str, DocumentTemplate] = {}

    @classmethod
    def register(cls, template_class: Type[DocumentTemplate]) -> None:
        """Registra um template de documento"""
        if not issubclass(template_class, DocumentTemplate):
            raise TypeError(
                f"{template_class.__name__} deve herdar de DocumentTemplate"
            )
        instance = template_class()
        cls._registry[instance.name.lower()] = instance
        logger.info(f"Template registrado: {instance.name}")

    @classmethod
    def get(cls, name: str) -> DocumentTemplate:
        """
        Obtém um template pelo nome.

        Raises:
            ValueError: Se o template não estiver registrado
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Template '{name}' não encontrado. "
                f"Nenhum template padrão disponível. "
                f"Envie um template customizado via request (campo 'template')."
            )
        return cls._registry[key]

    @classmethod
    def register_dynamic(cls, definition: Dict[str, Any]) -> DocumentTemplate:
        """
        Registra e retorna um template dinâmico a partir de uma definição JSON.

        Args:
            definition: Dict com chaves 'template_name', 'sections' e
                        opcionalmente 'description'.

        Returns:
            A instância de DynamicTemplate registrada.

        Raises:
            ValueError: Se a definição for inválida.
        """
        from ocr_pypi.chunking.templates.dynamic_template import DynamicTemplate

        template = DynamicTemplate(definition)
        cls._registry[template.name.lower()] = template
        logger.info(f"Dynamic template registered: {template.name}")
        return template

    @classmethod
    def list_templates(cls) -> List[str]:
        """Lista todos os templates registrados"""
        return list(cls._registry.keys())

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Não registra templates padrão - apenas aceita templates dinâmicos via request"""
        pass