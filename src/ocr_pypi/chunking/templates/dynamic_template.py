"""Dynamic template support: build a DocumentTemplate from a JSON definition."""
import logging
from typing import Dict, List, Any

from ocr_pypi.chunking.templates.base_template import DocumentTemplate

logger = logging.getLogger(__name__)


class DynamicTemplate(DocumentTemplate):
    """
    A DocumentTemplate constructed from a JSON definition at runtime.

    The definition dict must contain:
        - template_name (str): unique identifier
        - sections (list): list of section dicts compatible with DocumentTemplate
        - description (str, optional): human-readable description
    """

    def __init__(self, definition: Dict[str, Any]) -> None:
        self._validate(definition)
        self._name: str = definition["template_name"]
        self._description: str = definition.get(
            "description", f"Custom template: {self._name}"
        )
        self._sections: List[Dict[str, Any]] = definition.get("sections", [])

    # ------------------------------------------------------------------
    # DocumentTemplate interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def sections(self) -> List[Dict[str, Any]]:
        return self._sections

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(definition: Dict[str, Any]) -> None:
        """Raise ValueError if the definition is not usable."""
        if not isinstance(definition, dict):
            raise ValueError("Template definition must be a dict.")
        name = definition.get("template_name", "")
        if not name or not isinstance(name, str):
            raise ValueError(
                "Template definition must contain a non-empty 'template_name' string."
            )
        sections = definition.get("sections", [])
        if not isinstance(sections, list):
            raise ValueError("'sections' must be a list.")
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                raise ValueError(f"Section {i} must be a dict.")
            if "name" not in section:
                raise ValueError(f"Section {i} is missing the 'name' field.")
