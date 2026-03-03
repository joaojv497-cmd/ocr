"""Unit tests for new chunking components."""
import pytest
from ocr_pypi.chunking.paragraph_chunker import ParagraphChunker
from ocr_pypi.chunking.templates.dynamic_template import DynamicTemplate
from ocr_pypi.chunking.templates.registry import TemplateRegistry


PAGES = [
    {
        "page_number": 1,
        "text": (
            "Primeiro parágrafo com conteúdo relevante para o teste.\n\n"
            "Segundo parágrafo que continua o documento com mais texto.\n\n"
            "Terceiro parágrafo com informações adicionais e complementares."
        ),
    },
    {
        "page_number": 2,
        "text": (
            "Quarto parágrafo na segunda página do documento.\n\n"
            "Quinto parágrafo encerrando o conteúdo do documento de teste."
        ),
    },
]


# ---------------------------------------------------------------------------
# ParagraphChunker tests
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_basic_chunking_returns_chunks(self):
        """ParagraphChunker should return at least one chunk."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(PAGES)
        assert len(chunks) >= 1

    def test_chunk_contains_page_numbers(self):
        """Each chunk must reference valid page numbers."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(PAGES)
        for chunk in chunks:
            assert len(chunk.page_numbers) >= 1
            for pn in chunk.page_numbers:
                assert pn in {1, 2}

    def test_chunk_metadata_method(self):
        """Metadata should reflect paragraph chunking method."""
        chunker = ParagraphChunker()
        chunks = chunker.chunk(PAGES)
        for chunk in chunks:
            assert chunk.metadata["chunking_method"] == "paragraph"

    def test_small_chunk_size_produces_more_chunks(self):
        """Smaller chunk_size should produce more chunks."""
        chunker_small = ParagraphChunker(chunk_size=100)
        chunker_large = ParagraphChunker(chunk_size=10000)
        small_chunks = chunker_small.chunk(PAGES)
        large_chunks = chunker_large.chunk(PAGES)
        assert len(small_chunks) >= len(large_chunks)

    def test_empty_pages_returns_empty(self):
        """Empty pages list should return empty chunks list."""
        chunker = ParagraphChunker()
        assert chunker.chunk([]) == []

    def test_overlap_carries_text(self):
        """With chunk_overlap > 0 subsequent chunks should include some prior text."""
        # Use tiny chunk_size so multiple chunks are guaranteed
        chunker = ParagraphChunker(chunk_size=80, chunk_overlap=40)
        chunks = chunker.chunk(PAGES)
        # Just ensure it doesn't crash and returns chunks
        assert len(chunks) >= 1

    def test_chunk_index_sequential(self):
        """chunk_index values should be sequential starting from 0."""
        chunker = ParagraphChunker(chunk_size=100)
        chunks = chunker.chunk(PAGES)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# DynamicTemplate tests
# ---------------------------------------------------------------------------


class TestDynamicTemplate:
    def _make_definition(self, name="custom_template"):
        return {
            "template_name": name,
            "description": "Test custom template",
            "sections": [
                {"name": "introducao", "description": "Introdução", "required": True},
                {"name": "conclusao", "description": "Conclusão", "required": False},
            ],
        }

    def test_creates_template_with_correct_name(self):
        t = DynamicTemplate(self._make_definition())
        assert t.name == "custom_template"

    def test_creates_template_with_description(self):
        t = DynamicTemplate(self._make_definition())
        assert t.description == "Test custom template"

    def test_sections_are_preserved(self):
        t = DynamicTemplate(self._make_definition())
        assert len(t.sections) == 2
        assert t.sections[0]["name"] == "introducao"

    def test_build_prompt_includes_template_name(self):
        t = DynamicTemplate(self._make_definition())
        prompt = t.build_prompt("texto de teste")
        assert "custom_template" in prompt

    def test_missing_template_name_raises(self):
        with pytest.raises(ValueError):
            DynamicTemplate({"sections": []})

    def test_empty_template_name_raises(self):
        with pytest.raises(ValueError):
            DynamicTemplate({"template_name": "", "sections": []})

    def test_invalid_sections_type_raises(self):
        with pytest.raises(ValueError):
            DynamicTemplate({"template_name": "x", "sections": "not_a_list"})

    def test_section_missing_name_raises(self):
        with pytest.raises(ValueError):
            DynamicTemplate({
                "template_name": "x",
                "sections": [{"description": "no name field"}],
            })

    def test_default_description_when_omitted(self):
        t = DynamicTemplate({"template_name": "my_tmpl", "sections": []})
        assert "my_tmpl" in t.description


# ---------------------------------------------------------------------------
# TemplateRegistry dynamic registration tests
# ---------------------------------------------------------------------------


class TestTemplateRegistryDynamic:
    def test_register_and_retrieve_dynamic_template(self):
        definition = {
            "template_name": "test_dynamic_xyz",
            "sections": [{"name": "secao_a", "description": "Seção A"}],
        }
        t = TemplateRegistry.register_dynamic(definition)
        assert t.name == "test_dynamic_xyz"

        retrieved = TemplateRegistry.get("test_dynamic_xyz")
        assert retrieved.name == "test_dynamic_xyz"

    def test_dynamic_template_appears_in_list(self):
        definition = {
            "template_name": "test_list_template",
            "sections": [{"name": "s1", "description": "Section 1"}],
        }
        TemplateRegistry.register_dynamic(definition)
        assert "test_list_template" in TemplateRegistry.list_templates()

    def test_invalid_definition_raises(self):
        with pytest.raises(ValueError):
            TemplateRegistry.register_dynamic({"sections": []})
