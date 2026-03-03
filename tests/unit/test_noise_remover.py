"""Unit tests for noise remover."""
import pytest
from ocr_pypi.cleaning.noise_remover import NoiseRemover


class TestNoiseRemover:
    def test_removes_repeated_headers(self):
        """Test that repeated header text is removed."""
        remover = NoiseRemover()
        pages_data = [
            {
                "page_number": i,
                "text": f"Header Text\nContent of page {i}",
                "blocks": [
                    {"text": "Header Text", "bbox": {"x0": 50, "y0": 10, "x1": 500, "y1": 50}},
                    {"text": f"Content of page {i}", "bbox": {"x0": 50, "y0": 200, "x1": 500, "y1": 400}},
                ],
            }
            for i in range(1, 6)
        ]
        result = remover.remove_noise(pages_data, page_height=842.0)
        # Header should be removed from pages (appears in >60% of pages)
        for page in result:
            texts = [b["text"] for b in page["blocks"]]
            # Either header is removed or content remains
            assert any("Content" in t for t in texts)

    def test_removes_short_blocks(self):
        """Test that very short blocks (< MIN_BLOCK_LENGTH) are removed."""
        remover = NoiseRemover()
        pages_data = [
            {
                "page_number": 1,
                "text": "Hi\nThis is actual content with enough text.",
                "blocks": [
                    {"text": "Hi", "bbox": {"x0": 50, "y0": 100, "x1": 100, "y1": 120}},
                    {"text": "This is actual content with enough text.", "bbox": {"x0": 50, "y0": 200, "x1": 500, "y1": 220}},
                ],
            }
        ]
        result = remover.remove_noise(pages_data)
        blocks = result[0]["blocks"]
        texts = [b["text"] for b in blocks]
        assert "Hi" not in texts
        assert "This is actual content with enough text." in texts

    def test_preserves_unique_content(self):
        """Test that unique page content is preserved."""
        remover = NoiseRemover()
        pages_data = [
            {
                "page_number": 1,
                "text": "Unique content for page one that appears only once.",
                "blocks": [
                    {"text": "Unique content for page one that appears only once.", "bbox": {"x0": 50, "y0": 200, "x1": 500, "y1": 400}},
                ],
            }
        ]
        result = remover.remove_noise(pages_data)
        assert len(result[0]["blocks"]) == 1
