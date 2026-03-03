"""Unit tests for reading order reconstructor."""
import pytest
from ocr_pypi.layout.reading_order_reconstructor import ReadingOrderReconstructor


def make_block(x0, y0, x1, y1, text="text"):
    return {"text": text, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


class TestReadingOrderReconstructor:
    def test_empty_input(self):
        reconstructor = ReadingOrderReconstructor()
        result = reconstructor.reconstruct([])
        assert result == []

    def test_single_column_top_to_bottom(self):
        reconstructor = ReadingOrderReconstructor()
        blocks = [
            make_block(50, 200, 500, 220, "second"),
            make_block(50, 100, 500, 120, "first"),
            make_block(50, 300, 500, 320, "third"),
        ]
        result = reconstructor.reconstruct(blocks, page_width=595)
        assert result[0]["text"] == "first"
        assert result[1]["text"] == "second"
        assert result[2]["text"] == "third"

    def test_preserves_all_blocks(self):
        reconstructor = ReadingOrderReconstructor()
        blocks = [make_block(i * 10, i * 10, i * 10 + 100, i * 10 + 20) for i in range(5)]
        result = reconstructor.reconstruct(blocks)
        assert len(result) == len(blocks)
