"""Reading order reconstruction for document layout analysis."""
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

COLUMN_GAP_THRESHOLD = 0.4  # Fraction of page width that separates columns


class ReadingOrderReconstructor:
    """Reconstructs the correct reading order for document text blocks.

    Supports single-column and multi-column layouts using spatial sorting.
    """

    def reconstruct(
        self,
        blocks: List[Dict[str, Any]],
        page_width: float = 595.0,
    ) -> List[Dict[str, Any]]:
        """
        Sort text blocks into correct reading order.

        Args:
            blocks: List of block dicts with 'bbox' (x0, y0, x1, y1) and 'text'.
            page_width: Width of the page for column detection.

        Returns:
            Sorted list of blocks in reading order.
        """
        if not blocks:
            return []

        columns = self._detect_columns(blocks, page_width)

        if columns == 1:
            return self._sort_top_to_bottom(blocks)
        else:
            return self._sort_multi_column(blocks, page_width, columns)

    def _sort_top_to_bottom(
        self, blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple top-to-bottom, left-to-right sort."""
        return sorted(blocks, key=lambda b: (b["bbox"]["y0"], b["bbox"]["x0"]))

    def _detect_columns(
        self, blocks: List[Dict[str, Any]], page_width: float
    ) -> int:
        """Detect number of columns by analyzing x-coordinate distribution."""
        if not blocks:
            return 1
        mid_x_values = [
            (b["bbox"]["x0"] + b["bbox"]["x1"]) / 2 for b in blocks
        ]
        mid_page = page_width / 2
        left_count = sum(1 for x in mid_x_values if x < mid_page * (1 - COLUMN_GAP_THRESHOLD))
        right_count = sum(1 for x in mid_x_values if x > mid_page * (1 + COLUMN_GAP_THRESHOLD))
        min_column_blocks = max(2, len(blocks) // 4)
        if left_count >= min_column_blocks and right_count >= min_column_blocks:
            return 2
        return 1

    def _sort_multi_column(
        self,
        blocks: List[Dict[str, Any]],
        page_width: float,
        num_columns: int,
    ) -> List[Dict[str, Any]]:
        """Sort blocks in multi-column layout (left column first, top-to-bottom)."""
        mid_page = page_width / 2
        left_blocks = [b for b in blocks if b["bbox"]["x1"] <= mid_page * 1.1]
        right_blocks = [b for b in blocks if b["bbox"]["x0"] >= mid_page * 0.9]
        center_blocks = [
            b for b in blocks
            if b not in left_blocks and b not in right_blocks
        ]

        left_sorted = self._sort_top_to_bottom(left_blocks)
        right_sorted = self._sort_top_to_bottom(right_blocks)
        center_sorted = self._sort_top_to_bottom(center_blocks)

        return left_sorted + center_sorted + right_sorted
