"""Semantic chunker using sentence embeddings (no LLM required)."""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ocr_pypi.models.document import Chunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available; using simple chunker.")

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
PORTUGUESE_MODEL = "neuralmind/bert-base-portuguese-cased"
SIMILARITY_THRESHOLD = 0.7
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 200


class SemanticChunker:
    """
    Chunks document text using semantic similarity between paragraphs.

    Uses sentence-transformers to detect semantic breaks without requiring
    an LLM API call. Falls back to paragraph-based chunking when
    sentence-transformers is unavailable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
    ):
        self._model: Optional[Any] = None
        self._model_name = model_name or DEFAULT_MODEL
        self._max_chunk_chars = max_chunk_chars
        self._min_chunk_chars = min_chunk_chars

    def _load_model(self) -> bool:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return True
        if not _TRANSFORMERS_AVAILABLE:
            return False
        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model {self._model_name}: {e}")
            return False

    def chunk(
        self,
        pages: List[Dict[str, Any]],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Chunk]:
        """
        Chunk pages into semantically coherent chunks.

        Args:
            pages: List of page dicts with 'page_number' and 'text'.
            similarity_threshold: Cosine similarity threshold for grouping paragraphs.

        Returns:
            List of Chunk objects.
        """
        # Split all pages into paragraphs
        paragraphs = self._extract_paragraphs(pages)
        if not paragraphs:
            return []

        if self._load_model():
            return self._semantic_chunk(paragraphs, similarity_threshold)
        else:
            return self._simple_chunk(paragraphs)

    def _extract_paragraphs(
        self, pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split pages into individual paragraphs with page metadata."""
        paragraphs = []
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page_number", 1)
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) >= self._min_chunk_chars // 4:
                    paragraphs.append({"text": para, "page_number": page_num})
        return paragraphs

    def _semantic_chunk(
        self,
        paragraphs: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Chunk]:
        """Group paragraphs by semantic similarity."""
        texts = [p["text"] for p in paragraphs]
        embeddings = self._model.encode(texts, show_progress_bar=False)

        chunks: List[Chunk] = []
        current_texts: List[str] = [paragraphs[0]["text"]]
        current_pages: List[int] = [paragraphs[0]["page_number"]]
        chunk_idx = 0

        for i in range(1, len(paragraphs)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            total_chars = sum(len(t) for t in current_texts) + len(paragraphs[i]["text"])

            if sim >= threshold and total_chars <= self._max_chunk_chars:
                current_texts.append(paragraphs[i]["text"])
                current_pages.append(paragraphs[i]["page_number"])
            else:
                chunks.append(self._make_chunk(current_texts, current_pages, chunk_idx))
                chunk_idx += 1
                current_texts = [paragraphs[i]["text"]]
                current_pages = [paragraphs[i]["page_number"]]

        if current_texts:
            chunks.append(self._make_chunk(current_texts, current_pages, chunk_idx))

        return chunks

    def _simple_chunk(self, paragraphs: List[Dict[str, Any]]) -> List[Chunk]:
        """Simple chunking by character count."""
        chunks: List[Chunk] = []
        current_texts: List[str] = []
        current_pages: List[int] = []
        current_chars = 0
        chunk_idx = 0

        for para in paragraphs:
            text = para["text"]
            page_num = para["page_number"]

            if current_chars + len(text) > self._max_chunk_chars and current_texts:
                chunks.append(self._make_chunk(current_texts, current_pages, chunk_idx))
                chunk_idx += 1
                current_texts = []
                current_pages = []
                current_chars = 0

            current_texts.append(text)
            current_pages.append(page_num)
            current_chars += len(text)

        if current_texts:
            chunks.append(self._make_chunk(current_texts, current_pages, chunk_idx))

        return chunks

    def _make_chunk(
        self,
        texts: List[str],
        pages: List[int],
        idx: int,
    ) -> Chunk:
        """Create a Chunk from collected paragraph texts."""
        content = "\n\n".join(texts)
        unique_pages = sorted(set(pages))
        return Chunk(
            content=content,
            page_numbers=unique_pages,
            chunk_index=idx,
            metadata={
                "chunking_method": "semantic",
                "char_count": len(content),
                "word_count": len(content.split()),
                "paragraph_count": len(texts),
            },
            detected_areas=[],
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
