"""Smart image detection with size filtering, deduplication, and zone awareness."""
import hashlib
import io
import logging
from typing import List, Optional, Tuple

import fitz
from PIL import Image

from ocr_pypi.models.document import BoundingBox, ImageInfo

logger = logging.getLogger(__name__)

# Minimum image dimensions to be considered content-relevant
DEFAULT_MIN_IMAGE_WIDTH = 700
DEFAULT_MIN_IMAGE_HEIGHT = 200

# Page zones to exclude (header top 15%, footer bottom 15%)
HEADER_ZONE_RATIO = 0.15
FOOTER_ZONE_RATIO = 0.85


class SmartImageDetector:
    """Detects content-relevant images from PDF files.

    Compared to the basic ``ImageDetector`` this detector adds:

    * Minimum size filter — ignores icons and small decorative elements.
    * Visual hash deduplication — repeated logos across pages are counted only
      once (up to ``max_duplicates`` occurrences).
    * Header/footer zone exclusion — images whose centre falls within the
      top/bottom zone of the page are skipped, because they are typically
      logos or decorative separators embedded in headers and footers.
    """

    def __init__(
        self,
        min_size: Tuple[int, int] = (DEFAULT_MIN_IMAGE_WIDTH, DEFAULT_MIN_IMAGE_HEIGHT),
        max_duplicates: int = 1,
        header_zone_ratio: float = HEADER_ZONE_RATIO,
        footer_zone_ratio: float = FOOTER_ZONE_RATIO,
    ):
        """
        Args:
            min_size: Minimum (width, height) in pixels for an image to be kept.
            max_duplicates: How many times the same visual hash may appear before
                subsequent occurrences are dropped as duplicates.
            header_zone_ratio: Images whose vertical centre falls below this
                fraction of the page height are treated as header images.
            footer_zone_ratio: Images whose vertical centre falls above this
                fraction of the page height are treated as footer images.
        """
        self.min_width, self.min_height = min_size
        self.max_duplicates = max_duplicates
        self.header_zone_ratio = header_zone_ratio
        self.footer_zone_ratio = footer_zone_ratio

    def detect_images(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None,
    ) -> List[ImageInfo]:
        """
        Detect and extract content-relevant images from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: 1-based page numbers to process. ``None`` means all pages.

        Returns:
            Filtered list of ``ImageInfo`` objects (no duplicates, no tiny images,
            no header/footer images).
        """
        images: List[ImageInfo] = []
        # hash → count of occurrences seen so far
        seen_hashes: dict = {}

        pdf = fitz.open(pdf_path)
        try:
            pages_to_process = range(len(pdf))
            if page_numbers is not None:
                pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= len(pdf)]

            for page_idx in pages_to_process:
                page = pdf.load_page(page_idx)
                page_num = page_idx + 1
                page_height = page.rect.height

                page_images = self._extract_page_images(
                    pdf, page, page_num, page_height, seen_hashes
                )
                images.extend(page_images)
        finally:
            pdf.close()

        logger.info(
            f"SmartImageDetector: {len(images)} relevant images detected in '{pdf_path}'"
        )
        return images

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_page_images(
        self,
        pdf: fitz.Document,
        page: fitz.Page,
        page_num: int,
        page_height: float,
        seen_hashes: dict,
    ) -> List[ImageInfo]:
        """Extract and filter images from a single page."""
        page_images: List[ImageInfo] = []
        image_list = page.get_images(full=True)

        for img_idx, img_ref in enumerate(image_list):
            xref = img_ref[0]
            try:
                image_info = self._extract_single_image(
                    pdf, page, xref, img_idx, page_num, page_height, seen_hashes
                )
                if image_info is not None:
                    page_images.append(image_info)
            except Exception as e:
                logger.warning(
                    f"Error extracting image {img_idx} from page {page_num}: {e}"
                )

        return page_images

    def _extract_single_image(
        self,
        pdf: fitz.Document,
        page: fitz.Page,
        xref: int,
        img_idx: int,
        page_num: int,
        page_height: float,
        seen_hashes: dict,
    ) -> Optional[ImageInfo]:
        """Extract one image, applying all smart filters. Returns None if filtered."""
        # Obtain bounding box on page
        bbox: Optional[BoundingBox] = None
        rects = page.get_image_rects(xref)
        if rects:
            r = rects[0]
            bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)

        # --- Zone filter: skip header/footer images ---
        if bbox is not None and self._in_header_footer_zone(bbox, page_height):
            return None

        # Extract raw image bytes
        base_image = pdf.extract_image(xref)
        if not base_image:
            return None

        image_bytes = base_image.get("image", b"")
        colorspace = base_image.get("colorspace", 3)

        if not image_bytes:
            return None

        # Decode and validate image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size
        except Exception as e:
            logger.warning(f"Could not decode image (xref={xref}): {e}")
            return None

        # --- Size filter ---
        if width < self.min_width or height < self.min_height:
            return None

        # Normalise to PNG
        try:
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except Exception as e:
            logger.warning(f"Could not convert image to PNG (xref={xref}): {e}")
            return None

        # --- Hash deduplication ---
        img_hash = hashlib.md5(png_bytes).hexdigest()  # noqa: S324
        count = seen_hashes.get(img_hash, 0)
        if count >= self.max_duplicates:
            return None
        seen_hashes[img_hash] = count + 1

        colorspace_name = "rgb" if isinstance(colorspace, int) else str(colorspace)

        return ImageInfo(
            page_number=page_num,
            image_index=img_idx,
            bbox=bbox,
            width=width,
            height=height,
            image_data=png_bytes,
            colorspace=colorspace_name,
            format="png",
        )

    def _in_header_footer_zone(self, bbox: BoundingBox, page_height: float) -> bool:
        """Return True if the image centre sits in a header or footer zone."""
        center_y = (bbox.y0 + bbox.y1) / 2
        return (
            center_y < page_height * self.header_zone_ratio
            or center_y > page_height * self.footer_zone_ratio
        )
