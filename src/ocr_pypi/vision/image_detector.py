"""Detecção e extração de imagens de documentos PDF usando PyMuPDF."""
import io
import logging
from typing import List, Optional

import fitz
from PIL import Image

from ocr_pypi.models.document import BoundingBox, ImageInfo

logger = logging.getLogger(__name__)

# Minimum image dimensions to be considered relevant
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 300


class ImageDetector:
    """Detecta e extrai imagens de arquivos PDF usando PyMuPDF."""

    def detect_images(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None,
    ) -> List[ImageInfo]:
        """
        Detecta e extrai imagens de um PDF.

        Args:
            pdf_path: Caminho para o arquivo PDF.
            page_numbers: Lista de números de páginas (1-based) para processar.
                          Se None, processa todas as páginas.

        Returns:
            Lista de ImageInfo com dados das imagens encontradas.
        """
        images: List[ImageInfo] = []
        pdf = fitz.open(pdf_path)

        try:
            pages_to_process = range(len(pdf))
            if page_numbers is not None:
                pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= len(pdf)]

            for page_idx in pages_to_process:
                page = pdf.load_page(page_idx)
                page_num = page_idx + 1
                page_images = self._extract_page_images(pdf, page, page_num)
                images.extend(page_images)
        finally:
            pdf.close()

        logger.info(f"Detectadas {len(images)} imagens no PDF '{pdf_path}'")
        return images

    def _extract_page_images(
        self,
        pdf: fitz.Document,
        page: fitz.Page,
        page_num: int,
    ) -> List[ImageInfo]:
        """Extrai imagens de uma página específica."""
        page_images: List[ImageInfo] = []
        image_list = page.get_images(full=True)

        for img_idx, img_ref in enumerate(image_list):
            xref = img_ref[0]
            try:
                image_info = self._extract_single_image(
                    pdf, page, xref, img_idx, page_num
                )
                if image_info is not None:
                    page_images.append(image_info)
            except Exception as e:
                logger.warning(
                    f"Erro ao extrair imagem {img_idx} da página {page_num}: {e}"
                )

        return page_images

    def _extract_single_image(
        self,
        pdf: fitz.Document,
        page: fitz.Page,
        xref: int,
        img_idx: int,
        page_num: int,
    ) -> Optional[ImageInfo]:
        """Extrai uma imagem individual do PDF e retorna ImageInfo, ou None se inválida."""
        # Get bounding box on page
        bbox: Optional[BoundingBox] = None
        rects = page.get_image_rects(xref)
        if rects:
            r = rects[0]
            bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)

        # Extract raw image bytes
        base_image = pdf.extract_image(xref)
        if not base_image:
            return None

        image_bytes = base_image.get("image", b"")
        colorspace = base_image.get("colorspace", 3)
        image_ext = base_image.get("ext", "png")

        if not image_bytes:
            return None

        # Convert to PNG via Pillow for uniformity
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size

            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return None

            # Normalize to PNG
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except Exception as e:
            logger.warning(f"Erro ao processar imagem (xref={xref}): {e}")
            return None

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
