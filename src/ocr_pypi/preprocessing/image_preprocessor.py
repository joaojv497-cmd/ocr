"""Image preprocessing module for improving OCR quality."""
import io
import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("OpenCV not available; falling back to PIL-only preprocessing.")

TARGET_DPI = 300


class ImagePreprocessor:
    """Preprocesses images to improve OCR accuracy.

    Applies binarization, noise removal, deskew, contrast adjustment,
    and DPI normalization.
    """

    def preprocess(self, image: Image.Image, dpi: int = 72) -> Image.Image:
        """
        Full preprocessing pipeline.

        Args:
            image: Input PIL Image.
            dpi: Current DPI of the image (used for upscaling).

        Returns:
            Preprocessed PIL Image ready for OCR.
        """
        image = self.resize_to_target_dpi(image, dpi)
        image = self.adjust_contrast(image)
        if _CV2_AVAILABLE:
            image = self.deskew(image)
            image = self.remove_noise(image)
            image = self.binarize(image)
        else:
            image = self._pil_binarize(image)
        return image

    def preprocess_for_vision(
        self,
        image: Image.Image,
        max_width: int = 1024,
        max_height: int = 1024,
        jpeg_quality: int = 55,
    ) -> bytes:
        """
        Preprocess image for Vision LLM: resize to max dimensions and encode as JPEG.

        Maintains aspect ratio while reducing image size to minimise token costs.

        Args:
            image: Input PIL Image.
            max_width: Maximum width in pixels.
            max_height: Maximum height in pixels.
            jpeg_quality: JPEG compression quality (1–95). Values above 95
                disable JPEG compression algorithm optimisations and produce
                disproportionately large files with negligible quality gain.

        Returns:
            JPEG-encoded image bytes.
        """
        image = self._resize_for_vision(image, max_width, max_height)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()

    def _resize_for_vision(
        self,
        image: Image.Image,
        max_width: int,
        max_height: int,
    ) -> Image.Image:
        """Resize image to fit within max dimensions maintaining aspect ratio."""
        width, height = image.size
        if width <= max_width and height <= max_height:
            return image
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)

    def resize_to_target_dpi(self, image: Image.Image, current_dpi: int) -> Image.Image:
        """Resize image to TARGET_DPI for consistent OCR quality."""
        if current_dpi <= 0 or current_dpi == TARGET_DPI:
            return image
        scale = TARGET_DPI / current_dpi
        if scale == 1.0:
            return image
        new_size = (int(image.width * scale), int(image.height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def adjust_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Enhance image contrast."""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def binarize(self, image: Image.Image) -> Image.Image:
        """Apply adaptive (Otsu) binarization using OpenCV."""
        if not _CV2_AVAILABLE:
            return self._pil_binarize(image)
        gray = self._to_gray_cv2(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    def _pil_binarize(self, image: Image.Image) -> Image.Image:
        """Simple PIL-based binarization fallback."""
        gray = image.convert("L")
        return gray.point(lambda x: 255 if x > 128 else 0, "1").convert("L")

    def remove_noise(self, image: Image.Image) -> Image.Image:
        """Remove noise using morphological operations (requires OpenCV)."""
        if not _CV2_AVAILABLE:
            return image.filter(ImageFilter.MedianFilter(size=3))
        cv_image = self._to_gray_cv2(image)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(opened)

    def deskew(self, image: Image.Image) -> Image.Image:
        """Detect and correct image skew using OpenCV."""
        if not _CV2_AVAILABLE:
            return image
        try:
            gray = self._to_gray_cv2(image)
            coords = np.column_stack(np.where(gray < 128))
            if len(coords) < 10:
                return image
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 0.5:
                return image
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cv_image = np.array(image.convert("RGB") if image.mode != "L" else image)
            rotated = cv2.warpAffine(
                cv_image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return Image.fromarray(rotated)
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image

    def _to_gray_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV grayscale array."""
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
