"""Módulo de visão: detecção e descrição de imagens em documentos PDF."""
from ocr_pypi.vision.image_detector import ImageDetector
from ocr_pypi.vision.image_descriptor import ImageDescriptor
from ocr_pypi.vision.smart_image_detector import SmartImageDetector

__all__ = ["ImageDetector", "ImageDescriptor", "SmartImageDetector"]
