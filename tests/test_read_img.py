"""Pruebas unitarias para módulo de lectura de imágenes."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from PIL import Image

from src.processing.read_img import read_dicom_file, read_jpg_file, read_image


def test_read_dicom_returns_correct_types():
    """Verificar que la lectura de DICOM devuelve tupla de array e imagen PIL."""
    with patch("src.processing.read_img.pydicom.dcmread") as mock_dcm:
        # Simular archivo DICOM
        mock_data = Mock()
        mock_data.pixel_array = np.ones((512, 512), dtype=np.uint16)
        mock_dcm.return_value = mock_data

        with patch("src.processing.read_img.os.path.exists", return_value=True):
            array, pil_img = read_dicom_file("test.dcm")

        # Verificar tipos
        assert isinstance(array, np.ndarray)
        assert isinstance(pil_img, Image.Image)
        assert array.shape[-1] == 3  # Formato RGB


def test_read_dicom_file_not_found():
    """Prueba que se lanza FileNotFoundError para archivo faltante."""
    with pytest.raises(FileNotFoundError):
        read_dicom_file("nonexistent.dcm")


def test_read_jpg_returns_correct_types():
    """Prueba que la lectura de JPG devuelve tipos correctos."""
    with patch("src.processing.read_img.cv2.imread") as mock_cv:
        mock_cv.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        with patch("src.processing.read_img.os.path.exists", return_value=True):
            array, pil_img = read_jpg_file("test.jpg")

        assert isinstance(array, np.ndarray)
        assert array.dtype == np.uint8


def test_read_jpg_file_not_found():
    """Prueba que se lanza FileNotFoundError para JPG faltante."""
    with pytest.raises(FileNotFoundError):
        read_jpg_file("nonexistent.jpg")


def test_read_image_auto_detect_dicom():
    """Prueba que read_image detecta correctamente el formato DICOM."""
    with patch("src.processing.read_img.read_dicom_file") as mock_dicom:
        mock_dicom.return_value = (
            np.zeros((512, 512, 3)),
            Image.new("L", (512, 512)),
        )

        array, pil_img = read_image("scan.dcm")

        # Verificar que se llamó al lector DICOM
        mock_dicom.assert_called_once_with("scan.dcm")


def test_read_image_auto_detect_jpg():
    """Prueba que read_image detecta correctamente el formato JPG."""
    with patch("src.processing.read_img.read_jpg_file") as mock_jpg:
        mock_jpg.return_value = (
            np.zeros((512, 512, 3)),
            Image.new("RGB", (512, 512)),
        )

        array, pil_img = read_image("photo.jpg")

        # Verificar que se llamó al lector JPG
        mock_jpg.assert_called_once_with("photo.jpg")


def test_read_image_unsupported_format():
    """Prueba que se lanza ValueError para formatos no soportados."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_image("document.pdf")

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_image("data.txt")
