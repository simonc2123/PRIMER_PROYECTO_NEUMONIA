"""Módulo para leer imágenes médicas en varios formatos."""

import os
from typing import Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image

def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Leer archivo de imagen médica DICOM.

    Parameters
    ----------
    path : str
        Ruta al archivo DICOM (.dcm)

    Returns
    -------
    tuple
        - rgb_array : np.ndarray
            Array numpy RGB con forma (H, W, 3)
        - pil_image : PIL.Image.Image
            Objeto PIL Image para visualización

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe
    ValueError
        Si el archivo no es un DICOM válido

    Examples
    --------
    >>> rgb_arr, pil_img = read_dicom_file('chest_xray.dcm')
    >>> rgb_arr.shape
    (512, 512, 3)

    Notes
    -----
    DICOM (Digital Imaging and Communications in Medicine) es el
    formato estándar para datos de imágenes médicas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Leer archivo DICOM
        img = pydicom.dcmread(path)
        img_array = img.pixel_array

        # Crear imagen PIL para visualización
        img2show = Image.fromarray(img_array)

        # Normalizar y convertir a uint8
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        # Convertir a RGB
        img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        return img_rgb, img2show

    except Exception as e:
        raise ValueError(f"Invalid DICOM file: {e}")


def read_jpg_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Leer archivo de imagen estándar (JPG, JPEG, PNG).

    Parameters
    ----------
    path : str
        Ruta al archivo de imagen (.jpg, .jpeg, .png)

    Returns
    -------
    tuple
        - rgb_array : np.ndarray
            Array numpy RGB con forma (H, W, 3)
        - pil_image : PIL.Image.Image
            Objeto PIL Image para visualización

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe
    ValueError
        Si el archivo no se puede leer

    Examples
    --------
    >>> rgb_arr, pil_img = read_jpg_file('chest_xray.jpg')
    >>> rgb_arr.shape
    (1024, 1024, 3)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Leer imagen
        img = cv2.imread(path)
        if img is None:
            raise ValueError("No se pudo leer la imagen")

        img_array = np.asarray(img)

        # Crear imagen PIL para visualización
        img2show = Image.fromarray(img_array)

        # Normalizar
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        return img2, img2show

    except Exception as e:
        raise ValueError(f"Error reading image: {e}")
    
def read_image(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Leer archivo de imagen con detección automática de formato.

    Función de fábrica que determina el tipo de archivo basado en la
    extensión y llama al lector apropiado.

    Parameters
    ----------
    path : str
        Ruta al archivo de imagen (DICOM, JPG, JPEG, o PNG)

    Returns
    -------
    tuple
        - rgb_array : np.ndarray
            Array numpy RGB
        - pil_image : PIL.Image.Image
            Objeto PIL Image

    Raises
    ------
    ValueError
        Si el formato de archivo no está soportado
    FileNotFoundError
        Si el archivo no existe

    Examples
    --------
    >>> # Detecta automáticamente DICOM
    >>> arr, img = read_image('scan.dcm')
    >>> # Detecta automáticamente JPG
    >>> arr, img = read_image('photo.jpg')

    Notes
    -----
    Formatos soportados:
    - DICOM: .dcm
    - Imágenes: .jpg, .jpeg, .png
    """
    path_lower = path.lower()
    
    if path_lower.endswith('.dcm'):
        return read_dicom_file(path)
    elif path_lower.endswith(('.jpg', '.jpeg', '.png')):
        return read_jpg_file(path)
    else:
        raise ValueError(
            f"Unsupported file format. "
            f"Supported: .dcm, .jpg, .jpeg, .png. Got: {path}"
        )
