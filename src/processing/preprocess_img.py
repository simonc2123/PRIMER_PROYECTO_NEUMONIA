"""Modulo para preprocesamiento de las imagenes para entrada del modelo"""

from typing import Tuple

import cv2
import numpy as np


def preprocess_image(
    array: np.ndarray, target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Preprocesamos el arreglo de una imagen para que pueda ser ingresada al modelo.

    Aplica estas transformaciones:
    1. Redimensiona la imagne al tamaño objetivo (default 512x512).
    2. Convierte a escala de grises.
    3. Aplica ecualización de histograma CLAHE.
    4. Normaliza los valores de píxeles al rango [0, 1].
    5. Añade dimensiones adicionales para batch y canal.

    Parámetros:
        array : (np.ndarray):
            Arreglo de la imagen original en formato RGB.
        target_size : (Tuple[int, int], opcional):
            Tamaño objetivo para redimensionar. Por defecto es (512, 512).

    Retorna:
        np.ndarray:
            Arreglo preprocesado listo para ser ingresado al modelo con forma
            (1, target_size[0], target_size[1], 1).

    Ejemplo de uso:
        img = cv2.imread("ruta_imagen.jpg")
        processed = preprocess_image(img)
        processed.shape  # Debería ser (1, 512, 512, 1)
    """
    array = cv2.resize(array, target_size)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255  # Normalización
    array = np.expand_dims(array, axis=-1)  # Añadir canal
    array = np.expand_dims(array, axis=0)  # Añadir batch
    return array
