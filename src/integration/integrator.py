"""Módulo para integrar todos los pasos."""

from typing import Tuple

import numpy as np

from ..models.load_model import get_model
from ..processing.preprocess_img import preprocess_image
from ..processing.grad_cam import generate_gradcam


def predict_pneumonia(img_array: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Integrar todos los pasos para predecir el tipo de neumonía.

    Esta función orquesta el pipeline completo de predicción:
    1. Preprocesar la imagen
    2. Cargar el modelo CNN
    3. Realizar predicción
    4. Generar visualización Grad-CAM

    Parameters
    ----------
    img_array : np.ndarray
        Array de imagen original en formato RGB (H, W, 3)

    Returns
    -------
    tuple
        - label : str
            Clase predicha: 'bacteriana', 'normal', o 'viral'
        - probability : float
            Porcentaje de confianza (0-100)
        - heatmap_image : np.ndarray
            Array RGB con superposición Grad-CAM (512, 512, 3)

    Examples
    --------
    >>> import cv2
    >>> from src.processing.read_img import read_image
    >>> img, _ = read_image('chest_xray.dcm')
    >>> label, prob, heatmap = predict_pneumonia(img)
    >>> print(f"Predicción: {label} ({prob:.2f}%)")
    Predicción: normal (87.35%)

    Notes
    -----
    El modelo clasifica radiografías de tórax en tres categorías:
    - Clase 0: Neumonía bacteriana
    - Clase 1: Normal (sin neumonía)
    - Clase 2: Neumonía viral
    """
    batch_array_img = preprocess_image(img_array)

    model = get_model()

    prediction = np.argmax(model.predict(batch_array_img))
    probability = np.max(model.predict(batch_array_img)) * 100

    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = label_map.get(prediction, "desconocida")

    heatmap = generate_gradcam(model, batch_array_img, original_img=img_array)

    return label, probability, heatmap
