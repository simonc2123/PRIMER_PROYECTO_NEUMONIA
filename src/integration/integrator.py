"""Module for integrating all processing steps."""

from typing import Tuple

import numpy as np

from ..models.load_model import get_model
from ..processing.preprocess_img import preprocess_image
from ..processing.grad_cam import generate_gradcam


def predict_pneumonia(img_array: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Integrate all steps to predict pneumonia type.

    This function orchestrates the complete prediction pipeline:
    1. Preprocess the image
    2. Load the CNN model
    3. Make prediction
    4. Generate Grad-CAM visualization

    Parameters
    ----------
    img_array : np.ndarray
        Original image array in RGB format (H, W, 3)

    Returns
    -------
    tuple
        - label : str
            Predicted class: 'bacteriana', 'normal', or 'viral'
        - probability : float
            Confidence percentage (0-100)
        - heatmap_image : np.ndarray
            RGB array with Grad-CAM overlay (512, 512, 3)

    Examples
    --------
    >>> import cv2
    >>> from src.processing.read_img import read_image
    >>> img, _ = read_image('chest_xray.dcm')
    >>> label, prob, heatmap = predict_pneumonia(img)
    >>> print(f"Prediction: {label} ({prob:.2f}%)")
    Prediction: normal (87.35%)

    Notes
    -----
    The model classifies chest X-rays into three categories:
    - Class 0: Bacterial pneumonia
    - Class 1: Normal (no pneumonia)
    - Class 2: Viral pneumonia
    """
    # Preprocess image
    batch_array_img = preprocess_image(img_array)

    # Load model
    model = get_model()

    # Make prediction
    prediction = np.argmax(model.predict(batch_array_img))
    probability = np.max(model.predict(batch_array_img)) * 100

    # Map prediction to label
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = label_map.get(prediction, "desconocida")

    # Generate Grad-CAM heatmap with original image overlay
    heatmap = generate_gradcam(model, batch_array_img, original_img=img_array)

    return label, probability, heatmap
