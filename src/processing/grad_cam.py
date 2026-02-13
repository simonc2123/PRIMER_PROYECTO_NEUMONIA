"""Módulo para generar mapas de calor Grad-CAM."""

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str = "conv10_thisone",
    original_img: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Generar mapa de calor Grad-CAM para explicar predicciones del modelo.

    Grad-CAM (Mapeo de Activación de Clase Ponderado por Gradiente)
    produce explicaciones visuales para decisiones de CNN resaltando
    regiones importantes en la imagen de entrada.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo CNN entrenado
    img_array : np.ndarray
        Array de imagen preprocesado con forma (1, H, W, 1)
    last_conv_layer_name : str, optional
        Nombre de la última capa convolucional (por defecto: 'conv10_thisone')
    original_img : np.ndarray, optional
        Imagen RGB original para superponer el mapa de calor (H, W, 3)
        Si es None, devuelve solo el mapa de calor
    alpha : float, optional
        Opacidad de la superposición del mapa de calor (0.0 a 1.0, por defecto: 0.4)
        Valores más altos hacen el mapa de calor más visible

    Returns
    -------
    np.ndarray
        Mapa de calor RGB superpuesto en imagen original (512, 512, 3)

    Raises
    ------
    ValueError
        Si la capa especificada no se encuentra en el modelo

    Examples
    --------
    >>> from src.models.load_model import get_model
    >>> model = get_model()
    >>> processed_img = preprocess_image(img)
    >>> heatmap = generate_gradcam(model, processed_img)
    >>> cv2.imwrite('gradcam.jpg', heatmap)
    """

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(
            f"Capa '{last_conv_layer_name}' no encontrada en el modelo. "
            f"Capas disponibles: {[layer.name for layer in model.layers]}"
        )


    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[model.output, last_conv_layer.output]
    )


    with tf.GradientTape() as tape:
  
        tf_img = tf.convert_to_tensor(img_array, dtype=tf.float32)

        raw_model_output, last_conv_layer_output = grad_model(tf_img)


        if isinstance(raw_model_output, list):
            model_output_tensor = raw_model_output[0]
        else:
            model_output_tensor = raw_model_output


        tape.watch(model_output_tensor)

        predicted_class_idx = tf.argmax(model_output_tensor[0])

        class_channel_output = tf.gather(
            model_output_tensor, indices=predicted_class_idx, axis=1
        )

    grads = tape.gradient(class_channel_output, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_heatmap = tf.reduce_max(heatmap)
    if max_heatmap == 0:
        heatmap = heatmap
    else:
        heatmap /= max_heatmap

    heatmap = cv2.resize(heatmap.numpy(), (512, 512))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_img is None:
        return heatmap[:, :, ::-1]

    img_resized = cv2.resize(original_img, (512, 512))

    if img_resized.dtype != np.uint8:
        img_resized = np.uint8(img_resized)

    if len(img_resized.shape) == 2:  
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    else: 
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return superimposed_img[:, :, ::-1]
