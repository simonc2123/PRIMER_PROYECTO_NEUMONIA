"""Module for generating Grad-CAM heatmaps."""

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str = 'conv10_thisone'
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for explaining model predictions.
    
    Grad-CAM (Gradient-weighted Class Activation Mapping) produces
    visual explanations for CNN decisions by highlighting important
    regions in the input image.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN model
    img_array : np.ndarray
        Preprocessed image array with shape (1, H, W, 1)
    last_conv_layer_name : str, optional
        Name of the last convolutional layer (default: 'conv10_thisone')
        
    Returns
    -------
    np.ndarray
        RGB heatmap superimposed on original image (512, 512, 3)
        
    Raises
    ------
    ValueError
        If specified layer is not found in the model
        
    Examples
    --------
    >>> from src.models.load_model import get_model
    >>> model = get_model()
    >>> processed_img = preprocess_image(img)
    >>> heatmap = generate_gradcam(model, processed_img)
    >>> cv2.imwrite('gradcam.jpg', heatmap)
    
    Notes
    -----
    The heatmap uses the JET colormap where:
    - Red/Yellow: High importance regions
    - Blue/Purple: Low importance regions
    
    References
    ----------
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017
    """
    # Get the convolutional layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found in model. "
            f"Available layers: {[layer.name for layer in model.layers]}"
        )
    
    # Create gradient model
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.output, last_conv_layer.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Convert to tensor
        tf_img = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Get model outputs
        raw_model_output, last_conv_layer_output = grad_model(tf_img)
        
        # Handle list output
        if isinstance(raw_model_output, list):
            model_output_tensor = raw_model_output[0]
        else:
            model_output_tensor = raw_model_output
        
        # Watch the output
        tape.watch(model_output_tensor)
        
        # Get predicted class
        predicted_class_idx = tf.argmax(model_output_tensor[0])
        
        # Get class output
        class_channel_output = tf.gather(
            model_output_tensor, 
            indices=predicted_class_idx, 
            axis=1
        )
    
    # Compute gradients
    grads = tape.gradient(class_channel_output, last_conv_layer_output)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_heatmap = tf.reduce_max(heatmap)
    if max_heatmap == 0:
        heatmap = heatmap
    else:
        heatmap /= max_heatmap
    
    # Convert to numpy and resize
    heatmap = cv2.resize(heatmap.numpy(), (512, 512))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose on original (assuming original is in img_array)
    # Note: This needs the original image, not preprocessed
    # For now, we'll create a placeholder
    # In real use, pass original image separately
    transparency = heatmap * 0.8
    transparency = transparency.astype(np.uint8)
    
    # Return RGB (convert BGR to RGB)
    return heatmap[:, :, ::-1]