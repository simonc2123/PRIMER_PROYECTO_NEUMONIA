import os
from typing import Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image


def read_dicom_file(path):
    
    img = pydicom.dcmread(path) # Ref: REFACTOR_SUMMARY.md - Punto 2
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
    
def read_image(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Read image file with automatic format detection.
    
    Factory function that determines file type based on extension
    and calls the appropriate reader.
    
    Parameters
    ----------
    path : str
        Path to image file (DICOM, JPG, JPEG, or PNG)
        
    Returns
    -------
    tuple
        - rgb_array : np.ndarray
            RGB numpy array
        - pil_image : PIL.Image.Image
            PIL Image object
            
    Raises
    ------
    ValueError
        If file format is not supported
    FileNotFoundError
        If file doesn't exist
        
    Examples
    --------
    >>> # Automatically detects DICOM
    >>> arr, img = read_image('scan.dcm')
    >>> # Automatically detects JPG
    >>> arr, img = read_image('photo.jpg')
    
    Notes
    -----
    Supported formats:
    - DICOM: .dcm
    - Images: .jpg, .jpeg, .png
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
