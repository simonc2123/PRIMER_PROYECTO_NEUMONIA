""" "Pruebas unitarias para el modulo de preprocesamiento de imagenes"""

import numpy as np

from src.processing.preprocess_img import preprocess_image


def test_preprocess_image():
    """ "
    Verificamos que el preprocesamiento de la imagen se realice correctamente.

    la salida de la funcion debe ser un arreglo con forma (1, 512, 512, 1).
    """
    # Generamos una imagen de prueba 512x512 con 3 canales (RGB)
    test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    result = preprocess_image(test_img)

    # Revisamos la forma de la imagen preprocesada
    assert result.shape == (
        1,
        512,
        512,
        1,
    ), f"Forma esperada (1, 512, 512, 1) obtenida {result.shape}"


def test_preprocess_normalization():
    """ "
    Verificamos que los valores de píxeles estén normalizados en el rango [0, 1].
    """
    test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    result = preprocess_image(test_img)

    # Revisamos que los valores estén en el rango [0, 1]
    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "Los valores de píxeles no están normalizados en el rango [0, 1]"


def test_preprocess_grayscale_conversion():
    """
    Verificar que la salida tiene un solo canal (escala de grises),
    ya que el modelo espera una imagen en escala de grises.
    """
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    result = preprocess_image(dummy_img)

    # La última dimensión debe ser 1 (escala de grises)
    assert result.shape[-1] == 1, f"Se esperaba 1 canal, obtenido {result.shape[-1]}"


def test_preprocess_custom_size():
    """
    Verificar preprocesamiento con tamaño objetivo personalizado.
     El resultado debe tener la forma (1, target_size[0], target_size[1], 1).
    """
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Usar tamaño personalizado
    result = preprocess_image(dummy_img, target_size=(256, 256))

    assert result.shape == (
        1,
        256,
        256,
        1,
    ), f"Forma esperada (1, 256, 256, 1), obtenida {result.shape}"
