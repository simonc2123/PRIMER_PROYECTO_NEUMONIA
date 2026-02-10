# Resumen de Refactorización: `detector_neumonia.py`

**Fecha:** 2026-02-09
**Objetivo:** Habilitar la ejecución del proyecto de detección de neumonía y adaptarlo a un entorno moderno de Python/TensorFlow, corrigiendo errores de compatibilidad y lógica.

---

### 1. Importaciones de Librerías y Aliases Incorrectos o Faltantes

*   **Problema Original:**
    *   Las variables `tf` (TensorFlow) y `K` (Keras Backend) se utilizaban sin importaciones explícitas, causando `NameError`.
    *   La librería `pydicom` no se importaba correctamente (`dicom.read_file` causaba `NameError` o `AttributeError`).
    *   El `import cv2` estaba colocado de forma inconsistente, pudiendo generar `NameError` si se usaba antes de su definición.
*   **Razón del Fallo:** El intérprete de Python no podía encontrar los módulos o aliases requeridos en el ámbito global.
*   **Solución:**
    *   Se añadió `import tensorflow as tf`.
    *   Se añadió `from tensorflow import keras` y `from keras import backend as K`.
    *   Se estandarizó la importación de `pydicom` a `import pydicom` (eliminando `as dicom`).
    *   Se movió `import cv2` a la sección de importaciones iniciales.

---

### 2. Nombre Obsoleto de la Función `read_file` en `pydicom`

*   **Problema Original:** Después de corregir la importación de `pydicom`, el uso de `pydicom.read_file(path)` generaba un `AttributeError: module 'pydicom' has no attribute 'read_file'`.
*   **Razón del Fallo:** La función `read_file` de `pydicom` fue renombrada a `dcmread` en versiones modernas de la librería (a partir de la v1.0). El código estaba utilizando una convención de nombre antigua.
*   **Solución:** Se actualizó la llamada a la función en `read_dicom_file` a `img = pydicom.dcmread(path)`.

---

### 3. Función `model_fun()` no Definida

*   **Problema Original:** Las funciones `grad_cam` y `predict` llamaban a `model_fun()`, pero esta función no existía en el script.
*   **Razón del Fallo:** `NameError` al intentar invocar una función no definida.
*   **Solución:** Se definió la función `model_fun()` encargada de cargar el modelo Keras pre-entrenado (`conv_MLP_84.h5`).

---

### 4. Uso de `Image.ANTIALIAS` Obsoleto

*   **Problema Original:** Se utilizaba la constante `Image.ANTIALIAS` de la librería Pillow, la cual ha sido marcada como obsoleta en versiones recientes.
*   **Razón del Fallo:** Potenciales advertencias o futuros errores de compatibilidad con versiones de Pillow.
*   **Solución:** Se reemplazó `Image.ANTIALIAS` por `Image.LANCZOS` en `load_img_file` y `run_model`.

---

### 5. Borrado Incorrecto de Contenido en `delete()`

*   **Problema Original:** El método `delete()` intentaba limpiar los widgets `Text` utilizando referencias a imágenes (`self.img1`, `self.img2`) como argumentos para el método `delete()`, lo cual es incorrecto para Tkinter `Text` widgets.
*   **Razón del Fallo:** Comportamiento inesperado o `TypeError` en el método `delete()`.
*   **Solución:** Se cambió la forma de limpiar los widgets `Text` a `self.text_imgX.delete(1.0, "end")`, que es el método correcto para eliminar todo su contenido.

---

### 6. Incompatibilidad `reduction=auto` al Cargar el Modelo Keras

*   **Problema Original:** Al cargar el modelo (`.h5`), se producía un `ValueError` debido a que el parámetro `reduction=auto` (guardado con el modelo) no es un valor válido en las versiones recientes de TensorFlow/Keras.
*   **Razón del Fallo:** `ValueError` durante el proceso de deserialización del modelo, causado por un cambio en la API de Keras/TensorFlow.
*   **Solución:** Se añadió `compile=False` como argumento a `tf.keras.models.load_model('conv_MLP_84.h5', compile=False)`. Esto instruye a Keras a cargar solo la arquitectura y los pesos del modelo, sin intentar reconstruir la configuración de compilación (que no es necesaria para la inferencia).

---

### 7. Conflicto con la Ejecución Eager de TensorFlow

*   **Problema Original:** La línea `tf.compat.v1.disable_eager_execution()` deshabilitaba el modo eager de TensorFlow 2.x, lo que provocaba un `RuntimeError` cuando las funciones internas de Keras (como `model.predict` que usa `tf.data.Dataset`) esperaban el modo eager.
*   **Razón del Fallo:** Conflicto de modos de ejecución entre las API de TensorFlow 1.x (gráficos simbólicos) y TensorFlow 2.x (eager por defecto).
*   **Solución:** Se eliminaron las líneas `tf.compat.v1.disable_eager_execution()` y `tf.compat.v1.experimental.output_all_intermediates(True)`.

---

### 8. Manejo Incorrecto de Tipos de Archivo en `load_img_file`

*   **Problema Original:** La función `load_img_file` siempre llamaba a `read_dicom_file` independientemente de la extensión del archivo cargado.
*   **Razón del Fallo:** Intentos de leer archivos JPG/PNG como si fueran DICOMs con `pydicom`, lo que causaba errores en tiempo de ejecución.
*   **Solución:** Se implementó una lógica condicional (`if/elif/else`) basada en la extensión del archivo (`.dcm`, `.jpeg`, `.jpg`, `.png`) para llamar a `read_dicom_file` o `read_jpg_file` apropiadamente.

---

### 9. Incompatibilidad y Errores de Indexación en `grad_cam` (para Eager Mode)

*   **Problema Original:**
    *   La implementación original de `grad_cam` utilizaba construcciones de modo de gráfico simbólico (`model.output`, `K.gradients`, `K.function`) que no son compatibles con el modo eager de TensorFlow 2.x.
    *   Los intentos de adaptar la función resultaron en `TypeError: list indices must be integers or slices, not tuple` al intentar indexar `model_output` o `class_channel_output`.
*   **Razón del Fallo:** Uso de APIs obsoletas o incorrectas para el modo eager, y manipulación incorrecta de tensores de TensorFlow como si fueran listas de Python durante la indexación.
*   **Solución:**
    *   Se reescribió `grad_cam` para utilizar `tf.GradientTape`, el mecanismo de TensorFlow 2.x para el cálculo de gradientes en modo eager.
    *   Se creó un `grad_model` temporal que permite extraer tanto la salida final del modelo como las activaciones de la capa convolucional intermedia.
    *   Se aseguró la conversión de la imagen de entrada a `tf.Tensor` (`tf.convert_to_tensor`).
    *   Se añadió `tape.watch(model_output_tensor)` para asegurar que el `GradientTape` registre el tensor de salida.
    *   Se reemplazó la indexación problemática (`model_output[:, predicted_class]`) por `tf.gather(model_output_tensor, indices=predicted_class_idx, axis=1)`, que es la forma correcta y robusta de seleccionar elementos de un tensor utilizando un índice de tensor en TensorFlow.
    *   Se añadió un manejo de errores básico si la capa `"conv10_thisone"` no se encuentra en el modelo.

---
