"""Modulo que carga modelos CNN preentrenados"""

import os
from typing import Optional

import tensorflow as tf


class ModelLoader:
    """
    Clase implementada Singleton para cargar los modelos

    Esta clase se encarga de que solo se cargue una vez el modelo, y se
    pueda acceder a el desde cualquier parte del codigo sin necesidad de cargarlo nuevamente.

    Attributos:
        _instance: Instancia de la clase ModelLoader
        _model: Modelo cargado
    """

    _instance: Optional["ModelLoader"] = None
    _current_path: Optional[str] = None
    _model: Optional[tf.keras.Model] = None

    def __new__(cls):
        """ "
        Crear o devolver la instancia de la clase ModelLoader
        """
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = "conv_MLP_84.h5") -> tf.keras.Model:
        """ "
        Cargar el modelo pre-entrenado desde el .h5

        Se carga una sola vez y se guarda en la instancia para futuras referencias.

        Parametros:
            model_path: str, opcional
                Ruta al archivo .h5 del modelo pre-entrenado. Por defecto es "conv_MLP_84.h5"

        Retorna:
            tf.keras.Model:
                Modelo cargado listo para ser utilizado

        Excepciones:
            FileNotFoundError: Si el archivo del modelo no se encuentra en la ruta especificada.

        Ejemplo de uso:
            loader = ModelLoader()
            model = loader.load_model("ruta_al_modelo.h5")
        """
        if self._model is None or self._current_path != model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"El archivo del modelo no se encuentra en la ruta: {model_path}"
                )
            self._model = tf.keras.models.load_model(model_path, compile=False)
            self._current_path = model_path
        return self._model


def get_model(model_path: str = "conv_MLP_84.h5") -> tf.keras.Model:
    """
    Funcion de conveniencia para acceder al modelo cargado

    Parametros:
        model_path: str, opcional
            Ruta al archivo .h5 del modelo pre-entrenado. Por defecto es "conv_MLP_84.h5"

    Retorna:
        tf.keras.Model:
            Modelo cargado listo para ser utilizado

    Ejemplo de uso:
        loader = ModelLoader()
        predictions = model.predict(img_batch)
    """

    loader = ModelLoader()
    return loader.load_model(model_path)
