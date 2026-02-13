#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""API REST para detección de neumonía."""

import io
import os
from datetime import datetime

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image

from src.integration.integrator import predict_pneumonia
from src.processing.read_img import read_image

app = Flask(__name__, static_folder="static")
CORS(app)  # Permitir CORS para desarrollo


@app.route("/", methods=["GET"])
def home():
    """Servir interfaz web HTML."""
    return send_from_directory("static", "index.html")


@app.route("/api/info", methods=["GET"])
def api_info():
    """Información de la API."""
    return jsonify(
        {
            "nombre": "API de Detección de Neumonía",
            "versión": "1.0.0",
            "endpoints": {
                "/": "Interfaz web",
                "/api/info": "Información de la API",
                "/health": "Estado de salud",
                "/predict": "POST - Realizar predicción de neumonía",
                "/predict-with-heatmap": "POST - Predicción con mapa de calor",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Verificar estado de salud de la API."""
    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "modelo": "conv_MLP_84.h5",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Realizar predicción de neumonía en radiografía.

    Espera un archivo de imagen en formato multipart/form-data.

    Returns
    -------
    JSON con:
        - label: Tipo de diagnóstico (bacteriana, normal, viral)
        - probabilidad: Confianza de la predicción (0-100)
        - timestamp: Marca de tiempo de la predicción
    """
    # Verificar que se envió un archivo
    if "image" not in request.files:
        return (
            jsonify(
                {
                    "error": "No se encontró archivo de imagen",
                    "mensaje": "Debe enviar un archivo con la clave 'image'",
                }
            ),
            400,
        )

    file = request.files["image"]

    # Verificar que el archivo tiene nombre
    if file.filename == "":
        return (
            jsonify(
                {
                    "error": "Archivo sin nombre",
                    "mensaje": "El archivo debe tener un nombre válido",
                }
            ),
            400,
        )

    try:
        # Guardar temporalmente el archivo
        temp_path = f"temp_{datetime.now().timestamp()}_{file.filename}"
        file.save(temp_path)

        # Leer y procesar imagen
        img_array, _ = read_image(temp_path)

        # Realizar predicción
        label, probabilidad, heatmap = predict_pneumonia(img_array)

        # Limpiar archivo temporal
        os.remove(temp_path)

        # Retornar resultado
        return (
            jsonify(
                {
                    "label": label,
                    "probabilidad": round(float(probabilidad), 2),
                    "diagnostico": {
                        "bacteriana": "Neumonía Bacteriana",
                        "viral": "Neumonía Viral",
                        "normal": "Sin Neumonía",
                    }.get(label, "Desconocido"),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except FileNotFoundError as e:
        return jsonify({"error": "Archivo no encontrado", "mensaje": str(e)}), 404

    except ValueError as e:
        return jsonify({"error": "Error al procesar imagen", "mensaje": str(e)}), 400

    except Exception as e:
        return jsonify({"error": "Error interno del servidor", "mensaje": str(e)}), 500


@app.route("/predict-with-heatmap", methods=["POST"])
def predict_with_heatmap():
    """
    Realizar predicción y devolver imagen con mapa de calor Grad-CAM.

    Returns
    -------
    Imagen PNG con el mapa de calor superpuesto
    """
    if "image" not in request.files:
        return jsonify({"error": "No se encontró archivo de imagen"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Archivo sin nombre"}), 400

    try:
        # Guardar temporalmente el archivo
        temp_path = f"temp_{datetime.now().timestamp()}_{file.filename}"
        file.save(temp_path)

        # Leer y procesar imagen
        img_array, _ = read_image(temp_path)

        # Realizar predicción
        label, probabilidad, heatmap = predict_pneumonia(img_array)

        # Limpiar archivo temporal
        os.remove(temp_path)

        # Convertir heatmap a bytes
        heatmap_img = Image.fromarray(heatmap.astype("uint8"))
        img_io = io.BytesIO()
        heatmap_img.save(img_io, "PNG")
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"heatmap_{label}_{datetime.now().timestamp()}.png",
        )

    except Exception as e:
        return jsonify({"error": "Error al procesar imagen", "mensaje": str(e)}), 500


if __name__ == "__main__":
    # Ejecutar servidor
    app.run(host="0.0.0.0", port=5000, debug=False)
