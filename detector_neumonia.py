#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time

import cv2 # Ref: REFACTOR_SUMMARY.md - Punto 1
import tensorflow as tf # Ref: REFACTOR_SUMMARY.md - Punto 1
from tensorflow import keras # Ref: REFACTOR_SUMMARY.md - Punto 1
from keras import backend as K # Ref: REFACTOR_SUMMARY.md - Punto 1
import pydicom # Ref: REFACTOR_SUMMARY.md - Punto 1



def model_fun():  # Ref: REFACTOR_SUMMARY.md - Punto 3
    model_cnn = tf.keras.models.load_model('conv_MLP_84.h5',compile = False) # Ref: REFACTOR_SUMMARY.md - Punto 6
    return model_cnn


def grad_cam(array): # Ref: REFACTOR_SUMMARY.md - Punto 9
    img = preprocess(array)
    model = model_fun()
    last_conv_layer_name = 'conv10_thisone' #Esto puede representar un problema porque depende del modelo
    
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"Error: La capa '{last_conv_layer_name}' no se encontró en el modelo. Por favor, verifica el nombre de la capa.")
        return np.zeros((512, 512, 3), dtype=np.uint8) # Devuelve una imagen negra o maneja el error

    # Crear un nuevo modelo que toma la misma entrada y produce la salida final y la salida de la capa conv
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.output, last_conv_layer.output]
    )

    with tf.GradientTape() as tape:
        
        tf_img = tf.convert_to_tensor(img, dtype=tf.float32)

        raw_model_output, last_conv_layer_output = grad_model(tf_img)

        if isinstance(raw_model_output, list):
            model_output_tensor = raw_model_output[0]
        else:
            model_output_tensor = raw_model_output
        
        tape.watch(model_output_tensor)
        
        predicted_class_idx = tf.argmax(model_output_tensor[0])

        class_channel_output = tf.gather(model_output_tensor, indices=predicted_class_idx, axis=1)

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

    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)

    return superimposed_img[:, :, ::-1] # Convertir de BGR (OpenCV) a RGB


def predict(array):
    batch_array_img = preprocess(array)
    model = model_fun()
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    heatmap = grad_cam(array)
    return (label, proba, heatmap)


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


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.result = StringVar()

        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        self.ID_content = self.text1.get()

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array = None

        self.reportID = 0

        self.root.mainloop()

    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath: # Ref: REFACTOR_SUMMARY.md - Punto 8
            if filepath.lower().endswith('.dcm'):
                self.array, img2show = read_dicom_file(filepath)
            elif filepath.lower().endswith(('.jpeg', '.jpg', '.png')):
                self.array, img2show = read_jpg_file(filepath)
            else:
                showinfo(title="Error de Archivo", message="Tipo de archivo no soportado. Por favor, selecciona un archivo DICOM, JPEG, JPG o PNG.")
                return
                
            self.img1 = img2show.resize((250, 250), Image.LANCZOS) # Ref: REFACTOR_SUMMARY.md - Punto 4
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS) # Ref: REFACTOR_SUMMARY.md - Punto 4
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end") # Ref: REFACTOR_SUMMARY.md - Punto 5
            self.text_img2.delete(1.0, "end") # Ref: REFACTOR_SUMMARY.md - Punto 5
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
