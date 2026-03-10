#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
from datetime import datetime
from tkinter import (
    END,
    StringVar,
    Text,
    Tk,
    filedialog,
    font,
    ttk,
)
from tkinter.messagebox import askokcancel, showinfo, WARNING

from PIL import Image, ImageTk
import tkcap


from src.integration.integrator import predict_pneumonia
from src.processing.read_img import read_image
from pathlib import Path

# Configuración de directorios de salida
OUTPUT_DIR = Path("outputs")
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORTS_DIR = OUTPUT_DIR / "reports"
HISTORIAL_FILE = OUTPUT_DIR / "historial.csv"

# Crear directorios si no existen
OUTPUT_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class App:
    """
    Aplicación GUI principal para detección de neumonía.

    Proporciona una interfaz gráfica para cargar imágenes de rayos X de tórax,
    realizar predicciones usando un modelo CNN y mostrar resultados
    con visualizaciones Grad-CAM.

    Attributes
    ----------
    root : Tk
        Ventana principal de tkinter
    array : np.ndarray or None
        Array de imagen actualmente cargada
    """

    def __init__(self):
        """Inicializar la aplicación GUI."""
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
            self.root,
            text="Predecir",
            state="disabled",
            command=self.run_model,
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

        self.root.mainloop()

    def load_img_file(self):
        """
        Cargar archivo de imagen desde diálogo y mostrar en GUI.

        Soporta formatos DICOM (.dcm), JPEG (.jpg, .jpeg) y PNG (.png).
        """
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
        if filepath:
            try:
                # Usar función integrada read_image
                self.array, img2show = read_image(filepath)

                # Mostrar imagen
                self.img1 = img2show.resize((250, 250), Image.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                self.text_img1.image_create(END, image=self.img1)
                self.button1["state"] = "enabled"

            except ValueError as e:
                showinfo(title="Error de Archivo", message=str(e))

    def run_model(self):
        """
        Ejecutar modelo de predicción y mostrar resultados.

        Utiliza la función integrada predict_pneumonia para obtener
        clasificación y visualización Grad-CAM.
        """
        # Usar función de predicción integrada
        self.label, self.proba, self.heatmap = predict_pneumonia(self.array)

        # Mostrar mapa de calor
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)

        # Mostrar resultados
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        """Guardar resultados del paciente en archivo CSV."""
        with open(HISTORIAL_FILE, "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            w.writerow(
                [
                    fecha_hora,
                    self.text1.get(),
                    self.label,
                    "{:.2f}".format(self.proba) + "%",
                ]
            )
            showinfo(
                title="Guardar",
                message="Los datos se guardaron con éxito.",
            )

    def create_pdf(self):
        """Generar reporte PDF con información y resultados del paciente."""
        cedula = self.text1.get().strip()

        if not cedula:
            showinfo(
                title="Error",
                message="Por favor, ingresa la cédula del paciente.",
            )
            return

        base_filename = f"Reporte_{cedula}"
        jpg_filename = REPORTS_DIR / f"{base_filename}.jpg"
        pdf_filename = REPORTS_DIR / f"{base_filename}.pdf"

        counter = 2
        while jpg_filename.exists() or pdf_filename.exists():
            jpg_filename = REPORTS_DIR / f"{base_filename}_{counter}.jpg"
            pdf_filename = REPORTS_DIR / f"{base_filename}_{counter}.pdf"
            counter += 1

        cap = tkcap.CAP(self.root)
        img = cap.capture(str(jpg_filename))
        img = Image.open(str(jpg_filename))
        img = img.convert("RGB")
        img.save(str(pdf_filename))
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Limpiar todos los datos de la GUI."""
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            showinfo(
                title="Borrar",
                message="Los datos se borraron con éxito",
            )


def main():
    """Ejecutar la aplicación."""
    my_app = App()
    my_app.mainloop()
    return 0


if __name__ == "__main__":
    main()
