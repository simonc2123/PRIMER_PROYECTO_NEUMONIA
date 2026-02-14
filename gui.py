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

# Paleta de colores profesional
COLOR_PRIMARY = "#1e3a8a"      # Azul oscuro profesional
COLOR_SECONDARY = "#3b82f6"    # Azul más claro
COLOR_SUCCESS = "#10b981"      # Verde
COLOR_WARNING = "#f59e0b"      # Naranja
COLOR_DANGER = "#ef4444"       # Rojo
COLOR_BG = "#f8fafc"           # Fondo claro
COLOR_CARD = "#ffffff"         # Blanco para tarjetas
COLOR_TEXT = "#1f2937"         # Texto oscuro
COLOR_BORDER = "#e2e8f0"       # Borde gris claro


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
        self.root.geometry("900x680")
        self.root.resizable(0, 0)
        self.root.configure(bg=COLOR_BG)

        # Configurar estilo ttk
        self._setup_styles()

        # Header/Título principal
        header_frame = ttk.Frame(self.root, style="Header.TFrame")
        header_frame.place(x=0, y=0, width=900, height=70)

        title_font = font.Font(family="Segoe UI", size=16, weight="bold")
        title_label = ttk.Label(
            header_frame,
            text="🏥 DETECTOR DE NEUMONÍA - Apoyo al Diagnóstico Médico",
            font=title_font,
            style="Header.TLabel"
        )
        title_label.pack(pady=15)

        # Frame principal con dos columnas
        main_frame = ttk.Frame(self.root, style="Main.TFrame")
        main_frame.place(x=15, y=80, width=870, height=570)

        # Columna izquierda (Imagen original)
        left_frame = ttk.LabelFrame(
            main_frame,
            text="📋 Imagen Radiográfica Original",
            style="Card.TLabelframe"
        )
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.text_img1 = Text(
            left_frame,
            width=31,
            height=15,
            bg=COLOR_CARD,
            bd=0,
            relief="flat",
            state="disabled"
        )
        self.text_img1.pack(pady=10, padx=10)

        # Columna derecha (Heatmap)
        right_frame = ttk.LabelFrame(
            main_frame,
            text="🔥 Mapa de Calor (Grad-CAM)",
            style="Card.TLabelframe"
        )
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.text_img2 = Text(
            right_frame,
            width=31,
            height=15,
            bg=COLOR_CARD,
            bd=0,
            relief="flat",
            state="disabled"
        )
        self.text_img2.pack(pady=10, padx=10)

        # Frame para información del paciente y resultados
        info_frame = ttk.LabelFrame(
            self.root,
            text="📊 Información del Paciente y Resultados",
            style="Card.TLabelframe"
        )
        info_frame.place(x=15, y=470, width=870, height=120)

        # Fila 1: Cédula del paciente
        cedula_label = ttk.Label(
            info_frame,
            text="Cédula del Paciente:",
            style="Bold.TLabel"
        )
        cedula_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.ID = StringVar()
        self.text1 = ttk.Entry(info_frame, textvariable=self.ID, width=20)
        self.text1.grid(row=0, column=1, padx=15, pady=10, sticky="w")

        result_label = ttk.Label(
            info_frame,
            text="Resultado:",
            style="Bold.TLabel"
        )
        result_label.grid(row=0, column=2, padx=15, pady=10, sticky="w")

        self.text2 = Text(info_frame, width=15, height=1, bg=COLOR_CARD, bd=1, relief="solid", state="disabled")
        self.text2.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        # Fila 2: Probabilidad
        proba_label = ttk.Label(
            info_frame,
            text="Probabilidad:",
            style="Bold.TLabel"
        )
        proba_label.grid(row=1, column=0, padx=15, pady=10, sticky="w")

        self.text3 = Text(info_frame, width=15, height=1, bg=COLOR_CARD, bd=1, relief="solid", state="disabled")
        self.text3.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Frame para botones
        button_frame = ttk.Frame(self.root, style="Main.TFrame")
        button_frame.place(x=15, y=595, width=870, height=70)

        # Botones con colores y estilos mejorados
        self.button2 = ttk.Button(
            button_frame,
            text="📂 Cargar Imagen",
            command=self.load_img_file,
            style="Primary.TButton"
        )
        self.button2.grid(row=0, column=0, padx=8, pady=10)

        self.button1 = ttk.Button(
            button_frame,
            text="🔍 Predecir",
            state="disabled",
            command=self.run_model,
            style="Success.TButton"
        )
        self.button1.grid(row=0, column=1, padx=8, pady=10)

        self.button6 = ttk.Button(
            button_frame,
            text="💾 Guardar CSV",
            command=self.save_results_csv,
            style="Info.TButton"
        )
        self.button6.grid(row=0, column=2, padx=8, pady=10)

        self.button4 = ttk.Button(
            button_frame,
            text="📄 Generar PDF",
            command=self.create_pdf,
            style="Warning.TButton"
        )
        self.button4.grid(row=0, column=3, padx=8, pady=10)

        self.button3 = ttk.Button(
            button_frame,
            text="🗑️  Limpiar",
            command=self.delete,
            style="Danger.TButton"
        )
        self.button3.grid(row=0, column=4, padx=8, pady=10)

        self.text1.focus_set()
        self.array = None
        self.result = StringVar()
        self.ID_content = self.text1.get()

        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        info_frame.grid_columnconfigure(3, weight=1)

        self.root.mainloop()

    def _setup_styles(self):
        """Configurar estilos personalizados para ttk."""
        style = ttk.Style()
        style.theme_use("clam")

        # Header
        style.configure(
            "Header.TFrame",
            background=COLOR_PRIMARY,
            relief="flat"
        )
        style.configure(
            "Header.TLabel",
            background=COLOR_PRIMARY,
            foreground="white",
            font=("Segoe UI", 14, "bold")
        )

        # Main Frame
        style.configure("Main.TFrame", background=COLOR_BG)

        # LabelFrame (Cards)
        style.configure(
            "Card.TLabelframe",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            borderwidth=2,
            relief="solid"
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLOR_BG,
            foreground=COLOR_PRIMARY,
            font=("Segoe UI", 11, "bold")
        )

        # Labels
        style.configure(
            "Bold.TLabel",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10, "bold")
        )

        # Entry
        style.configure(
            "TEntry",
            fieldbackground=COLOR_CARD,
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            relief="solid",
            borderwidth=1
        )

        # Botones primarios
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            borderwidth=1
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLOR_SECONDARY)],
            foreground=[("active", "white")]
        )

        # Botones success (verde)
        style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            borderwidth=1
        )
        style.map(
            "Success.TButton",
            background=[("active", "#059669")]
        )

        # Botones info (azul claro)
        style.configure(
            "Info.TButton",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            borderwidth=1
        )
        style.map(
            "Info.TButton",
            background=[("active", "#0891b2")]
        )

        # Botones warning (naranja)
        style.configure(
            "Warning.TButton",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            borderwidth=1
        )
        style.map(
            "Warning.TButton",
            background=[("active", "#d97706")]
        )

        # Botones danger (rojo)
        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            borderwidth=1
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#dc2626")]
        )

    def load_img_file(self):
        """
        Cargar archivo de imagen desde diálogo y mostrar en GUI.

        Soporta formatos DICOM (.dcm), JPEG (.jpg, .jpeg) y PNG (.png).
        """
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen radiográfica",
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
                self.text_img1.config(state="normal")
                self.text_img1.image_create(END, image=self.img1)
                self.text_img1.config(state="disabled")
                self.button1["state"] = "enabled"

            except ValueError as e:
                showinfo(title="❌ Error de Archivo", message=str(e))

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
        self.text_img2.config(state="normal")
        self.text_img2.image_create(END, image=self.img2)
        self.text_img2.config(state="disabled")

        # Mostrar resultados (habilitar escritura temporal)
        self.text2.config(state="normal")
        self.text2.insert(END, self.label)
        self.text2.config(state="disabled")
        
        self.text3.config(state="normal")
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")
        self.text3.config(state="disabled")

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
                title="✅ Guardar",
                message="Los datos se guardaron con éxito.",
            )

    def create_pdf(self):
        """Generar reporte PDF con información y resultados del paciente."""
        cedula = self.text1.get().strip()

        if not cedula:
            showinfo(
                title="⚠️  Error",
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
        showinfo(title="✅ PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Limpiar todos los datos de la GUI."""
        answer = askokcancel(
            title="⚠️  Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if answer:
            self.text1.delete(0, "end")
            
            self.text2.config(state="normal")
            self.text2.delete(1.0, "end")
            self.text2.config(state="disabled")
            
            self.text3.config(state="normal")
            self.text3.delete(1.0, "end")
            self.text3.config(state="disabled")
            
            self.text_img1.config(state="normal")
            self.text_img1.delete(1.0, "end")
            self.text_img1.config(state="disabled")
            
            self.text_img2.config(state="normal")
            self.text_img2.delete(1.0, "end")
            self.text_img2.config(state="disabled")
            
            showinfo(
                title="✅ Limpiar",
                message="Los datos se borraron con éxito",
            )


def main():
    """Ejecutar la aplicación."""
    my_app = App()
    my_app.mainloop()
    return 0


if __name__ == "__main__":
    main()
