#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI para detección de neumonía.

Interfaz de línea de comandos equivalente a la GUI.
Análisis de imágenes individuales con guardado de resultados.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from src.integration.integrator import predict_pneumonia
from src.processing.read_img import read_image


OUTPUT_DIR = Path('outputs')
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
REPORTS_DIR = OUTPUT_DIR / 'reports'
HISTORIAL_FILE = OUTPUT_DIR / 'historial.csv'

OUTPUT_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def analyze_image(image_path: str, patient_id: str = None, save_heatmap: bool = False, 
                  save_csv: bool = False, save_pdf: bool = False):
    """
    Analizar una imagen de rayos X.

    Parameters
    ----------
    image_path : str
        Ruta a la imagen (DICOM, JPG, PNG)
    patient_id : str, optional
        Cédula o ID del paciente
    save_heatmap : bool
        Guardar mapa de calor Grad-CAM
    save_csv : bool
        Guardar resultados en historial CSV
    save_pdf : bool
        Generar reporte PDF
    """
    # Validar archivo
    if not Path(image_path).exists():
        print(f"Error: Archivo no encontrado - {image_path}")
        sys.exit(1)

    print(f"\nAnalizando: {image_path}")
    print("-" * 60)

    try:
        # Leer imagen
        img_array, img_original = read_image(image_path)

        # Realizar predicción
        label, probabilidad, heatmap = predict_pneumonia(img_array)

        # Mostrar resultados
        print(f"\n{'='*60}")
        print(f"DIAGNÓSTICO: {label.upper()}")
        print(f"Confianza:   {probabilidad:.2f}%")
        print(f"{'='*60}\n")

        heatmap_path = None

        # Guardar heatmap si se solicita
        if save_heatmap:
            heatmap_path = HEATMAP_DIR / f"heatmap_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(heatmap_path), heatmap[:, :, ::-1])
            print(f"✓ Heatmap guardado: {heatmap_path}")

        # Guardar en CSV si se solicita
        if save_csv:
            with open(HISTORIAL_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='-')
                fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([
                    fecha_hora,
                    patient_id or 'N/A',
                    label,
                    f"{probabilidad:.2f}%"
                ])
            print(f"✓ Guardado en historial: {HISTORIAL_FILE}")

        # Generar PDF si se solicita
        if save_pdf:
            if not patient_id:
                print("⚠ Advertencia: Se requiere cédula del paciente para generar PDF")
                print("  Use: python cli.py imagen.jpg -p <cedula> --pdf")
            else:
                # Generar nombre del PDF
                base_filename = f"Reporte_{patient_id}"
                pdf_filename = REPORTS_DIR / f"{base_filename}.pdf"
                
                counter = 2
                while pdf_filename.exists():
                    pdf_filename = REPORTS_DIR / f"{base_filename}_{counter}.pdf"
                    counter += 1

                # Crear imagen combinada: original + heatmap
                img_pil = img_original.resize((400, 400), Image.LANCZOS)
                heatmap_pil = Image.fromarray(heatmap).resize((400, 400), Image.LANCZOS)
                
                # Crear imagen combinada horizontal
                combined = Image.new('RGB', (820, 500), 'white')
                combined.paste(img_pil, (10, 50))
                combined.paste(heatmap_pil, (410, 50))
                
                # Agregar texto con resultado
                from PIL import ImageDraw
                draw = ImageDraw.Draw(combined)
                
                # Título
                draw.text((250, 10), "REPORTE DE DIAGNÓSTICO", fill='black')
                draw.text((230, 30), f"ID Paciente: {patient_id}", fill='black')
                
                # Resultados abajo
                draw.text((100, 460), f"Diagnóstico: {label.upper()}", fill='black')
                draw.text((500, 460), f"Confianza: {probabilidad:.2f}%", fill='black')
                
                # Guardar como PDF
                combined.save(str(pdf_filename), 'PDF', resolution=100.0)
                print(f"✓ PDF generado: {pdf_filename}")

        return 0

    except Exception as e:
        print(f"\nError al procesar imagen: {e}")
        return 1


def interactive_mode():
    """
    Modo interactivo con menú (equivalente a GUI).
    
    Permite usar el sistema como la GUI pero en terminal.
    """
    print("\n" + "="*60)
    print("🏥 SISTEMA DE DETECCIÓN DE NEUMONÍA")
    print("Modo Interactivo - CLI")
    print("="*60)
    
    while True:
        print("\n📋 MENÚ PRINCIPAL:")
        print("  1. Analizar imagen")
        print("  2. Ver historial")
        print("  3. Limpiar historial")
        print("  4. Salir")
        
        choice = input("\n👉 Seleccione opción (1-4): ").strip()
        
        if choice == '1':
            # Análisis de imagen
            print("\n--- ANÁLISIS DE IMAGEN ---")
            img_path = input("📁 Ruta de la imagen: ").strip().strip('"').strip("'")
            
            if not img_path:
                print("❌ Ruta vacía")
                continue
                
            if not Path(img_path).exists():
                print(f"❌ Archivo no encontrado: {img_path}")
                continue
            
            patient_id = input("👤 Cédula del paciente (Enter para omitir): ").strip() or None
            
            # Opciones de guardado
            print("\n📋 Opciones de guardado:")
            save_heatmap = input("💾 ¿Guardar heatmap? (s/n) [n]: ").lower() == 's'
            save_csv = input("💾 ¿Guardar en historial CSV? (s/n) [s]: ").lower() != 'n'
            save_pdf = input("📄 ¿Generar reporte PDF? (s/n) [n]: ").lower() == 's'
            
            # Realizar análisis
            analyze_image(img_path, patient_id, save_heatmap, save_csv, save_pdf)
            
        elif choice == '2':
            # Ver historial
            print("\n--- HISTORIAL ---")
            if HISTORIAL_FILE.exists():
                with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"\nTotal de registros: {len(lines) - 1}\n")
                        # Mostrar últimos 10 registros
                        for line in lines[-11:]:
                            print(line.strip())
                    else:
                        print("Historial vacío")
            else:
                print("❌ No existe archivo historial")
                
        elif choice == '3':
            # Limpiar historial
            if HISTORIAL_FILE.exists():
                confirm = input("⚠️  ¿Seguro que desea limpiar el historial? (s/n): ").lower()
                if confirm == 's':
                    HISTORIAL_FILE.unlink()
                    print("✅ Historial eliminado")
            else:
                print("No hay historial para limpiar")
                
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción inválida")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Sistema de Detección de Neumonía - CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s -i                                      # Modo interactivo (recomendado)
  %(prog)s imagen.dcm                              # Análisis básico
  %(prog)s imagen.jpg -p 123456789                 # Con cédula del paciente
  %(prog)s imagen.png --heatmap                    # Guardar mapa de calor
  %(prog)s imagen.dcm -p 123 --heatmap --csv       # Guardar heatmap y CSV
  %(prog)s imagen.jpg -p 123 --heatmap --csv --pdf # Guardar todo (como GUI)

Formatos soportados: DICOM (.dcm), JPEG (.jpg, .jpeg), PNG (.png)
        """
    )

    parser.add_argument(
        'image',
        nargs='?',
        help='Ruta a la imagen de rayos X'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Modo interactivo (recomendado para Docker)'
    )

    parser.add_argument(
        '-p', '--paciente',
        metavar='ID',
        help='Cédula o ID del paciente'
    )

    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Guardar mapa de calor (Grad-CAM) en outputs/heatmaps/'
    )

    parser.add_argument(
        '--csv',
        action='store_true',
        help='Guardar resultados en historial CSV (outputs/historial.csv)'
    )

    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Generar reporte PDF (outputs/reports/) - requiere -p/--paciente'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 2.0 - Equivalente a GUI'
    )

    args = parser.parse_args()
    
    
    try:
        if args.interactive:
            interactive_mode()
            return 0
        elif args.image:
            return analyze_image(
                args.image, 
                args.paciente, 
                args.heatmap, 
                args.csv, 
                args.pdf
            )
        else:
            print("❌ Debe especificar una imagen o usar modo interactivo (-i)")
            print("\nUso básico:")
            print("  python cli.py -i                 # Modo interactivo (recomendado)")
            print("  python cli.py imagen.jpg         # Análisis directo")
            print("\nUse -h para ver ayuda completa")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
