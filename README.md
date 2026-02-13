# Sistema de Detección de Neumonía con Deep Learning

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Sistema de apoyo al diagnóstico médico que utiliza Deep Learning para clasificar radiografías de tórax (DICOM/JPG/PNG) en tres categorías: **Neumonía Bacteriana**, **Neumonía Viral** o **Sin Neumonía**.

Se realizó la reestructuración completa del sistema, se refactorizó el código que originalmente era un Monolito, y se transformó en módulos, además se agregaron dos interfaces más además de la GUI original, para poder usarlo en Docker por medio de CLI y API.

---

## Estructura del Proyecto

```
PRIMER_PROYECTO_NEUMONIA/
├── src/                 # Módulos principales (models, processing, integration)
├── outputs/             # Archivos generados
│   ├── heatmaps/        # Mapas de calor Grad-CAM
│   ├── reports/         # Reportes PDF
│   └── historial.csv    # Registro de análisis
├── tests/               # Pruebas unitarias
├── gui.py               # Interfaz gráfica (PRINCIPAL)
├── cli.py               # Interfaz línea de comandos (PRINCIPAL)
├── api.py               # API REST (opcional)
├── static/              # Interfaz web para API (opcional)
└── conv_MLP_84.h5       # Modelo CNN pre-entrenado
```

**Carpeta `outputs/`:** Ambas interfaces (GUI y CLI) guardan aquí:
- `heatmaps/` - Mapas de calor Grad-CAM (.jpg)
- `reports/` - Reportes PDF generados
- `historial.csv` - Registro histórico de análisis

---

## Instalación

### Con uv (Recomendado)

```bash
# Instalar uv
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar e instalar
git clone <tu-repositorio>
cd PRIMER_PROYECTO_NEUMONIA
uv sync
```

**Archivos de dependencias:**
- `pyproject.toml` - Dependencias principales (optimizado)
- `uv.lock` - Lockfile con versiones exactas  
- `requirements-docker.txt` - Dependencias mínimas para Docker

---

## Uso

### GUI (Interfaz Gráfica) - Principal

Interfaz visual para uso clínico interactivo.

```bash
uv run gui.py
```

**Funcionalidades:**
- ✅ Cargar imagen (DICOM/JPG/PNG)
- ✅ Realizar predicción con Grad-CAM
- ✅ Guardar heatmap en outputs/heatmaps/
- ✅ Guardar resultados en outputs/historial.csv
- ✅ Generar reporte PDF en outputs/reports/

---

### CLI (Línea de Comandos) - Principal

Interfaz de terminal con **modo interactivo** (ideal para Docker) y análisis directo.

**Modo Interactivo (Recomendado para Docker):**
```bash
python cli.py -i
```
Menú interactivo que replica toda la funcionalidad de la GUI:
- Analizar imágenes paso a paso
- Ver y limpiar historial
- Opciones de guardado personalizables

**Análisis Directo:**
```bash
# Uso básico
python cli.py imagen.dcm

# Con todas las opciones (equivalente a GUI)
python cli.py imagen.jpg -p 123456789 --heatmap --csv --pdf

# Ver ayuda
python cli.py --help
```

**Opciones:**
- `-i, --interactive` - Modo interactivo (menú en terminal)
- `-p, --paciente ID` - Cédula del paciente
- `--heatmap` - Guardar mapa de calor en outputs/heatmaps/
- `--csv` - Guardar en outputs/historial.csv
- `--pdf` - Generar reporte PDF en outputs/reports/

**Funcionalidades (mismo que GUI):**
- ✅ Análisis de imagen individual
- ✅ Mostrar diagnóstico y confianza
- ✅ Guardar heatmap
- ✅ Guardar en historial CSV
- ✅ Generar reporte PDF

---

### Docker (Despliegue)

Containerización para despliegue del sistema usando CLI interactivo.

**Pre-requisitos:**
- Docker Desktop instalado y **ejecutándose** 
- Verificar con: `docker --version`
- Si no funciona → **Problema común**: Docker no está en PATH

**Solución rápida para PATH (Windows):**
```powershell
# Agregar Docker al PATH permanentemente
$env:PATH += ";C:\Program Files\Docker\Docker\resources\bin"
```
O configurar manualmente: `Este equipo > Propiedades > Variables de entorno > Path > Nuevo`
Agregar: `C:\Program Files\Docker\Docker\resources\bin`

**Construir imagen:**
```bash
docker build -t neumonia-cli .
```

**Ejecutar CLI interactivo:**
```bash
# Windows - Con volúmenes para persistir datos
docker run -it --rm -v "${PWD}\imagenes:/app/imagenes" -v "${PWD}\outputs:/app/outputs" neumonia-cli

# Linux/Mac - Con volúmenes para persistir datos  
docker run -it --rm -v "./imagenes:/app/imagenes" -v "./outputs:/app/outputs" neumonia-cli

# Ejecución básica (sin persistir archivos)
docker run -it --rm neumonia-cli
```

**Con docker-compose:**
```bash
# Solo CLI interactivo
docker compose run --rm cli

# API (opcional)
docker compose up api
```

**Estructura en contenedor:**
- `/app/` - Código fuente
- `/app/imagenes/` - Montaje para imágenes de entrada  
- `/app/outputs/` - Montaje para archivos generados
- **Punto de entrada:** `python cli.py -i` (modo interactivo)

---

### API REST (Opcional)

Interfaz web para integración con sistemas externos.

```bash
uv run api.py
```

Abre http://localhost:5000 para interfaz web.

**Endpoints:**
- `GET /` - Interfaz web
- `POST /predict` - Predicción (JSON)
- `POST /predict-with-heatmap` - Predicción + imagen

**Docker (opcional):**
```bash
# Iniciar API
docker-compose --profile api up -d

# Iniciar CLI en contenedor
docker-compose --profile cli run --rm cli /data/imagen.dcm
```

---

## Pruebas

Las pruebas unitarias se ejecutan con **pytest**, un framework de testing robusto y flexible para Python que permite escribir tests concisos y expresivos.

**Recursos:**
- [Documentación oficial de pytest](https://docs.pytest.org/)
- [Guía de inicio rápido](https://docs.pytest.org/en/stable/getting-started.html)

**Características principales:**
- Sintaxis simple con funciones `test_*`
- Fixtures para configuración y limpieza
- Cobertura de código con `pytest-cov`
- Ejecución paralela y reportes detallados

**Ejecutar pruebas:**

```bash
uv pip install pytest pytest-cov
uv run pytest tests/ -v
```

---

## Modelo

**Arquitectura:** CNN basada en F. Pasa et al. (2019)

- 5 bloques convolucionales con skip connections
- Filtros: 16, 32, 48, 64, 80 (kernels 3x3)
- FC: 1024 → 1024 → 3
- Precisión: 84% en validación

**Grad-CAM:** Visualización de regiones relevantes para la clasificación.

---

## Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.
✅ Distribución  
✅ Uso privado

---

## Agradecimientos

Proyecto original desarrollado por:
- **Isabella Torres Revelo** - [GitHub](https://github.com/isa-tr)
- **Nicolas Diaz Salazar** - [GitHub](https://github.com/nicolasdiazsalazar)

---

##  Referencias

- Pasa, F., et al. (2019). "Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization"
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---
