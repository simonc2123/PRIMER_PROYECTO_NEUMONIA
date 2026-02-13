FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt


COPY . .

RUN mkdir -p outputs/heatmaps outputs/reports

# Comando CLI interactivo
CMD ["python", "cli.py", "-i"]

