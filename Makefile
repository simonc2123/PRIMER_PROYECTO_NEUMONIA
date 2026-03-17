.PHONY: install lint test build push all clean run-cli run-api run-gui docker-cli docker-api

# ── Configuración ──────────────────────────────────────────────────────────────
IMAGE_NAME ?= neumonia
IMAGE_TAG  ?= latest

# ── CI/CD: instalación y calidad ───────────────────────────────────────────────
install:
	uv sync --frozen

lint:
	uvx flake8 src/ tests/ api.py cli.py --max-line-length=100 --exclude=__pycache__,.git,.venv --statistics

test:
	uv run pytest tests/ -v --tb=short --ignore=tests/test_grad_cam.py

# ── CI/CD: Docker ──────────────────────────────────────────────────────────────
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

push:
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

# ── Local: correr la aplicación ────────────────────────────────────────────────
run-cli:
	uv run python cli.py -i

run-api:
	uv run python api.py

run-gui:
	uv run python gui.py

# ── Local: correr con Docker ───────────────────────────────────────────────────
docker-cli:
	docker compose run --rm cli

docker-api:
	docker compose --profile api up api

# ── Atajos ─────────────────────────────────────────────────────────────────────
all: install lint test build

clean:
	uv cache prune
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .venv -exec rm -rf {} +
