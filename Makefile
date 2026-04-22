.PHONY: install dev test lint run demo docker-build docker-run clean

PY ?= python3
PIP ?= pip

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

test:
	pytest -q tests/

lint:
	ruff check src/ tests/ app.py demo.py

run:
	streamlit run app.py

demo:
	$(PY) scripts/download_sample_video.py
	$(PY) demo.py --source data/sample.mp4 --output results/sample_tracked.mp4 --max-frames 200

docker-build:
	docker build -t visiontrack -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8501:8501 visiontrack

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache **/__pycache__ results/ runs/
