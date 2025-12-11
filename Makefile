
# Smart Spot Trading System - Makefile

.PHONY: build up down logs live train help

help:
	@echo "🦅 Smart Spot - Command Center"
	@echo "------------------------------"
	@echo "make init     Initialize System (Generate Data & Models)"
	@echo "make build    Build the Docker image"
	@echo "make up       Start Bot (Uses .env/IS_PAPER_TRADING) + Dashboard"
	@echo "make down     Stop all services"
	@echo "make logs     Tail bot logs"
	@echo "make shell    Enter Bot Container (Bash)"
	@echo "make clean    Remove artifacts (Caution!)"

init:
	@echo "🌱 Initializing System Artifacts (Data + Models)..."
	docker compose build bot
	docker compose run --rm bot python src/data/pipeline.py
	docker compose run --rm -v $$(pwd):/app bot python scripts/init_onnx.py
	@echo "✅ Initialization Complete. You can now run 'make up'."

build:
	docker compose build

up:
	docker compose up -d
	@echo "Dashboard available at http://localhost:8501"

down:
	docker compose down

logs:
	docker compose logs -f bot

shell:
	docker compose run --rm -v $$(pwd):/app bot bash

clean:
	rm -f models/*.pkl models/*.onnx data/*.db
	@echo "Artifacts cleaned."

# Developer Tools
train:
	docker compose run --rm -v $$(pwd):/app bot python scripts/train.py --epochs 10 --pretrain
