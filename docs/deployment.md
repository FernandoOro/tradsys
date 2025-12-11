# Deployment Guide: Smart Spot V1

## Prerequisites
- **Git**
- **Docker** & **Docker Compose**
- **NVIDIA Drivers** (Optional, only if training on GPU, but Inference runs on CPU).

## 1. Setup Environment
1. Clone the repository.

## 2. The Cloud Workflow (Train on RunPod GPU -> Run on VPS CPU)
As per project guidelines (Contexto Maestro), heavy training should happen on RunPod (GPU), while inference runs on a cheaper VPS (CPU).

### Phase A: Training (Ephemeral GPU Pod)
1.  **Rent a GPU Pod** (e.g., RTX 3090 on RunPod).
2.  **Clone Repo** & Setup `.env`.
3.  **Increase Data Limit** in `src/data/pipeline.py` (e.g., 100,000 candles).
4.  **Run Training with Auto-Kill**:
    ```bash
    # This downloads data, trains, saves ONNX, and kills the pod to save money.
    # Ensure RUNPOD_API_KEY and RUNPOD_POD_ID are in .env
    make init 
    docker compose run --rm -v $(pwd):/app bot python scripts/train.py --pretrain --epochs 50 --autokill
    ```
5.  **Retrieve Artifacts**:
    Before the pod dies (or if you didn't use autokill), download `models/agent1.onnx` to your local machine (via SCP or Magic Wormhole).

### Phase B: Deployment (Permanent CPU VPS)
1.  **Rent a CPU VPS** (e.g., AWS, DigitalOcean).
2.  **Upload** the `agent1.onnx` file you got from Phase A to `models/`.
3.  **Run Bot**:
    ```bash
    docker compose up -d
    ```

## 3. Quick Start (Local Prototype)
2. Create `.env` file from example:
   ```bash
   cp .env.example .env
   # Edit .env with your Binance API Keys
   ```
3. Ensure required directories exist:
   ```bash
   mkdir -p data models logs
   ```

## 2. Quick Start
To launch the **Bot** (Paper Mode) and **Dashboard**:
```bash
docker-compose up -d --build
```

## 3. Operations
### Check Logs
```bash
docker-compose logs -f bot
```

### Access Dashboard
- **Local**: Open `http://localhost:8501`
- **VPS (Secure Tunnel)**:
  If running on a remote VPS, do NOT open port 8501 firewall. Tunnel instead:
  ```bash
  ssh -L 8501:localhost:8501 user@vps-ip
  ```
  Then open `http://localhost:8501` locally.

### Switch to Live Trading
1. Edit `docker-compose.yml`:
   Change `command: python scripts/run_bot.py --mode paper` to `--mode live`.
2. Restart:
   ```bash
   docker-compose up -d --force-recreate
   ```

## 4. Updates
To update the code and redeploy:
```bash
git pull
docker-compose up -d --build
```
