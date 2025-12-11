#!/bin/bash

echo "🚀 Iniciando Configuración de RunPod para Smart Spot Trader..."

# 1. Configurar PYTHONPATH (Para que Python encuentre 'src')
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "✅ PYTHONPATH configurado: $PYTHONPATH"

# 2. Actualizar Código
echo "📥 Bajando últimos cambios de Git..."
git pull

# 3. Instalar Dependencias (Fix de Memoria y Sklearn)
echo "📦 Verificando librerías..."
# Fix: Desinstalar sklearn corrupto
pip uninstall -y scikit-learn
# Instalar solo lo necesario SIN caché para no explotar la RAM (Torch ya viene instalado)
pip install --no-cache-dir scikit-learn ccxt psutil
# Instalar resto de requirements sin forzar actualización de Torch
pip install --no-cache-dir -r requirements.txt

# 4. Crear .env si no existe (Plantilla básica)
if [ ! -f .env ]; then
    echo "⚠️ No se encontró .env. Creando uno básico..."
    cat <<EOF > .env
# --- Exchange Secrets (Binance) ---
EXCHANGE_ID=binance
# Si es Paper Trading, estas keys no envían órdenes reales, pero se necesitan para leer datos
API_KEY=tu_binance_api_key
SECRET_KEY=tu_binance_secret_key

# --- Trading Configuration ---
SYMBOL=BTC/USDT
# IMPORTANTE: Cambiado a 1m para trading intradía real
TIMEFRAME=1m  
# ¿Es dinero real? False = Dinero Real (CUIDADO), True = Simulación
IS_PAPER_TRADING=True

# --- Strategy Profile ---
# Opciones: SNIPER, AUDITED, RECKLESS
STRATEGY_PROFILE=AUDITED

# --- Risk Management (Safety Nets) ---
MAX_RISK_PER_TRADE=0.02  # 2% de la cuenta
MAX_LEVERAGE=1           # Spot trading (sin apalancamiento)
STOP_LOSS_ATR_MULT=2.0   # Multiplicador de volatilidad para SL

# --- Infrastructure ---
# AWS Tokyo para mínima latencia con Binance
AWS_REGION=ap-northeast-1 
RUNPOD_API_KEY=tu_runpod_key

# --- Logging & Monitoring ---
# Necesario para ver las gráficas de entrenamiento
WANDB_API_KEY=3e0049090c8b79811b6abb59b319d4ee12f58611
LOG_LEVEL=INFO
EOF
    echo "✅ .env creado. ¡Recuerda editarlo con tus claves!"
fi

echo "==================================================="
echo "🦁 ¡Entorno Listo! Ya puedes ejecutar:"
echo "   python src/data/pipeline.py"
echo "   python scripts/train.py --pretrain --epochs 100"
echo "==================================================="
