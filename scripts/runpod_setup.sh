#!/bin/bash

echo "🚀 Iniciando Configuración de RunPod para Smart Spot Trader..."

# 1. Configurar PYTHONPATH (Para que Python encuentre 'src')
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "✅ PYTHONPATH configurado: $PYTHONPATH"

# 2. Actualizar Código
echo "📥 Bajando últimos cambios de Git..."
git pull

# 3. Instalar Dependencias (Solo si falta alguna)
echo "📦 Verificando librerías..."
pip install -r requirements.txt

# 4. Crear .env si no existe (Plantilla básica)
if [ ! -f .env ]; then
    echo "⚠️ No se encontró .env. Creando uno básico..."
    cat <<EOF > .env
EXCHANGE_ID=binance
SYMBOL=BTC/USDT
TIMEFRAME=1m
IS_PAPER_TRADING=True
LOG_LEVEL=INFO
# Añade tus API KEYS aquí
EOF
    echo "✅ .env creado. ¡Recuerda editarlo con tus claves!"
fi

echo "==================================================="
echo "🦁 ¡Entorno Listo! Ya puedes ejecutar:"
echo "   python src/data/pipeline.py"
echo "   python scripts/train.py --pretrain --epochs 100"
echo "==================================================="
