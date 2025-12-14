
# 🚀 Guía de Despliegue en RunPod - Agente 2 (Mean Reversion)

Sigue estos pasos exactos para llevar tu código local a la nube (RTX 3090).

## 1. Sincronizar Código (Local -> GitHub)
Primero, sube los nuevos scripts (`src/agent2` y `scripts/agent2`) a tu repositorio.

```bash
# En tu terminal local:
git add .
git commit -m "feat: Agent 2 architecture and training scripts"
git push origin main
```

## 2. Preparar RunPod (Instalación desde Cero)
1.  Abre tu instancia en RunPod (Jupyter Lab).
2.  Abre una **Terminal** en Jupyter.
3.  **Clonar Repositorio:**
    ```bash
    git clone https://github.com/FernandoOro/tradsys.git
    cd tradsys
    ```
4.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Crear Carpetas Necesarias:**
    Como git ignora las carpetas de datos, debemos crearlas manualmente antes de subir los archivos.
    ```bash
    mkdir -p data/processed models/agent2
    ```

## 3. Subir los "Datos Cocinados" (Manual)
Dado que `train_reversion.parquet` (5MB) está ignorado por git (para no ensuciar el repo con datos pesados), debes subirlo manualmente.

1.  En la interfaz de **Jupyter Lab (RunPod)** (panel izquierdo de archivos).
2.  Navega a la carpeta: `tradsys/data/processed/`.
3.  Arrastra y suelta desde tu PC los archivos:
    *   `data/processed/train_reversion.parquet`
    *   `data/processed/val_reversion.parquet`
    *   `models/pca_reversion.pkl` (Opcional, pero recomendado si quieres consistencia).

## 4. Ejecutar el Entrenamiento (¡Fuego! 🔥)

### Paso A: Entrenamiento No Supervisado (DAE)
Enséñale la "Física del Mercado" usando la GPU.
```bash
# En terminal de RunPod:
python scripts/agent2/train_dae.py
```
*   *Salida esperada:* Un archivo `models/dae_reversion.pt`.

### Paso B: Entrenamiento Supervisado (Classifier + Optuna)
Afina la puntería buscando los mejores hiperparámetros.
```bash
# En terminal de RunPod:
python scripts/agent2/train_classifier.py
```
*   *Resultado:* Verás los logs de Optuna mejorando el `Val Acc`.
*   El mejor modelo quedará listo.

## 5. (Opcional) Descargar el Cerebro
Cuando termine, descarga el modelo entrenado a tu máquina local para guardarlo o desplegarlo en producción.
*   Click derecho en `models/agent2_best.pt` (o similar) -> Download.

## ⚠️ Solución de Problemas (Troubleshooting)

### Error: `wandb.errors.Error: You must call wandb.init() before wandb.log()`
Este error ya fue corregido en el repositorio.
1.  Ejecuta `git pull` en la terminal de RunPod.
2.  Vuelve a ejecutar `python scripts/agent2/train_classifier.py`.

### Error: `No such file or directory: train_reversion.parquet`
1.  Asegúrate de haber subido **manualmente** los archivos a `data/processed/`.
2.  Usa `ls -lh data/processed/` para verificar que pesan ~5MB.
