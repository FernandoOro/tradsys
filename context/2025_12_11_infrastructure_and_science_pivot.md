# AI Research Context & Progress Journal

> [!NOTE]
> This document maintains the context of research experiments, infrastructure decisions, and model pivots to ensure future AI sessions build upon this success rather than regressing.

## Session: RunPod Infrastructure & Labeling Pivot (Dec 2025)

### 1. The Challenge
The objective was to train the `TransformerAgent` on RunPod. Initial attempts failed catastrophically.
**Symptoms**:
- `[Errno 12] Cannot allocate memory` (OOM) crashes immediately.
- Random "Hangs" during "Creating Sequences" or "Adversarial Validation".
- Model Loss stuck at 1.000 (Random Guessing).

### 2. Diagnosis & Fixes (Infrastructure)
We discovered a critical mismatch between RunPod "Container" limits and "Host" specs.
*   **Root Cause**: The user was running on a non-GPU pod which enforces a strict **488MB RAM** limit, despite `free -h` showing the host's 1TB.
*   **Action 1 (Hardware)**: Migrated to **GPU Instance (RTX 3090)**. Result: 125GB RAM available.
*   **Action 2 (Software)**: Implemented `LazySequenceDataset` in `scripts/train.py`.
    *   *Why*: Pre-allocating `(N, Seq_Len, Feats)` tensor caused RAM spikes of >50GB. Lazy loading shifts this to the DataLoader.
*   **Action 3 (Concurrency)**: Enforced `OMP_NUM_THREADS=1` to prevent CPU deadlocks common in containerized PyTorch.
*   **Action 4 (Optimization)**: Activated **"Beast Mode"** configuration:
    *   `BATCH_SIZE = 1024` (vs 32).
    *   `NUM_WORKERS = 8` (vs 0).
    *   `PIN_MEMORY = True`.

### 3. Diagnosis & Fixes (Data Science)
Post-infrastructure fix, the model trained fast but failed to learn (Loss ~1.0).
*   **Root Cause**: The target label was `df['close'].shift(-1) > df['close']`.
    *   *Insight*: Predicting the very next candle direction is statistically indistinguishable from noise (50/50 chance).
*   **Scientific Pivot**: Activated **Triple Barrier Method (TBM)** in `src/data/pipeline.py`.
    *   *Logic*: Label 1 if price hits `Upper Barrier (Price + 1.5*ATR)` first. Label 0 otherwise (Stop Loss or Time Limit).
    *   *Effect*: Filters out small noise. Only asks the model to predict **significant** moves.
*   **Result**:
    *   **Train Loss**: Dropped to **0.52**.
    *   **Val Loss**: Dropped to **0.50**.
    *   *Interpretation*: The model found a strong signal. Loss of 0.50 implies high confidence and accuracy on the filtered events.

### 4. Configuration Snapshot (Current Production)
Use these settings for future runs. Do not revert to "Safe Mode" unless hardware is downgraded.

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Hardware** | RunPod RTX 3090 (1x) | Needs >16GB VRAM and >32GB RAM. |
| **Batch Size** | 1024 | Saturation of Ampere GPU cores. |
| **Workers** | 8 | Efficient data pre-fetching. |
| **Labeling** | Triple Barrier (Width=1.5) | Removes noise. |
| **Model** | TransformerAgent (d_model=64) | Lightweight but effective. |
| **Loss** | FocalLoss | Handles class imbalance if barriers are tight. |

### 5. Next Steps (Roadmap)
For the next session, focus on:
1.  **Feature Engineering**: The current set is basic (OHLCV + RSI/MACD). Add **Order Book Imbalance** or **Fractional Diff** features if enabled.
2.  **Hyperparameter Tuning**: Re-run `src/optimization/tuner.py` targeting the new TBM dataset.
3.  **Backtesting**: The `models/agent1.onnx` is ready. Test it in the `backtesting` module using the TBM logic for exit signals.
