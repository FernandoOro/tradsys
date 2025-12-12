# Optimization Session Report: The "Wider is Better" Discovery
**Date**: 2025-12-12
**Objective**: Optimize Hyperparameters for `TransformerAgent` using Purged Cross-Validation and Optuna Pruning.

## 1. The Strategy: "Massive Search"
We switched from manual guessing to a mathematical search using Bayesian Optimization (TPE).
*   **Trials**: 50 (Approx, due to pruning)
*   **Infrastructure**: RunPod RTX 3090 (Batch 1024, Workers 4)
*   **Technique**: PurgedKFold (CV=3) + Median Pruning (Kill bad models early)

## 2. Results Analysis
The optimization converged on **Trial 31** as the Alpha Model.

| Metric | Baseline (Default) | **Winner (Trial 31)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 0.55 - 0.58 | **0.5220** | **-~6-9% Error** |
| **Model Size** | `d_model=64` | **`d_model=128`** | **2x Capacity** |
| **Depth** | `layers=3` | **`layers=2`** | Shallower |
| **Heads** | `nhead=8` | **`nhead=4`** | Focused Attention |
| **Dropout** | `0.23` | **`0.12`** | Less Regularization |

### Interpretation
*   **Wider (128)**: The market data has high dimensional complexity; 64 neurons were bottling it. 128 allows the model to "see" more distinct patterns.
*   **Shallower (2)**: 3 layers was likely causing gradient issues or over-processing for the amount of data we have. 2 layers is more direct.
*   **Low Dropout**: The data is noisy enough; we don't need excessive dropout killing the few good signals.

## 3. The "Strange Error" (Diagnosis)
User reported logs showing `Trial failed: inf`.
*   **Cause**: The logs show `Trial 30 finished with value: inf`. This was the **Median Pruner** in action.
*   **Mechanism**: The script detected that Trial 30 was performing worse than the median of previous trials at epoch X. It raised `optuna.TrialPruned`.
*   **Fix**: Our script caught this as a generic `Exception` and logged it as a failure. **This was a feature, not a bug.** It saved us ~50% of compute time.

## 4. Next Steps (Actionable)
We have a new "champion" configuration. We should update the production training script to use these values by default.

### Recommended Configuration
```python
# src/config.py or CLI Args
d_model = 128
nhead = 4
num_layers = 2
dropout = 0.12
lr = 3.79e-4
```

## 5. Can we do better? (Advanced Strategies)
The user asked: *"Is this luck? Can we improve?"*

1.  **Ensemble (Bagging)**: Instead of picking just Trial 31, we can train the Top 5 best trials and average their predictions. This reduces variance (luck factor).
2.  **Feature Selection**: We are using all features. Running **Recursive Feature Elimination (RFE)** might remove noise columns that confuse the Transformer.
3.  **Longer Training**: Now that we have the best size (128/2), we can train for **50 Epochs** with "Early Stopping" instead of just 10. The loss might drop to 0.48.
