# Biogas Transfer Learning System — v3

> **Physics-Informed Neural Network with Domain-Adaptive Transfer Learning for Biogas Production Prediction**

---

## Validation Results

### Per-Domain Metrics

| Dataset | R² | RMSE | MAE |
|---------|------|------|-----|
| Muscatine (Iowa WWTP) | **+0.8335** | 25.29 | 19.34 |
| DataONE (research scale) | +0.0112 | 0.293 | 0.237 |
| EDI SSAD (batch reactor) | −0.2033 | 0.323 | 0.270 |
| **Weighted Average** | **+0.8310** | **25.22** | **19.29** |

### Temporal-Block Cross-Validation (5-fold, per-dataset)

| Dataset | RMSE | R² |
|---------|------|----|
| Muscatine | 0.3592 ± 0.0147 | **0.8721 ± 0.0086** |

### Leave-One-Dataset-Out (LODO)

| Left-Out Dataset | RMSE | R² |
|-----------------|------|----|
| Muscatine | 0.9422 | 0.1122 |
| DataONE | 0.3607 | −0.4233 |
| EDI SSAD | 0.4657 | −1.8260 |

### Deployment Backtest (rolling simulation)

| Metric | Value |
|--------|-------|
| Initial RMSE | 14.68 |
| Final RMSE | 14.72 |
| RMSE Drift | **+0.035 (stable)** |

### Diagnostics

| Check | Value | Status |
|-------|-------|--------|
| Prediction variance | 0.460 | ✅ No collapse |
| Mean \|residual\| | 0.257 | ✅ OK |
| LSTM gradient \|grad\| | 8.3 × 10⁻³ | ✅ Flowing |
| Physics loss | 1.205 | ✅ Non-zero |

---

## Architecture v3

```
Input (B, seq_len, feat_dim)
       │
  SensorDropout ─────────── randomly masks sensor groups (simulate failures)
       │
  CrossSensorAttention ───── attend across sensors to infer failed readings
       │
  ┌────────────────────────────────────────────────────────┐
  │           Domain-Specific Encoders                      │
  │  MuscatineEncoder  │  DataONEEncoder  │  EDIEncoder     │
  │  CNN→LSTM→Attn     │  GRU+Mask chan.  │  Transformer    │
  └──────────────────────────┬─────────────────────────────┘
                              │   DomainSimilarityRouter
                              │   (cosine similarity → weighted mix)
                         LatentBiokineticsDecoder
                         (X̂, Ŝ, VFÂ — physics constraints here)
                              │
                    ┌─────────┴─────────┐
             EvidentialHead        ProcessStateClassifier
             (γ, ν, α, β NIG)     (startup/stable/unstable)
                    │                   │
              mean prediction     conditions DANN
              aleatoric unc.            │
              epistemic unc.   DomainDiscriminator ← GRL
              rejection flag   (DANN adversarial)
                              MahalanobisGate (OOD rejection)
```

**Key v3 fixes:**
- Physics loss now acts on *latent biokinetic states* (X, S, VFA) — not raw biogas — ensuring gradients reach the LSTM
- Domain-specific encoders replace a single shared encoder; router selects by cosine similarity
- Per-dataset `StandardScaler` fitted only on training split (prevents val/test leakage)
- Physics residual weight scheduled 0 → 1 over first 50% of training

---

## Project Structure

```
BIO/
├── config.py                    ← all settings (edit before running)
├── requirements.txt
├── data/
│   ├── muscatine_datasets/      ← SCADA-raw.csv, LABS-raw.csv
│   ├── DataONE_ad_datasets/     ← DataONE sensor CSVs
│   └── edi_ssad_datasets/       ← EDI SSAD CSVs
└── src/
    ├── model.py                 ← v3 architecture (domain encoders + router)
    ├── physics_loss.py          ← LatentODEResiduals + PhysicsNormLayer
    ├── dataset.py               ← per-dataset scalers + MaskedSequenceDataset
    ├── train_source.py          ← Phase 1: source pre-training
    ├── adapt_target.py          ← Phase 2: few-shot target adaptation
    ├── validation.py            ← TemporalBlockCV + LODO + backtest
    ├── evaluation.py            ← TemporalBlockCV + DomainMetricsReporter
    ├── evidential_loss.py       ← NIG evidential regression
    ├── data_quality.py          ← correlation filter + MI selection
    ├── active_learning.py       ← epistemic / gradient / CoreSet samplers
    ├── meta_learning.py         ← FOMAML across 3 datasets
    ├── continual_learning.py    ← EWC + experience replay
    ├── edge_deployment.py       ← INT8 + ONNX + latency benchmark
    └── predict.py               ← inference
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure (set TARGET_COL and data paths)
# Edit config.py

# 3. Pre-train on Muscatine source domain
python src/train_source.py

# 4. Adapt to a new plant (few-shot)
python src/adapt_target.py

# 5. Predict
python src/predict.py --demo          # synthetic demo
python src/predict.py --csv data/new_plant.csv

# 6. Full validation (TSCV + LODO + backtest)
python src/validation.py
```

> ⚠️ **Note:** Run `train_source.py` first — the v3 checkpoint format is incompatible with v2.

---

## Research Gaps Addressed

| # | Gap | Module | Method |
|---|-----|--------|--------|
| 1 | PINN physics on latent states | `physics_loss.py` | Monod ODE residuals on X/S/VFA + PhysicsNormLayer (SI units) |
| 2 | Uncertainty quantification | `evidential_loss.py` | Normal-Inverse-Gamma (NIG) evidential regression |
| 3 | Sensor robustness | `model.py` | `SensorDropout` + `CrossSensorAttention` |
| 4 | Temporal hierarchy | `model.py` | CNN → LSTM → Temporal attention |
| 5 | Domain-specific encoding | `model.py` | MuscatineEncoder / DataONEEncoder / EDIEncoder + `DomainSimilarityRouter` |
| 6 | OOD rejection | `model.py` | `MahalanobisGate` fitted on source embeddings |
| 7 | Active few-shot | `active_learning.py` | Epistemic / Gradient / CoreSet samplers |
| 8 | Meta-learning | `meta_learning.py` | FOMAML across Muscatine / DataONE / EDI |
| 9 | Data quality | `data_quality.py` | Correlation filter + MCAR/MAR test + MI feature selection |
| 10 | Industrial evaluation | `evaluation.py` | Economic cost + regulatory compliance + `TemporalBlockCV` + `DomainMetricsReporter` |
| 11 | Continual adaptation | `continual_learning.py` | EWC + experience replay buffer + online adapter |
| 12 | Edge deployment | `edge_deployment.py` | INT8 quantization + ONNX export + latency benchmark |

---

## Output Files

| File | Contents |
|------|----------|
| `outputs/validation_report.txt` | Per-domain R², TSCV, LODO, backtest summary |
| `outputs/domain_metrics.csv` | Per-dataset R² / RMSE / MAE breakdown |
| `outputs/tscv_results.csv` | 5-fold temporal cross-validation per dataset |
| `outputs/lodo_results.csv` | Leave-one-dataset-out results |
| `outputs/backtest_results.csv` | Rolling RMSE / R² over deployment window |
| `outputs/source_training.png` | Training curves + physics weight schedule |
| `outputs/backtest_plot.png` | RMSE and R² drift over time |
| `outputs/evaluation_plots.png` | Prediction vs actual + uncertainty bands |
| `outputs/predictions.csv` | mean, aleatoric, epistemic, 95% CI, stability flag |
| `outputs/evaluation.csv` | RMSE, MAE, R², economic cost, compliance score |
