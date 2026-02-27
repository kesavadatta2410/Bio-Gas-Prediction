# Biogas Transfer Learning System v2

## Complete Folder Structure
```
biogas_transfer/
├── config.py                    ← ALL settings here (edit before running)
├── requirements.txt
├── data/
│   ├── muscatine/               ← SCADA-raw.csv, LABS-raw.csv
│   ├── dataone_ad/              ← DataONE sensor CSVs
│   └── edi_ssad/                ← EDI SSAD CSVs
├── src/
│   ├── dataset.py               ← Data loading, preprocessing, sequences
│   ├── model.py                 ← Full architecture (v2)
│   ├── evidential_loss.py       ← NIG evidential regression [Gap 2]
│   ├── physics_loss.py          ← Monod + Mass balance + Thermo [Gap 1]
│   ├── data_quality.py          ← Correlation filter + MI selection [Gap 8]
│   ├── active_learning.py       ← Epistemic / Gradient / CoreSet [Gap 6]
│   ├── meta_learning.py         ← MAML across datasets [Gap 7]
│   ├── evaluation.py            ← Economic + Compliance + SHAP [Gap 9]
│   ├── continual_learning.py    ← EWC + Replay Buffer [Gap 10]
│   ├── edge_deployment.py       ← INT8 + ONNX + Latency + Maintenance [Gap 11]
│   ├── train_source.py          ← Phase 1: pre-train
│   ├── adapt_target.py          ← Phase 2: adapt to new plant
│   ├── predict.py               ← Inference
│   └── validation.py            ← LODO + TSCV + Backtest [Gaps 8,12]
└── outputs/
    ├── models/
    │   ├── source_best.pt
    │   └── adapted_best.pt
    ├── predictions.csv
    ├── evaluation.csv
    ├── latency_benchmark.csv
    ├── validation_report.txt
    └── *.png
```

## Quick Start (VS Code)

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Configure
Open `config.py` and set:
- `TARGET_COL` to your biogas column name
- `DATA["target_plant"]` to your new plant CSV (or leave `None` for demo)
- `SCADA_FEATURES` to match your actual column names

### Step 3 — Pre-train on source (Iowa/Muscatine)
```bash
python src/train_source.py
```

### Step 4 — Adapt to target plant
```bash
python src/adapt_target.py
```

### Step 5 — Predict
```bash
python src/predict.py --demo            # synthetic demo
python src/predict.py --csv data/new_plant.csv
python src/predict.py --benchmark       # latency + ONNX export
```

### Step 6 — Full validation
```bash
python src/validation.py
```

---

## Architecture v2

```
Input (B, seq_len, feat_dim)
       │
  SensorDropout ──────────── randomly masks sensor groups (simulate failures)
       │
  CrossSensorAttention ────── attend across sensors to infer failed readings
       │
  HierarchicalEncoder
    ├── CNN   (local short-term)
    ├── LSTM  (hourly process dynamics)
    └── Attention (day-level trends)
       │
  ┌────┴────────────────────────────┐
  │                                 │
ProcessStateClassifier         EvidentialHead (NIG)
 (startup/stable/unstable)      (γ, ν, α, β)
  │                                 │
  └──── conditions DANN ────┐      outputs:
                             │       • mean prediction
DomainDiscriminator ←────GRL│       • aleatoric uncertainty
(DANN adversarial)           │       • epistemic uncertainty
                             │       • rejection flag
MMD loss ───────────────────┘
```

## Research Gaps Addressed

| # | Gap | Module | Method |
|---|-----|--------|--------|
| 1 | PINN physics | `physics_loss.py` | Monod kinetics ODE + mass balance + thermodynamic feasibility |
| 2 | Uncertainty | `evidential_loss.py` | Normal-Inverse-Gamma (NIG) evidential regression |
| 3 | Sensor robustness | `model.py` | `SensorDropout` + `CrossSensorAttention` |
| 4 | Temporal hierarchy | `model.py` | CNN → LSTM → Temporal attention |
| 5 | Conditional adaptation | `model.py` + `adapt_target.py` | `ProcessStateClassifier` + conditional DANN + dynamic α schedule |
| 6 | Active few-shot | `active_learning.py` | Epistemic / Gradient / CoreSet samplers |
| 7 | Meta-learning | `meta_learning.py` | FOMAML across Muscatine/DataONE/EDI |
| 8 | Data quality | `data_quality.py` | Correlation filter + MCAR/MAR test + MI feature selection |
| 9 | Industrial metrics | `evaluation.py` | Economic cost + regulatory compliance + gradient SHAP |
|10 | Continual adaptation | `continual_learning.py` | EWC + experience replay buffer + online adapter |
|11 | Edge deployment | `edge_deployment.py` | INT8 quantization + ONNX export + latency benchmark + maintenance mode |
|12 | Validation | `validation.py` | LODO + 5-fold TSCV + rolling backtest |

## Output Files

| File | Contents |
|------|----------|
| `outputs/predictions.csv` | mean, aleatoric, epistemic, 95% CI, stability flag |
| `outputs/evaluation.csv` | RMSE, MAE, R², economic cost, compliance score |
| `outputs/validation_report.txt` | LODO + TSCV + backtest summary |
| `outputs/latency_benchmark.csv` | Inference latency per batch size |
| `outputs/biogas_model.onnx` | Exported model for edge/PLC deployment |
| `outputs/feature_importance.png` | Integrated gradients attribution |
