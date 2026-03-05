"""
config.py  (v4)
Central configuration for Biogas Transfer Learning System.
Edit this file before running any scripts.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA = {
    "scada_raw":    os.path.join(BASE_DIR, "data", "muscatine_datasets", "SCADA-raw.csv"),
    "lab_raw":      os.path.join(BASE_DIR, "data", "muscatine_datasets", "LABS-raw.csv"),
    "dataone_ad":   os.path.join(BASE_DIR, "data", "dataone_ad_datasets"),
    "edi_ssad":     os.path.join(BASE_DIR, "data", "edi_ssad_datasets"),
    "target_plant": None,   # set to CSV path of your new plant
}

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")

# ─── Feature Columns (matched to actual CSV column names) ────────────────────
SCADA_FEATURES = [
    "D1_TEMPERATURE", "D2_TEMPERATURE",
    "Q-PS_MGD", "Q-influent_MGD", "Q-TWAS_GPM",
    "V-burner_FT3", "V-Boiler_FT3",
    "H-Dig1_FT", "H-Dig2_FT",
    "Biogas_burner", "Biogas_boiler",
]
LAB_FEATURES = [
    "Dig1-pH", "Dig2-pH",
    "Dig1-T_degF", "Dig2-T_degF",
    "Dig1-alk_mgL", "Dig2-alk_mgL",
    "Dig1-VFA_mgL", "Dig2-VFA_mgL",
    "HSW-COD_mgL", "SRT",
]
TARGET_COL = "Biogas"   # ← actual column name found in both SCADA and LABS CSVs

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL = {
    "input_dim":    10,       # auto-updated at runtime
    "hidden_dims":  [256, 128, 64],
    "dropout_rate": 0.3,
    # Legacy MC-Dropout kept for reference; evidential is now default
    "num_mc_samples": 30,
}

# Asymmetric encoder capacity (v4): small domains use smaller hidden dim
# to prevent 100-sample DataONE/EDI from competing with 75K-sample Muscatine
SMALL_DOMAIN_HIDDEN = 64   # DataONE + EDI encoder hidden_dim

# Distillation loss weight: how strongly small encoders align to Muscatine latents
DISTILLATION_WEIGHT = 0.5

# ─── Process Types ────────────────────────────────────────────────────────────
PROCESS_TYPE = {"continuous": 0, "batch": 1, "pilot": 2}
DOMAIN_PROCESS_MAP = {"muscatine": "continuous", "dataone": "pilot", "edi": "batch"}

# ─── Safety & Recall ──────────────────────────────────────────────────────────
SAFETY = {
    "RECALL_TARGET":           0.95,   # target upset-detection recall
    "UPSET_CONSECUTIVE_STEPS": 3,      # require N consecutive steps before alerting
    "MAPE_MIN_PRODUCTION":     0.1,    # mask MAPE when y_true < this (startup phase)
    "UPSET_COST_WEIGHT":       10.0,   # loss weighting for missed upsets vs false alarms
}

# ─── Unit Conversion ──────────────────────────────────────────────────────────
# Target: Nm³ CH₄ / kg VS·added / day  (specific methane yield)
UNIT_CONVERSION = {
    "muscatine_divide_by_cod":  True,   # Muscatine: divide by influent COD load
    "dataone_normalize_volume": True,   # DataONE:   normalise by reactor working volume
    "edi_batch_to_daily":       True,   # EDI:       convert cumulative batch → daily rate
    "ch4_fraction":             0.60,   # typical biogas CH₄ fraction (60%)
}

# ─── Training ─────────────────────────────────────────────────────────────────
TRAIN = {
    "source_epochs":  120,    # total across all 3 phases (v4)
    "adapt_epochs":   50,
    "batch_size":     64,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "lambda_mmd":     0.1,
    "lambda_adv":     0.1,
    "patience":       20,
    "val_split":      0.15,
    "test_split":     0.15,
    "seq_len":        48,      # ≥48 h for continuous (2× HRT); full batch for EDI
    # Per-domain LRs — keys match actual model param name prefixes (v4 fix)
    "lr_per_domain": {
        "encoder_muscatine": 1e-5,   # low LR to preserve Phase-1 learnt repr
        "encoder_dataone":   1e-3,   # higher LR for small domains in Phase 3
        "encoder_edi":       1e-3,
    },
    # Teacher forcing: 1.0 = always use ground truth; decays to 0.0 (autoregressive)
    "teacher_forcing_init":  1.0,
    "teacher_forcing_final": 0.0,
    # Delta prediction: train on y(t)-y(t-1) instead of absolute y(t)
    "predict_delta":         False,   # set True to enable stationary targets
    # Sequence length minimums
    "seq_len_min_continuous": 48,    # 2 × typical HRT (hours)
}

# ─── Curriculum Training (v4) ──────────────────────────────────────────────────
# Phases split across source_epochs total.
# Phase 1: Muscatine-only (establish solid base representation)
# Phase 2: Frozen Muscatine teacher + small-domain distillation
# Phase 3: All unfrozen with graduated per-domain LR
CURRICULUM = {
    "phase1_epochs": 50,    # Muscatine only — small encoders frozen
    "phase2_epochs": 30,    # Small domains distill from frozen Muscatine teacher
    "phase3_epochs": 40,    # All unfrozen, graduated LR
    "phase2_small_lr": 1e-3,   # LR for small encoders during distillation
    "phase3_muscatine_lr": 1e-5,  # Fine-tune LR for Muscatine (preserve repr)
    "phase3_small_lr": 1e-3,
}

# ─── Physics Constraints ──────────────────────────────────────────────────────
PHYSICS = {
    "biogas_min":   0.0,
    "biogas_max":   5000.0,
    "ph_optimal":   (6.8, 7.4),
    "temp_optimal": (35.0, 38.0),
}

# ─── Evidential Uncertainty ───────────────────────────────────────────────────
EVIDENTIAL = {
    "coeff_reg":      1e-2,
    "reject_thresh":  2.0,    # evidence below this → flag as uncertain
}

# ─── Active Learning ──────────────────────────────────────────────────────────
ACTIVE_LEARNING = {
    "strategy": "epistemic",   # "epistemic" | "gradient" | "coreset"
    "budget":   20,
    "n_rounds": 3,
}

# ─── MAML Meta-Learning ───────────────────────────────────────────────────────
MAML = {
    "inner_lr":    0.01,
    "outer_lr":    1e-3,
    "inner_steps": 5,
    "n_tasks":     4,
    "n_epochs":    30,
    "n_episodes":  50,
}

# ─── Continual Learning ───────────────────────────────────────────────────────
CONTINUAL = {
    "replay_capacity": 2000,
    "replay_ratio":    0.5,
    "lambda_ewc":      0.4,
    "online_lr":       5e-5,
    "anomaly_thresh":  4.0,
}

# ─── Edge Deployment ──────────────────────────────────────────────────────────
EDGE = {
    "max_latency_ms":    900.0,
    "quantize_dynamic":  True,
    "export_onnx":       True,
    "flatline_window":   12,
}

# ─── Economic Evaluation ──────────────────────────────────────────────────────
ECONOMICS = {
    "cost_fp":          0.05,    # €/m³ over-prediction
    "cost_fn":          0.15,    # €/m³ under-prediction
    "cost_upset_miss":  500.0,   # € per missed upset
    "alert_threshold":  0.4,
}

SEED   = 42
DEVICE = "cuda"   # auto-falls back to cpu if unavailable
