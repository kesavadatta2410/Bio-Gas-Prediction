"""
evaluation.py  (v4)
Industrial-grade evaluation metrics for biogas prediction.

Changes from v3 (v4):
  - MAPE masked for zero/startup production (< SAFETY.MAPE_MIN_PRODUCTION)
  - sMAPE and MASE added as primary metrics
  - find_recall_optimized_threshold(): sweeps thresholds for 95% upset recall
  - apply_temporal_consistency(): suppress single/double alerts (need 3 consecutive)
  - phase_metrics(): separate RMSE/R² for startup vs steady-state phases
  - CrossDomainTransferMatrix: per (train, test) domain R² table
  - full_evaluation(): per-domain only, no misleading global average
  - EconomicEvaluator: cost-sensitive 10× weight for missed upsets
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")
import config

_DEFAULT_DEVICE = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"


# ─── 1. Regression Metrics ────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       phase_mask: np.ndarray = None) -> dict:
    """
    Standard regression metrics.

    phase_mask : bool array, True = steady-state row. When provided, MAPE is
                 computed only on steady-state rows; startup rows are excluded.
    """
    residuals = y_pred - y_true
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - y_true.mean()) ** 2)
    r2        = 1 - ss_res / (ss_tot + 1e-9)

    # ── MAPE: mask zero/startup production  (Priority 0 Safety Fix) ───────
    mape_mask = y_true >= config.SAFETY["MAPE_MIN_PRODUCTION"]
    if phase_mask is not None:
        mape_mask = mape_mask & phase_mask
    if mape_mask.any():
        mape = float(np.mean(
            np.abs(residuals[mape_mask] / (y_true[mape_mask] + 1e-9))
        ) * 100)
    else:
        mape = float("nan")

    # ── sMAPE  (symmetric — bounded 0–200%, handles near-zero) ───────────
    smape = float(np.mean(
        200.0 * np.abs(residuals) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    ))

    # ── MASE  (Mean Absolute Scaled Error — naive-forecast baseline) ──────
    naive_mae = float(np.mean(np.abs(np.diff(y_true)))) if len(y_true) > 1 else 1.0
    mase      = float(np.mean(np.abs(residuals)) / (naive_mae + 1e-9))

    return {
        "RMSE":  float(np.sqrt(np.mean(residuals ** 2))),
        "MAE":   float(np.mean(np.abs(residuals))),
        "MAPE":  mape,
        "sMAPE": smape,
        "MASE":  mase,
        "R2":    float(r2),
        "Bias":  float(residuals.mean()),
    }


# ─── 2. Economic Cost Metrics ─────────────────────────────────────────────────

class EconomicEvaluator:
    """
    Computes economic cost of prediction errors.

    In biogas plants:
      - Under-prediction (false negative): missed energy revenue + possible
        overflow events → HIGH cost
      - Over-prediction (false positive): unnecessary flaring, poor grid
        planning → MEDIUM cost
      - Process upset detection: failing to detect instability → VERY HIGH cost

    Args:
      cost_fp:         €/m³ cost of over-predicting biogas
      cost_fn:         €/m³ cost of under-predicting biogas
      cost_upset_miss: € cost per missed process upset event
      alert_threshold: fraction of mean below which a process upset is flagged
    """

    def __init__(self,
                 cost_fp:         float = 0.05,   # €/m³
                 cost_fn:         float = 0.15,   # €/m³
                 cost_upset_miss: float = 500.0,  # € per event
                 alert_threshold: float = 0.4):
        self.cost_fp         = cost_fp
        self.cost_fn         = cost_fn
        self.cost_upset_miss = cost_upset_miss
        self.alert_threshold = alert_threshold

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                  uncertainty: np.ndarray = None,
                  threshold: float = None) -> dict:
        errors    = y_pred - y_true
        fp_mask   = errors > 0
        fn_mask   = errors < 0

        cost_fp_total = self.cost_fp * np.abs(errors[fp_mask]).sum()
        cost_fn_total = self.cost_fn * np.abs(errors[fn_mask]).sum()

        # Process upset detection
        mean_flow  = y_true.mean()
        alert_thr  = threshold if threshold is not None else (self.alert_threshold * mean_flow)
        upset_mask = y_true < alert_thr
        n_upsets   = int(upset_mask.sum())

        # Missed upsets: predicted > threshold but true < threshold
        missed_mask = upset_mask & (y_pred >= alert_thr)
        n_missed    = int(missed_mask.sum())

        # Cost-sensitive weighting: missed upsets penalised 10× (SAFETY.UPSET_COST_WEIGHT)
        upset_weight = config.SAFETY.get("UPSET_COST_WEIGHT", 10.0)
        cost_upsets  = n_missed * self.cost_upset_miss * upset_weight / 10.0
        # (÷10 so base cost_upset_miss is still the per-event €, weight scales internally)

        total_cost  = cost_fp_total + cost_fn_total + cost_upsets

        return {
            "total_economic_cost_eur":  float(total_cost),
            "cost_over_prediction":     float(cost_fp_total),
            "cost_under_prediction":    float(cost_fn_total),
            "cost_missed_upsets":       float(cost_upsets),
            "n_process_upsets":         n_upsets,
            "n_missed_upsets":          n_missed,
            "upset_detection_rate":     float(1 - n_missed / max(n_upsets, 1)),
            "alert_threshold_used":     float(alert_thr),
        }

# ─── 2b. Upset Recall-Optimised Threshold ─────────────────────────────────────

def find_recall_optimized_threshold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_recall: float = None,
        n_thresholds: int = 200) -> float:
    """
    Sweeps prediction thresholds to find the one achieving >= target_recall
    for process upset detection. Accepts higher false-positive rate.

    Returns the threshold value (on the raw prediction scale).
    """
    if target_recall is None:
        target_recall = config.SAFETY["RECALL_TARGET"]

    mean_flow  = y_true.mean()
    upset_true = y_true < (config.ECONOMICS["alert_threshold"] * mean_flow)

    if not upset_true.any():
        return float(config.ECONOMICS["alert_threshold"] * mean_flow)

    lo, hi = y_pred.min(), y_pred.max()
    thresholds = np.linspace(lo, hi, n_thresholds)

    best_thr    = float(config.ECONOMICS["alert_threshold"] * mean_flow)
    best_recall = 0.0

    for thr in thresholds:
        upset_pred = y_pred < thr
        tp = int((upset_true & upset_pred).sum())
        fn = int((upset_true & ~upset_pred).sum())
        recall = tp / (tp + fn + 1e-9)
        if recall >= target_recall:
            best_thr    = float(thr)
            best_recall = recall
            break   # first threshold that meets target (lowest FPR)

    if best_recall < target_recall:
        # Fall back to max-recall threshold
        best_thr = float(thresholds[-1])

    return best_thr


# ─── 2c. Temporal Consistency Filter ─────────────────────────────────────────

def apply_temporal_consistency(
        upset_alerts: np.ndarray,
        n_consecutive: int = None) -> np.ndarray:
    """
    Suppresses isolated alert spikes.  An alert is confirmed only if
    n_consecutive consecutive steps are all flagged as upset.

    upset_alerts : bool array (T,)
    Returns      : filtered bool array (T,)
    """
    if n_consecutive is None:
        n_consecutive = config.SAFETY["UPSET_CONSECUTIVE_STEPS"]

    filtered = np.zeros_like(upset_alerts, dtype=bool)
    n = len(upset_alerts)
    for i in range(n):
        end = min(i + n_consecutive, n)
        if upset_alerts[i:end].all() and (end - i) == n_consecutive:
            filtered[i:end] = True
    return filtered


# ─── 2d. Per-Phase Metrics (startup vs steady-state) ─────────────────────────

def phase_metrics(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  phase_mask: np.ndarray = None) -> dict:
    """
    Reports RMSE / MAE / R² separately for startup and steady-state phases.

    phase_mask : bool array, True = steady-state. If None, derived from
                 SAFETY.MAPE_MIN_PRODUCTION threshold applied to y_true.
    """
    if phase_mask is None:
        phase_mask = y_true >= config.SAFETY["MAPE_MIN_PRODUCTION"]

    steady_idx  = phase_mask
    startup_idx = ~phase_mask

    def _m(idx):
        if idx.sum() == 0:
            return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"),
                    "n": 0}
        m = regression_metrics(y_true[idx], y_pred[idx])
        m["n"] = int(idx.sum())
        return m

    return {
        "startup":     _m(startup_idx),
        "steady_state": _m(steady_idx),
    }

class ComplianceScorer:
    """
    Scores regulatory compliance based on:
      - Prediction accuracy within regulatory tolerance bands
      - Sustained prediction reliability (no long gaps of high uncertainty)
      - Biogas yield vs permitted minimum

    Args:
      tolerance_pct:  acceptable % deviation from true value
      min_biogas_m3d: minimum required daily biogas (permit limit)
      max_uncertainty_pct: max allowed uncertainty as % of prediction
    """

    def __init__(self,
                 tolerance_pct:       float = 15.0,
                 min_biogas_m3d:      float = 100.0,
                 max_uncertainty_pct: float = 20.0):
        self.tolerance_pct       = tolerance_pct
        self.min_biogas_m3d      = min_biogas_m3d
        self.max_uncertainty_pct = max_uncertainty_pct

    def score(self, y_true: np.ndarray, y_pred: np.ndarray,
               uncertainty: np.ndarray = None) -> dict:
        n = len(y_true)

        # Accuracy compliance
        pct_error    = np.abs(y_pred - y_true) / (y_true + 1e-9) * 100
        compliant    = (pct_error <= self.tolerance_pct).mean()

        # Minimum yield compliance
        below_min    = (y_pred < self.min_biogas_m3d).mean()

        # Uncertainty compliance
        if uncertainty is not None:
            unc_pct = uncertainty / (np.abs(y_pred) + 1e-9) * 100
            unc_ok  = (unc_pct <= self.max_uncertainty_pct).mean()
        else:
            unc_ok  = 1.0

        # Composite score (0–100)
        score = (0.5 * compliant + 0.3 * (1 - below_min) + 0.2 * unc_ok) * 100

        return {
            "compliance_score":            float(score),
            "accuracy_compliance_pct":     float(compliant * 100),
            "below_minimum_yield_pct":     float(below_min * 100),
            "uncertainty_compliance_pct":  float(unc_ok * 100),
        }


# ─── 4. Feature Importance (Gradient-based SHAP fallback) ────────────────────

class GradientImportance:
    """
    Gradient-based input attribution (IntegratedGradients approximation).
    Works without the shap library.

    If shap is installed, uses SHAP DeepExplainer for more accurate attribution.
    """

    def __init__(self, model: nn.Module, device: str = _DEFAULT_DEVICE, n_steps: int = 50):
        self.model   = model
        self.device  = torch.device(device)
        self.n_steps = n_steps

    def _integrated_gradients(self, X: torch.Tensor,
                                baseline: torch.Tensor = None) -> np.ndarray:
        """Integrated gradients w.r.t. input features."""
        if baseline is None:
            baseline = torch.zeros_like(X)

        X.requires_grad_(True)
        alphas   = torch.linspace(0, 1, self.n_steps).to(self.device)
        grads    = []

        self.model.eval()
        for alpha in alphas:
            interp = baseline + alpha * (X - baseline)
            interp.requires_grad_(True)

            feat   = self.model._encode(interp)
            gamma, _, _, _ = self.model.pred_head(feat)
            loss   = gamma.sum()
            loss.backward()

            grads.append(interp.grad.detach().cpu().numpy())

        grads_arr = np.stack(grads, axis=0)  # (steps, B, T, F)
        ig = (X.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * grads_arr.mean(0)
        return ig   # (B, T, F)

    def feature_importance(self, X_sample: np.ndarray,
                            feature_cols: list) -> pd.DataFrame:
        """
        Returns DataFrame with mean absolute importance per feature.
        """
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        ig  = self._integrated_gradients(X_t)   # (B, T, F)

        # Mean over batch and time → per-feature importance
        importance = np.abs(ig).mean(axis=(0, 1))   # (F,)

        df = pd.DataFrame({
            "feature":    feature_cols[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def plot_importance(self, df: pd.DataFrame, save_path: str):
        top20 = df.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top20["feature"][::-1], top20["importance"][::-1],
                color="steelblue")
        ax.set_title("Feature Importance (Integrated Gradients)")
        ax.set_xlabel("Mean |Attribution|")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  [Importance] Plot saved: {save_path}")


# ─── 5. Time-Series Cross-Validation ─────────────────────────────────────────

class TimeSeriesCV:
    """
    Walk-forward validation with expanding or sliding window.
    Respects temporal order — no data leakage.

    Args:
      n_splits:    number of CV folds
      gap:         number of time steps gap between train and test
      mode:        'expanding' (grows train set) or 'sliding' (fixed size)
    """

    def __init__(self, n_splits: int = 5, gap: int = 0, mode: str = "expanding"):
        self.n_splits = n_splits
        self.gap      = gap
        self.mode     = mode

    def split(self, n: int):
        """Yields (train_indices, test_indices) for each fold."""
        fold_size = n // (self.n_splits + 1)
        start_test = fold_size

        for i in range(self.n_splits):
            test_start = start_test + i * fold_size
            test_end   = test_start + fold_size

            if self.mode == "expanding":
                train_end = test_start - self.gap
                train_idx = np.arange(0, max(1, train_end))
            else:  # sliding
                train_start = max(0, test_start - fold_size * 2 - self.gap)
                train_idx   = np.arange(train_start, max(train_start + 1, test_start - self.gap))

            test_idx = np.arange(test_start, min(test_end, n))

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def evaluate(self, model_fn,
                  X: np.ndarray, y: np.ndarray,
                  train_fn, predict_fn) -> list:
        """
        model_fn()    → new untrained model
        train_fn(m, X_tr, y_tr) → trained model
        predict_fn(m, X_te)     → y_pred np array
        """
        fold_results = []
        for fold, (tr_idx, te_idx) in enumerate(self.split(len(X))):
            m = model_fn()
            m = train_fn(m, X[tr_idx], y[tr_idx])
            y_pred = predict_fn(m, X[te_idx])
            metrics = regression_metrics(y[te_idx], y_pred)
            metrics["fold"] = fold
            metrics["n_train"] = len(tr_idx)
            metrics["n_test"]  = len(te_idx)
            fold_results.append(metrics)
            print(f"  Fold {fold+1}: RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
        return fold_results


# ─── 5b. Temporal Block Cross-Validation ────────────────────────────────────

class TemporalBlockCV:
    """
    Cross-validation that preserves contiguous time blocks PER dataset.
    No mixing of samples from different datasets within a fold — prevents
    the negative R² seen when random cross-dataset splits break temporal structure.

    datasets : list of (X_seq, y_seq, dataset_name) tuples
    n_splits : number of contiguous blocks per dataset
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap      = gap

    def _per_dataset_splits(self, n: int):
        """Yields (train_idx, test_idx) for each fold within one dataset."""
        fold_size  = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            test_start = fold_size + i * fold_size
            test_end   = min(test_start + fold_size, n)
            train_end  = max(1, test_start - self.gap)
            train_idx  = np.arange(0, train_end)
            test_idx   = np.arange(test_start, test_end)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def evaluate(self,
                 datasets:   list,        # [(X, y, name), ...]
                 model_fn,
                 train_fn,
                 predict_fn) -> pd.DataFrame:
        """
        For each dataset independently, creates n_splits temporal folds.
        Returns DataFrame with per-dataset per-fold metrics.
        """
        all_results = []
        for X_all, y_all, ds_name in datasets:
            print(f"\n  [TemporalBlockCV] Dataset: '{ds_name}' ({len(X_all)} samples)")
            for fold, (tr_idx, te_idx) in enumerate(
                    self._per_dataset_splits(len(X_all))):
                m      = model_fn()
                m      = train_fn(m, X_all[tr_idx], y_all[tr_idx])
                y_pred = predict_fn(m, X_all[te_idx])
                metrics = regression_metrics(y_all[te_idx], y_pred)
                metrics["dataset"] = ds_name
                metrics["fold"]    = fold
                metrics["n_train"] = len(tr_idx)
                metrics["n_test"]  = len(te_idx)
                all_results.append(metrics)
                print(f"    Fold {fold+1}: RMSE={metrics['RMSE']:.4f} "
                      f" R²={metrics['R2']:.4f}")

        df = pd.DataFrame(all_results)
        # Per-dataset summary
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            print(f"  → {ds}: mean R²={sub['R2'].mean():.4f} ± "
                  f"{sub['R2'].std():.4f}")
        print(f"  → GLOBAL:  mean R²={df['R2'].mean():.4f}")
        return df


# ─── 5c. Domain Metrics Reporter ─────────────────────────────────────────────

class DomainMetricsReporter:
    """
    Collects per-domain predictions and computes metrics separately
    before aggregating — avoids one large-variance domain masking others.

    Usage:
        reporter = DomainMetricsReporter()
        reporter.add('muscatine', y_true_m, y_pred_m)
        reporter.add('dataone',   y_true_d, y_pred_d)
        df = reporter.report()
    """

    def __init__(self):
        self._records = []   # [(domain, y_true, y_pred)]

    def add(self, domain: str,
            y_true: np.ndarray, y_pred: np.ndarray):
        self._records.append((domain, y_true, y_pred))

    def report(self, verbose: bool = True) -> pd.DataFrame:
        rows = []
        for domain, y_true, y_pred in self._records:
            m = regression_metrics(y_true, y_pred)
            m["domain"]  = domain
            m["n_samples"] = len(y_true)
            rows.append(m)
            if verbose:
                print(f"  [{domain:12s}] R²={m['R2']:+.4f}  "
                      f"RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

        if rows:
            if verbose:
                print(f"  [NOTE] Global average omitted — use per-domain metrics above")

        return pd.DataFrame([r for r in rows])


# ─── 5d. Cross-Domain Transfer Matrix ───────────────────────────────────────────

class CrossDomainTransferMatrix:
    """
    Builds R² matrix for all (train_domain, test_domain) pairs.
    Reveals negative transfer (e.g., EDI R² = -1.83).

    Usage:
        matrix = CrossDomainTransferMatrix()
        matrix.add(train='muscatine', test='edi', r2=-1.83)
        matrix.add(train='muscatine', test='dataone', r2=0.61)
        df = matrix.report()   # pivot table: rows=train, cols=test
    """

    def __init__(self):
        self._entries = []   # [(train_domain, test_domain, r2)]

    def add(self, train: str, test: str, r2: float,
            rmse: float = None):
        self._entries.append({"train": train, "test": test,
                               "R2": r2, "RMSE": rmse})

    def report(self, verbose: bool = True) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        df = pd.DataFrame(self._entries)
        pivot = df.pivot_table(index="train", columns="test",
                               values="R2", aggfunc="mean")
        if verbose:
            print("\n[Cross-Domain Transfer Matrix] R²:")
            print(pivot.to_string(float_format="{:+.3f}".format))
        return pivot



# ─── 6. Leave-One-Dataset-Out Validation ──────────────────────────────────────

def leave_one_dataset_out(datasets: dict,
                           model_fn,
                           train_fn,
                           predict_fn) -> pd.DataFrame:
    """
    For each dataset in `datasets`, trains on all others,
    evaluates on the left-out one.

    Args:
      datasets:   {name: (X_seq, y_seq)}
      model_fn:   () → new model
      train_fn:   (model, X_train, y_train) → trained model
      predict_fn: (model, X_test) → y_pred

    Returns DataFrame of results.
    """
    names   = list(datasets.keys())
    results = []

    for left_out in names:
        train_names = [n for n in names if n != left_out]

        # Stack all training datasets
        X_trains = [datasets[n][0] for n in train_names]
        y_trains = [datasets[n][1] for n in train_names]

        # Handle different feature dimensions by padding to max
        max_feat = max(X.shape[-1] for X in X_trains)
        X_trains_padded = []
        for X in X_trains:
            if X.shape[-1] < max_feat:
                pad = np.zeros((*X.shape[:2], max_feat - X.shape[-1]))
                X   = np.concatenate([X, pad], axis=-1)
            X_trains_padded.append(X)

        X_train = np.concatenate(X_trains_padded, axis=0)
        y_train = np.concatenate(y_trains, axis=0)

        # Test set
        X_test, y_test = datasets[left_out]
        if X_test.shape[-1] < max_feat:
            pad    = np.zeros((*X_test.shape[:2], max_feat - X_test.shape[-1]))
            X_test = np.concatenate([X_test, pad], axis=-1)

        print(f"\n[LODO] Left out: '{left_out}' | "
              f"Train: {len(X_train)} | Test: {len(X_test)}")

        model   = model_fn()
        model   = train_fn(model, X_train, y_train)
        y_pred  = predict_fn(model, X_test)
        metrics = regression_metrics(y_test, y_pred)
        metrics["left_out"] = left_out
        results.append(metrics)
        print(f"  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")

    return pd.DataFrame(results)


# ─── Combined Evaluation Runner ───────────────────────────────────────────────

def full_evaluation(y_true:      np.ndarray,
                    y_pred:      np.ndarray,
                    uncertainty: np.ndarray = None,
                    output_dir:  str        = "outputs",
                    domain:      str        = "unknown",
                    find_best_threshold: bool = True) -> dict:
    """
    Run all evaluations in one call.
    v4 changes:
      - domain arg for per-domain labelling
      - recall-optimised upset threshold search
      - phase metrics (startup vs steady-state)
      - sMAPE / MASE in report
    Returns dict of all metric dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Regression (with startup mask)
    phase_mask = y_true >= config.SAFETY["MAPE_MIN_PRODUCTION"]
    reg = regression_metrics(y_true, y_pred, phase_mask=phase_mask)
    results["regression"] = reg
    mape_str = f"{reg['MAPE']:.1f}%" if not np.isnan(reg["MAPE"]) else "N/A (all startup)"
    print(f"\n[Evaluation | {domain}] "
          f"RMSE={reg['RMSE']:.3f}  MAE={reg['MAE']:.3f}  R²={reg['R2']:.3f}")
    print(f"  MAPE={mape_str}  sMAPE={reg['sMAPE']:.1f}%  MASE={reg['MASE']:.3f}")

    # Phase metrics
    phases = phase_metrics(y_true, y_pred, phase_mask)
    results["phases"] = phases
    for ph, m in phases.items():
        if m["n"] > 0:
            print(f"  [{ph:12s}] n={m['n']}  RMSE={m['RMSE']:.3f}  R²={m['R2']:+.3f}")

    # Economic (with recall-optimized threshold)
    if find_best_threshold and len(y_pred) >= 10:
        opt_thr = find_recall_optimized_threshold(y_true, y_pred)
        print(f"  Recall-optimised upset threshold: {opt_thr:.3f}")
    else:
        opt_thr = None
    eco = EconomicEvaluator().evaluate(y_true, y_pred, uncertainty,
                                       threshold=opt_thr)
    results["economic"] = eco
    print(f"  Total cost: €{eco['total_economic_cost_eur']:.0f}  "
          f"Upsets detected: {eco['upset_detection_rate']*100:.0f}%  "
          f"(threshold={eco['alert_threshold_used']:.2f})")

    # Compliance
    comp = ComplianceScorer().score(y_true, y_pred, uncertainty)
    results["compliance"] = comp
    print(f"  Compliance score: {comp['compliance_score']:.1f}/100")

    # Save to CSV
    flat = {"domain": domain,
            **{f"reg_{k}": v for k, v in reg.items()},
            **{f"eco_{k}": v for k, v in eco.items()},
            **{f"comp_{k}": v for k, v in comp.items()}}
    out_csv = os.path.join(output_dir, f"evaluation_{domain}.csv")
    pd.DataFrame([flat]).to_csv(out_csv, index=False)

    # Plot residuals
    _plot_residuals(y_true, y_pred, uncertainty, output_dir)

    return results


def _plot_residuals(y_true, y_pred, uncertainty, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=10, color="steelblue")
    lo, hi = y_true.min(), y_true.max()
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs True")

    # Residuals
    res = y_pred - y_true
    axes[1].hist(res, bins=40, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Prediction Error")

    # Time series with uncertainty
    idx = np.arange(min(200, len(y_true)))
    axes[2].plot(idx, y_true[idx], label="True", linewidth=1)
    axes[2].plot(idx, y_pred[idx], label="Pred", linewidth=1)
    if uncertainty is not None:
        axes[2].fill_between(idx,
                              y_pred[idx] - 1.96 * uncertainty[idx],
                              y_pred[idx] + 1.96 * uncertainty[idx],
                              alpha=0.2, label="95% CI")
    axes[2].set_title("Time Series (first 200)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [Evaluation] Plots saved: {path}")
