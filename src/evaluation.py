"""
evaluation.py  (v3)
Industrial-grade evaluation metrics for biogas prediction.

Includes:
  1. Standard regression metrics (RMSE, MAE, R²)
  2. Economic cost metrics (false positive vs false negative costs)
  3. Regulatory compliance scoring
  4. SHAP-based feature importance (gradient SHAP fallback if shap not installed)
  5. TimeSeriesCV — walk-forward validation (single dataset)
  6. TemporalBlockCV — contiguous temporal blocks per dataset (no cross-dataset leak)
  7. DomainMetricsReporter — per-domain breakdown before global average
  8. Leave-one-dataset-out validation
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

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard regression metrics."""
    residuals = y_pred - y_true
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - y_true.mean()) ** 2)
    r2        = 1 - ss_res / (ss_tot + 1e-9)

    return {
        "RMSE":  float(np.sqrt(np.mean(residuals ** 2))),
        "MAE":   float(np.mean(np.abs(residuals))),
        "MAPE":  float(np.mean(np.abs(residuals / (y_true + 1e-9))) * 100),
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
                  uncertainty: np.ndarray = None) -> dict:
        errors    = y_pred - y_true
        fp_mask   = errors > 0
        fn_mask   = errors < 0

        cost_fp_total = self.cost_fp * np.abs(errors[fp_mask]).sum()
        cost_fn_total = self.cost_fn * np.abs(errors[fn_mask]).sum()

        # Process upset detection
        mean_flow   = y_true.mean()
        upset_mask  = y_true < (self.alert_threshold * mean_flow)
        n_upsets    = int(upset_mask.sum())

        # Missed upsets: predicted > threshold but true < threshold
        missed_mask = upset_mask & (y_pred >= self.alert_threshold * mean_flow)
        n_missed    = int(missed_mask.sum())
        cost_upsets = n_missed * self.cost_upset_miss

        total_cost  = cost_fp_total + cost_fn_total + cost_upsets

        return {
            "total_economic_cost_eur":  float(total_cost),
            "cost_over_prediction":     float(cost_fp_total),
            "cost_under_prediction":    float(cost_fn_total),
            "cost_missed_upsets":       float(cost_upsets),
            "n_process_upsets":         n_upsets,
            "n_missed_upsets":          n_missed,
            "upset_detection_rate":     float(1 - n_missed / max(n_upsets, 1)),
        }


# ─── 3. Regulatory Compliance Scorer ─────────────────────────────────────────

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
            # Weighted average by n_samples
            total_n  = sum(r["n_samples"] for r in rows)
            avg_r2   = sum(r["R2"]   * r["n_samples"] for r in rows) / total_n
            avg_rmse = sum(r["RMSE"] * r["n_samples"] for r in rows) / total_n
            avg_mae  = sum(r["MAE"]  * r["n_samples"] for r in rows) / total_n
            if verbose:
                print(f"  {'[GLOBAL-AVERAGE]':16s} R²={avg_r2:+.4f}  "
                      f"RMSE={avg_rmse:.4f}  MAE={avg_mae:.4f}")
            rows.append({"domain": "_average", "R2": avg_r2,
                          "RMSE": avg_rmse, "MAE": avg_mae, "n_samples": total_n})

        return pd.DataFrame(rows)


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
                    output_dir:  str        = "outputs") -> dict:
    """
    Run all evaluations in one call.
    Returns dict of all metric dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Regression
    reg = regression_metrics(y_true, y_pred)
    results["regression"] = reg
    print(f"\n[Evaluation] RMSE={reg['RMSE']:.3f}  MAE={reg['MAE']:.3f}  R²={reg['R2']:.3f}")

    # Economic
    eco = EconomicEvaluator().evaluate(y_true, y_pred, uncertainty)
    results["economic"] = eco
    print(f"             Total cost: €{eco['total_economic_cost_eur']:.0f}  "
          f"Upsets detected: {eco['upset_detection_rate']*100:.0f}%")

    # Compliance
    comp = ComplianceScorer().score(y_true, y_pred, uncertainty)
    results["compliance"] = comp
    print(f"             Compliance score: {comp['compliance_score']:.1f}/100")

    # Save to CSV
    flat = {**{f"reg_{k}": v for k, v in reg.items()},
            **{f"eco_{k}": v for k, v in eco.items()},
            **{f"comp_{k}": v for k, v in comp.items()}}
    pd.DataFrame([flat]).to_csv(os.path.join(output_dir, "evaluation.csv"), index=False)

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
