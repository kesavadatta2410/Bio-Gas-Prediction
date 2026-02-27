"""
predict.py  (v2)
Inference with evidential uncertainty, edge deployment, and full evaluation.

Run:
  python src/predict.py --csv path/to/plant.csv   # real data
  python src/predict.py --demo                     # synthetic demo
  python src/predict.py --benchmark                # latency benchmark
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.dataset        import smart_load_csv, fill_and_clip, make_sequences
from src.model          import BiogasTransferModel
from src.data_quality   import assign_sensor_groups
from src.evaluation     import full_evaluation, GradientImportance
from src.edge_deployment import LatencyBenchmarker, EdgeInferenceEngine, export_onnx

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
device = torch.device(
    "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
)


def load_model(adapted: bool = True):
    fname = "adapted_best.pt" if adapted else "source_best.pt"
    path  = os.path.join(config.MODEL_DIR, fname)
    if not os.path.exists(path):
        alt = os.path.join(config.MODEL_DIR, "source_best.pt")
        if os.path.exists(alt):
            print(f"[WARN] {fname} not found, using source_best.pt")
            path = alt
        else:
            raise FileNotFoundError(f"No model found. Run train_source.py first.")

    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own checkpoint
    config.MODEL["input_dim"] = ckpt["input_dim"]
    group_indices = assign_sensor_groups(ckpt["feature_cols"])
    model = BiogasTransferModel(group_indices=group_indices).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["scaler_X"], ckpt["scaler_y"], ckpt["feature_cols"]


def predict_csv(csv_path: str):
    model, scaler_X, scaler_y, feature_cols = load_model()
    df = smart_load_csv(csv_path)
    df = fill_and_clip(df, feature_cols, config.TARGET_COL, config.PHYSICS)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X   = df[feature_cols].values.astype("float32")
    X_s = scaler_X.transform(X)
    X_seq, _ = make_sequences(X_s, np.zeros(len(X_s)), config.TRAIN["seq_len"])
    X_t = torch.tensor(X_seq, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        mean, aleat, epist, reject = model.predict(X_t, scaler_y)

    _save_and_plot(mean, aleat, epist, reject, config.OUTPUT_DIR)

    # Feature importance
    gi  = GradientImportance(model, device=str(device))
    imp = gi.feature_importance(X_seq[:50], feature_cols)
    gi.plot_importance(imp, os.path.join(config.OUTPUT_DIR, "feature_importance.png"))
    print(f"\nTop features:\n{imp.head(10).to_string(index=False)}")


def predict_demo():
    print("[predict v2] Demo mode")
    model, scaler_X, scaler_y, feature_cols = load_model()
    np.random.seed(42)
    n, seq_len, fdim = 300, config.TRAIN["seq_len"], config.MODEL["input_dim"]
    X_seq = np.random.randn(n, seq_len, fdim).astype("float32")
    X_t   = torch.tensor(X_seq, device=device)

    with torch.no_grad():
        mean, aleat, epist, reject = model.predict(X_t, scaler_y)

    _save_and_plot(mean, aleat, epist, reject, config.OUTPUT_DIR, suffix="_demo")

    # Evaluate
    y_true = mean.cpu().numpy().ravel() + np.random.randn(n) * 50
    y_pred = mean.cpu().numpy().ravel()
    total_unc = (aleat + epist).cpu().numpy().ravel()
    full_evaluation(y_true, y_pred, total_unc, config.OUTPUT_DIR)

    print("[predict v2] Demo complete.")


def run_benchmark():
    model, _, _, feature_cols = load_model()
    bench = LatencyBenchmarker(model, device="cpu")   # benchmark on CPU (edge target)
    results = bench.benchmark(
        seq_len=config.TRAIN["seq_len"],
        feat_dim=config.MODEL["input_dim"],
        batch_sizes=[1, 8, 32],
    )
    # Save results
    pd.DataFrame(results).T.to_csv(
        os.path.join(config.OUTPUT_DIR, "latency_benchmark.csv")
    )
    # Try ONNX export
    export_onnx(
        model,
        seq_len=config.TRAIN["seq_len"],
        feat_dim=config.MODEL["input_dim"],
        save_path=os.path.join(config.OUTPUT_DIR, "biogas_model.onnx"),
    )


def _save_and_plot(mean, aleat, epist, reject, out_dir, suffix=""):
    mean_np   = mean.cpu().numpy().ravel()
    aleat_np  = aleat.cpu().numpy().ravel()
    epist_np  = epist.cpu().numpy().ravel()
    total_np  = aleat_np + epist_np
    reject_np = reject.cpu().numpy().ravel()

    def stability(m, e):
        cv = e / (np.abs(m) + 1e-6)
        return np.where(cv < 0.1, "STABLE", np.where(cv < 0.3, "WARNING", "UNSTABLE"))

    results = pd.DataFrame({
        "biogas_mean_m3d":    mean_np,
        "biogas_lower_95":    mean_np - 1.96 * total_np,
        "biogas_upper_95":    mean_np + 1.96 * total_np,
        "aleatoric_unc":      aleat_np,
        "epistemic_unc":      epist_np,
        "total_unc":          total_np,
        "uncertain_flag":     reject_np,
        "stability":          stability(mean_np, epist_np),
    })
    csv_path = os.path.join(out_dir, f"predictions{suffix}.csv")
    results.to_csv(csv_path, index=False)
    print(f"[predict] Saved: {csv_path}")

    # Plot
    idx = np.arange(len(mean_np))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Prediction + CI
    axes[0].fill_between(idx, mean_np - 1.96 * total_np, mean_np + 1.96 * total_np,
                          alpha=0.2, color="steelblue", label="95% CI")
    axes[0].plot(idx, mean_np, color="steelblue", linewidth=1.5, label="Prediction")
    # Mark uncertain predictions
    unc_mask = reject_np.astype(bool)
    if unc_mask.any():
        axes[0].scatter(idx[unc_mask], mean_np[unc_mask], color="red", s=15, zorder=5,
                        label="Flagged uncertain")
    axes[0].set_title("Biogas Prediction + Uncertainty")
    axes[0].set_ylabel("Biogas (m³/day)")
    axes[0].legend(fontsize=8)

    # Aleatoric vs Epistemic
    axes[1].stackplot(idx, aleat_np, epist_np, labels=["Aleatoric", "Epistemic"],
                       colors=["coral", "orchid"], alpha=0.7)
    axes[1].set_title("Uncertainty Decomposition")
    axes[1].set_ylabel("Std Dev")
    axes[1].legend(fontsize=8)

    # Stability classification
    stab = stability(mean_np, epist_np)
    colors_map = {"STABLE": "green", "WARNING": "orange", "UNSTABLE": "red"}
    for label, color in colors_map.items():
        mask = stab == label
        axes[2].bar(idx[mask], np.ones(mask.sum()), color=color,
                    label=label, alpha=0.7, width=1.0)
    axes[2].set_title("Process Stability Classification")
    axes[2].set_yticks([])
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"prediction_plot{suffix}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       type=str,            help="Path to new plant CSV")
    parser.add_argument("--demo",      action="store_true", help="Run synthetic demo")
    parser.add_argument("--benchmark", action="store_true", help="Latency + ONNX benchmark")
    args = parser.parse_args()

    if args.csv:
        predict_csv(args.csv)
    elif args.benchmark:
        run_benchmark()
    else:
        predict_demo()
