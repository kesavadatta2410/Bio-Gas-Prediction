"""
train_source.py  (v3)
Phase 1: Pre-train on source domain (Iowa/Muscatine) with:
  - Evidential regression (NIG loss)
  - PINN physics loss on LATENT biokinetic states (X, S, VFA)
  - Residual weight scheduler: physics_weight linear 0 → 1 over 50% of epochs
  - Gradient-flow diagnostics (flags dead LSTM weights)
  - Prediction variance diagnostics (flags prediction collapse)
  - MahalanobisGate + DomainSimilarityRouter fitting after training

Run:  python src/train_source.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.dataset         import SourceDataLoader
from src.model           import BiogasTransferModel
from src.evidential_loss import EvidentialLoss
from src.physics_loss    import PhysicsInformedLoss, build_si_batch
from src.data_quality    import run_data_quality_pipeline, assign_sensor_groups
from src.evaluation      import full_evaluation
from src.continual_learning import ReplayBuffer, EWC

# ─── Setup ────────────────────────────────────────────────────────────────────

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
os.makedirs(config.MODEL_DIR,  exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
)
print(f"[train_source v3] Device: {device}")


# ─── Schedulers ───────────────────────────────────────────────────────────────

def grl_alpha_schedule(epoch: int, total_epochs: int) -> float:
    p = epoch / total_epochs
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def physics_weight_schedule(epoch: int, total_epochs: int) -> float:
    """
    Residual weight scheduler: starts at 0, reaches 1.0 linearly
    over the first 50% of training.  Stays at 1.0 thereafter.
    This prevents physics loss from overpowering data loss early on
    when latent states are still randomly initialised.
    """
    warmup_epochs = max(1, total_epochs // 2)
    return min(1.0, epoch / warmup_epochs)


# ─── Gradient Diagnostic ──────────────────────────────────────────────────────

def check_gradient_flow(model: nn.Module) -> dict:
    """
    Returns dict of {param_name → mean_abs_grad}.
    Warns if any LSTM weight has |grad| < 1e-7 (dead gradient).
    Returns None values for params with no grad yet.
    """
    report = {}
    dead   = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.abs().mean().item()
            report[name] = g
            # Flag LSTM weights specifically
            if "lstm" in name and "weight" in name and g < 1e-7:
                dead.append(f"{name} (grad={g:.2e})")
        else:
            report[name] = None
    if dead:
        print(f"  [GRAD WARN] Dead gradients in: {dead}")
    return report


def check_prediction_variance(model: nn.Module,
                               val_loader, device) -> float:
    """Returns variance of gamma over the validation set."""
    model.eval()
    gammas = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            (g, _, _, _), _ = model.source_forward(X_b, domain_id="muscatine")
            gammas.append(g.cpu())
    all_g = torch.cat(gammas)
    var   = all_g.var().item()
    if var < 1e-4:
        print(f"  [PRED WARN] Prediction collapse detected! var={var:.2e}")
    return var


# ─── Main Training ────────────────────────────────────────────────────────────

def train_source():
    # 1. Load data (per-dataset scaler, temporal splits)
    loader = SourceDataLoader()
    tr_dl, val_dl, te_dl = loader.load()

    # 2. Assign sensor groups for structured dropout
    group_indices = assign_sensor_groups(loader.feature_cols)
    print(f"[SensorGroups] {len(group_indices)} groups: {list(group_indices.keys())}")

    # 3. Build model
    config.MODEL["input_dim"] = loader.input_dim
    model = BiogasTransferModel(group_indices=group_indices).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model v3] Parameters: {n_params:,}")

    # 4. Losses & optimiser
    evid_loss = EvidentialLoss(coeff_reg=1e-2)
    pinn_loss = PhysicsInformedLoss(w_ode=0.20, w_vfa=0.10, w_thermo=0.05)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAIN["lr"],
        weight_decay=config.TRAIN["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=20, T_mult=2
    )

    # 5. Experience replay buffer
    replay_buf = ReplayBuffer(capacity=2000, strategy="reservoir")

    best_val   = float("inf")
    patience   = 0
    epochs     = config.TRAIN["source_epochs"]
    history    = {
        "train_task": [], "train_phys": [], "train_phys_weight": [],
        "val": [], "pred_var": [], "grad_lstm": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        physics_w = physics_weight_schedule(epoch, epochs)
        model.set_grl_alpha(grl_alpha_schedule(epoch, epochs))

        tr_task, tr_phys = [], []

        for X_batch, y_batch in tr_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optim.zero_grad()

            # Evidential prediction + latent biokinetic states
            (gamma, nu, alpha, beta), latent_t1 = model.source_forward(
                X_batch, domain_id="muscatine"
            )
            task_l = evid_loss(y_batch, gamma, nu, alpha, beta)

            # ── Physics loss on LATENT states (not raw biogas) ─────────────
            # Use last-step latent as t1; shift input by 1 for t0 approximation
            with torch.no_grad():
                # t0: same batch slightly shifted (use zero-grad proxy)
                X_shift = torch.roll(X_batch, shifts=1, dims=1)
                X_shift[:, 0, :] = X_batch[:, 0, :]  # wrap-around fix
            (_, _, _, _), latent_t0 = model.source_forward(
                X_shift, domain_id="muscatine"
            )

            latent_states = {
                "t0": {k: v.detach() for k, v in latent_t0.items()},
                "t1": latent_t1,
            }

            # Optional sensor dict for thermodynamic loss
            sensor_batch = build_si_batch(X_batch, loader.feature_cols)

            phys_l = pinn_loss(latent_states, sensor_batch,
                               dt=1.0, weight=physics_w)

            loss = task_l + phys_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tr_task.append(task_l.item())
            tr_phys.append(phys_l.item())

            replay_buf.add_batch(
                X_batch.cpu().numpy(),
                y_batch.cpu().numpy().ravel()
            )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                (g, n, a, b), _ = model.source_forward(X_b, domain_id="muscatine")
                val_losses.append(evid_loss(y_b, g, n, a, b).item())

        tr_t = np.mean(tr_task)
        tr_p = np.mean(tr_phys)
        vl   = np.mean(val_losses)

        history["train_task"].append(tr_t)
        history["train_phys"].append(tr_p)
        history["train_phys_weight"].append(physics_w)
        history["val"].append(vl)

        # ── Diagnostics (every 10 epochs) ────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            grad_report = check_gradient_flow(model)
            lstm_grads  = [v for k, v in grad_report.items()
                           if v is not None and "lstm" in k and "weight" in k]
            mean_lstm_g = np.mean(lstm_grads) if lstm_grads else 0.0
            pred_var    = check_prediction_variance(model, val_dl, device)

            history["grad_lstm"].append(mean_lstm_g)
            history["pred_var"].append(pred_var)

            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Task={tr_t:.4f} | Phys={tr_p:.4f} (w={physics_w:.2f}) | "
                  f"Val={vl:.4f} | LR={scheduler.get_last_lr()[0]:.2e} | "
                  f"LSTM|grad|={mean_lstm_g:.2e} | PredVar={pred_var:.4f}")
        else:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Task={tr_t:.4f} | Phys={tr_p:.4f} (w={physics_w:.2f}) | "
                  f"Val={vl:.4f}")

        # ── Checkpoint ──────────────────────────────────────────────────
        if vl < best_val:
            best_val  = vl
            patience  = 0
            _save_checkpoint(model, loader, replay_buf, epoch)
        else:
            patience += 1
            if patience >= config.TRAIN["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Post-training: fit router centroids and Mahalanobis gate ──────────
    print("\n[Post-Training] Fitting DomainSimilarityRouter centroids …")
    model.eval()
    src_feats = []
    with torch.no_grad():
        for X_b, _ in tr_dl:
            f = model._encode(X_b.to(device), "muscatine")
            src_feats.append(f.cpu())
    src_emb = torch.cat(src_feats)
    model.fit_domain_router({"muscatine": src_emb})

    print("[Post-Training] Fitting MahalanobisGate …")
    model.fit_maha_gate(tr_dl, device)

    # ── EWC Fisher ────────────────────────────────────────────────────────
    print("\n[EWC] Computing Fisher Information Matrix …")
    ckpt = torch.load(
        os.path.join(config.MODEL_DIR, "source_best.pt"),
        map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    ewc = EWC(model, tr_dl, device=str(device), n_batches=30)
    ewc.compute_fisher()
    ckpt["ewc_fisher"]     = ewc.fisher_
    ckpt["ewc_theta_star"] = ewc.theta_star_
    # Also persist domain router centroids and gate
    ckpt["domain_centroids"] = model.router.centroids.cpu()
    ckpt["maha_mean"]        = model.maha_gate.mean_.cpu()
    ckpt["maha_inv_cov"]     = model.maha_gate.inv_cov_.cpu()
    torch.save(ckpt, os.path.join(config.MODEL_DIR, "source_best.pt"))

    # ── Test Evaluation ───────────────────────────────────────────────────
    _evaluate_test(model, te_dl, loader, evid_loss, history)
    print(f"\n[DONE] Model saved: {config.MODEL_DIR}/source_best.pt")
    return model, loader


def _save_checkpoint(model, loader, replay_buf, epoch):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "scaler_X":       loader.scaler_X,
        "scaler_y":       loader.scaler_y,
        "scalers":        loader.scalers,
        "feature_cols":   loader.feature_cols,
        "input_dim":      loader.input_dim,
        "replay_buffer_X": replay_buf.buffer_X[:500],
        "replay_buffer_y": replay_buf.buffer_y[:500],
    }, os.path.join(config.MODEL_DIR, "source_best.pt"))


def _evaluate_test(model, te_dl, loader, evid_loss, history):
    model.eval()
    all_preds, all_true, all_epist = [], [], []

    with torch.no_grad():
        for X_b, y_b in te_dl:
            X_b = X_b.to(device)
            mean, aleat, epist, reject = model.predict(
                X_b, loader.scaler_y, domain_id="muscatine"
            )
            all_preds.extend(mean.cpu().numpy().ravel())
            all_true.extend(
                y_b.numpy().ravel() * loader.scaler_y.scale_[0]
                + loader.scaler_y.mean_[0]
            )
            all_epist.extend(epist.cpu().numpy().ravel())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    unc    = np.array(all_epist)

    full_evaluation(y_true, y_pred, unc, config.OUTPUT_DIR)
    _plot_training(history)


def _plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curves
    axes[0].plot(history["train_task"], label="Task Loss")
    axes[0].plot(history["train_phys"], label="Physics Loss", linestyle="--")
    axes[0].plot(history["val"],        label="Val Loss",     linestyle=":")
    axes[0].set_title("Training Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Physics weight schedule
    axes[1].plot(history["train_phys_weight"], color="green")
    axes[1].set_title("Physics Residual Weight Schedule")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weight (0 → 1)")

    # Diagnostics (sampled every 10 epochs)
    if history["pred_var"]:
        ax2 = axes[2]
        ax2.plot(history["pred_var"], label="Pred Variance", color="coral")
        ax2.set_ylabel("Prediction Variance", color="coral")
        ax2.set_xlabel("Diagnostic Checkpoint (every 10 epochs)")
        ax2.set_title("Diagnostics")
        if history["grad_lstm"]:
            ax3 = ax2.twinx()
            ax3.plot(history["grad_lstm"], label="|LSTM grad|",
                     color="steelblue", linestyle="--")
            ax3.set_ylabel("|LSTM grad|", color="steelblue")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "source_training.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved: {path}")


if __name__ == "__main__":
    train_source()
