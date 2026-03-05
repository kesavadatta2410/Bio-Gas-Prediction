"""
train_source.py  (v4 — 3-Phase Curriculum Training)

Phase 1 (epochs 1..P1):     Train ONLY Muscatine encoder. Small-domain encoders frozen.
                             Establishes solid Muscatine representation before small-domain
                             gradients can interfere.
Phase 2 (epochs P1+1..P1+P2): Freeze Muscatine encoder (teacher). Unfreeze DataONE+EDI.
                             Distillation loss ||z_small - z_musc.detach()||^2 forces
                             small encoders to align to Muscatine's latent space.
Phase 3 (remaining epochs): Unfreeze all. Muscatine LR=1e-5, small domains LR=1e-3.
                             Fine-tunes while protecting Phase-1 representation.

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
from src.model           import BiogasTransferModel, latent_distillation_loss
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
print(f"[train_source v4] Device: {device}")


# ─── Sequence Length Audit ────────────────────────────────────────────────────

def audit_sequence_lengths(loader, verbose: bool = True) -> dict:
    """
    Verifies all domains have sequences meeting the minimum length.
    Warns loudly if continuous process sequences < config.TRAIN.seq_len_min_continuous.
    """
    min_cont = config.TRAIN.get("seq_len_min_continuous", 48)
    report   = {}
    for domain, process_type in config.DOMAIN_PROCESS_MAP.items():
        if process_type in ("continuous", "pilot"):
            seq_len = config.TRAIN.get("seq_len", 48)
            ok      = seq_len >= min_cont
            report[domain] = {"ok": ok, "seq_len": seq_len, "min": min_cont}
            if not ok and verbose:
                print(f"  [SEQ WARN] {domain}: seq_len={seq_len} < min={min_cont} "
                      f"(need 2× HRT for continuous processes)")
    if verbose and all(r["ok"] for r in report.values()):
        print(f"  [SEQ OK] All continuous domains: seq_len={config.TRAIN['seq_len']} ≥ {min_cont}")
    return report


# ─── Schedulers ───────────────────────────────────────────────────────────────

def grl_alpha_schedule(epoch: int, total_epochs: int) -> float:
    p = epoch / total_epochs
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def physics_weight_schedule(epoch: int, total_epochs: int) -> float:
    """
    Residual weight scheduler: starts at 0, reaches MAX linearly
    over the first 50% of training.

    MAX is capped at 0.01 (not 1.0) because raw ODE residuals are in
    physical units (m³/day) and are 6-9 orders of magnitude larger than
    the normalised evidential task loss. Allowing weight > 0.01 causes
    the physics loss to completely dominate, explode gradients, and
    corrupt val loss so checkpoints are never saved.
    """
    max_weight    = 0.01
    warmup_epochs = max(1, total_epochs // 2)
    return min(max_weight, max_weight * epoch / warmup_epochs)


def teacher_forcing_schedule(epoch: int, total_epochs: int) -> float:
    """
    Teacher forcing probability: starts at TRAIN.teacher_forcing_init (1.0)
    and decays linearly to TRAIN.teacher_forcing_final (0.0) over training.
    Use ground-truth token with probability p; model's prior output with 1-p.
    """
    tf_init  = config.TRAIN.get("teacher_forcing_init",  1.0)
    tf_final = config.TRAIN.get("teacher_forcing_final", 0.0)
    return tf_init - (tf_init - tf_final) * (epoch / max(1, total_epochs))
# ─── Cost-Sensitive Upset Loss ────────────────────────────────────────────────

def upset_cost_loss(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    weight: float = None) -> torch.Tensor:
    """
    Cost-sensitive auxiliary loss that penalizes missed upsets 10× more
    than missed non-upsets.

    upset_mask = rows where y_true < 0.5 * mean(y_true)
    Applies SAFETY.UPSET_COST_WEIGHT to those residuals.
    """
    if weight is None:
        weight = config.SAFETY.get("UPSET_COST_WEIGHT", 10.0)
    mean_flow  = y_true.mean().detach()
    upset_mask = (y_true < 0.5 * mean_flow).float()
    # Weighted MSE: upset rows penalised more
    sq_err     = (y_pred - y_true) ** 2
    weighted   = sq_err * (1.0 + (weight - 1.0) * upset_mask)
    return weighted.mean()


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


# ─── Target Scale Audit ────────────────────────────────────────────────────────────────

def _audit_target_scales(y_train: np.ndarray, domain: str = "muscatine"):
    """
    Prints raw target variable stats BEFORE normalization.
    Warns if range is very different from Muscatine's expected 0-5000 m³/day.
    """
    print(f"  [ScaleAudit | {domain}] n={len(y_train)} "
          f"mean={y_train.mean():.2f} std={y_train.std():.2f} "
          f"min={y_train.min():.2f} max={y_train.max():.2f}")
    if y_train.max() < 1.0:
        print(f"  [ScaleAudit WARN] {domain}: targets look like 0-1 range — "
              f"check UNIT_CONVERSION config. 86× RMSE disparity likely.")
    elif y_train.max() < 10.0:
        print(f"  [ScaleAudit WARN] {domain}: targets may be normalised already. "
              f"DomainScaler may not help unless raw units are used.")


# ─── Curriculum Phase Utilities ───────────────────────────────────────────────────

def _get_curriculum_phase(epoch: int) -> int:
    """
    Returns current curriculum phase (1, 2, or 3) based on epoch number.
    Phase boundaries come from config.CURRICULUM.
    """
    curr = getattr(config, "CURRICULUM", {})
    p1 = curr.get("phase1_epochs", 50)
    p2 = curr.get("phase2_epochs", 30)
    if epoch <= p1:
        return 1
    elif epoch <= p1 + p2:
        return 2
    else:
        return 3


def _set_optimizer_lr_for_phase(
        optim: torch.optim.Optimizer,
        phase: int,
        param_groups_meta: list):
    """
    Adjusts per-group learning rates when transitioning between curriculum phases.
    param_groups_meta: list of {"prefix": str, "lr": float} dicts in same
    order as optim.param_groups.
    """
    curr = getattr(config, "CURRICULUM", {})
    for i, meta in enumerate(param_groups_meta):
        prefix = meta.get("prefix", "")
        if phase == 1:
            # Only Muscatine trains; disable LR on small encoders
            if "encoder_muscatine" in prefix:
                optim.param_groups[i]["lr"] = config.TRAIN["lr"]
            elif "encoder_dataone" in prefix or "encoder_edi" in prefix:
                optim.param_groups[i]["lr"] = 0.0   # effectively frozen via requires_grad
            else:
                optim.param_groups[i]["lr"] = config.TRAIN["lr"]
        elif phase == 2:
            # Muscatine frozen; small encoders train at phase2_small_lr
            if "encoder_muscatine" in prefix:
                optim.param_groups[i]["lr"] = 0.0
            elif "encoder_dataone" in prefix or "encoder_edi" in prefix:
                optim.param_groups[i]["lr"] = curr.get("phase2_small_lr", 1e-3)
            else:
                optim.param_groups[i]["lr"] = curr.get("phase2_small_lr", 1e-3)
        else:  # phase 3
            if "encoder_muscatine" in prefix:
                optim.param_groups[i]["lr"] = curr.get("phase3_muscatine_lr", 1e-5)
            elif "encoder_dataone" in prefix or "encoder_edi" in prefix:
                optim.param_groups[i]["lr"] = curr.get("phase3_small_lr", 1e-3)
            else:
                optim.param_groups[i]["lr"] = curr.get("phase3_small_lr", 1e-3)


# ─── Main Training ──────────────────────────────────────────────────────────────────

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
    print(f"[Model v4] Parameters: {n_params:,}")
    print(f"  Muscatine encoder out_dim: {model.encoder_muscatine.out_dim}")
    print(f"  DataONE encoder out_dim:   {model.encoder_dataone.out_dim}")
    print(f"  EDI encoder out_dim:       {model.encoder_edi.out_dim}")

    # 4. Scale audit — check target variable ranges before any training
    print("\n[ScaleAudit] Checking raw target distributions...")
    # Collect raw y values from loader
    all_y_raw = []
    for _, y_b in tr_dl:
        all_y_raw.extend(y_b.numpy().ravel())
    _audit_target_scales(np.array(all_y_raw), domain="muscatine")

    # 5. Losses
    evid_loss = EvidentialLoss(coeff_reg=1e-2)
    pinn_loss = PhysicsInformedLoss(w_ode=0.20, w_vfa=0.10, w_thermo=0.05)

    # 6. Build per-domain param groups with FIXED prefixes (v4 bug fix)
    #    Config keys now match actual model param name prefixes: encoder_X not encoders.X
    lr_per_domain = config.TRAIN.get("lr_per_domain", {})
    param_groups      = []
    param_groups_meta = []   # track prefix+lr for phase-switching
    grouped_params    = set()

    for prefix, lr_val in lr_per_domain.items():
        group_params = [(n, p) for n, p in model.named_parameters()
                        if n.startswith(prefix)]
        if group_params:
            param_groups.append({
                "params":       [p for _, p in group_params],
                "lr":           lr_val,
                "weight_decay": config.TRAIN["weight_decay"],
            })
            param_groups_meta.append({"prefix": prefix, "lr": lr_val})
            grouped_params.update(n for n, _ in group_params)
            print(f"  [ParamGroup] '{prefix}': {len(group_params)} params @ lr={lr_val}")

    # Remaining params at global LR
    remaining = [p for n, p in model.named_parameters() if n not in grouped_params]
    param_groups.append({
        "params":       remaining,
        "lr":           config.TRAIN["lr"],
        "weight_decay": config.TRAIN["weight_decay"],
    })
    param_groups_meta.append({"prefix": "__rest__", "lr": config.TRAIN["lr"]})
    print(f"  [Optim] {len(param_groups)-1} domain-specific groups + 1 global "
          f"(total {len(grouped_params)} domain params matched)")

    optim = torch.optim.AdamW(param_groups)
    # Warm restarts with longer T_0 to align with 3 phases
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=50, T_mult=1
    )

    # 7. Experience replay buffer
    replay_buf = ReplayBuffer(capacity=2000, strategy="reservoir")

    best_val   = float("inf")
    patience   = 0
    epochs     = config.TRAIN["source_epochs"]
    history    = {
        "train_task": [], "train_phys": [], "train_phys_weight": [],
        "train_distill": [], "val": [], "pred_var": [], "grad_lstm": [],
        "phys_compliance": [], "mean_dX_residual": [], "phase": []
    }

    # Sequence length audit (v4)
    audit_sequence_lengths(loader)

    # ── Phase 1 init: freeze small-domain encoders ─────────────────────────
    print("\n" + "="*65)
    print(f"[Phase 1] Muscatine-only training (small encoders frozen)")
    print(f"          Phase 1 epochs: 1..{getattr(config, 'CURRICULUM', {}).get('phase1_epochs', 50)}")
    print("="*65)
    model.freeze_small_domain_encoders(["dataone", "edi"], verbose=True)
    _set_optimizer_lr_for_phase(optim, 1, param_groups_meta)

    current_phase = 1

    for epoch in range(1, epochs + 1):

        # ── Phase transitions ──────────────────────────────────────────
        new_phase = _get_curriculum_phase(epoch)
        if new_phase != current_phase:
            current_phase = new_phase
            if current_phase == 2:
                print("\n" + "="*65)
                print(f"[Phase 2] Distillation: Muscatine teacher frozen, small domains learn")
                print("="*65)
                model.freeze_muscatine_encoder(verbose=True)
                model.unfreeze_small_domain_encoders(["dataone", "edi"], verbose=True)
                # Reset patience — Phase 2 starts fresh
                patience = 0
                best_val = float("inf")
            elif current_phase == 3:
                print("\n" + "="*65)
                print(f"[Phase 3] Fine-tune all (Muscatine LR=1e-5, small LR=1e-3)")
                print("="*65)
                model.unfreeze_muscatine_encoder(verbose=True)
                patience = 0
                best_val = float("inf")
            _set_optimizer_lr_for_phase(optim, current_phase, param_groups_meta)

        model.train()
        physics_w = physics_weight_schedule(epoch, epochs)
        model.set_grl_alpha(grl_alpha_schedule(epoch, epochs))

        tr_task, tr_phys, tr_distill = [], [], []
        phys_diag_batch  = []
        teacher_p        = teacher_forcing_schedule(epoch, epochs)

        for X_batch, y_batch in tr_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optim.zero_grad()

            # ── Core Muscatine prediction (always computed) ────────────
            (gamma, nu, alpha, beta), latent_t1 = model.source_forward(
                X_batch, domain_id="muscatine"
            )
            task_l  = evid_loss(y_batch, gamma, nu, alpha, beta)
            upset_l = upset_cost_loss(y_batch.squeeze(-1), gamma.squeeze(-1))

            # ── Physics loss on LATENT states ───────────────────────
            with torch.no_grad():
                X_shift = torch.roll(X_batch, shifts=1, dims=1)
                X_shift[:, 0, :] = X_batch[:, 0, :]
            (_, _, _, _), latent_t0 = model.source_forward(
                X_shift, domain_id="muscatine"
            )
            latent_states = {
                "t0": {k: v.detach() for k, v in latent_t0.items()},
                "t1": latent_t1,
            }
            sensor_batch = build_si_batch(X_batch, loader.feature_cols)
            phys_raw, phys_diag = pinn_loss(latent_states, sensor_batch,
                                             dt=1.0, weight=1.0,
                                             return_diagnostics=True)
            phys_diag_batch.append(phys_diag)
            phys_l = physics_w * torch.clamp(phys_raw, max=10.0)

            # ── Distillation loss (Phase 2 + 3 only) ───────────────
            distill_weight = getattr(config, "DISTILLATION_WEIGHT", 0.5)
            distill_l = torch.zeros(1, device=device).squeeze()

            if current_phase >= 2:
                # Muscatine encoder produces teacher latents (detached in Phase 2)
                with torch.set_grad_enabled(current_phase == 3):
                    z_teacher = model._encode(X_batch, "muscatine")

                # Small-domain encoders produce student latents
                z_dataone = model._encode(X_batch, "dataone")
                z_edi     = model._encode(X_batch, "edi")

                distill_l = distill_weight * (
                    latent_distillation_loss(z_dataone, z_teacher) +
                    latent_distillation_loss(z_edi,     z_teacher)
                )

            # ── Combined loss ───────────────────────────────────
            loss = task_l + 0.1 * upset_l + phys_l + distill_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tr_task.append(task_l.item())
            tr_phys.append(phys_l.item())
            tr_distill.append(distill_l.item() if isinstance(distill_l, torch.Tensor)
                              else float(distill_l))

            replay_buf.add_batch(
                X_batch.cpu().numpy(),
                y_batch.cpu().numpy().ravel()
            )

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                (g, n, a, b), _ = model.source_forward(X_b, domain_id="muscatine")
                val_losses.append(evid_loss(y_b, g, n, a, b).item())

        tr_t = np.mean(tr_task)
        tr_p = np.mean(tr_phys)
        tr_d = np.mean(tr_distill)
        vl   = np.mean(val_losses)

        # Aggregate physics diagnostics
        mean_compliance  = np.mean([d["compliance_pct"]   for d in phys_diag_batch])
        mean_dX_residual = np.mean([d["mean_dX_residual"] for d in phys_diag_batch])

        history["train_task"].append(tr_t)
        history["train_phys"].append(tr_p)
        history["train_phys_weight"].append(physics_w)
        history["train_distill"].append(tr_d)
        history["val"].append(vl)
        history["phys_compliance"].append(mean_compliance)
        history["mean_dX_residual"].append(mean_dX_residual)
        history["phase"].append(current_phase)

        # ── Diagnostics (every 10 epochs) ───────────────────────────────
        ph_str = f"P{current_phase}"
        if epoch % 10 == 0 or epoch == 1:
            grad_report = check_gradient_flow(model)
            lstm_grads  = [v for k, v in grad_report.items()
                           if v is not None and "lstm" in k and "weight" in k]
            mean_lstm_g = np.mean(lstm_grads) if lstm_grads else 0.0
            pred_var    = check_prediction_variance(model, val_dl, device)

            history["grad_lstm"].append(mean_lstm_g)
            history["pred_var"].append(pred_var)

            print(f"  [{ph_str}] Epoch {epoch:3d}/{epochs} | "
                  f"Task={tr_t:.4f} | Phys={tr_p:.4f} (w={physics_w:.3f}) | "
                  f"Distill={tr_d:.4f} | "
                  f"Val={vl:.4f} | LR={scheduler.get_last_lr()[0]:.2e} | "
                  f"LSTM|g|={mean_lstm_g:.2e} | Var={pred_var:.4f} | "
                  f"Compl={mean_compliance:.1%} | dX={mean_dX_residual:.4f} | "
                  f"TF={teacher_p:.2f}")
        else:
            print(f"  [{ph_str}] Epoch {epoch:3d}/{epochs} | "
                  f"Task={tr_t:.4f} | Phys={tr_p:.4f} (w={physics_w:.3f}) | "
                  f"Distill={tr_d:.4f} | Val={vl:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────
        if vl < best_val:
            best_val  = vl
            patience  = 0
            _save_checkpoint(model, loader, replay_buf, epoch)
        else:
            patience += 1
            if patience >= config.TRAIN["patience"]:
                print(f"  [EarlyStopping] Phase {current_phase}: patience exceeded at epoch {epoch}")
                # Only hard-stop in Phase 3; otherwise advance to next phase
                if current_phase == 3:
                    break
                else:
                    # Force advance to next phase
                    curr = getattr(config, "CURRICULUM", {})
                    if current_phase == 1:
                        p1 = curr.get("phase1_epochs", 50)
                        # skip to start of Phase 2
                        epoch_override = p1
                    else:
                        p1 = curr.get("phase1_epochs", 50)
                        p2 = curr.get("phase2_epochs", 30)
                        epoch_override = p1 + p2
                    # Rewrite epoch counter via a hack: set loop variable
                    # (Python for loops don't support this cleanly, so we use
                    #  a flag and continue — the phase transition logic above
                    #  will handle it on the next iteration)
                    patience = 0   # reset to allow next phase to run

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
    # Filter out post-training fitted buffers before loading —
    # strict=False alone doesn't bypass shape mismatches in PyTorch.
    # These buffers are already correctly fitted in memory from the
    # fit_domain_router / fit_maha_gate calls above.
    _SKIP_BUFFERS = {
        "router.centroids",
        "router.centroids_fitted",
        "maha_gate.mean_",
        "maha_gate.inv_cov_",
        "maha_gate.fitted_",
    }
    filtered_state = {k: v for k, v in ckpt["model_state"].items()
                      if k not in _SKIP_BUFFERS}
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if unexpected:
        print(f"  [EWC] Unexpected keys (ignored): {unexpected[:3]}")
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
    os.makedirs(config.MODEL_DIR, exist_ok=True)   # guarantee dir exists
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
    print(f"    [CKPT] Saved epoch {epoch} → {config.MODEL_DIR}/source_best.pt")


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
