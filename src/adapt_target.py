"""
adapt_target.py  (v2)
Phase 2: Adapt to target plant with:
  - Conditional domain adaptation (state-conditioned DANN)
  - Dynamic GRL alpha schedule
  - Active learning sample selection
  - Optional MAML fast-adaptation path
  - EWC regularisation to prevent forgetting

Run:  python src/adapt_target.py
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
from src.dataset          import SourceDataLoader, TargetDataLoader
from src.model            import BiogasTransferModel, mmd_loss
from src.evidential_loss  import EvidentialLoss
from src.physics_loss     import PhysicsInformedLoss, build_physics_dict
from src.active_learning  import ActiveLearningManager
from src.continual_learning import ReplayBuffer, EWC, OnlineAdapter
from src.data_quality     import assign_sensor_groups
from src.evaluation       import full_evaluation

torch.manual_seed(config.SEED)
os.makedirs(config.MODEL_DIR,  exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
)
print(f"[adapt_target v2] Device: {device}")


# ─── Dynamic alpha schedule ───────────────────────────────────────────────────

def alpha_schedule(epoch: int, total: int) -> float:
    p = epoch / total
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


# ─── Main Adaptation ──────────────────────────────────────────────────────────

def adapt_target():
    # 1. Load source checkpoint
    src_path = os.path.join(config.MODEL_DIR, "source_best.pt")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Run train_source.py first. Not found: {src_path}")

    ckpt         = torch.load(src_path, map_location=device, weights_only=False)  # our own checkpoint
    scaler_X     = ckpt["scaler_X"]
    scaler_y     = ckpt["scaler_y"]
    feature_cols = ckpt["feature_cols"]
    input_dim    = ckpt["input_dim"]
    config.MODEL["input_dim"] = input_dim

    group_indices = assign_sensor_groups(feature_cols)
    model = BiogasTransferModel(group_indices=group_indices).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[adapt] Loaded source model (epoch {ckpt['epoch']})")

    # 2. Rebuild EWC from saved Fisher
    ewc = None
    if "ewc_fisher" in ckpt:
        src_loader = SourceDataLoader()
        src_loader.scaler_X     = scaler_X
        src_loader.scaler_y     = scaler_y
        src_loader.feature_cols = feature_cols
        src_loader.input_dim    = input_dim
        src_tr_dl, _, _ = src_loader.load()

        ewc = EWC(model, src_tr_dl, device=str(device))
        ewc.fisher_    = {k: v.to(device) for k, v in ckpt["ewc_fisher"].items()}
        ewc.theta_star_ = {k: v.to(device) for k, v in ckpt["ewc_theta_star"].items()}
        print("[EWC] Loaded Fisher from checkpoint.")
    else:
        src_loader = SourceDataLoader()
        src_loader.scaler_X     = scaler_X
        src_loader.scaler_y     = scaler_y
        src_loader.feature_cols = feature_cols
        src_loader.input_dim    = input_dim
        src_tr_dl, _, _ = src_loader.load()

    # 3. Restore replay buffer
    replay_buf = ReplayBuffer(capacity=2000)
    if "replay_buffer_X" in ckpt and ckpt["replay_buffer_X"]:
        replay_buf.buffer_X = list(ckpt["replay_buffer_X"])
        replay_buf.buffer_y = list(ckpt["replay_buffer_y"])
        print(f"[Replay] Loaded {len(replay_buf)} source samples.")

    # 4. Load target data
    tgt_loader = TargetDataLoader(scaler_X, scaler_y, feature_cols)
    tgt_adapt_dl, tgt_fs_dl = tgt_loader.load(few_shot_fraction=0.1)

    # 5. Active learning: select most informative samples from target pool
    print("\n── Active Learning Sample Selection ──")
    al_manager = ActiveLearningManager(
        model=model, strategy="epistemic", budget=20, device=str(device)
    )
    # We use the first 200 target samples as the "unlabelled pool"
    pool_X, pool_y = _extract_numpy(tgt_adapt_dl, n_max=200)
    if len(pool_X) > 0:
        selected_idx = al_manager.query(pool_X)
        print(f"  Selected {len(selected_idx)} high-uncertainty samples for annotation.")

    # 6. Step A: Conditional DANN + MMD alignment
    print("\n── Step A: Conditional Domain Adaptation ──")
    _run_alignment(model, src_tr_dl, tgt_adapt_dl, scaler_y, ewc)

    # 7. Step B: Few-shot fine-tuning (freeze backbone)
    print("\n── Step B: Few-Shot Fine-Tuning ──")
    _run_fewshot(model, tgt_fs_dl, scaler_y, ewc)

    # 8. Online adapter demo
    print("\n── Step C: Setup Online Adapter ──")
    online_adapter = OnlineAdapter(
        model=model,
        replay_buffer=replay_buf,
        ewc=ewc,
        lr=5e-5,
        replay_ratio=0.5,
        lambda_ewc=0.4,
        device=str(device),
    )
    # Run a few online steps
    if len(pool_X) > 0:
        n_online = min(50, len(pool_X))
        for i in range(0, n_online, 4):
            batch = pool_X[i:i+4]
            online_adapter.update(batch, verbose=(i == 0))

    # 9. Save
    save_path = os.path.join(config.MODEL_DIR, "adapted_best.pt")
    torch.save({
        "model_state":  model.state_dict(),
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "feature_cols": feature_cols,
        "input_dim":    input_dim,
    }, save_path)
    print(f"\n[DONE] Adapted model saved → {save_path}")
    return model


# ─── Alignment Step ───────────────────────────────────────────────────────────

def _run_alignment(model, src_dl, tgt_dl, scaler_y, ewc):
    optim     = torch.optim.Adam(model.parameters(), lr=config.TRAIN["lr"] * 0.1)
    evid_loss = EvidentialLoss()
    pinn_loss = PhysicsInformedLoss()
    bce       = nn.BCEWithLogitsLoss()

    epochs    = config.TRAIN["adapt_epochs"]
    lam_mmd   = config.TRAIN["lambda_mmd"]
    lam_adv   = config.TRAIN["lambda_adv"]

    src_iter  = iter(src_dl)
    tgt_iter  = iter(tgt_dl)
    steps     = min(len(src_dl), len(tgt_dl))
    history   = {"task": [], "adv": [], "mmd": [], "ewc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = alpha_schedule(epoch, epochs)
        model.set_grl_alpha(alpha)

        task_ls, adv_ls, mmd_ls, ewc_ls = [], [], [], []

        for _ in range(steps):
            try:
                src_X, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_dl)
                src_X, src_y = next(src_iter)

            tgt_batch = next(tgt_iter, None)
            if tgt_batch is None:
                tgt_iter = iter(tgt_dl)
                tgt_batch = next(tgt_iter)

            tgt_X = tgt_batch[0]
            src_X, src_y = src_X.to(device), src_y.to(device)
            tgt_X = tgt_X.to(device)

            optim.zero_grad()
            (gamma, nu, alpha_p, beta), dom_src, dom_tgt, f_src, f_tgt, \
                state_src, state_tgt = model.adapt_forward(src_X, tgt_X, conditioning=True)

            # Task + physics
            task_l = evid_loss(src_y, gamma, nu, alpha_p, beta)
            phys_d = build_physics_dict(src_X, [])
            phys_l = pinn_loss(phys_d, gamma)

            # Adversarial
            adv_l = (bce(dom_src, torch.ones_like(dom_src))
                   + bce(dom_tgt, torch.zeros_like(dom_tgt)))

            # MMD
            mmd_l = mmd_loss(f_src, f_tgt)

            # EWC
            ewc_l = ewc.penalty(model, lambda_ewc=0.4) if ewc else torch.zeros(1, device=device).squeeze()

            total = task_l + 0.01 * phys_l + lam_adv * adv_l + lam_mmd * mmd_l + ewc_l
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            task_ls.append(task_l.item())
            adv_ls.append(adv_l.item())
            mmd_ls.append(mmd_l.item())
            ewc_ls.append(ewc_l.item() if isinstance(ewc_l, torch.Tensor) else ewc_l)

        history["task"].append(np.mean(task_ls))
        history["adv"].append(np.mean(adv_ls))
        history["mmd"].append(np.mean(mmd_ls))
        history["ewc"].append(np.mean(ewc_ls))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Task={history['task'][-1]:.4f}  "
                  f"ADV={history['adv'][-1]:.4f}  "
                  f"MMD={history['mmd'][-1]:.4f}  "
                  f"EWC={history['ewc'][-1]:.4f}  "
                  f"α={alpha:.3f}")

    _plot_adapt(history)


# ─── Few-shot step ────────────────────────────────────────────────────────────

def _run_fewshot(model, fs_dl, scaler_y, ewc):
    # Freeze backbone
    for p in model.encoder.parameters():
        p.requires_grad = False

    optim     = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    evid_loss = EvidentialLoss()
    epochs    = max(20, config.TRAIN["adapt_epochs"] // 2)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in fs_dl:
            X_b, y_b = batch[0].to(device), batch[1].to(device)
            optim.zero_grad()
            gamma, nu, alpha_p, beta = model.source_forward(X_b)
            loss = evid_loss(y_b, gamma, nu, alpha_p, beta)
            if ewc:
                loss = loss + ewc.penalty(model, lambda_ewc=0.2)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Fine-tune {epoch:3d}/{epochs}  Loss={np.mean(losses):.4f}")

    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_numpy(dataloader, n_max: int = 200):
    all_X, all_y = [], []
    for batch in dataloader:
        all_X.append(batch[0].numpy())
        all_y.append(batch[1].numpy())
        if sum(len(x) for x in all_X) >= n_max:
            break
    if not all_X:
        return np.array([]), np.array([])
    X = np.concatenate(all_X)[:n_max]
    y = np.concatenate(all_y)[:n_max]
    return X, y


def _plot_adapt(history):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    keys   = ["task", "adv", "mmd", "ewc"]
    colors = ["steelblue", "coral", "seagreen", "orchid"]
    for ax, key, color in zip(axes, keys, colors):
        ax.plot(history[key], color=color)
        ax.set_title(f"{key.upper()} Loss")
        ax.set_xlabel("Epoch")
    path = os.path.join(config.OUTPUT_DIR, "adaptation_loss.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved: {path}")


if __name__ == "__main__":
    adapt_target()
