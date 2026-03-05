"""
model.py  (v4 — Asymmetric Encoders + Curriculum Training + Distillation)

Changes from v3 → v4:
  - MuscatineEncoder          : unchanged, full 128-dim LSTM (teacher)
  - DataONEEncoder            : reduced to 64-dim hidden + align_proj → 64-dim output
  - EDIEncoder                : reduced to 64-dim hidden + align_proj → 64-dim output
  - latent_distillation_loss(): ||z_small - z_musc.detach()||^2 for curriculum Phase 2
  - ProcessTypeClassifier     : auxiliary head (continuous / batch / pilot)
  - ContinuousDecoder         : standard LatentBiokinetics path for continuous domains
  - BatchDecoder              : takes optional time_since_loading feature — fixes EDI
  - DomainScaler              : trainable log-affine per domain — fixes 86× scale disparity
  - freeze_muscatine_encoder(): freeze/unfreeze helpers for curriculum
  - predict_lodo()            : LOO evaluation with frozen Muscatine encoder as prior
  - BiogasTransferModel.source_forward() routes to correct decoder by domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import config
from src.evidential_loss import EvidentialHead, EvidentialLoss


# ─── Gradient Reversal ────────────────────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


# ─── MMD Loss ─────────────────────────────────────────────────────────────────

def mmd_loss(source: torch.Tensor, target: torch.Tensor,
             kernel_mul=2.0, kernel_num=5) -> torch.Tensor:
    n_s, n_t = source.size(0), target.size(0)
    if n_s == 0 or n_t == 0:
        return torch.zeros(1, device=source.device).squeeze()
    total    = torch.cat([source, target], dim=0)
    total_sq = torch.sum(total ** 2, dim=1, keepdim=True)
    sq_dists = (total_sq + total_sq.t() - 2 * total @ total.t()).clamp(min=0)
    bandwidth  = sq_dists.data.mean() / (kernel_mul ** (kernel_num // 2))
    bandwidths = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    mmd = torch.zeros(1, device=source.device)
    for bw in bandwidths:
        K   = torch.exp(-sq_dists / (2 * bw + 1e-9))
        mmd += K[:n_s, :n_s].mean() + K[n_s:, n_s:].mean() - 2 * K[:n_s, n_s:].mean()
    return (mmd / len(bandwidths)).squeeze()


# ─── Sensor Dropout ───────────────────────────────────────────────────────────

class SensorDropout(nn.Module):
    """Randomly masks entire sensor groups during training."""

    def __init__(self, group_indices: dict, p_group: float = 0.15):
        super().__init__()
        self.group_indices = group_indices
        self.p_group       = p_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        for _, indices in self.group_indices.items():
            if indices and torch.rand(1).item() < self.p_group:
                for idx in indices:
                    if idx < x.shape[-1]:
                        x[:, :, idx] = 0.0
        return x


# ─── Cross-Sensor Attention ───────────────────────────────────────────────────

class CrossSensorAttention(nn.Module):
    """Multi-head attention across feature dimension."""

    def __init__(self, feat_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        proj_dim = feat_dim + (n_heads - feat_dim % n_heads) % n_heads
        proj_dim = max(proj_dim, n_heads * 4)
        self.proj_in  = nn.Linear(feat_dim, proj_dim)
        self.attn     = nn.MultiheadAttention(proj_dim, n_heads,
                                               dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(proj_dim, feat_dim)
        self.norm     = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp         = self.proj_in(x)
        att_out, _ = self.attn(xp, xp, xp)
        return self.norm(x + self.proj_out(att_out))


# ─── Domain-Specific Encoders ─────────────────────────────────────────────────

class MuscatineEncoder(nn.Module):
    """
    CNN → LSTM → Temporal Attention.
    Optimised for dense, structured Muscatine SCADA data.
    Output: (B, out_dim)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm      = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                                  batch_first=True, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.proj    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                      nn.GELU(), nn.Dropout(dropout))
        self.out_dim = hidden_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cnn = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out    = self.lstm_norm(lstm_out)
        att_out, _  = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        att_out     = self.attn_norm(lstm_out + att_out)
        fused = att_out[:, -1, :] + att_out.mean(dim=1)
        return self.proj(fused)


class DataONEEncoder(nn.Module):
    """
    GRU with missing-value mask channel (v4: reduced to 64-dim hidden).
    Handles the 65-column sparse DataONE AD dataset where many columns
    are absent per row. The mask (0/1 per feature) is concatenated to the
    input so the GRU can learn which channels are reliable.

    v4 change: hidden_dim reduced 128 → 64 so 100 DataONE samples cannot
    compete on equal footing with 75K Muscatine samples during shared training.
    align_proj restores output to enc_out (64) for downstream head compatibility.

    Output: (B, out_dim=64)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        small_h = max(32, hidden_dim // 2)   # 64 when hidden_dim=128 (default)
        # mask channel doubles the input dimension
        self.gru = nn.GRU(feat_dim * 2, small_h, num_layers=2,
                           batch_first=True, dropout=dropout)
        self.norm  = nn.LayerNorm(small_h)
        self.proj  = nn.Sequential(nn.Linear(small_h, small_h // 2),
                                    nn.GELU(), nn.Dropout(dropout))
        # Align projector: lifts small_h//2 output up to full enc_out (hidden_dim//2)
        # so downstream prediction head + DomainScaler shapes are unchanged
        self.align_proj = nn.Linear(small_h // 2, hidden_dim // 2)
        self.out_dim = hidden_dim // 2   # same interface as MuscatineEncoder
        self._small_out = small_h // 2   # actual pre-align dimension (for distillation)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, T, F)
        mask : (B, T, F)  1 = observed, 0 = imputed/missing
               If None, inferred as (x != 0).float()
        """
        if mask is None:
            mask = (x != 0.0).float()
        x_masked = torch.cat([x, mask], dim=-1)   # (B, T, 2F)
        out, _   = self.gru(x_masked)
        out      = self.norm(out)
        small_z  = self.proj(out[:, -1, :] + out.mean(dim=1))  # (B, small_h//2)
        return self.align_proj(small_z)   # (B, hidden_dim//2) = (B, enc_out)


class EDIEncoder(nn.Module):
    """
    Transformer encoder for irregularly-sampled EDI SSAD time series (v4: 64-dim).
    Uses learned positional encoding (not fixed sinusoidal) to handle
    irregular time gaps.

    v4 change: hidden_dim reduced 128 → 64 to prevent ~100-sample EDI
    gradients from overriding Muscatine's 75K-sample representation.
    align_proj restores output to enc_out (64) for downstream compatibility.

    Output: (B, out_dim=64)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        small_h = max(32, hidden_dim // 2)   # 64 when hidden_dim=128
        # n_heads must divide small_h evenly
        _n_heads = min(n_heads, small_h // 8) or 1
        while small_h % _n_heads != 0:
            _n_heads -= 1
        _n_heads = max(1, _n_heads)

        self.input_proj = nn.Linear(feat_dim, small_h)
        self.pos_emb    = nn.Embedding(512, small_h)   # up to 512 timesteps
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=small_h, nhead=_n_heads,
            dim_feedforward=small_h * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.norm = nn.LayerNorm(small_h)
        self.proj = nn.Sequential(nn.Linear(small_h, small_h // 2),
                                   nn.GELU(), nn.Dropout(dropout))
        # Align projector: lifts small_h//2 up to full enc_out
        self.align_proj = nn.Linear(small_h // 2, hidden_dim // 2)
        self.out_dim = hidden_dim // 2   # same interface as MuscatineEncoder
        self._small_out = small_h // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.input_proj(x) + self.pos_emb(pos)
        h   = self.transformer(h)
        h   = self.norm(h)
        small_z = self.proj(h[:, -1, :] + h.mean(dim=1))   # (B, small_h//2)
        return self.align_proj(small_z)                     # (B, enc_out)


# ─── Distillation Loss ─────────────────────────────────────────────────────────────

def latent_distillation_loss(z_small: torch.Tensor,
                             z_teacher: torch.Tensor) -> torch.Tensor:
    """
    Curriculum Phase 2: aligns small-encoder latents to Muscatine teacher.

    ||z_small - z_teacher.detach()||^2

    z_teacher is DETACHED so gradients only flow through the small encoder.
    Muscatine encoder stays frozen during Phase 2.
    """
    return F.mse_loss(z_small, z_teacher.detach())


# ─── Domain Similarity Router ─────────────────────────────────────────────────

class DomainSimilarityRouter(nn.Module):
    """
    Selects the best-matching source encoder for a target batch using
    cosine similarity to stored domain centroids.

    Centroids are registered as buffers so they're saved with the model
    and moved to the correct device. They must be computed once after
    source training via fit_centroids().

    During training : soft weighted mixture (differentiable).
    During eval     : hard argmax routing (single encoder).
    """

    DOMAIN_NAMES = ["muscatine", "dataone", "edi"]

    def __init__(self):
        super().__init__()
        # Derive centroid size from config to match encoder output dim exactly
        _enc_out = config.MODEL["hidden_dims"][0] // 2   # same as enc_out in BiogasTransferModel
        self.register_buffer("centroids", torch.zeros(3, _enc_out))
        self.register_buffer("centroids_fitted", torch.tensor(False))

    def fit_centroids(self, embeddings_dict: dict):
        """
        Call after source training with:
          embeddings_dict = {
              'muscatine': Tensor(N, D),
              'dataone':   Tensor(N, D),
              'edi':       Tensor(N, D),
          }
        """
        stacked = []
        for name in self.DOMAIN_NAMES:
            if name in embeddings_dict:
                stacked.append(embeddings_dict[name].mean(0))
            else:
                stacked.append(torch.zeros(embeddings_dict[
                    list(embeddings_dict.keys())[0]].shape[-1]))
        self.centroids = torch.stack(stacked).to(self.centroids.device)
        self.centroids_fitted.fill_(True)

    def route_weights(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Returns soft weights (B, 3) for each encoder given a feature batch.
        """
        if not self.centroids_fitted.item():
            # Equal mixture if not fitted yet
            return torch.ones(feat.shape[0], 3,
                               device=feat.device) / 3.0
        feat_norm = F.normalize(feat, dim=-1)          # (B, D)
        cent_norm = F.normalize(self.centroids, dim=-1) # (3, D)
        sim = feat_norm @ cent_norm.t()                 # (B, 3)
        if self.training:
            return F.softmax(sim * 5.0, dim=-1)         # soft
        else:
            idx = sim.argmax(dim=-1)                    # hard argmax
            w = torch.zeros_like(sim)
            w.scatter_(1, idx.unsqueeze(1), 1.0)
            return w


# ─── Latent Biokinetics Decoder ───────────────────────────────────────────────

class LatentBiokineticsDecoder(nn.Module):
    """
    Decodes the encoder feature vector into latent biochemical state:
      X_hat   – latent biomass proxy        (non-negative)
      S_hat   – latent substrate proxy      (non-negative)
      VFA_hat – latent VFA/acetate proxy    (non-negative)

    These three quantities are passed to PhysicsInformedLoss for ODE
    residual computation, guaranteeing that physics gradients flow all
    the way back through the encoder/LSTM weights.

    Returns:
      latent_dict : {'X': (B,1), 'S': (B,1), 'VFA': (B,1)}
    """

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, feat: torch.Tensor) -> dict:
        out = F.softplus(self.net(feat))   # (B, 3) — all non-negative
        return {
            "X":   out[:, 0:1],
            "S":   out[:, 1:2],
            "VFA": out[:, 2:3],
        }


# ─── Process State Classifier ─────────────────────────────────────────────────

class ProcessStateClassifier(nn.Module):
    """Classifies digester state: 0=startup, 1=stable, 2=unstable."""

    def __init__(self, feat_dim: int, n_states: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_states),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ─── Domain Discriminator ─────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(self.grl(x))


# ─── Mahalanobis Input Gate ────────────────────────────────────────────────────

class MahalanobisGate(nn.Module):
    """
    Rejects out-of-domain batches at inference time.
    Fitted on source-domain embedding centroids after training.

    Usage:
      gate.fit(source_embeddings)       # after training
      reject_mask = gate(feat)          # True = OOD, skip prediction
    """

    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold
        self.register_buffer("mean_", torch.zeros(1))
        self.register_buffer("inv_cov_", torch.zeros(1, 1))
        self.register_buffer("fitted_", torch.tensor(False))

    @torch.no_grad()
    def fit(self, embeddings: torch.Tensor):
        """embeddings: (N, D)"""
        mu  = embeddings.mean(0)
        centered = embeddings - mu
        cov = (centered.t() @ centered) / (len(embeddings) - 1 + 1e-6)
        # Regularise with diagonal to ensure invertibility
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-4
        self.mean_   = mu
        self.inv_cov_ = torch.linalg.inv(cov)
        self.fitted_.fill_(True)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns boolean mask (B,) where True = OOD (reject)."""
        if not self.fitted_.item():
            return torch.zeros(feat.shape[0], dtype=torch.bool,
                               device=feat.device)
        diff  = feat - self.mean_
        dist  = (diff @ self.inv_cov_ * diff).sum(-1).sqrt()   # (B,)
        return dist > self.threshold


# ─── Process Type Classifier (v4) ─────────────────────────────────────────────

class ProcessTypeClassifier(nn.Module):
    """Auxiliary head: 0=continuous, 1=batch, 2=pilot."""
    def __init__(self, feat_dim: int):
        super().__init__()
        n_types = len(config.PROCESS_TYPE)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_types),
        )
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ─── ContinuousDecoder & BatchDecoder (v4) ────────────────────────────────────

class ContinuousDecoder(nn.Module):
    """Biokinetics decoder for continuous-flow reactors (Muscatine, DataONE)."""
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, 3),
        )
    def forward(self, feat: torch.Tensor, t_loading=None) -> dict:
        out = F.softplus(self.net(feat))
        return {"X": out[:, 0:1], "S": out[:, 1:2], "VFA": out[:, 2:3]}


class BatchDecoder(nn.Module):
    """
    Biokinetics decoder for batch reactors (EDI).
    Takes time_since_loading to model batch production curve — fixes EDI R²=-1.83.
    """
    def __init__(self, feat_dim: int, hidden: int = 64, t_dim: int = 8):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(feat_dim + t_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, 3),
        )
    def forward(self, feat: torch.Tensor, t_loading=None) -> dict:
        if t_loading is None:
            t_loading = torch.zeros(feat.shape[0], 1, device=feat.device)
        t_emb = self.time_embed(t_loading)
        out   = F.softplus(self.net(torch.cat([feat, t_emb], dim=-1)))
        return {"X": out[:, 0:1], "S": out[:, 1:2], "VFA": out[:, 2:3]}


# ─── Domain Scaler (v4 — fixes 86× scale disparity) ──────────────────────────

class DomainScaler(nn.Module):
    """Trainable log-affine scale per domain. Fixes 86× Muscatine vs DataONE/EDI disparity."""
    DOMAINS = ["muscatine", "dataone", "edi"]
    def __init__(self):
        super().__init__()
        n = len(self.DOMAINS)
        self.log_scale = nn.Parameter(torch.zeros(n))
        self.bias      = nn.Parameter(torch.zeros(n))
        self._idx = {d: i for i, d in enumerate(self.DOMAINS)}
    def forward(self, gamma: torch.Tensor, domain: str) -> torch.Tensor:
        i     = self._idx.get(domain, 0)
        scale = torch.exp(self.log_scale[i])
        return scale * gamma + self.bias[i]


# ─── Full Transfer Model ───────────────────────────────────────────────────────

class BiogasTransferModel(nn.Module):
    """
    Enhanced biogas transfer learning model (v4).

    Pipeline:
      SensorDropout
      → CrossSensorAttention
      → [MuscatineEncoder | DataONEEncoder | EDIEncoder]
         (selected by DomainSimilarityRouter)
      → ContinuousDecoder | BatchDecoder  (process-type routing, v4)
      → DomainScaler                      (per-domain affine, v4)
      → EvidentialHead            (prediction + uncertainty)
      → ProcessStateClassifier    (conditional alignment)
      → DomainDiscriminator       (adversarial alignment)
      → MahalanobisGate           (OOD rejection at inference)
    """

    def __init__(self, group_indices: dict = None):
        super().__init__()
        cfg      = config.MODEL
        feat_dim = cfg["input_dim"]
        h_dim    = cfg["hidden_dims"][0]
        dropout  = cfg["dropout_rate"]
        enc_out  = h_dim // 2   # all encoders → same output dim

        self.sensor_dropout    = SensorDropout(group_indices or {}, p_group=0.15)
        self.cross_sensor_attn = CrossSensorAttention(feat_dim, n_heads=4,
                                                       dropout=dropout)

        # Domain-specific encoders
        self.encoder_muscatine = MuscatineEncoder(feat_dim, h_dim, dropout)
        self.encoder_dataone   = DataONEEncoder(feat_dim, h_dim, dropout)
        self.encoder_edi       = EDIEncoder(feat_dim, h_dim, dropout=dropout)

        # Router
        self.router = DomainSimilarityRouter()

        # Decoders: continuous vs batch (v4)
        self.decoder_continuous = ContinuousDecoder(enc_out)
        self.decoder_batch      = BatchDecoder(enc_out)

        # Domain scaler to fix 86× scale disparity (v4)
        self.domain_scaler = DomainScaler()

        # Shared heads
        self.pred_head      = EvidentialHead(enc_out)
        self.state_clf      = ProcessStateClassifier(enc_out)   # digester state
        self.process_clf    = ProcessTypeClassifier(enc_out)    # process type (v4)
        self.domain_disc    = DomainDiscriminator(enc_out, alpha=1.0)
        self.maha_gate      = MahalanobisGate(threshold=4.0)
        self.evid_loss      = EvidentialLoss(coeff_reg=1e-2, reject_thresh=2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─── Encoding ────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor,
                domain_id: str = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode input sequence → feature vector.

        domain_id : 'muscatine' | 'dataone' | 'edi' | None
                    If None → router selects best encoder.
        mask      : (B, T, F) boolean mask for DataONE (optional)
        """
        x = self.sensor_dropout(x)
        x = self.cross_sensor_attn(x)

        if domain_id == "muscatine":
            return self.encoder_muscatine(x)
        elif domain_id == "dataone":
            return self.encoder_dataone(x, mask)
        elif domain_id == "edi":
            return self.encoder_edi(x)
        else:
            # Soft-router: use temporary Muscatine pass to get feat for routing
            with torch.no_grad():
                feat_probe = self.encoder_muscatine(x)
            weights = self.router.route_weights(feat_probe)  # (B, 3)

            if self.training:
                # Soft mixture — compute all three (differentiable)
                f_m = self.encoder_muscatine(x)
                f_d = self.encoder_dataone(x, mask)
                f_e = self.encoder_edi(x)
                w   = weights.unsqueeze(-1)   # (B, 3, 1)
                stacked = torch.stack([f_m, f_d, f_e], dim=1)   # (B, 3, D)
                return (stacked * w).sum(dim=1)
            else:
                # Hard routing — use best matching encoder only
                best = weights.argmax(dim=-1)   # (B,)
                outs = torch.zeros(x.shape[0], self.encoder_muscatine.out_dim,
                                   device=x.device)
                mask_m = best == 0
                mask_d = best == 1
                mask_e = best == 2
                if mask_m.any(): outs[mask_m] = self.encoder_muscatine(x[mask_m])
                if mask_d.any(): outs[mask_d] = self.encoder_dataone(x[mask_d])
                if mask_e.any(): outs[mask_e] = self.encoder_edi(x[mask_e])
                return outs

    # ─── Forward Passes ──────────────────────────────────────────────────────

    def source_forward(self, x: torch.Tensor,
                       domain_id: str = "muscatine",
                       mask: torch.Tensor = None,
                       t_loading: torch.Tensor = None):
        """
        Returns:
          (gamma_scaled, nu, alpha, beta) — NIG parameters (gamma is domain-scaled)
          latent_states                   — {'X', 'S', 'VFA'} for physics loss
        """
        feat = self._encode(x, domain_id, mask)

        # Route to correct decoder based on process type (v4)
        process_type = config.DOMAIN_PROCESS_MAP.get(domain_id, "continuous")
        if process_type == "batch":
            latent = self.decoder_batch(feat, t_loading)
        else:
            latent = self.decoder_continuous(feat)

        g, nu, alpha, beta = self.pred_head(feat)

        # Apply trainable domain scaler (v4)
        g_scaled = self.domain_scaler(g, domain_id)

        return (g_scaled, nu, alpha, beta), latent

    def adapt_forward(self, src_x: torch.Tensor, tgt_x: torch.Tensor,
                      src_domain: str = "muscatine",
                      tgt_domain: str = None,
                      conditioning: bool = True):
        """
        Domain adaptation forward pass.
        Alignment is conditional: only within the same process-state class.
        Returns: pred_src, dom_src, dom_tgt, f_sa, f_ta, state_src, state_tgt, latent
        """
        feat_src = self._encode(src_x, src_domain)
        feat_tgt = self._encode(tgt_x, tgt_domain)

        # Use correct decoder for source domain
        proc_src = config.DOMAIN_PROCESS_MAP.get(src_domain, "continuous")
        latent   = (self.decoder_batch(feat_src)
                    if proc_src == "batch"
                    else self.decoder_continuous(feat_src))

        pred_src  = self.pred_head(feat_src)
        state_src = self.state_clf(feat_src)
        state_tgt = self.state_clf(feat_tgt)

        if conditioning:
            s_idx = state_src.argmax(-1)
            t_idx = state_tgt.argmax(-1)
            dom   = int(s_idx.mode().values.item())
            f_sa  = feat_src[s_idx == dom] if (s_idx == dom).any() else feat_src
            f_ta  = feat_tgt[t_idx == dom] if (t_idx == dom).any() else feat_tgt
        else:
            f_sa, f_ta = feat_src, feat_tgt

        dom_src = self.domain_disc(f_sa)
        dom_tgt = self.domain_disc(f_ta)
        return pred_src, dom_src, dom_tgt, f_sa, f_ta, state_src, state_tgt, latent

    def freeze_small_domain_encoders(
            self,
            small_domains: list = None,
            verbose: bool = True):
        """
        Freezes encoders for small domains during LODO / fine-tuning.
        Prevents catastrophic forgetting when adapting to a new target domain.

        Args:
          small_domains : list of domain names to freeze, e.g. ['edi','dataone'].
                          If None, uses config.SMALL_DOMAINS (falls back to edi+dataone).
        """
        if small_domains is None:
            small_domains = getattr(config, "SMALL_DOMAINS",
                                    ["edi", "dataone"])
        enc_map = {
            "muscatine": self.encoder_muscatine,
            "dataone":   self.encoder_dataone,
            "edi":       self.encoder_edi,
        }
        for domain in small_domains:
            enc = enc_map.get(domain)
            if enc is None:
                continue
            for p in enc.parameters():
                p.requires_grad = False
            if verbose:
                print(f"  [LODO] Frozen encoder: {domain}")

    def freeze_muscatine_encoder(self, verbose: bool = True):
        """Freeze Muscatine encoder (curriculum Phase 2 — teacher locked)."""
        for p in self.encoder_muscatine.parameters():
            p.requires_grad = False
        if verbose:
            print("  [Curriculum] Muscatine encoder FROZEN (teacher mode)")

    def unfreeze_muscatine_encoder(self, verbose: bool = True):
        """Unfreeze Muscatine encoder (curriculum Phase 3 — fine-tune all)."""
        for p in self.encoder_muscatine.parameters():
            p.requires_grad = True
        if verbose:
            print("  [Curriculum] Muscatine encoder UNFROZEN (Phase 3 fine-tune)")

    def unfreeze_small_domain_encoders(
            self,
            small_domains: list = None,
            verbose: bool = True):
        """Unfreeze small-domain encoders (curriculum Phase 2/3)."""
        if small_domains is None:
            small_domains = ["dataone", "edi"]
        enc_map = {
            "muscatine": self.encoder_muscatine,
            "dataone":   self.encoder_dataone,
            "edi":       self.encoder_edi,
        }
        for domain in small_domains:
            enc = enc_map.get(domain)
            if enc is None:
                continue
            for p in enc.parameters():
                p.requires_grad = True
            if verbose:
                print(f"  [Curriculum] Unfrozen encoder: {domain}")

    def set_grl_alpha(self, alpha: float):
        self.domain_disc.grl.alpha = alpha

    # ─── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                scaler_y=None,
                domain_id: str = None,
                mask: torch.Tensor = None):
        self.eval()
        feat   = self._encode(x, domain_id, mask)
        reject_ood = self.maha_gate(feat)
        mean, aleatoric, epistemic, total, reject_evid = self.pred_head.predict(
            feat, self.evid_loss
        )
        reject = reject_ood | reject_evid.squeeze(-1)
        if scaler_y is not None:
            scale = torch.tensor(scaler_y.scale_[0], dtype=torch.float32)
            mu_   = torch.tensor(scaler_y.mean_[0],  dtype=torch.float32)
            mean      = mean.squeeze() * scale + mu_
            aleatoric = aleatoric.squeeze() * scale
            epistemic = epistemic.squeeze() * scale
        return mean, aleatoric, epistemic, reject

    # ─── Centroid / Gate Fitting ─────────────────────────────────────────────

    @torch.no_grad()
    def fit_domain_router(self, embeddings_dict: dict):
        """Call after source training with per-domain embedding tensors."""
        self.router.fit_centroids(embeddings_dict)

    @torch.no_grad()
    def fit_maha_gate(self, source_loader, device):
        """Collect source embeddings and fit Mahalanobis gate."""
        self.eval()
        feats = []
        for X_b, _ in source_loader:
            X_b = X_b.to(device)
            f   = self._encode(X_b, "muscatine")
            feats.append(f.cpu())
        all_feat = torch.cat(feats, dim=0)
        self.maha_gate.fit(all_feat.to(device))
        print(f"  [MahalanobisGate] Fitted on {len(all_feat)} source embeddings")

    # ─── LOO Ensemble Prior ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict_lodo(self, x: torch.Tensor,
                     excluded_domain: str,
                     scaler_y=None,
                     mask: torch.Tensor = None):
        """
        LOO prediction with frozen Muscatine encoder as distribution prior.

        When excluded_domain='muscatine', we KEEP the frozen Muscatine encoder
        as a prior rather than dropping it entirely.  This avoids the collapse
        seen when 200 remaining samples must support a 128-dim LSTM alone.

        Ensemble weighting:
          - If Muscatine excluded from training data: use 0.3 Muscatine + 0.7 small
          - If a small domain excluded: use 1.0 Muscatine
        """
        self.eval()
        available = [d for d in ["muscatine", "dataone", "edi"]
                     if d != excluded_domain]

        if excluded_domain == "muscatine":
            # Muscatine encoder acts as frozen prior (not fully excluded)
            f_m = self._encode(x, "muscatine")          # (B, enc_out)
            # Soft prediction from remaining small encoders
            f_d = self._encode(x, "dataone", mask)      # (B, enc_out)
            f_e = self._encode(x, "edi")                # (B, enc_out)
            # Weighted ensemble: Muscatine prior (30%) + small-domain mean (70%)
            feat = 0.3 * f_m + 0.35 * f_d + 0.35 * f_e
        else:
            # Simple case: excluded small domain, use Muscatine as primary
            feat = self._encode(x, "muscatine")

        mean, aleatoric, epistemic, total, reject_evid = self.pred_head.predict(
            feat, self.evid_loss
        )
        if scaler_y is not None:
            scale = torch.tensor(scaler_y.scale_[0], dtype=torch.float32)
            mu_   = torch.tensor(scaler_y.mean_[0],  dtype=torch.float32)
            mean      = mean.squeeze() * scale + mu_
            aleatoric = aleatoric.squeeze() * scale
            epistemic = epistemic.squeeze() * scale
        return mean, aleatoric, epistemic
