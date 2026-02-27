"""
model.py  (v3 — Domain-Specific Encoders + Latent Biokinetics)

Architecture overview:
  Three domain-specific encoders:
    MuscatineEncoder  – CNN → LSTM (for dense, structured SCADA)
    DataONEEncoder    – GRU with missing-value mask channel
    EDIEncoder        – Transformer with learned positional encoding

  DomainSimilarityRouter:
    Routes target batches to the best matching source encoder using
    cosine similarity between batch embedding and stored domain centroids.
    Avoids averaging all three encoders which degrades cross-dataset R².

  LatentBiokineticsDecoder:
    Inserted between encoder and prediction head.
    Produces (X_hat, S_hat, VFA_hat) — latent biochemical concentrations.
    Physics constraints attach here (guarantees LSTM gradient flow).

  MahalanobisGate:
    Fitted on source training embeddings at end of training.
    At inference, rejects batches with Mahalanobis distance > threshold.

  EvidentialHead:
    Shared across all domain encoders.
    Produces (gamma, nu, alpha, beta) for NIG uncertainty.
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
    GRU with missing-value mask channel.
    Handles the 65-column sparse DataONE AD dataset where many columns
    are absent per row. The mask (0/1 per feature) is concatenated to the
    input so the GRU can learn which channels are reliable.
    Output: (B, out_dim)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        # mask channel doubles the input dimension
        self.gru = nn.GRU(feat_dim * 2, hidden_dim, num_layers=2,
                           batch_first=True, dropout=dropout)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.proj  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.GELU(), nn.Dropout(dropout))
        self.out_dim = hidden_dim // 2

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
        return self.proj(out[:, -1, :] + out.mean(dim=1))


class EDIEncoder(nn.Module):
    """
    Transformer encoder for irregularly-sampled EDI SSAD time series.
    Uses learned positional encoding (not fixed sinusoidal) to handle
    irregular time gaps.
    Output: (B, out_dim)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_emb    = nn.Embedding(512, hidden_dim)   # up to 512 timesteps
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                   nn.GELU(), nn.Dropout(dropout))
        self.out_dim = hidden_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.input_proj(x) + self.pos_emb(pos)
        h   = self.transformer(h)
        h   = self.norm(h)
        return self.proj(h[:, -1, :] + h.mean(dim=1))


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
        # Centroids set by fit_centroids(); zeros until then
        self.register_buffer("centroids", torch.zeros(3, 64))
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


# ─── Full Transfer Model ───────────────────────────────────────────────────────

class BiogasTransferModel(nn.Module):
    """
    Enhanced biogas transfer learning model (v3).

    Pipeline:
      SensorDropout
      → CrossSensorAttention
      → [MuscatineEncoder | DataONEEncoder | EDIEncoder]
         (selected by DomainSimilarityRouter)
      → LatentBiokineticsDecoder  (physics gradient path)
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

        # Shared decoder + heads
        self.latent_decoder = LatentBiokineticsDecoder(enc_out)
        self.pred_head      = EvidentialHead(enc_out)
        self.state_clf      = ProcessStateClassifier(enc_out)
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
                       mask: torch.Tensor = None):
        """
        Returns:
          (gamma, nu, alpha, beta)  — NIG parameters for evidential loss
          latent_states             — {'X', 'S', 'VFA'} for physics loss
        """
        feat   = self._encode(x, domain_id, mask)
        latent = self.latent_decoder(feat)
        g, nu, alpha, beta = self.pred_head(feat)
        return (g, nu, alpha, beta), latent

    def adapt_forward(self, src_x: torch.Tensor, tgt_x: torch.Tensor,
                      src_domain: str = "muscatine",
                      tgt_domain: str = None,
                      conditioning: bool = True):
        """
        Domain adaptation forward pass.
        Alignment is conditional: only within the same process-state class.
        Returns: pred_src, dom_src, dom_tgt, f_sa, f_ta, state_src, state_tgt, latent
        """
        feat_src  = self._encode(src_x, src_domain)
        feat_tgt  = self._encode(tgt_x, tgt_domain)
        latent    = self.latent_decoder(feat_src)
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
