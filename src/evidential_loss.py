"""
evidential_loss.py
Evidential Deep Learning for uncertainty quantification.

Replaces MC-Dropout with a single forward pass that predicts:
  gamma (μ)  – predicted mean
  nu         – pseudo-count (evidence for mean)
  alpha      – shape parameter of Gamma (evidence for variance)
  beta       – scale parameter of Gamma

The model learns a Normal-Inverse-Gamma (NIG) prior over (μ, σ²),
giving both aleatoric (data noise) and epistemic (model) uncertainty.

References:
  Amini et al. 2020 "Deep Evidential Regression" (NeurIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── NIG Loss ─────────────────────────────────────────────────────────────────

def nig_nll(y: torch.Tensor,
            gamma: torch.Tensor,
            nu:    torch.Tensor,
            alpha: torch.Tensor,
            beta:  torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood under the NIG distribution.
    All inputs: shape (B, 1) or (B,).
    """
    # ensure alpha > 1 for finite NLL
    alpha = alpha.clamp(min=1.0 + 1e-6)
    nu    = nu.clamp(min=1e-6)
    beta  = beta.clamp(min=1e-6)

    two_b_lam = 2.0 * beta * (1.0 + nu)

    nll = (0.5 * torch.log(torch.tensor(torch.pi, device=y.device) / nu)
           - alpha * torch.log(two_b_lam)
           + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b_lam)
           + torch.lgamma(alpha)
           - torch.lgamma(alpha + 0.5))

    return nll.mean()


def nig_regulariser(y:     torch.Tensor,
                    gamma: torch.Tensor,
                    nu:    torch.Tensor,
                    alpha: torch.Tensor,
                    coeff: float = 1e-2) -> torch.Tensor:
    """
    Evidence regularisation: penalises high evidence (nu + alpha) on
    samples where the model is wrong, preventing over-confident wrong answers.
    """
    error     = torch.abs(y - gamma)
    evidence  = nu + alpha          # total evidence
    reg       = error * evidence
    return coeff * reg.mean()


class EvidentialLoss(nn.Module):
    """
    Combined NIG NLL + regularisation.

    Args:
      coeff_reg: weight on the regularisation term (default 1e-2)
      reject_thresh: evidence threshold below which predictions are flagged
                     as uncertain (used at inference, not in loss)
    """

    def __init__(self, coeff_reg: float = 1e-2, reject_thresh: float = 2.0):
        super().__init__()
        self.coeff_reg      = coeff_reg
        self.reject_thresh  = reject_thresh

    def forward(self,
                y:     torch.Tensor,
                gamma: torch.Tensor,
                nu:    torch.Tensor,
                alpha: torch.Tensor,
                beta:  torch.Tensor) -> torch.Tensor:

        nll = nig_nll(y, gamma, nu, alpha, beta)
        reg = nig_regulariser(y, gamma, nu, alpha, self.coeff_reg)
        return nll + reg

    def uncertainty(self,
                    nu:    torch.Tensor,
                    alpha: torch.Tensor,
                    beta:  torch.Tensor):
        """
        Decompose uncertainty into aleatoric and epistemic components.

        Returns:
          aleatoric   – irreducible data noise  (beta / (alpha - 1))
          epistemic   – model uncertainty       (beta / (nu * (alpha - 1)))
          total       – sum
          reject_mask – True where epistemic uncertainty > threshold
        """
        alpha_safe = alpha.clamp(min=1.0 + 1e-6)
        nu_safe    = nu.clamp(min=1e-6)
        beta_safe  = beta.clamp(min=1e-6)

        aleatoric  = beta_safe / (alpha_safe - 1.0)
        epistemic  = beta_safe / (nu_safe * (alpha_safe - 1.0))
        total      = aleatoric + epistemic

        # evidence = 2*alpha + nu  (total pseudo-counts)
        evidence    = 2.0 * alpha_safe + nu_safe
        reject_mask = evidence < self.reject_thresh

        return aleatoric, epistemic, total, reject_mask


# ─── Evidential Prediction Head ───────────────────────────────────────────────

class EvidentialHead(nn.Module):
    """
    Drop-in replacement for PredictionHead.
    Outputs 4 parameters (gamma, nu, alpha, beta) instead of 1.

    Use forward() for training (returns all 4).
    Use predict() for inference (returns mean + uncertainty breakdown).
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.SiLU(),
        )
        self.gamma_head = nn.Linear(64, 1)   # mean prediction
        self.nu_head    = nn.Linear(64, 1)   # evidence for mean
        self.alpha_head = nn.Linear(64, 1)   # shape
        self.beta_head  = nn.Linear(64, 1)   # scale

    def forward(self, feat: torch.Tensor):
        h = self.shared(feat)

        gamma = self.gamma_head(h)
        # nu, alpha, beta must be positive; alpha > 1 for finite variance
        nu    = F.softplus(self.nu_head(h))    + 1e-6
        alpha = F.softplus(self.alpha_head(h)) + 1.0 + 1e-6
        beta  = F.softplus(self.beta_head(h))  + 1e-6

        return gamma, nu, alpha, beta

    @torch.no_grad()
    def predict(self, feat: torch.Tensor, loss_fn: EvidentialLoss = None):
        """
        Returns:
          mean       – point prediction (gamma)
          aleatoric  – data noise
          epistemic  – model uncertainty
          reject     – bool mask: True = prediction should be flagged
        """
        gamma, nu, alpha, beta = self.forward(feat)

        if loss_fn is None:
            loss_fn = EvidentialLoss()

        aleatoric, epistemic, total, reject = loss_fn.uncertainty(nu, alpha, beta)

        return gamma, aleatoric, epistemic, total, reject
