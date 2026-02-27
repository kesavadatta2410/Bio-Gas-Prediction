"""
active_learning.py
Uncertainty-based active learning for few-shot label selection.

Selects the most informative unlabelled samples from the target domain
for human annotation, maximising information gain per labelling budget.

Two strategies:
  1. EpistemicSampler    – picks samples with highest epistemic uncertainty
  2. GradientSampler     – picks samples where gradient magnitude is largest
                           (most sensitive to model parameters)
  3. CoreSetSampler      – greedy coreset: maximises coverage in feature space
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config

_DEFAULT_DEVICE = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"


# ─── 1. Epistemic Uncertainty Sampler ─────────────────────────────────────────

class EpistemicSampler:
    """
    Select unlabelled samples where the model is most epistemically uncertain.
    Uses the evidential uncertainty output directly.

    Args:
      model  : BiogasTransferModel (must have .predict())
      budget : number of samples to select per round
    """

    def __init__(self, model: nn.Module, budget: int = 20, device: str = _DEFAULT_DEVICE):
        self.model  = model
        self.budget = budget
        self.device = device

    def select(self, X_pool: torch.Tensor) -> np.ndarray:
        """
        Args:
          X_pool: (N, seq_len, feat_dim) – unlabelled pool
        Returns:
          indices: top-K most uncertain sample indices (sorted descending)
        """
        self.model.eval()
        all_epistemic = []

        ds     = TensorDataset(X_pool)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                _, _, epistemic, _ = self.model.predict(x_batch)
                all_epistemic.append(epistemic.cpu().numpy().ravel())

        epist_arr = np.concatenate(all_epistemic)
        # Top-K highest epistemic uncertainty
        top_k = np.argsort(epist_arr)[-self.budget:][::-1]
        return top_k

    def active_learning_round(self, X_pool: np.ndarray,
                               y_pool_oracle,   # callable(indices) → labels
                               X_labeled: np.ndarray,
                               y_labeled: np.ndarray):
        """
        One full active learning round.
        Returns updated X_labeled, y_labeled with newly annotated samples.
        """
        X_t  = torch.tensor(X_pool, dtype=torch.float32)
        idx  = self.select(X_t)

        new_X = X_pool[idx]
        new_y = y_pool_oracle(idx)   # request labels from oracle/annotator

        X_labeled = np.concatenate([X_labeled, new_X], axis=0)
        y_labeled = np.concatenate([y_labeled, new_y], axis=0)

        print(f"[ActiveLearning] Added {len(idx)} samples. "
              f"Total labelled: {len(y_labeled)}")
        return X_labeled, y_labeled, idx


# ─── 2. Gradient-Based Importance Sampler ────────────────────────────────────

class GradientSampler:
    """
    Selects samples that cause the largest gradient magnitude
    w.r.t. model parameters (highest learning signal).
    """

    def __init__(self, model: nn.Module, budget: int = 20, device: str = _DEFAULT_DEVICE):
        self.model  = model
        self.budget = budget
        self.device = device

    def select(self, X_pool: torch.Tensor) -> np.ndarray:
        self.model.train()   # need gradients
        grad_norms = []

        for i in range(len(X_pool)):
            x = X_pool[i:i+1].to(self.device)
            self.model.zero_grad()

            gamma, nu, alpha, beta = self.model.source_forward(x)
            # Use the mean prediction as a proxy loss
            loss = gamma.abs().mean()
            loss.backward()

            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

            if i > 0 and i % 50 == 0:
                print(f"  [GradSampler] Scored {i}/{len(X_pool)}", end="\r")

        self.model.zero_grad()
        grad_arr = np.array(grad_norms)
        return np.argsort(grad_arr)[-self.budget:][::-1]


# ─── 3. CoreSet Sampler ───────────────────────────────────────────────────────

class CoreSetSampler:
    """
    Greedy k-Center coreset selection in feature space.
    Maximises coverage: minimises the maximum distance from any
    unlabelled point to its nearest labelled centre.

    More diverse than pure uncertainty sampling — avoids clustered queries.
    """

    def __init__(self, model: nn.Module, budget: int = 20, device: str = _DEFAULT_DEVICE):
        self.model  = model
        self.budget = budget
        self.device = device

    @torch.no_grad()
    def _extract_features(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        feats  = []
        loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        for (x_b,) in loader:
            feat = self.model._encode(x_b.to(self.device))
            feats.append(feat.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def select(self,
               X_pool:    torch.Tensor,
               X_labeled: torch.Tensor = None) -> np.ndarray:
        """
        Args:
          X_pool:    (N_pool,  seq, feat) – unlabelled pool
          X_labeled: (N_label, seq, feat) – already labelled (optional)
        Returns:
          indices into X_pool of selected samples
        """
        pool_feats = self._extract_features(X_pool)

        if X_labeled is not None and len(X_labeled) > 0:
            labeled_feats = self._extract_features(X_labeled)
        else:
            # Start from a random point
            seed_idx      = np.random.randint(len(pool_feats))
            labeled_feats = pool_feats[seed_idx:seed_idx+1]

        selected = []
        remaining = np.arange(len(pool_feats))

        for _ in range(self.budget):
            if len(remaining) == 0:
                break
            # Distance from each remaining point to nearest labelled centre
            dists = np.full(len(remaining), np.inf)
            for lf in labeled_feats:
                d = np.linalg.norm(pool_feats[remaining] - lf, axis=1)
                dists = np.minimum(dists, d)

            # Pick the most distant (least covered) point
            pick_rel = np.argmax(dists)
            pick_abs = remaining[pick_rel]
            selected.append(pick_abs)

            # Update labelled set
            labeled_feats = np.vstack([labeled_feats, pool_feats[pick_abs]])
            remaining     = np.delete(remaining, pick_rel)

        return np.array(selected)


# ─── Round Manager ────────────────────────────────────────────────────────────

class ActiveLearningManager:
    """
    Orchestrates multiple rounds of active learning.

    Usage:
        mgr = ActiveLearningManager(model, strategy="epistemic", budget=20)
        idx = mgr.query(X_pool)
        # Get labels for idx from your annotation process
        mgr.update(X_pool[idx], y_new)
        # Retrain model on mgr.X_labeled, mgr.y_labeled
    """

    STRATEGIES = {
        "epistemic": EpistemicSampler,
        "gradient":  GradientSampler,
        "coreset":   CoreSetSampler,
    }

    def __init__(self,
                 model:    nn.Module,
                 strategy: str  = "epistemic",
                 budget:   int  = 20,
                 device:   str  = _DEFAULT_DEVICE):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {list(self.STRATEGIES)}")
        self.sampler   = self.STRATEGIES[strategy](model, budget, device)
        self.X_labeled = None
        self.y_labeled = None
        self.round_num = 0

    def query(self, X_pool: np.ndarray,
               X_labeled_tensor: torch.Tensor = None) -> np.ndarray:
        """Return indices of samples to annotate next."""
        X_t = torch.tensor(X_pool, dtype=torch.float32)
        if isinstance(self.sampler, CoreSetSampler):
            return self.sampler.select(X_t, X_labeled_tensor)
        return self.sampler.select(X_t)

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add newly labelled samples to the pool."""
        if self.X_labeled is None:
            self.X_labeled = X_new
            self.y_labeled = y_new
        else:
            self.X_labeled = np.concatenate([self.X_labeled, X_new])
            self.y_labeled = np.concatenate([self.y_labeled, y_new])
        self.round_num += 1
        print(f"[AL Round {self.round_num}] Labelled set size: {len(self.y_labeled)}")
