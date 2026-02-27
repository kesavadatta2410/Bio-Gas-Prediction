"""
continual_learning.py
Continual adaptation without catastrophic forgetting.

Implements:
  1. Experience Replay Buffer  – stores source domain samples
  2. EWC (Elastic Weight Consolidation) – penalises changes to important weights
  3. OnlineAdapter – streams new target data and updates model incrementally

References:
  Kirkpatrick et al. 2017 "Overcoming catastrophic forgetting in NNs" (EWC)
  Rolnick et al. 2019 "Experience Replay for Continual Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy
import os
import config

_DEFAULT_DEVICE = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"


# ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer that stores (X, y) pairs from the source domain.
    Randomly sampled during target-domain updates to prevent forgetting.

    Args:
      capacity:   max number of samples to store
      strategy:   'random' | 'reservoir' (reservoir sampling = unbiased)
    """

    def __init__(self, capacity: int = 2000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy
        self.buffer_X = []
        self.buffer_y = []
        self._n_seen  = 0   # total samples seen (for reservoir sampling)

    def add_batch(self, X: np.ndarray, y: np.ndarray):
        """Add a batch of source-domain samples."""
        for xi, yi in zip(X, y):
            self._n_seen += 1
            if len(self.buffer_X) < self.capacity:
                self.buffer_X.append(xi)
                self.buffer_y.append(yi)
            elif self.strategy == "reservoir":
                # Reservoir sampling: replace with decreasing probability
                j = np.random.randint(0, self._n_seen)
                if j < self.capacity:
                    self.buffer_X[j] = xi
                    self.buffer_y[j] = yi
            else:
                # Random replacement
                j = np.random.randint(0, self.capacity)
                self.buffer_X[j] = xi
                self.buffer_y[j] = yi

    def sample(self, n: int) -> tuple:
        """Sample n items from the buffer. Returns (X, y) np arrays."""
        n   = min(n, len(self.buffer_X))
        idx = np.random.choice(len(self.buffer_X), n, replace=False)
        X   = np.stack([self.buffer_X[i] for i in idx])
        y   = np.array([self.buffer_y[i] for i in idx])
        return X, y

    def __len__(self):
        return len(self.buffer_X)

    def is_ready(self, min_size: int = 64) -> bool:
        return len(self) >= min_size


# ─── Elastic Weight Consolidation (EWC) ───────────────────────────────────────

class EWC:
    """
    Computes Fisher Information Matrix diagonal over the source domain
    and penalises parameter changes during target adaptation.

    Usage:
      ewc = EWC(model, src_dataloader)
      ewc.compute_fisher()
      # ... during target training:
      loss = task_loss + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, dataloader, device: str = _DEFAULT_DEVICE,
                 n_batches: int = 50):
        self.model       = model
        self.dataloader  = dataloader
        self.device      = torch.device(device)
        self.n_batches   = n_batches
        self.fisher_     = {}   # param_name → diagonal Fisher
        self.theta_star_ = {}   # param_name → optimal source-trained weights

    def compute_fisher(self):
        """
        Estimate Fisher diagonal via squared gradients on source data.
        Call once after source pre-training.
        """
        # cuDNN RNNs require train() mode for backward() — switch temporarily
        self.model.train()

        # Store optimal parameters
        for name, param in self.model.named_parameters():
            self.theta_star_[name] = param.data.clone()
            self.fisher_[name]     = torch.zeros_like(param.data)

        criterion = nn.HuberLoss(delta=0.5)
        n_batches = 0

        for X_batch, y_batch in self.dataloader:
            if n_batches >= self.n_batches:
                break

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.model.zero_grad()
            gamma, nu, alpha, beta = self.model.source_forward(X_batch)
            loss = criterion(gamma, y_batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_[name] += param.grad.data.clone() ** 2

            n_batches += 1

        # Normalise
        for name in self.fisher_:
            self.fisher_[name] /= n_batches

        self.model.eval()  # restore eval mode after Fisher computation
        print(f"[EWC] Fisher computed over {n_batches} source batches.")

    def penalty(self, model: nn.Module, lambda_ewc: float = 1.0) -> torch.Tensor:
        """
        EWC penalty term:  λ/2 * Σ F_i * (θ_i - θ*_i)²
        Add to task loss during target domain adaptation.
        """
        if not self.fisher_:
            return torch.zeros(1, device=next(model.parameters()).device).squeeze()

        penalty = torch.zeros(1, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher_ and name in self.theta_star_:
                diff    = param - self.theta_star_[name].to(param.device)
                fisher  = self.fisher_[name].to(param.device)
                penalty += (fisher * diff ** 2).sum()

        return (lambda_ewc / 2) * penalty.squeeze()


# ─── Online Adapter ───────────────────────────────────────────────────────────

class OnlineAdapter:
    """
    Streaming adaptation: processes incoming target-domain batches
    and updates the model incrementally.

    Combines:
      - Experience replay (replay_ratio of each batch from source buffer)
      - EWC regularisation (optional, set lambda_ewc=0 to disable)
      - Anomaly detection (suspends update during sensor anomalies)
    """

    def __init__(self,
                 model:         nn.Module,
                 replay_buffer: ReplayBuffer,
                 ewc:           EWC = None,
                 lr:            float = 5e-5,
                 replay_ratio:  float = 0.5,
                 lambda_ewc:    float = 0.4,
                 device:        str   = _DEFAULT_DEVICE,
                 anomaly_thresh: float = 4.0):
        self.model          = model
        self.buffer         = replay_buffer
        self.ewc            = ewc
        self.lr             = lr
        self.replay_ratio   = replay_ratio
        self.lambda_ewc     = lambda_ewc
        self.device         = torch.device(device)
        self.anomaly_thresh = anomaly_thresh
        self.criterion      = nn.HuberLoss(delta=0.5)
        self.optim          = torch.optim.Adam(model.parameters(), lr=lr)
        self.step_count     = 0
        self.loss_history   = []

    def _detect_anomaly(self, X_batch: np.ndarray) -> bool:
        """
        Simple z-score anomaly check: flag batch if any feature has
        |z-score| > threshold (sensor spike / sensor failure).
        Returns True if anomaly detected (update should be skipped).
        """
        z = np.abs((X_batch - X_batch.mean(axis=(0, 1), keepdims=True))
                   / (X_batch.std(axis=(0, 1), keepdims=True) + 1e-8))
        return bool(z.max() > self.anomaly_thresh)

    def update(self, X_new: np.ndarray, y_new: np.ndarray = None,
               verbose: bool = False) -> float:
        """
        One online adaptation step on a new batch.

        Args:
          X_new: (B, seq_len, feat_dim)
          y_new: (B,) optional labels. If None, uses pseudo-labels.
        Returns:
          loss value
        """
        # Anomaly gate
        if self._detect_anomaly(X_new):
            if verbose:
                print(f"  [OnlineAdapter] Anomaly detected at step {self.step_count}. Skipping.")
            return 0.0

        self.model.train()
        self.optim.zero_grad()

        X_t = torch.tensor(X_new, dtype=torch.float32).to(self.device)

        # Generate pseudo-labels if none provided
        if y_new is None:
            with torch.no_grad():
                gamma, _, _, _ = self.model.source_forward(X_t)
            y_t = gamma.detach()
        else:
            y_t = torch.tensor(y_new, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # Forward on new batch
        gamma, nu, alpha, beta = self.model.source_forward(X_t)
        loss_new = self.criterion(gamma, y_t)

        # Replay: mix in source samples
        total_loss = loss_new
        if self.buffer.is_ready():
            n_replay = max(1, int(len(X_new) * self.replay_ratio))
            X_rep, y_rep = self.buffer.sample(n_replay)
            X_rep_t = torch.tensor(X_rep, dtype=torch.float32).to(self.device)
            y_rep_t = torch.tensor(y_rep, dtype=torch.float32).unsqueeze(-1).to(self.device)
            gamma_r, *_ = self.model.source_forward(X_rep_t)
            loss_rep    = self.criterion(gamma_r, y_rep_t)
            total_loss  = total_loss + loss_rep

        # EWC penalty
        if self.ewc is not None and self.lambda_ewc > 0:
            total_loss = total_loss + self.ewc.penalty(self.model, self.lambda_ewc)

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        self.step_count += 1
        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        if verbose and self.step_count % 10 == 0:
            recent_loss = np.mean(self.loss_history[-10:])
            print(f"  [OnlineAdapter] Step {self.step_count:4d}  "
                  f"Loss={recent_loss:.4f}")

        return loss_val

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state":  self.model.state_dict(),
            "step_count":   self.step_count,
            "loss_history": self.loss_history,
        }, path)
        print(f"[OnlineAdapter] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # our own checkpoint
        self.model.load_state_dict(ckpt["model_state"])
        self.step_count  = ckpt["step_count"]
        self.loss_history = ckpt["loss_history"]
        print(f"[OnlineAdapter] Loaded checkpoint: {path} (step {self.step_count})")
