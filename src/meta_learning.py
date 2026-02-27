"""
meta_learning.py
Model-Agnostic Meta-Learning (MAML) initialization.

Trains a meta-learner across Muscatine, DataONE, and EDI datasets
so the model finds an initialization that adapts quickly to any new
anaerobic digestion plant with minimal labelled data.

Algorithm: MAML (Finn et al., 2017)
  For each episode:
    1. Sample a task (dataset / operating condition)
    2. Compute inner-loop gradient update on support set
    3. Evaluate adapted model on query set
    4. Meta-update outer parameters via second-order gradients (or first-order approx)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import os
import config

_DEFAULT_DEVICE = "cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"


# ─── Task Sampler ─────────────────────────────────────────────────────────────

class TaskSampler:
    """
    Creates episodic tasks from multiple datasets.
    Each task = (support_set, query_set) from one dataset/operating_condition.
    """

    def __init__(self,
                 task_datasets: list,        # list of (X_seq, y_seq) np arrays
                 task_names:    list = None,
                 n_support:     int  = 16,
                 n_query:       int  = 32,
                 seed:          int  = 42):
        self.tasks      = task_datasets
        self.names      = task_names or [f"task_{i}" for i in range(len(task_datasets))]
        self.n_support  = n_support
        self.n_query    = n_query
        self.rng        = np.random.RandomState(seed)

    def sample_task(self):
        """
        Returns:
          (sup_X, sup_y, qry_X, qry_y, task_name)
          All as torch.FloatTensor.
        """
        task_idx = self.rng.randint(len(self.tasks))
        X, y     = self.tasks[task_idx]
        n        = len(X)

        needed = self.n_support + self.n_query
        if n < needed:
            idx = self.rng.choice(n, needed, replace=True)
        else:
            idx = self.rng.choice(n, needed, replace=False)

        sup_idx = idx[:self.n_support]
        qry_idx = idx[self.n_support:]

        def to_t(arr, idxs):
            return torch.tensor(arr[idxs], dtype=torch.float32)

        return (to_t(X, sup_idx), to_t(y, sup_idx),
                to_t(X, qry_idx), to_t(y, qry_idx),
                self.names[task_idx])

    def sample_batch(self, n_tasks: int = 4):
        return [self.sample_task() for _ in range(n_tasks)]


# ─── MAML Trainer ─────────────────────────────────────────────────────────────

class MAMLTrainer:
    """
    First-Order MAML (FOMAML) trainer — avoids expensive second-order gradients
    while retaining most of the meta-learning benefit.

    Args:
      model        : BiogasTransferModel
      task_sampler : TaskSampler
      inner_lr     : step size for inner loop (task-specific adaptation)
      outer_lr     : step size for meta-update
      inner_steps  : gradient steps during inner loop
      n_tasks      : tasks per meta-update
      device       : torch device
    """

    def __init__(self,
                 model:        nn.Module,
                 task_sampler: TaskSampler,
                 inner_lr:     float = 0.01,
                 outer_lr:     float = 1e-3,
                 inner_steps:  int   = 5,
                 n_tasks:      int   = 4,
                 device:       str   = _DEFAULT_DEVICE):
        self.model        = model
        self.sampler      = task_sampler
        self.inner_lr     = inner_lr
        self.outer_lr     = outer_lr
        self.inner_steps  = inner_steps
        self.n_tasks      = n_tasks
        self.device       = torch.device(device)
        self.meta_optim   = torch.optim.Adam(model.parameters(), lr=outer_lr)
        self.model.to(self.device)

    def _inner_update(self, model_copy: nn.Module,
                      sup_X: torch.Tensor,
                      sup_y: torch.Tensor) -> nn.Module:
        """
        Perform inner-loop gradient steps on support set.
        Returns adapted model (does NOT modify original).
        """
        inner_optim = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr)
        criterion   = nn.HuberLoss(delta=0.5)

        for _ in range(self.inner_steps):
            inner_optim.zero_grad()
            gamma, nu, alpha, beta = model_copy.source_forward(sup_X)
            loss = criterion(gamma, sup_y)
            loss.backward()
            inner_optim.step()

        return model_copy

    def _task_loss(self, adapted: nn.Module,
                   qry_X: torch.Tensor,
                   qry_y: torch.Tensor) -> torch.Tensor:
        gamma, nu, alpha, beta = adapted.source_forward(qry_X)
        return F.huber_loss(gamma, qry_y, delta=0.5)

    def train_epoch(self, n_episodes: int = 50) -> float:
        """One meta-training epoch. Returns mean meta-loss."""
        self.model.train()
        epoch_losses = []

        for ep in range(n_episodes):
            tasks         = self.sampler.sample_batch(self.n_tasks)
            meta_loss     = torch.zeros(1, device=self.device).squeeze()

            for sup_X, sup_y, qry_X, qry_y, name in tasks:
                sup_X = sup_X.to(self.device)
                sup_y = sup_y.unsqueeze(-1).to(self.device)
                qry_X = qry_X.to(self.device)
                qry_y = qry_y.unsqueeze(-1).to(self.device)

                # Clone model for inner loop (FOMAML: no second-order)
                model_copy = copy.deepcopy(self.model)
                adapted    = self._inner_update(model_copy, sup_X, sup_y)

                # Compute query loss with adapted params
                task_loss  = self._task_loss(adapted, qry_X, qry_y)
                meta_loss  = meta_loss + task_loss / self.n_tasks

            self.meta_optim.zero_grad()
            meta_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.meta_optim.step()

            epoch_losses.append(meta_loss.item())

        return float(np.mean(epoch_losses))

    def train(self, n_epochs: int = 30, n_episodes: int = 50,
              save_path: str = None) -> list:
        """Full meta-training loop."""
        history = []
        print(f"[MAML] Meta-training for {n_epochs} epochs × {n_episodes} episodes …")

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(n_episodes)
            history.append(loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Meta-Epoch {epoch:3d}/{n_epochs}  Meta-Loss={loss:.4f}")

        if save_path:
            torch.save({
                "model_state": self.model.state_dict(),
                "meta_loss":   history,
            }, save_path)
            print(f"  [MAML] Saved meta-initialized model → {save_path}")

        return history

    def adapt_to_task(self, X_support: np.ndarray,
                       y_support: np.ndarray,
                       n_steps:   int = 10) -> nn.Module:
        """
        Fast adaptation to a new task given a small support set.
        Returns an adapted copy of the model (original unchanged).
        """
        sup_X = torch.tensor(X_support, dtype=torch.float32).to(self.device)
        sup_y = torch.tensor(y_support, dtype=torch.float32).unsqueeze(-1).to(self.device)

        adapted = copy.deepcopy(self.model)
        adapted.train()
        optim = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        criterion = nn.HuberLoss(delta=0.5)

        for step in range(n_steps):
            optim.zero_grad()
            gamma, nu, alpha, beta = adapted.source_forward(sup_X)
            loss = criterion(gamma, sup_y)
            loss.backward()
            optim.step()

        print(f"[MAML] Adapted to new task in {n_steps} gradient steps.")
        return adapted


# ─── Dataset Builder for Meta-Learning ───────────────────────────────────────

def build_meta_tasks(dataset_loaders: dict) -> tuple:
    """
    Build task list from {name: (X_seq, y_seq)} dict.
    Pads/trims all tasks to same feature dimension.

    Returns (task_list, task_names) suitable for TaskSampler.
    """
    task_list  = []
    task_names = []

    # Find minimum feature dimension across all tasks
    min_feat = min(X.shape[-1] for X, _ in dataset_loaders.values())

    for name, (X, y) in dataset_loaders.items():
        if X.shape[-1] > min_feat:
            X = X[:, :, :min_feat]   # truncate extra features
        task_list.append((X.astype(np.float32), y.astype(np.float32)))
        task_names.append(name)
        print(f"  Task '{name}': {len(X)} sequences, {X.shape[-1]} features")

    return task_list, task_names
