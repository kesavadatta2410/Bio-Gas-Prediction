"""
dataset.py  (v3 — Per-Dataset Scalers + Temporal Splits + Masking)

Changes from v2:
  - Per-dataset StandardScaler (separate for Muscatine / DataONE / EDI)
    instead of one global scaler that distorted cross-domain distributions.
  - MaskedSequenceDataset: carries (X, mask, y) for DataONE sparse handling.
  - Temporal-block train/val/test split (no random shuffle — preserves
    time structure and prevents data leakage between contiguous blocks).
  - SourceDataLoader.load() returns scalers dict + feature_col dict per domain.
  - TargetDataLoader.load(target_type) does asymmetric source selection:
    uses the closest domain's scaler rather than the global one.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import config


# ─── Utilities ────────────────────────────────────────────────────────────────

def smart_load_csv(path, **kwargs):
    """Load CSV, auto-detect datetime index, drop all-NaN columns."""
    df = pd.read_csv(path, **kwargs)
    df.dropna(axis=1, how="all", inplace=True)
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.sort_values(by=df.columns[0], inplace=True)
        df.set_index(df.columns[0], inplace=True)
    except Exception:
        pass
    return df


def fill_and_clip(df, feature_cols, target_col, physics):
    """Fill missing values and clip targets to physical bounds."""
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    if target_col in df.columns:
        df[target_col] = df[target_col].clip(
            lower=physics["biogas_min"],
            upper=physics["biogas_max"]
        ).ffill().fillna(0)
    return df


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding-window sequences. Returns (X_seq, y_seq)."""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_sequences_with_mask(X: np.ndarray, mask: np.ndarray,
                              y: np.ndarray, seq_len: int):
    """Like make_sequences but also windows the observation mask."""
    xs, ms, ys = [], [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ms.append(mask[i:i + seq_len])
        ys.append(y[i + seq_len])
    return (np.array(xs, dtype=np.float32),
            np.array(ms, dtype=np.float32),
            np.array(ys, dtype=np.float32))


def temporal_block_split(n: int, val_frac: float = 0.15,
                          test_frac: float = 0.15):
    """
    Returns (train_idx, val_idx, test_idx) as contiguous blocks.
    No shuffling — preserves temporal structure.
    """
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx


# ─── Dataset Classes ──────────────────────────────────────────────────────────

class BiogasSequenceDataset(Dataset):
    """Sliding-window sequence dataset for time-series modelling."""

    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MaskedSequenceDataset(Dataset):
    """
    Like BiogasSequenceDataset but carries an observation mask.
    Used for DataONE sparse data passed to DataONEEncoder.
    Returns (X, mask, y).
    """

    def __init__(self, X_seq: np.ndarray,
                 mask_seq: np.ndarray,
                 y: np.ndarray):
        self.X    = torch.tensor(X_seq,    dtype=torch.float32)
        self.mask = torch.tensor(mask_seq, dtype=torch.float32)
        self.y    = torch.tensor(y,        dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


class DomainAdaptDataset(Dataset):
    """
    Wraps source + target features for domain adaptation training.
    Returns (src_x, src_y, tgt_x) tuples; tgt_y optional.
    """

    def __init__(self, src_X, src_y, tgt_X, tgt_y=None):
        self.src_X = torch.tensor(src_X, dtype=torch.float32)
        self.src_y = torch.tensor(src_y, dtype=torch.float32).unsqueeze(-1)
        self.tgt_X = torch.tensor(tgt_X, dtype=torch.float32)
        self.tgt_y = (torch.tensor(tgt_y, dtype=torch.float32).unsqueeze(-1)
                      if tgt_y is not None else None)
        self._len  = max(len(self.src_X), len(self.tgt_X))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        s = idx % len(self.src_X)
        t = idx % len(self.tgt_X)
        if self.tgt_y is not None:
            return self.src_X[s], self.src_y[s], self.tgt_X[t], self.tgt_y[t]
        return self.src_X[s], self.src_y[s], self.tgt_X[t]


# ─── Source Data Loader ───────────────────────────────────────────────────────

class SourceDataLoader:
    """
    Loads Muscatine SCADA + LAB data and prepares train/val/test loaders.

    v3 changes:
      - Uses per-dataset StandardScaler fitted only on training split
        (prevents val/test leakage and domain distribution distortion).
      - Temporal-block splits (no random shuffle).
      - Exposes self.scalers dict: {'muscatine': (scaler_X, scaler_y), ...}
    """

    def __init__(self):
        self.scaler_X     = StandardScaler()   # kept for backward compatibility
        self.scaler_y     = StandardScaler()
        self.scalers      = {}                 # {'muscatine': (sx, sy)}
        self.feature_cols = None
        self.input_dim    = None

    def _auto_select_features(self, df, candidates, target):
        available = [c for c in candidates if c in df.columns]
        if not available:
            available = [c for c in df.select_dtypes(include=np.number).columns
                         if c != target]
        return available

    def load(self):
        print("[SourceDataLoader v3] Loading SCADA data …")
        scada = smart_load_csv(config.DATA["scada_raw"])
        lab   = smart_load_csv(config.DATA["lab_raw"])

        # Resample SCADA to daily to match LAB frequency
        if isinstance(scada.index, pd.DatetimeIndex):
            scada = scada.resample("D").mean()
        if isinstance(lab.index, pd.DatetimeIndex):
            lab = lab.resample("D").mean()
            df  = scada.join(lab, how="outer", rsuffix="_lab")
        else:
            df = scada.copy()

        candidates        = config.SCADA_FEATURES + config.LAB_FEATURES
        target            = config.TARGET_COL

        self.feature_cols = self._auto_select_features(df, candidates, target)

        if target not in df.columns:
            gas_cols = [c for c in df.columns
                        if "gas" in c.lower() or "methane" in c.lower()]
            if gas_cols:
                target = gas_cols[0]
                print(f"  [INFO] Using '{target}' as target column")
            else:
                raise ValueError(
                    f"Target column '{target}' not found. "
                    "Set TARGET_COL in config.py to an existing column."
                )

        df = fill_and_clip(df, self.feature_cols, target, config.PHYSICS)
        df.dropna(subset=[target], inplace=True)

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[target].values.astype(np.float32)

        # ── Temporal-block split (before fitting scaler!) ──────────────────
        X_seq_raw, y_seq_raw = make_sequences(X, y, config.TRAIN["seq_len"])
        tr_idx, val_idx, te_idx = temporal_block_split(
            len(X_seq_raw),
            val_frac=config.TRAIN["val_split"],
            test_frac=config.TRAIN["test_split"]
        )

        X_tr_raw, y_tr_raw   = X_seq_raw[tr_idx], y_seq_raw[tr_idx]
        X_val_raw, y_val_raw = X_seq_raw[val_idx], y_seq_raw[val_idx]
        X_te_raw,  y_te_raw  = X_seq_raw[te_idx],  y_seq_raw[te_idx]

        # ── Per-dataset scaler fitted ONLY on training split ───────────────
        sx = StandardScaler()
        sy = StandardScaler()

        n_tr, T, F = X_tr_raw.shape
        sx.fit(X_tr_raw.reshape(-1, F))
        sy.fit(y_tr_raw.reshape(-1, 1))

        def transform(X_seq, y_seq):
            n, t, f = X_seq.shape
            Xs = sx.transform(X_seq.reshape(-1, f)).reshape(n, t, f)
            ys = sy.transform(y_seq.reshape(-1, 1)).ravel()
            return Xs, ys

        X_tr,  y_tr  = transform(X_tr_raw,  y_tr_raw)
        X_val, y_val = transform(X_val_raw, y_val_raw)
        X_te,  y_te  = transform(X_te_raw,  y_te_raw)

        # Store scalers per domain
        self.scaler_X = sx
        self.scaler_y = sy
        self.scalers["muscatine"] = (sx, sy)

        self.input_dim = F
        config.MODEL["input_dim"] = F

        bs = config.TRAIN["batch_size"]
        train_loader = DataLoader(BiogasSequenceDataset(X_tr, y_tr),
                                  batch_size=bs, shuffle=True, drop_last=True)
        val_loader   = DataLoader(BiogasSequenceDataset(X_val, y_val),
                                  batch_size=bs, shuffle=False)
        test_loader  = DataLoader(BiogasSequenceDataset(X_te, y_te),
                                  batch_size=bs, shuffle=False)

        print(f"  Features: {self.feature_cols}")
        print(f"  [TemporalSplit] Train={len(X_tr)} | Val={len(X_val)} | "
              f"Test={len(X_te)} sequences (no shuffle)")
        print(f"  [PerDomainScaler] Muscatine scaler fitted on {len(X_tr)} "
              f"training samples only")
        return train_loader, val_loader, test_loader

    def get_domain_scaler(self, domain: str = "muscatine"):
        return self.scalers.get(domain, (self.scaler_X, self.scaler_y))


# ─── Target Data Loader ───────────────────────────────────────────────────────

class TargetDataLoader:
    """
    Loads target plant data.
    Asymmetric few-shot: uses the closest source domain's scaler.

    target_type: 'industrial' | 'pilot' | 'batch'
      - 'industrial' → use Muscatine (Iowa WWTP) as closest source
      - 'pilot'      → use DataONE (research scale)
      - 'batch'      → use EDI (batch reactor data)
    """

    DOMAIN_MAP = {
        "industrial": "muscatine",
        "pilot":      "dataone",
        "batch":      "edi",
    }

    def __init__(self, scalers: dict, feature_cols: list):
        self.scalers      = scalers       # {'muscatine': (sx, sy), ...}
        self.feature_cols = feature_cols

    def load(self, few_shot_fraction: float = 0.1,
             target_type: str = "industrial"):
        """
        few_shot_fraction : portion of target data used as labelled fine-tune set.
        target_type       : selects which domain scaler to apply.
        """
        path = config.DATA["target_plant"]

        # Select appropriate source scaler
        source_domain = self.DOMAIN_MAP.get(target_type, "muscatine")
        if source_domain in self.scalers:
            scaler_X, scaler_y = self.scalers[source_domain]
        elif "muscatine" in self.scalers:
            scaler_X, scaler_y = self.scalers["muscatine"]
            print(f"  [WARN] No scaler for '{source_domain}', using muscatine")
        else:
            raise RuntimeError("No source scalers available. Run SourceDataLoader first.")

        if path is None or not os.path.exists(str(path)):
            print("[TargetDataLoader] No target data → synthetic demo")
            return self._synthetic_demo(scaler_X, scaler_y, few_shot_fraction)

        print(f"[TargetDataLoader] Loading {path} (type='{target_type}') …")
        df     = smart_load_csv(path)
        target = config.TARGET_COL

        # Align features — fill missing columns with 0 (masked by DataONEEncoder)
        available = [c for c in self.feature_cols if c in df.columns]
        missing   = set(self.feature_cols) - set(available)
        if missing:
            print(f"  [WARN] Missing {len(missing)} features → zeroed & masked")
            for col in missing:
                df[col] = 0.0

        # Build observation mask (0 = imputed/missing, 1 = observed)
        obs_mask = np.ones((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            if col in missing:
                obs_mask[:, i] = 0.0

        df = fill_and_clip(df, self.feature_cols, target, config.PHYSICS)

        X    = df[self.feature_cols].values.astype(np.float32)
        y    = (df[target].values.astype(np.float32)
                if target in df.columns else np.zeros(len(X)))
        seq_len = config.TRAIN["seq_len"]

        X_s  = scaler_X.transform(X)
        y_s  = scaler_y.transform(y.reshape(-1, 1)).ravel()

        X_seq, mask_seq, y_seq = make_sequences_with_mask(X_s, obs_mask,
                                                           y_s, seq_len)

        n_fs = max(1, int(len(X_seq) * few_shot_fraction))

        bs = config.TRAIN["batch_size"]
        adapt_ds   = MaskedSequenceDataset(X_seq, mask_seq, y_seq)
        fewshot_ds = MaskedSequenceDataset(X_seq[:n_fs], mask_seq[:n_fs],
                                            y_seq[:n_fs])

        return (DataLoader(adapt_ds,   batch_size=bs, shuffle=True, drop_last=True),
                DataLoader(fewshot_ds, batch_size=bs, shuffle=True))

    def _synthetic_demo(self, scaler_X, scaler_y, few_shot_fraction):
        np.random.seed(config.SEED + 1)
        n       = 200
        seq_len = config.TRAIN["seq_len"]
        f       = config.MODEL["input_dim"]

        X = np.random.randn(n + seq_len, f).astype(np.float32) * 1.3 + 0.5
        y = (np.sin(np.linspace(0, 8 * np.pi, n + seq_len)) * 0.5
             + np.random.randn(n + seq_len) * 0.1).astype(np.float32)

        mask = np.ones_like(X)   # fully observed for synthetic

        X_seq, mask_seq, y_seq = make_sequences_with_mask(X, mask, y, seq_len)
        n_fs = max(1, int(len(X_seq) * few_shot_fraction))

        bs = config.TRAIN["batch_size"]
        adapt_loader   = DataLoader(
            MaskedSequenceDataset(X_seq, mask_seq, y_seq),
            batch_size=bs, shuffle=True, drop_last=True
        )
        fewshot_loader = DataLoader(
            MaskedSequenceDataset(X_seq[:n_fs], mask_seq[:n_fs], y_seq[:n_fs]),
            batch_size=bs, shuffle=True
        )
        return adapt_loader, fewshot_loader
