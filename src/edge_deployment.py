"""
edge_deployment.py
Edge deployment optimization for PLC / industrial controller deployment.

Covers:
  1. INT8 quantization (static + dynamic)
  2. Latency benchmarking (<1s requirement)
  3. Model export to ONNX
  4. Maintenance mode detection (anomaly-based suspension)
  5. Lightweight inference wrapper suitable for edge devices
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import warnings
warnings.filterwarnings("ignore")


# ─── 1. Quantization ──────────────────────────────────────────────────────────

class ModelQuantizer:
    """
    Quantizes the model to INT8 for reduced memory and faster inference.

    Two modes:
      'dynamic' – quantizes weights to INT8, activations dynamically (easier, no calibration)
      'static'  – quantizes both weights and activations (faster, needs calibration data)
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model    = model
        self.device   = device
        self.quant_model = None

    def dynamic_quantize(self) -> nn.Module:
        """
        Dynamic INT8 quantization (recommended for LSTM-based models on CPU).
        Works on CPU only — most PLCs/edge devices run on CPU.
        """
        import torch.quantization as tq

        self.model.eval()
        self.model.cpu()

        # Quantize Linear layers (LSTM quantization has limited support)
        quant = tq.quantize_dynamic(
            self.model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )
        self.quant_model = quant
        print("[Quantize] Dynamic INT8 quantization applied.")
        return quant

    def benchmark_size(self) -> dict:
        """Compare model sizes before/after quantization."""
        def model_size_mb(m):
            total = sum(p.numel() * p.element_size() for p in m.parameters())
            return total / (1024 ** 2)

        orig_size = model_size_mb(self.model)
        quant_size = model_size_mb(self.quant_model) if self.quant_model else None

        result = {
            "original_mb":    round(orig_size, 3),
            "quantized_mb":   round(quant_size, 3) if quant_size else None,
            "compression":    round(orig_size / quant_size, 2) if quant_size else None,
        }
        print(f"[Quantize] Original: {orig_size:.2f} MB  "
              f"Quantized: {quant_size:.2f} MB  "
              f"({result['compression']}×)")
        return result


# ─── 2. ONNX Export ───────────────────────────────────────────────────────────

def export_onnx(model: nn.Module,
                seq_len:  int,
                feat_dim: int,
                save_path: str) -> bool:
    """
    Export model to ONNX format for deployment in non-PyTorch environments.
    Returns True on success.
    """
    try:
        import torch.onnx
        model.eval()
        dummy = torch.randn(1, seq_len, feat_dim)

        # Wrap model to return only the mean prediction (gamma)
        class ExportWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                gamma, _, _, _ = self.m.source_forward(x)
                return gamma

        wrapper = ExportWrapper(model)
        torch.onnx.export(
            wrapper, dummy, save_path,
            input_names=["sensor_sequence"],
            output_names=["biogas_prediction"],
            dynamic_axes={
                "sensor_sequence":  {0: "batch_size"},
                "biogas_prediction": {0: "batch_size"},
            },
            opset_version=13,
        )
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        print(f"[ONNX] Exported to {save_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        return False


# ─── 3. Latency Benchmarker ───────────────────────────────────────────────────

class LatencyBenchmarker:
    """
    Measures inference latency for real-time deployment validation.
    Target: <1 second for single-sample prediction on CPU.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model  = model
        self.device = torch.device(device)

    def benchmark(self,
                  seq_len:    int,
                  feat_dim:   int,
                  batch_sizes: list = [1, 8, 32],
                  n_warmup:   int  = 20,
                  n_runs:     int  = 100) -> dict:
        """
        Run latency benchmark over multiple batch sizes.
        Returns dict with latency stats per batch size.
        """
        self.model.eval()
        self.model.to(self.device)
        results = {}

        print(f"\n[Latency] Benchmarking on {self.device} …")
        for bs in batch_sizes:
            dummy = torch.randn(bs, seq_len, feat_dim).to(self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = self.model.source_forward(dummy)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _  = self.model.source_forward(dummy)
                    times.append(time.perf_counter() - t0)

            times = np.array(times) * 1000  # ms
            stats = {
                "mean_ms":    round(float(times.mean()), 3),
                "p50_ms":     round(float(np.percentile(times, 50)), 3),
                "p95_ms":     round(float(np.percentile(times, 95)), 3),
                "p99_ms":     round(float(np.percentile(times, 99)), 3),
                "per_sample_ms": round(float(times.mean() / bs), 3),
                "meets_1s_sla": bool(np.percentile(times, 99) < 1000),
            }
            results[f"batch_{bs}"] = stats
            sla = "✓ PASS" if stats["meets_1s_sla"] else "✗ FAIL"
            print(f"  BS={bs:3d}  P99={stats['p99_ms']:7.2f}ms  "
                  f"per-sample={stats['per_sample_ms']:6.3f}ms  SLA:{sla}")

        return results

    def profile_layers(self, seq_len: int, feat_dim: int) -> dict:
        """Profile time per major module (encoder, pred head, etc.)."""
        self.model.eval()
        dummy    = torch.randn(1, seq_len, feat_dim).to(self.device)
        timings  = {}
        hooks    = []

        def make_hook(name):
            def hook(module, inp, out):
                timings[name] = timings.get(name, 0) + time.perf_counter()
            return hook

        # Register hooks on major modules
        for name, module in self.model.named_children():
            module.register_forward_hook(make_hook(name))

        with torch.no_grad():
            self.model.source_forward(dummy)

        print("\n[Latency] Per-module timings:")
        for name, t in timings.items():
            print(f"  {name}: {t*1000:.3f} ms")

        return timings


# ─── 4. Maintenance Mode Detector ────────────────────────────────────────────

class MaintenanceModeDetector:
    """
    Suspends model predictions during sensor maintenance / plant shutdown.

    Detection criteria:
      1. Flat-line detection: sensor values don't change for N timesteps
      2. Out-of-range detection: sensor value outside physical bounds
      3. Rapid change detection: step change larger than expected process dynamics
      4. Multi-sensor failure: too many sensors simultaneously missing/zero

    Returns a maintenance flag that the inference pipeline checks.
    """

    def __init__(self,
                 flatline_window:   int   = 12,    # timesteps
                 flatline_tol:      float = 1e-4,
                 oob_bounds:        dict  = None,
                 step_change_sigma: float = 5.0,
                 max_zero_fraction: float = 0.4):
        self.flatline_window   = flatline_window
        self.flatline_tol      = flatline_tol
        self.oob_bounds        = oob_bounds or {}
        self.step_change_sigma = step_change_sigma
        self.max_zero_fraction = max_zero_fraction
        self._history          = []   # rolling window of recent readings

    def check(self, x_latest: np.ndarray) -> dict:
        """
        Args:
          x_latest: 1D array of shape (feat_dim,) — latest sensor reading

        Returns dict with:
          'maintenance': bool
          'reason':      str or None
        """
        self._history.append(x_latest.copy())
        if len(self._history) > self.flatline_window * 2:
            self._history.pop(0)

        # 1. Multi-sensor failure
        zero_frac = (np.abs(x_latest) < 1e-6).mean()
        if zero_frac > self.max_zero_fraction:
            return {"maintenance": True,
                    "reason": f"Multi-sensor failure ({zero_frac*100:.0f}% zeros)"}

        # 2. Out-of-range
        for col_idx, (lo, hi) in self.oob_bounds.items():
            if col_idx < len(x_latest):
                val = x_latest[col_idx]
                if val < lo or val > hi:
                    return {"maintenance": True,
                            "reason": f"Sensor {col_idx} out of range: {val:.2f} ∉ [{lo}, {hi}]"}

        if len(self._history) >= self.flatline_window:
            recent = np.array(self._history[-self.flatline_window:])

            # 3. Flat-line detection
            ranges = recent.max(axis=0) - recent.min(axis=0)
            flat   = (ranges < self.flatline_tol).mean()
            if flat > 0.5:
                return {"maintenance": True,
                        "reason": f"Flat-line detected on {flat*100:.0f}% of sensors"}

            # 4. Rapid step change
            if len(self._history) >= 2:
                diffs = np.abs(np.diff(recent, axis=0))
                stds  = recent.std(axis=0) + 1e-8
                zscore_changes = (diffs / stds).max()
                if zscore_changes > self.step_change_sigma:
                    return {"maintenance": True,
                            "reason": f"Step change detected (z={zscore_changes:.1f})"}

        return {"maintenance": False, "reason": None}


# ─── 5. Edge Inference Wrapper ────────────────────────────────────────────────

class EdgeInferenceEngine:
    """
    Production-ready inference wrapper for edge deployment.
    Combines:
      - Maintenance mode detection (auto-suspend)
      - Latency guard (skip if >budget ms)
      - Output caching (return last valid prediction if current fails)
      - Structured JSON output for PLC/SCADA integration
    """

    def __init__(self,
                 model:      nn.Module,
                 scaler_X,
                 scaler_y,
                 feature_cols: list,
                 seq_len:      int,
                 max_latency_ms: float = 900.0,
                 device:      str = "cpu"):
        self.model          = model
        self.scaler_X       = scaler_X
        self.scaler_y       = scaler_y
        self.feature_cols   = feature_cols
        self.seq_len        = seq_len
        self.max_latency_ms = max_latency_ms
        self.device         = torch.device(device)
        self.maintenance    = MaintenanceModeDetector()
        self._buffer        = []
        self._last_output   = None
        self.model.eval()
        self.model.to(self.device)

    def ingest(self, sensor_reading: dict) -> dict:
        """
        Process one sensor reading (dict of {col_name: value}).
        Returns prediction dict or maintenance/error status.
        """
        # Build feature vector
        x = np.array([sensor_reading.get(c, 0.0) for c in self.feature_cols],
                     dtype=np.float32)

        # Maintenance check
        maint = self.maintenance.check(x)
        if maint["maintenance"]:
            return {
                "status":  "MAINTENANCE",
                "reason":  maint["reason"],
                "biogas_m3d": self._last_output.get("biogas_m3d") if self._last_output else None,
                "source":  "cached",
            }

        # Accumulate sequence
        x_scaled = self.scaler_X.transform(x.reshape(1, -1))[0]
        self._buffer.append(x_scaled)
        if len(self._buffer) > self.seq_len:
            self._buffer.pop(0)

        if len(self._buffer) < self.seq_len:
            return {"status": "BUFFERING",
                    "buffer_fill": f"{len(self._buffer)}/{self.seq_len}"}

        # Inference
        X_seq = np.array(self._buffer, dtype=np.float32)[np.newaxis]  # (1, T, F)
        X_t   = torch.tensor(X_seq).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            mean, aleatoric, epistemic, reject = self.model.predict(X_t, self.scaler_y)
        latency_ms = (time.perf_counter() - t0) * 1000

        if latency_ms > self.max_latency_ms:
            return {"status": "LATENCY_EXCEEDED",
                    "latency_ms": round(latency_ms, 1),
                    **({} if self._last_output is None else self._last_output)}

        mean_val      = float(mean.cpu().numpy().ravel()[0])
        aleatoric_val = float(aleatoric.cpu().numpy().ravel()[0])
        epistemic_val = float(epistemic.cpu().numpy().ravel()[0])
        rejected      = bool(reject.cpu().numpy().ravel()[0])

        output = {
            "status":          "OK",
            "biogas_m3d":      round(mean_val, 2),
            "uncertainty_std": round(aleatoric_val + epistemic_val, 3),
            "aleatoric":       round(aleatoric_val, 3),
            "epistemic":       round(epistemic_val, 3),
            "prediction_flag": "UNCERTAIN" if rejected else "RELIABLE",
            "latency_ms":      round(latency_ms, 1),
            "stability":       self._classify_stability(mean_val, epistemic_val),
        }
        self._last_output = output
        return output

    def _classify_stability(self, mean: float, epistemic: float) -> str:
        cv = epistemic / (abs(mean) + 1e-6)
        if cv < 0.1:
            return "STABLE"
        elif cv < 0.3:
            return "WARNING"
        return "UNSTABLE"
