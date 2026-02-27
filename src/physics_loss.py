"""
physics_loss.py  (v3 — Latent State Physics)
Physics-Informed Neural Network (PINN) loss module.

Key changes from v2:
  - PhysicsNormLayer : converts raw sensor units → SI before residuals
  - LatentODEResiduals: Monod/mass balance on LATENT biokinetic states
    (X_hat, S_hat, VFA_hat) decoded from LSTM hidden state, not on
    raw biogas flow (which had no gradient path to LSTM weights)
  - PhysicsInformedLoss.forward() now accepts `latent_states` dict and
    a scalar `weight` for the residual weight schedule (0 → 1)
  - build_physics_dict replaced by build_si_batch (unit-safe extraction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Physical Constants ────────────────────────────────────────────────────────

MU_MAX   = 0.40      # max growth rate        (day⁻¹)
K_S      = 0.50      # half-saturation COD    (g/L)
K_D      = 0.03      # decay rate             (day⁻¹)
Y        = 0.05      # biomass yield           (g VSS / g COD)
K_I      = 0.30      # Andrews inhibition     (g NH₃-N / L)
DELTA_G0 = -31.0     # kJ/mol  (acetoclastic methanogenesis)
R_GAS    = 8.314e-3  # kJ / (mol·K)

# Unit conversion factors
VFA_MW      = 60.052   # g/mol  acetate
COD_TO_MOL  = 1.0 / 32.0  # mol O₂ per g COD  (for normalisation)


# ─── 1. Physics Normalisation Layer ───────────────────────────────────────────

class PhysicsNormLayer(nn.Module):
    """
    Converts sensor readings to SI units before residual computation.

    Supported keys (all optional – missing keys silently skipped):
      'T_celsius' or 'T_fahrenheit' → 'T_kelvin'    (K)
      'VFA_mgL'                     → 'VFA_molL'    (mol/L)
      'COD_mgL'                     → 'COD_gL'      (g/L)
      'pH'                          → 'H_molL'      (mol/L)
      'pressure_atm'                → 'pressure_pa' (Pa)
    """

    def forward(self, sensor_dict: dict) -> dict:
        out = dict(sensor_dict)  # shallow copy

        # Temperature → Kelvin
        if "T_celsius" in out:
            out["T_kelvin"] = out["T_celsius"] + 273.15
        elif "T_fahrenheit" in out:
            out["T_kelvin"] = (out["T_fahrenheit"] - 32.0) * (5.0 / 9.0) + 273.15

        # VFA mg/L → mol/L  (acetate proxy)
        if "VFA_mgL" in out:
            out["VFA_molL"] = out["VFA_mgL"] / (VFA_MW * 1000.0)

        # COD mg/L → g/L
        if "COD_mgL" in out:
            out["COD_gL"] = out["COD_mgL"] / 1000.0

        # pH → [H⁺] mol/L
        if "pH" in out:
            out["H_molL"] = 10.0 ** (-out["pH"])

        # Pressure atm → Pa
        if "pressure_atm" in out:
            out["pressure_pa"] = out["pressure_atm"] * 101325.0

        return out


# ─── 2. Monod Kinetics Helper ─────────────────────────────────────────────────

def _monod_rate(S: torch.Tensor, I: torch.Tensor = None) -> torch.Tensor:
    """Specific growth rate μ(S) with optional Andrews inhibition."""
    base = MU_MAX * S / (K_S + S + 1e-8)
    if I is not None:
        base = base / (1.0 + I / (K_I + 1e-8))
    return base


# ─── 3. Latent ODE Residuals ─────────────────────────────────────────────────

class LatentODEResiduals(nn.Module):
    """
    Enforces Monod ODEs on the LATENT biokinetic state decoded from the LSTM:

      dX/dt = μ(S)·X - Kd·X          (biomass growth-decay)
      dS/dt = -(1/Y)·μ(S)·X          (substrate consumption)

    Uses finite differences between consecutive latent state snapshots
    to approximate the time derivative → ensures gradient flows through
    the LSTM/encoder weights (previous approach computed residuals on raw
    sensor columns that had no gradient connection to LSTM parameters).

    Args:
      latent_t0 : dict with 'X', 'S', optionally 'I' — shape (B,)
      latent_t1 : dict — same keys, next time step
      dt        : time step in days (default 1.0)
    """

    def forward(self,
                latent_t0: dict,
                latent_t1: dict,
                dt: float = 1.0) -> torch.Tensor:
        if "X" not in latent_t0 or "S" not in latent_t0:
            return torch.tensor(0.0, requires_grad=False)

        X0 = F.relu(latent_t0["X"])
        S0 = F.relu(latent_t0["S"])
        X1 = F.relu(latent_t1["X"])
        S1 = F.relu(latent_t1["S"])
        I0 = latent_t0.get("I", None)

        mu = _monod_rate(S0, I0)

        # Finite-difference approximations of derivatives
        dX_obs = (X1 - X0) / (dt + 1e-8)
        dS_obs = (S1 - S0) / (dt + 1e-8)

        # ODE right-hand sides
        dX_pred = mu * X0 - K_D * X0
        dS_pred = -(1.0 / Y) * mu * X0

        res_X = F.mse_loss(dX_pred, dX_obs)
        res_S = F.mse_loss(dS_pred, dS_obs)

        return res_X + res_S


# ─── 4. VFA / Acetoclastic Mass Balance ───────────────────────────────────────

class VFAMassBalance(nn.Module):
    """
    Acetate consumption drives CH₄ production:
      dVFA/dt ≈ -k_ace · VFA     (first-order degradation, simplified)
    Penalty = residual between predicted and observed VFA rate.
    """
    K_ACE = 0.12   # day⁻¹ first-order acetate degradation

    def forward(self, vfa_t0: torch.Tensor,
                       vfa_t1: torch.Tensor,
                       dt: float = 1.0) -> torch.Tensor:
        vfa_t0 = F.relu(vfa_t0)
        vfa_t1 = F.relu(vfa_t1)
        dVFA_obs  = (vfa_t1 - vfa_t0) / (dt + 1e-8)
        dVFA_pred = -self.K_ACE * vfa_t0
        return F.mse_loss(dVFA_pred, dVFA_obs)


# ─── 5. Thermodynamic Feasibility ─────────────────────────────────────────────

class ThermodynamicLoss(nn.Module):
    """
    ΔG_reaction = ΔG0 + RT·ln(Q)  must be < 0 (spontaneous).
    Penalty = ReLU(ΔG).mean()
    Inputs (all SI units from PhysicsNormLayer):
      T_kelvin   : (B,)
      VFA_molL   : (B,)  acetate proxy
      ch4_partial: (B,)  partial pressure atm
      co2_partial: (B,)  partial pressure atm
    """

    def forward(self,
                T_kelvin:    torch.Tensor,
                VFA_molL:    torch.Tensor,
                ch4_partial: torch.Tensor,
                co2_partial: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        Q = (ch4_partial * co2_partial) / (VFA_molL + eps)
        Q = Q.clamp(min=eps)
        delta_G = DELTA_G0 + R_GAS * T_kelvin * torch.log(Q)
        return F.relu(delta_G).mean()


# ─── 6. Combined PINN Loss ────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    Combines all physics sub-losses.

    IMPORTANT CHANGE from v2:
      - `latent_states` replaces `biogas_pred` as the primary argument.
        It must be a dict with two keys:
            't0' : {'X': (B,), 'S': (B,), 'VFA': (B,)}
            't1' : {'X': (B,), 'S': (B,), 'VFA': (B,)}
        produced by the LatentBiokineticsDecoder (not raw sensor data).
      - `weight` (float 0→1) scales the total physics loss, driven by
        the residual weight scheduler in train_source.py.
      - Thermodynamic loss still uses the raw sensor batch (optional).

    Usage in train_source.py:
        physics_weight = min(1.0, epoch / (epochs * 0.5))
        phys_l = pinn_loss(latent_states, sensor_batch, weight=physics_weight)
    """

    def __init__(self,
                 w_ode:    float = 0.20,
                 w_vfa:    float = 0.10,
                 w_thermo: float = 0.05):
        super().__init__()
        self.ode_residuals = LatentODEResiduals()
        self.vfa_balance   = VFAMassBalance()
        self.thermo        = ThermodynamicLoss()
        self.norm_layer    = PhysicsNormLayer()
        self.w_ode         = w_ode
        self.w_vfa         = w_vfa
        self.w_thermo      = w_thermo

    def forward(self,
                latent_states: dict,
                sensor_batch:  dict = None,
                dt:            float = 1.0,
                weight:        float = 1.0) -> torch.Tensor:
        """
        latent_states : {
            't0': {'X': Tensor(B,), 'S': Tensor(B,), 'VFA': Tensor(B,)},
            't1': {'X': Tensor(B,), 'S': Tensor(B,), 'VFA': Tensor(B,)},
        }
        sensor_batch  : raw sensor dict (optional, for thermodynamic loss)
        weight        : physics residual weight from scheduler
        """
        if weight <= 0.0:
            device = next(iter(latent_states.get("t0",
                          {None: torch.zeros(1)}).values())).device
            return torch.zeros(1, device=device).squeeze()

        loss = torch.zeros(1).squeeze()

        # ── ODE residuals on latent states ─────────────────────────────────
        if "t0" in latent_states and "t1" in latent_states:
            loss = loss + self.w_ode * self.ode_residuals(
                latent_states["t0"], latent_states["t1"], dt
            )
            # VFA balance
            if "VFA" in latent_states["t0"] and "VFA" in latent_states["t1"]:
                loss = loss + self.w_vfa * self.vfa_balance(
                    latent_states["t0"]["VFA"],
                    latent_states["t1"]["VFA"], dt
                )

        # ── Thermodynamic feasibility (optional sensor data) ────────────────
        if sensor_batch is not None:
            sb = self.norm_layer(sensor_batch)
            if all(k in sb for k in ("T_kelvin", "VFA_molL",
                                      "ch4_partial", "co2_partial")):
                loss = loss + self.w_thermo * self.thermo(
                    sb["T_kelvin"], sb["VFA_molL"],
                    sb["ch4_partial"], sb["co2_partial"]
                )

        return weight * loss


# ─── 7. Sensor-to-SI Batch Builder ────────────────────────────────────────────

def build_si_batch(X_batch: torch.Tensor, feature_cols: list) -> dict:
    """
    Extracts physics variables from a SCADA batch and converts to SI units.
    Returns a sensor dict usable by PhysicsNormLayer.

    Matched with actual Muscatine column names (Dig1-VFA_mgL, D1_TEMPERATURE,
    HSW-COD_mgL, Dig1-T_degF, etc.)
    """
    col_lower = [c.lower() for c in feature_cols]
    d = {}

    def get_col(keywords):
        for kw in keywords:
            for i, c in enumerate(col_lower):
                if kw in c:
                    return X_batch[:, -1, i]   # last timestep
        return None

    # Temperature  (°F preferred due to Muscatine column names)
    T_f = get_col(["_degf", "temp_f"])
    T_c = get_col(["temperature", "temp_c", "d1_temp", "d2_temp"])
    if T_f is not None:
        d["T_fahrenheit"] = T_f
    elif T_c is not None:
        d["T_celsius"] = T_c

    # VFA mg/L
    vfa = get_col(["vfa_mgl", "vfa"])
    if vfa is not None:
        d["VFA_mgL"] = vfa

    # COD mg/L
    cod = get_col(["cod_mgl", "cod_in", "hsw-cod"])
    if cod is not None:
        d["COD_mgL"] = cod

    # pH
    ph = get_col(["ph"])
    if ph is not None:
        d["pH"] = ph

    return d
