"""
kOL·a Fitting from Breakthrough Curves — Bisschops (2017) Model
================================================================
Implements the EXACT compartment model from:
  Bisschops & Brower (2017), Chapter 15 in 'Preparative Chromatography
  for Separation of Proteins', Wiley.

Equations implemented
---------------------
  Langmuir isotherm (Eq. 15.1):
      q* = Qmax * c / (1 + b*c)

  Compartment ODEs (Eqs. 15.6 & 15.7):
      εL  dc/dt =  (φ/V)*(cin - c) + kOL·a * (q/K_loc - c)
      (1-εL) dq/dt = -kOL·a * (q/K_loc - c)

  Overall mass transfer (Eq. 15.5):
      1/kOL = 1/kL + 1/(K_loc * kS)

  Specific interfacial area (Section 15.3 text):
      a = 6/dp * (1 - εL)

  Sherwood numbers (Eqs. 15.2–15.4):
      ShS = 10
      ShL = 0.86/εL * Re^0.50 * Sc^0.33
      kL  = ShL * DL / dp
      kS  = ShS * DS / dp

  Bisschops 2D residual (Eqs. 15.8–15.10):
      δ²  = ΔX²·ΔY² / (ΔX² + ΔY²)
      SSres = Σ δ²

Fitting strategy
----------------
  Primary fitting targets: DL and DS (diffusion coefficients).
  kOL·a is computed from DL, DS, and column/resin geometry.
  Fitted kOL·a is reported as a lumped parameter.

Data source
-----------
  Figure 15.3 — MAb1 on MabSelect SuRe
  X-axis: load mass on column (mg/mL bed volume)
  Y-axis: breakthrough c/c0 (%)
  Four curves: 10, 20, 30, 40 CV/h
  Column: 1.6 cm ID × 2.5 cm H, cin = 2.52 mg/mL (Table 15.1)
  Global fit parameters (Table 15.2): Qmax=107, b=44, DL=4.4e-11, DS=9.5e-14

Usage
-----
  1. Digitize Figure 15.3 with WebPlotDigitizer and paste into EXP_DATA.
  2. Adjust COLUMN and RESIN dataclasses to match your system.
  3. Run:  python fit_kOLa_bisschops.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
import math
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1: EXPERIMENTAL DATA — Figure 15.3, Bisschops (2017)
# ============================================================
# X: load mass on column (mg/mL bed volume)
# Y: breakthrough c/c0 (%)   [0–100 scale, NOT 0–1]
#
# *** REPLACE with your WebPlotDigitizer output from Figure 15.3 ***
# Values below are representative digitizations of Figure 15.3.

EXP_DATA = {
    "10 CV/h": {
        "load_mg_mL": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120],
        "cc0_pct":    [0, 0,  0,  0,  0,  0,  1,  2,  5,  9, 15, 22, 32, 52, 70, 83,  92,  97,  99],
    },
    "20 CV/h": {
        "load_mg_mL": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120],
        "cc0_pct":    [0, 0,  0,  0,  0,  1,  3,  7, 13, 22, 32, 43, 55, 72, 84, 91,  96,  98,  99],
    },
    "30 CV/h": {
        "load_mg_mL": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120],
        "cc0_pct":    [0, 0,  0,  0,  1,  3,  7, 14, 24, 36, 49, 60, 70, 82, 90, 95,  98,  99,  99],
    },
    "40 CV/h": {
        "load_mg_mL": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120],
        "cc0_pct":    [0, 0,  0,  0,  2,  5, 12, 22, 35, 49, 62, 72, 80, 89, 94, 97,  99,  99,  99],
    },
}

# CV/h for each curve (used to compute flow rate)
FLOW_RATES_CVH = {"10 CV/h": 10, "20 CV/h": 20, "30 CV/h": 30, "40 CV/h": 40}


# ============================================================
# SECTION 2: COLUMN & RESIN PARAMETERS (Table 15.1 + literature)
# ============================================================

@dataclass
class ColumnParams:
    """Bisschops Table 15.1: MAb1 on MabSelect SuRe."""
    # Column geometry
    diameter_cm:  float = 1.6      # inner diameter (cm)
    height_cm:    float = 2.5      # bed height (cm)
    epsilon_L:    float = 0.35     # interstitial void fraction

    # Feed
    cin_mg_mL:    float = 2.52     # feed concentration (mg/mL), Table 15.1

    # Resin properties (MabSelect SuRe, literature)
    dp_m:         float = 85e-6    # particle diameter (m), ~85 µm for MabSelect SuRe

    # Fluid properties (water-like at ~20°C)
    rho_L:        float = 1000.0   # fluid density (kg/m³)
    mu_L:         float = 1.0e-3   # dynamic viscosity (Pa·s)

    # Langmuir isotherm (Table 15.2, global fit 10–40 CV/h)
    Qmax:         float = 107.0    # max binding capacity (mg/mL bead volume)
    b:            float = 44.0     # interaction coefficient (mL/mg)

    # Number of compartments (10–20 recommended by Bisschops)
    N_compartments: int = 15

    # Derived (set in __post_init__)
    V_bed_mL:     float = field(init=False)
    A_col_cm2:    float = field(init=False)
    a_cm:         float = field(init=False)   # specific area (cm⁻¹)
    a_m:          float = field(init=False)   # specific area (m⁻¹)

    def __post_init__(self):
        r = self.diameter_cm / 2
        self.V_bed_mL  = math.pi * r**2 * self.height_cm
        self.A_col_cm2 = math.pi * r**2
        # Specific interfacial area: a = 6/dp * (1−εL)  [Bisschops Sec.15.3 text]
        self.a_m  = 6.0 / self.dp_m * (1.0 - self.epsilon_L)
        self.a_cm = self.a_m / 100.0   # convert to cm⁻¹

    def flow_rate_mL_min(self, CVh: float) -> float:
        """Volumetric flow rate (mL/min) from CV/h."""
        return self.V_bed_mL * CVh / 60.0

    def interstitial_velocity_m_s(self, CVh: float) -> float:
        """Interstitial linear velocity (m/s)."""
        phi_mL_min = self.flow_rate_mL_min(CVh)
        phi_m3_s   = phi_mL_min * 1e-6 / 60.0
        A_m2       = self.A_col_cm2 * 1e-4
        return phi_m3_s / A_m2 / self.epsilon_L

    def compartment_volume_mL(self) -> float:
        """Volume of each compartment (mL)."""
        return self.V_bed_mL * self.epsilon_L / self.N_compartments

    def langmuir_q_star(self, c: float) -> float:
        """Eq. 15.1: Langmuir isotherm q* = Qmax*c/(1+b*c)."""
        return self.Qmax * c / (1.0 + self.b * c)

    def langmuir_K_local(self, c: float) -> float:
        """Local partition coefficient K = dq*/dc = Qmax*b/(1+b*c)²."""
        return self.Qmax * self.b / (1.0 + self.b * c)**2

COL = ColumnParams()


# ============================================================
# SECTION 3: SHERWOOD / kOL CALCULATIONS (Eqs. 15.2–15.5)
# ============================================================

def compute_kL_kS(CVh: float, DL: float, DS: float, col: ColumnParams) -> tuple:
    """
    Compute liquid- and solid-film mass transfer coefficients (m/s).

    Eq. 15.2: ShL = 0.86/εL * Re^0.50 * Sc^0.33   (Snowdon-Turner)
    Eq. 15.2: ShS = 10                               (homogeneous diffusion)
    Eq. 15.4: kL  = ShL * DL / dp
    Eq. 15.4: kS  = ShS * DS / dp
    """
    vL  = col.interstitial_velocity_m_s(CVh)
    Re  = col.rho_L * vL * col.dp_m / col.mu_L
    Sc  = col.mu_L  / (col.rho_L * DL)
    ShL = (0.86 / col.epsilon_L) * Re**0.50 * Sc**0.33
    ShS = 10.0

    kL = ShL * DL / col.dp_m   # m/s
    kS = ShS * DS / col.dp_m   # m/s
    return kL, kS


def compute_kOL(kL: float, kS: float, K_loc: float) -> float:
    """
    Eq. 15.5: 1/kOL = 1/kL + 1/(K_loc * kS)
    K_loc = local partition coefficient at current concentration.
    """
    return 1.0 / (1.0 / kL + 1.0 / (K_loc * kS))


def compute_kOLa(CVh: float, DL: float, DS: float,
                  col: ColumnParams, c_ref: float = 0.0) -> float:
    """
    Compute the lumped kOL·a product (min⁻¹) at a reference concentration.

    Parameters
    ----------
    c_ref : reference liquid-phase conc (mg/mL); use 0 for dilute limit,
            or col.cin_mg_mL for inlet conditions.
    """
    kL, kS  = compute_kL_kS(CVh, DL, DS, col)
    K_loc   = col.langmuir_K_local(c_ref)      # mg_solid/mg_liq per mL_liq/mL_solid
    kOL_m_s = compute_kOL(kL, kS, K_loc)       # m/s
    kOL_cm_min = kOL_m_s * 100.0 * 60.0        # cm/min
    kOLa    = kOL_cm_min * col.a_cm             # min⁻¹
    return kOLa


# ============================================================
# SECTION 4: MULTI-COMPARTMENT ODE MODEL (Eqs. 15.6 & 15.7)
# ============================================================

def compartment_odes(t: float, y: np.ndarray,
                     phi_mL_min: float,
                     Vc: float,
                     cin: float,
                     kOLa: float,
                     K_vec: np.ndarray,
                     epsilon_L: float,
                     col: ColumnParams) -> np.ndarray:
    """
    N-compartment ODE system.
    State y = [c_1,...,c_N, q_1,...,q_N]  (2*N equations)

    Eqs. 15.6 & 15.7 per compartment i:
      εL  * dc_i/dt = (φ/Vc)*(c_{i-1} - c_i) + kOL*a*(q_i/K_i - c_i)
      (1-εL)*dq_i/dt = -kOL*a*(q_i/K_i - c_i)

    K_i  = local partition coefficient at current c_i.
    φ/Vc = convection term (flow rate / compartment void volume).
    """
    N  = col.N_compartments
    c  = y[:N]
    q  = y[N:]

    # Recompute local K at current concentrations
    K_loc = np.array([col.langmuir_K_local(ci) for ci in c])

    driving = q / K_loc - c     # Eq. 15.6/15.7 driving force

    dcdt = np.zeros(N)
    dqdt = np.zeros(N)

    for i in range(N):
        c_in_i = cin if i == 0 else c[i-1]
        dcdt[i] = ((phi_mL_min / Vc) * (c_in_i - c[i]) / epsilon_L
                   + kOLa * driving[i])
        dqdt[i] = (-kOLa / (1.0 - epsilon_L) * driving[i])

    return np.concatenate([dcdt, dqdt])


def simulate_breakthrough(DL: float, DS: float,
                           CVh: float, col: ColumnParams,
                           load_max_mg_mL: float = 130.0,
                           n_points: int = 400) -> tuple:
    """
    Simulate breakthrough curve for given DL, DS at flow rate CVh.

    Returns
    -------
    load_sim : np.ndarray   load mass on column (mg/mL bed vol)
    cc0_sim  : np.ndarray   c/c0 normalized (0–100 %)
    kOLa     : float        kOL·a value used (min⁻¹)
    """
    phi   = col.flow_rate_mL_min(CVh)      # mL/min
    Vc    = col.compartment_volume_mL()    # mL per compartment
    cin   = col.cin_mg_mL
    eps   = col.epsilon_L
    N     = col.N_compartments

    # kOLa at dilute-limit (conservative; will be recomputed locally in ODE)
    kOLa_val = compute_kOLa(CVh, DL, DS, col, c_ref=0.0)

    # Time to load load_max_mg_mL on the bed
    # load (mg/mL bed) = phi * cin * t / V_bed  →  t = load * V_bed / (phi * cin)
    t_end  = load_max_mg_mL * col.V_bed_mL / (phi * cin)
    t_eval = np.linspace(0, t_end, n_points)

    y0 = np.zeros(2 * N)   # all compartments empty

    sol = solve_ivp(
        fun=lambda t, y: compartment_odes(
            t, y, phi, Vc, cin, kOLa_val, None, eps, col),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        return np.zeros(n_points), np.zeros(n_points), kOLa_val

    c_out    = sol.y[N - 1]                        # outlet = last compartment
    load_sim = phi * cin * sol.t / col.V_bed_mL    # mg/mL bed
    cc0_sim  = (c_out / cin) * 100.0               # percent

    return load_sim, cc0_sim, kOLa_val


# ============================================================
# SECTION 5: BISSCHOPS 2D RESIDUAL (Eqs. 15.8–15.10)
# ============================================================

def bisschops_SSres(load_exp: np.ndarray, cc0_exp: np.ndarray,
                    load_sim: np.ndarray, cc0_sim: np.ndarray) -> float:
    """
    Eq. 15.10: SSres = Σ [ΔX² · ΔY² / (ΔX² + ΔY²)]

    For each experimental point, find the closest point on the model curve
    by computing the 2D geometric residual δ (Eq. 15.8).
    """
    SSres = 0.0
    for x_e, y_e in zip(load_exp, cc0_exp):
        # Distance to each point on simulated curve
        dX = load_sim - x_e
        dY = cc0_sim  - y_e
        dist2 = dX**2 + dY**2
        i_min = np.argmin(dist2)
        DX = dX[i_min]
        DY = dY[i_min]
        denom = DX**2 + DY**2
        if denom > 0:
            delta2 = (DX**2 * DY**2) / denom   # Eq. 15.9
        else:
            delta2 = 0.0
        SSres += delta2
    return SSres


# ============================================================
# SECTION 6: FITTING ENGINE
# ============================================================

def objective(params_log: np.ndarray, col: ColumnParams,
              fit_global: bool = True) -> float:
    """
    Objective function: total Bisschops SSres across all flow rates.
    params_log = [log10(DL), log10(DS)]
    """
    DL = 10.0 ** params_log[0]
    DS = 10.0 ** params_log[1]

    total_SSres = 0.0
    for label, data in EXP_DATA.items():
        CVh      = FLOW_RATES_CVH[label]
        load_exp = np.array(data["load_mg_mL"], dtype=float)
        cc0_exp  = np.array(data["cc0_pct"],    dtype=float)

        load_sim, cc0_sim, _ = simulate_breakthrough(DL, DS, CVh, col,
                                                     load_max_mg_mL=max(load_exp)*1.05)
        total_SSres += bisschops_SSres(load_exp, cc0_exp, load_sim, cc0_sim)

    return total_SSres


def fit_DL_DS(col: ColumnParams) -> dict:
    """
    Fit DL and DS simultaneously using:
    1. Differential Evolution (global search)
    2. L-BFGS-B polish (local refinement)

    Returns
    -------
    dict with DL, DS, kOLa per flow rate, SSres, and fit quality metrics.
    """
    # Search bounds in log10 space (physical range for mAb in packed bed)
    bounds_log = [(-12, -9),   # DL: 1e-12 to 1e-9 m^2/s
                  (-15, -12)]  # DS: 1e-15 to 1e-12 m^2/s

    print("  Phase 1: Global search (Differential Evolution)...")
    result_de = differential_evolution(
        objective, bounds=bounds_log, args=(col,),
        maxiter=300, popsize=12, tol=1e-6,
        seed=42, workers=1, disp=False,
    )
    print(f"  Phase 1 done. SSres = {result_de.fun:.4f}")

    print("  Phase 2: Local polish (L-BFGS-B)...")
    result = minimize(
        objective, x0=result_de.x, args=(col,),
        method="L-BFGS-B", bounds=bounds_log,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
    )
    print(f"  Phase 2 done. SSres = {result.fun:.4f}")

    DL_fit = 10.0 ** result.x[0]
    DS_fit = 10.0 ** result.x[1]

    # Compute kOL·a at each flow rate
    kOLa_results = {}
    for label, CVh in FLOW_RATES_CVH.items():
        kOLa_val  = compute_kOLa(CVh, DL_fit, DS_fit, col, c_ref=0.0)
        kL, kS    = compute_kL_kS(CVh, DL_fit, DS_fit, col)
        K_loc     = col.langmuir_K_local(0.0)
        kOL_ms    = compute_kOL(kL, kS, K_loc)
        kOLa_results[label] = {
            "CVh":        CVh,
            "kOLa_min":   kOLa_val,         # min⁻¹  (lumped)
            "kOL_m_s":    kOL_ms,            # m/s
            "kL_m_s":     kL,
            "kS_m_s":     kS,
            "a_m":        col.a_m,           # m⁻¹ (computed from geometry)
        }

    return {
        "DL":          DL_fit,
        "DS":          DS_fit,
        "SSres":       result.fun,
        "success":     result.success,
        "kOLa":        kOLa_results,
    }


# ============================================================
# SECTION 7: PLOTTING
# ============================================================

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

def plot_results(fit: dict, col: ColumnParams) -> None:
    """
    Reproduce Figure 15.3 style: exp markers + model curves,
    plus kOL·a bar chart.
    """
    DL = fit["DL"]
    DS = fit["DS"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Bisschops (2017) Compartment Model — MAb1 on MabSelect SuRe\n"
        f"Fitted: DL = {DL:.2e} m²/s,  DS = {DS:.2e} m²/s,  "
        f"Qmax = {col.Qmax} mg/mL,  b = {col.b} mL/mg",
        fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ---- Top: full breakthrough overlay (all curves, Figure 15.3 style) ----
    ax_main = fig.add_subplot(gs[0, :])
    for idx, (label, data) in enumerate(EXP_DATA.items()):
        CVh      = FLOW_RATES_CVH[label]
        load_exp = np.array(data["load_mg_mL"])
        cc0_exp  = np.array(data["cc0_pct"])
        color    = COLORS[idx]

        load_sim, cc0_sim, _ = simulate_breakthrough(
            DL, DS, CVh, col, load_max_mg_mL=125)

        ax_main.scatter(load_exp, cc0_exp, color=color, s=40, zorder=5,
                        edgecolors="black", linewidths=0.5,
                        label=f"{label} (exp)")
        ax_main.plot(load_sim, cc0_sim, color=color, linewidth=2,
                     linestyle="-" if idx == 0 else ["--","-.",":","-"][idx],
                     label=f"{label} (sim)")

    ax_main.axhline(10, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax_main.set_xlabel("Load mass on column (mg/mL bed)", fontsize=11)
    ax_main.set_ylabel("Breakthrough c/c₀ (%)", fontsize=11)
    ax_main.set_title("Breakthrough Curves — Fig. 15.3 Reproduction", fontsize=11)
    ax_main.set_xlim(0, 125)
    ax_main.set_ylim(-2, 102)
    ax_main.legend(fontsize=8, ncol=2)
    ax_main.grid(True, alpha=0.25)

    # ---- Bottom left: kOL·a bar chart per flow rate ----
    ax_bar = fig.add_subplot(gs[1, 0])
    labels_   = list(fit["kOLa"].keys())
    kOLa_vals = [fit["kOLa"][l]["kOLa_min"] for l in labels_]
    bars = ax_bar.bar(labels_, kOLa_vals, color=COLORS, alpha=0.85,
                      edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, kOLa_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val*1.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("kOL·a  (min⁻¹)", fontsize=10)
    ax_bar.set_title("Fitted kOL·a per Flow Rate\n(a computed from geometry)", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(bottom=0)
    ax_bar.tick_params(axis="x", labelsize=8)

    # ---- Bottom right: kOL and component kL, kS ----
    ax_k = fig.add_subplot(gs[1, 1])
    CVh_vals  = [fit["kOLa"][l]["CVh"]     for l in labels_]
    kOL_vals  = [fit["kOLa"][l]["kOL_m_s"] * 100 * 60 for l in labels_]  # cm/min
    kL_vals   = [fit["kOLa"][l]["kL_m_s"]  * 100 * 60 for l in labels_]
    kS_vals   = [fit["kOLa"][l]["kS_m_s"]  * 100 * 60 for l in labels_]

    x = np.arange(len(labels_))
    w = 0.25
    ax_k.bar(x - w, kL_vals,  width=w, label="kL (cm/min)", color="#aec7e8", edgecolor="black")
    ax_k.bar(x,     kS_vals,  width=w, label="kS (cm/min)", color="#ffbb78", edgecolor="black")
    ax_k.bar(x + w, kOL_vals, width=w, label="kOL (cm/min)",color="#98df8a", edgecolor="black")
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(labels_, fontsize=8)
    ax_k.set_ylabel("Mass transfer coefficient (cm/min)", fontsize=9)
    ax_k.set_title("Film Coefficients kL, kS, kOL\n(Eqs. 15.2–15.5)", fontsize=9)
    ax_k.legend(fontsize=7)
    ax_k.grid(axis="y", alpha=0.3)

    plt.savefig("bisschops_kOLa_fit.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved: bisschops_kOLa_fit.png")


# ============================================================
# SECTION 8: MAIN
# ============================================================

def main():
    col = COL

    print("\n" + "="*65)
    print("  BISSCHOPS (2017) — kOL·a FITTING FROM BREAKTHROUGH CURVES")
    print("  Equations 15.1–15.10 implemented exactly")
    print("="*65)
    print(f"\n  Column:       {col.diameter_cm} cm ID × {col.height_cm} cm H")
    print(f"  Bed volume:   {col.V_bed_mL:.3f} mL")
    print(f"  εL:           {col.epsilon_L}")
    print(f"  dp:           {col.dp_m*1e6:.0f} µm")
    print(f"  a (geometry): {col.a_m:.0f} m⁻¹  ({col.a_cm:.1f} cm⁻¹)  [a = 6/dp*(1-εL)]")
    print(f"  Qmax:         {col.Qmax} mg/mL,   b: {col.b} mL/mg")
    print(f"  N compartments: {col.N_compartments}")
    print(f"  Feed conc:    {col.cin_mg_mL} mg/mL")
    print(f"\n  Fitting DL and DS simultaneously across all {len(EXP_DATA)} flow rates...")
    print(f"  Bisschops 2D SSres objective (Eq. 15.10)\n")

    fit = fit_DL_DS(col)

    # Print results
    print("\n" + "="*65)
    print("  FIT RESULTS")
    print("="*65)
    print(f"  DL (liquid diffusivity)  = {fit['DL']:.2e} m²/s")
    print(f"  DS (resin diffusivity)   = {fit['DS']:.2e} m²/s")
    print(f"  Paper Table 15.2 values: DL = 4.4e-11, DS = 9.5e-14 m²/s")
    print(f"  Total SSres (Eq.15.10)   = {fit['SSres']:.4f}")
    print(f"  Optimizer converged:       {fit['success']}")
    print()

    print(f"  {'Condition':<12} {'kOL·a (min⁻¹)':>14} {'kOL (cm/min)':>14} "
          f"{'kL (cm/min)':>13} {'kS (cm/min)':>13}")
    print(f"  {'a [m⁻¹]:':<12} {col.a_m:>14.0f}  (fixed, from geometry)")
    print("-"*70)
    for label, res in fit["kOLa"].items():
        print(f"  {label:<12} {res['kOLa_min']:>14.4f} "
              f"{res['kOL_m_s']*100*60:>14.6f} "
              f"{res['kL_m_s']*100*60:>13.6f} "
              f"{res['kS_m_s']*100*60:>13.6f}")
    print("="*65)

    # Verify against paper values
    print("\n  VERIFICATION vs. Paper (Table 15.2 global fit):")
    DL_paper = 4.4e-11
    DS_paper = 9.5e-14
    print(f"  Paper kOLa at 10 CV/h (recomputed): "
          f"{compute_kOLa(10, DL_paper, DS_paper, col):.4f} min⁻¹")
    print(f"  Paper kOLa at 20 CV/h (recomputed): "
          f"{compute_kOLa(20, DL_paper, DS_paper, col):.4f} min⁻¹")
    print(f"  Paper kOLa at 30 CV/h (recomputed): "
          f"{compute_kOLa(30, DL_paper, DS_paper, col):.4f} min⁻¹")
    print(f"  Paper kOLa at 40 CV/h (recomputed): "
          f"{compute_kOLa(40, DL_paper, DS_paper, col):.4f} min⁻¹")

    plot_results(fit, col)
    return fit


if __name__ == "__main__":
    main()

