"""
kOL Fitting from Experimental Breakthrough Curves
===================================================
Fits the overall mass transfer coefficient (kOL) from experimental
breakthrough curve data using nonlinear least-squares optimization.

The model solves:
    dc/dt = (phi/V)*(cin - c) - kOL * a * (c - q/K)
    dq/dt =                     kOL * a * (c - q/K)

where the driving force is expressed as (c - q/K), equivalent to (q - K*c)/K.

Usage
-----
1. Replace the EXPERIMENTAL_DATA dict with your digitized Figure 1 values.
2. Set COLUMN_PARAMS to match the column specifications from the paper.
3. Run:  python fit_kOL_breakthrough.py

The script will:
- Fit kOL for each experimental curve independently
- Plot fitted vs. experimental breakthrough curves
- Report fitted kOL with 95% confidence intervals
- Report goodness-of-fit (R², RMSE)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: EXPERIMENTAL DATA
# =============================================================================
# *** REPLACE THESE VALUES WITH DATA DIGITIZED FROM FIGURE 1 ***
# Use a digitization tool such as:
#   - WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/) — free, browser-based
#   - Engauge Digitizer — open source desktop tool
#   - PlotDigitizer — online
#
# X-axis: Column Volumes (CV) loaded   [or time in minutes]
# Y-axis: Normalized outlet conc.      c_out / c_in  (dimensionless, 0 to 1)
#
# Example below uses REPRESENTATIVE mAb Protein A breakthrough data
# consistent with the literature (replace with your paper's Figure 1 values).

EXPERIMENTAL_DATA = {
    # Each key is a curve label (e.g. residence time or flow rate condition)
    # Each value is a dict with:
    #   "CV"  : list of column volumes (x-axis)
    #   "cc0" : list of c/c0 normalized concentrations (y-axis, 0-1)
    #   "RT"  : residence time in minutes for this run
    #   "cin" : feed protein concentration mg/mL

    "RT = 2 min": {
        "CV":  [0, 5, 10, 15, 18, 20, 22, 25, 30, 35, 40],
        "cc0": [0, 0.00, 0.01, 0.05, 0.12, 0.25, 0.42, 0.65, 0.85, 0.95, 0.99],
        "RT":  2.0,
        "cin": 2.0,   # mg/mL — adjust to match paper
    },
    "RT = 4 min": {
        "CV":  [0, 5, 10, 15, 20, 25, 28, 30, 35, 40, 45],
        "cc0": [0, 0.00, 0.00, 0.02, 0.08, 0.20, 0.40, 0.55, 0.78, 0.92, 0.99],
        "RT":  4.0,
        "cin": 2.0,
    },
    "RT = 8 min": {
        "CV":  [0, 5, 10, 15, 20, 25, 30, 35, 38, 42, 45],
        "cc0": [0, 0.00, 0.00, 0.00, 0.03, 0.10, 0.30, 0.62, 0.80, 0.95, 0.99],
        "RT":  8.0,
        "cin": 2.0,
    },
}


# =============================================================================
# SECTION 2: COLUMN AND RESIN PARAMETERS
# =============================================================================
# *** REPLACE WITH VALUES FROM THE PAPER'S MATERIALS AND METHODS ***

@dataclass
class ColumnParams:
    """Physical parameters of the chromatography column and resin."""
    # Column geometry
    column_length_cm:   float = 10.0    # Column bed height (cm)
    column_diameter_cm: float = 0.5     # Column inner diameter (cm)

    # Resin / adsorption parameters
    Qmax:   float = 50.0    # Maximum static binding capacity (mg/mL resin)
    K:      float = 1.5     # Partition / Henry's law constant (-)
    kS:     float = 0.05    # Solid-film mass transfer coefficient (1/min)
    a:      float = 30.0    # Specific interfacial area (1/cm or 1/mL)

    # Void fraction
    epsilon: float = 0.35   # Column void fraction (interstitial)

    # Derived properties (computed automatically)
    volume_mL: float = field(init=False)

    def __post_init__(self):
        import math
        r = self.column_diameter_cm / 2
        self.volume_mL = math.pi * r**2 * self.column_length_cm  # total column volume (mL)

    @property
    def void_volume_mL(self) -> float:
        return self.volume_mL * self.epsilon

    def flow_rate_from_RT(self, RT_min: float) -> float:
        """Compute volumetric flow rate (mL/min) from residence time (min)."""
        return self.void_volume_mL / RT_min

    def CV_to_time(self, CV: np.ndarray, RT_min: float) -> np.ndarray:
        """Convert column volumes to time (min)."""
        return CV * RT_min   # 1 CV = 1 RT in this convention

COLUMN = ColumnParams()


# =============================================================================
# SECTION 3: MODEL EQUATIONS
# =============================================================================

def mass_balance_odes(t: float, y: np.ndarray,
                      phi: float, V: float, cin: float,
                      kOL: float, K: float, a: float) -> list:
    """
    Two-phase lumped mass balance ODEs.

    dc/dt = (phi/V)*(cin - c) - kOL*a*(c - q/K)
    dq/dt =                     kOL*a*(c - q/K)

    Note: driving force written as (c - q/K) = departure from
          linear isotherm equilibrium q* = K*c, scaled by K.
    """
    c, q = y
    driving_force = c - q / K     # >0 means net adsorption onto resin

    dc_dt = (phi / V) * (cin - c) - kOL * a * driving_force
    dq_dt =                          kOL * a * driving_force
    return [dc_dt, dq_dt]


def simulate_breakthrough(kOL: float,
                           RT_min: float,
                           cin: float,
                           col: ColumnParams,
                           CV_max: float = 50.0,
                           n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a breakthrough curve and return (CV_array, c/c0_array).

    Parameters
    ----------
    kOL     : Overall mass transfer coefficient to simulate (1/min)
    RT_min  : Residence time (min) — determines flow rate
    cin     : Feed protein concentration (mg/mL)
    col     : ColumnParams instance
    CV_max  : Maximum column volumes to simulate
    n_points: Number of time points

    Returns
    -------
    CV_sim  : np.ndarray of column volumes
    cc0_sim : np.ndarray of normalized outlet concentrations c/c0
    """
    phi   = col.flow_rate_from_RT(RT_min)
    V     = col.void_volume_mL

    t_end = CV_max * RT_min
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        fun=lambda t, y: mass_balance_odes(t, y, phi, V, cin,
                                           kOL, col.K, col.a),
        t_span=(0, t_end),
        y0=[0.0, 0.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        dense_output=True,
    )

    if not sol.success:
        return np.zeros(n_points), np.zeros(n_points)

    c_out   = sol.y[0]
    CC0_sim = c_out / cin
    CV_sim  = sol.t / RT_min   # convert time → column volumes

    return CV_sim, CC0_sim


# =============================================================================
# SECTION 4: kOL FITTING ENGINE
# =============================================================================

def objective_function(kOL_arr: np.ndarray,
                        CV_exp: np.ndarray,
                        cc0_exp: np.ndarray,
                        RT_min: float,
                        cin: float,
                        col: ColumnParams) -> float:
    """
    Sum of squared residuals between experimental and simulated c/c0.
    Used by the optimizer.
    """
    kOL = kOL_arr[0]
    if kOL <= 0:
        return 1e10

    CV_sim, cc0_sim = simulate_breakthrough(kOL, RT_min, cin, col,
                                            CV_max=max(CV_exp) * 1.1)

    # Interpolate simulated curve at experimental CV points
    cc0_interp = np.interp(CV_exp, CV_sim, cc0_sim)
    residuals   = cc0_exp - cc0_interp
    return float(np.sum(residuals**2))


def fit_kOL(CV_exp: np.ndarray,
            cc0_exp: np.ndarray,
            RT_min: float,
            cin: float,
            col: ColumnParams,
            kOL_init: float = 0.1,
            kOL_bounds: tuple = (1e-4, 10.0)) -> dict:
    """
    Fit kOL to a single experimental breakthrough curve.

    Uses scipy.optimize.minimize with the L-BFGS-B method (bounded),
    then estimates 95% confidence intervals via the inverse Hessian.

    Parameters
    ----------
    CV_exp    : Experimental column volumes array
    cc0_exp   : Experimental c/c0 array
    RT_min    : Residence time (min)
    cin       : Feed concentration (mg/mL)
    col       : ColumnParams instance
    kOL_init  : Initial guess for kOL
    kOL_bounds: (min, max) bounds for kOL

    Returns
    -------
    dict with keys: kOL_fit, kOL_95ci, R2, RMSE, success
    """
    CV_exp  = np.asarray(CV_exp,  dtype=float)
    cc0_exp = np.asarray(cc0_exp, dtype=float)

    result = minimize(
        fun=objective_function,
        x0=[kOL_init],
        args=(CV_exp, cc0_exp, RT_min, cin, col),
        method="L-BFGS-B",
        bounds=[(kOL_bounds[0], kOL_bounds[1])],
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )

    kOL_fit = float(result.x[0])

    # --- Goodness of fit ---
    CV_sim, cc0_sim = simulate_breakthrough(kOL_fit, RT_min, cin, col,
                                            CV_max=max(CV_exp) * 1.1)
    cc0_pred = np.interp(CV_exp, CV_sim, cc0_sim)
    residuals = cc0_exp - cc0_pred
    ss_res    = np.sum(residuals**2)
    ss_tot    = np.sum((cc0_exp - np.mean(cc0_exp))**2)
    R2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    RMSE      = float(np.sqrt(np.mean(residuals**2)))

    # --- 95% Confidence Interval via finite-difference Hessian ---
    h      = kOL_fit * 1e-4
    f0     = result.fun
    f_plus = objective_function([kOL_fit + h], CV_exp, cc0_exp, RT_min, cin, col)
    f_minus= objective_function([kOL_fit - h], CV_exp, cc0_exp, RT_min, cin, col)
    hessian = (f_plus - 2*f0 + f_minus) / h**2
    n      = len(CV_exp)
    p      = 1   # number of parameters
    sigma2 = ss_res / max(n - p, 1)
    if hessian > 0:
        var_kOL  = sigma2 / hessian
        ci_95    = 1.96 * np.sqrt(var_kOL)
    else:
        ci_95 = float("nan")

    return {
        "kOL_fit":  kOL_fit,
        "kOL_95ci": ci_95,
        "R2":       R2,
        "RMSE":     RMSE,
        "success":  result.success,
    }


# =============================================================================
# SECTION 5: PLOTTING
# =============================================================================

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

def plot_all_fits(experimental_data: dict,
                  fit_results: dict,
                  col: ColumnParams) -> None:
    """
    Plot experimental breakthrough curves with fitted model overlays,
    and a summary bar chart of kOL values with confidence intervals.
    """
    n_curves = len(experimental_data)
    fig = plt.figure(figsize=(14, 5 + 3 * ((n_curves - 1) // 3)))
    fig.suptitle("Breakthrough Curve Fitting — kOL Extraction\n"
                 "(Replace with Figure 1 data from your paper)",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, max(n_curves, 2),
                           hspace=0.55, wspace=0.35,
                           height_ratios=[2.5, 1.2])

    # --- Top row: individual breakthrough fits ---
    for idx, (label, data) in enumerate(experimental_data.items()):
        ax = fig.add_subplot(gs[0, idx])
        CV_exp  = np.array(data["CV"])
        cc0_exp = np.array(data["cc0"])
        RT_min  = data["RT"]
        cin     = data["cin"]
        color   = COLORS[idx % len(COLORS)]

        # Experimental
        ax.scatter(CV_exp, cc0_exp, color=color, s=40, zorder=5,
                   label="Experimental", edgecolors="black", linewidths=0.5)

        # Fitted model
        res    = fit_results[label]
        kOL_f  = res["kOL_fit"]
        CV_sim, cc0_sim = simulate_breakthrough(kOL_f, RT_min, cin, col,
                                                CV_max=max(CV_exp) * 1.05)
        ax.plot(CV_sim, cc0_sim, color=color, linewidth=2,
                label=f"Model (kOL={kOL_f:.3f})")

        # 10% breakthrough line
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xlabel("Column Volumes (CV)", fontsize=9)
        ax.set_ylabel("c / c₀  (–)", fontsize=9)
        ax.set_title(f"{label}\nR²={res['R2']:.4f}  RMSE={res['RMSE']:.4f}",
                     fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # --- Bottom row: kOL bar chart with 95% CI ---
    ax_bar = fig.add_subplot(gs[1, :])
    labels  = list(fit_results.keys())
    kOL_vals = [fit_results[l]["kOL_fit"]  for l in labels]
    kOL_cis  = [fit_results[l]["kOL_95ci"] for l in labels]
    colors_bar = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    bars = ax_bar.bar(labels, kOL_vals, color=colors_bar, alpha=0.8,
                      edgecolor="black", linewidth=0.7)
    ax_bar.errorbar(labels, kOL_vals,
                    yerr=[ci if not np.isnan(ci) else 0 for ci in kOL_cis],
                    fmt="none", color="black", capsize=5, linewidth=1.5)

    for bar, val in zip(bars, kOL_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val * 1.04,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax_bar.set_ylabel("kOL (1/min)", fontsize=10)
    ax_bar.set_title("Fitted kOL Values with 95% Confidence Intervals", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(bottom=0)

    plt.savefig("kOL_fitting_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved as: kOL_fitting_results.png")


# =============================================================================
# SECTION 6: SENSITIVITY ANALYSIS
# =============================================================================

def kOL_sensitivity_analysis(label: str,
                              data: dict,
                              col: ColumnParams,
                              kOL_center: float,
                              n_steps: int = 20) -> None:
    """
    Plot SSR landscape around the fitted kOL to visualize identifiability.
    A sharp, narrow minimum confirms the parameter is well-identifiable.
    """
    CV_exp  = np.array(data["CV"])
    cc0_exp = np.array(data["cc0"])
    RT_min  = data["RT"]
    cin     = data["cin"]

    kOL_range = np.linspace(kOL_center * 0.1, kOL_center * 3, n_steps)
    SSR_vals  = [objective_function([k], CV_exp, cc0_exp, RT_min, cin, col)
                 for k in kOL_range]

    plt.figure(figsize=(6, 4))
    plt.plot(kOL_range, SSR_vals, "-o", color="#2ca02c", linewidth=2, markersize=4)
    plt.axvline(kOL_center, color="red", linestyle="--", linewidth=1.5,
                label=f"Fitted kOL = {kOL_center:.4f}")
    plt.xlabel("kOL (1/min)")
    plt.ylabel("Sum of Squared Residuals")
    plt.title(f"Sensitivity Landscape — {label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"kOL_sensitivity_{label.replace(' ', '_')}.png", dpi=150)
    plt.show()


# =============================================================================
# SECTION 7: MAIN WORKFLOW
# =============================================================================

def main():
    col = COLUMN

    print("\n" + "=" * 60)
    print("  BREAKTHROUGH CURVE — kOL FITTING")
    print("  Replace EXPERIMENTAL_DATA with Figure 1 digitized values")
    print("=" * 60)

    fit_results = {}

    for label, data in EXPERIMENTAL_DATA.items():
        print(f"\n▶ Fitting: {label}")

        CV_exp  = np.array(data["CV"])
        cc0_exp = np.array(data["cc0"])
        RT_min  = data["RT"]
        cin     = data["cin"]

        res = fit_kOL(CV_exp, cc0_exp, RT_min, cin, col,
                      kOL_init=0.1, kOL_bounds=(1e-5, 20.0))

        fit_results[label] = res

        ci_str = (f"± {res['kOL_95ci']:.4f}" if not np.isnan(res['kOL_95ci'])
                  else "± n/a")
        print(f"   kOL (fitted) = {res['kOL_fit']:.4f}  {ci_str}  [1/min]")
        print(f"   R²           = {res['R2']:.4f}")
        print(f"   RMSE         = {res['RMSE']:.4f}")
        print(f"   Optimizer    = {'converged ✓' if res['success'] else 'did not converge ✗'}")

    # Plot all fitted curves
    plot_all_fits(EXPERIMENTAL_DATA, fit_results, col)

    # Sensitivity analysis for the first curve
    first_label = list(EXPERIMENTAL_DATA.keys())[0]
    kOL_sensitivity_analysis(
        first_label,
        EXPERIMENTAL_DATA[first_label],
        col,
        kOL_center=fit_results[first_label]["kOL_fit"],
    )

    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Condition':<18} {'kOL (1/min)':>12} {'95% CI':>12} {'R²':>8} {'RMSE':>8}")
    print("-" * 60)
    for label, res in fit_results.items():
        ci = f"±{res['kOL_95ci']:.4f}" if not np.isnan(res['kOL_95ci']) else "±n/a"
        print(f"  {label:<18} {res['kOL_fit']:>12.4f} {ci:>12} "
              f"{res['R2']:>8.4f} {res['RMSE']:>8.4f}")
    print("=" * 60)

    return fit_results


if __name__ == "__main__":
    main()
