"""
dispatch_optimizer.py
Bayesian Dispatch & Storage Optimizer (Chapter 6) + Extension Module (Chapter 7).

Note on backend choice: scikit-optimize (skopt.gp_minimize) was specified in
the brief but is not installable in this offline sandbox. A from-scratch
Bayesian Optimisation loop is implemented instead using
sklearn.gaussian_process.GaussianProcessRegressor (Matern kernel) as the
surrogate and an Expected Improvement acquisition function -- the same
GP + EI algorithm skopt.gp_minimize wraps. All convergence curves below are
produced by genuinely running this loop, not hand-authored.

Search-space design: rather than optimising 24 independent hourly battery
setpoints directly (which is a very high-dimensional, sample-inefficient
space for a ~60-evaluation BO budget), the dispatch is parameterised by a
small set of policy parameters that a rule-based dispatch simulator expands
into an hourly schedule. This is a standard way to make BO tractable for
control problems and is documented as a deliberate design choice.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "microgrid_timeseries.csv"
SPECS = ROOT / "data" / "microgrid_specs.csv"
FIG = ROOT / "figures"
MODELS = ROOT / "models"

# ---------------- Microgrid physical parameters (from specs table) --------
specs_df = pd.read_csv(SPECS)
S = {row.parameter: row.value for row in specs_df.itertuples()}

BESS_CAP_KWH = S["usable_capacity_kwh"]
BESS_CHG_KW = S["max_charge_power_kw"]
BESS_DIS_KW = S["max_discharge_power_kw"]
BESS_EFF = S["round_trip_efficiency"] ** 0.5   # per-direction efficiency
SOC_MIN = S["min_soc_fraction"]
SOC_MAX = S["max_soc_fraction"]
CYCLE_LIFE = S["cycle_life_at_80pct_dod"]
BESS_REPL_COST = S["replacement_cost_inr_per_kwh"]
DIESEL_CAP_KW = S["rated_capacity_kw"]
DIESEL_FUEL_COST = S["fuel_cost_inr_per_kwh"]
DIESEL_CO2 = S["co2_emission_kg_per_kwh"]

CARBON_PRICE_INR_PER_KG = 2.0     # shadow price used to weight emissions in the cost
DEG_COST_PER_KWH_THROUGHPUT = BESS_REPL_COST / (2 * CYCLE_LIFE * 0.8)  # INR per kWh cycled

# ---------------- Search space: 7 policy parameters ------------------------
# [charge_price_threshold, discharge_price_threshold, max_charge_frac,
#  max_discharge_frac, reserve_soc, diesel_trigger_frac, deg_weight]
BOUNDS = np.array([
    [3.0, 6.5],     # charge below this tariff (INR/kWh)
    [6.0, 9.5],     # discharge above this tariff (INR/kWh)
    [0.3, 1.0],     # fraction of max charge power usable
    [0.3, 1.0],     # fraction of max discharge power usable
    [0.15, 0.45],   # reserve SOC fraction kept for emergencies
    [0.6, 1.0],     # net-demand fraction of diesel capacity that triggers diesel
    [0.0, 3.0],     # relative weight on battery degradation cost
])
PARAM_NAMES = ["charge_price_thr", "discharge_price_thr", "max_chg_frac",
               "max_dis_frac", "reserve_soc", "diesel_trigger_frac", "deg_weight"]


def simulate_day(theta, day_df, w_drag_unused=None):
    """Rule-based dispatch simulator driven by policy parameters `theta`.
    Returns total 24h cost (INR) and a full hourly log for inspection."""
    (charge_thr, discharge_thr, max_chg_frac, max_dis_frac,
     reserve_soc, diesel_trigger_frac, deg_weight) = theta

    soc = 0.5 * BESS_CAP_KWH   # start at 50% SOC
    total_cost = 0.0
    total_co2 = 0.0
    total_deg_cost = 0.0
    total_grid_cost = 0.0
    total_diesel_cost = 0.0
    log = []

    for _, row in day_df.iterrows():
        net = row.load_demand_kw - row.pv_output_kw - row.wind_output_kw  # +ve = deficit
        tariff = row.grid_tariff_inr_per_kwh
        batt_kw = 0.0   # +ve = discharge, -ve = charge
        diesel_kw = 0.0

        soc_frac = soc / BESS_CAP_KWH

        curtailed_kw = 0.0
        if net < 0:
            # surplus renewable -> store what the battery can absorb; the rest is curtailed
            surplus = -net
            room = (SOC_MAX * BESS_CAP_KWH - soc) / BESS_EFF
            chg = min(surplus, BESS_CHG_KW * max_chg_frac, room)
            chg = max(chg, 0.0)
            curtailed_kw = surplus - chg
            batt_kw = -chg
            soc += chg * BESS_EFF
            grid_import = 0.0
        else:
            deficit = net
            # 1) discharge battery if tariff is expensive enough and SOC above reserve
            available = max(0.0, (soc - reserve_soc * BESS_CAP_KWH) * BESS_EFF)
            if tariff >= discharge_thr and available > 0:
                dis = min(deficit, BESS_DIS_KW * max_dis_frac, available)
                batt_kw = dis
                soc -= dis / BESS_EFF
                deficit -= dis
            # 2) if remaining deficit is large, use diesel backup
            if deficit > diesel_trigger_frac * DIESEL_CAP_KW:
                diesel_kw = min(deficit, DIESEL_CAP_KW)
                deficit -= diesel_kw
            grid_import = max(deficit, 0.0)

        # cheap-tariff opportunistic charging from grid (only if very cheap & SOC low)
        if tariff <= charge_thr and soc_frac < SOC_MAX and net >= 0 and diesel_kw == 0:
            room = (SOC_MAX * BESS_CAP_KWH - soc) / BESS_EFF
            extra_chg = min(BESS_CHG_KW * max_chg_frac * 0.3, room)
            if extra_chg > 0:
                soc += extra_chg * BESS_EFF
                grid_import += extra_chg
                batt_kw -= extra_chg   # this energy goes to storage, not to serving load

        grid_cost = grid_import * tariff
        diesel_cost = diesel_kw * DIESEL_FUEL_COST
        co2 = diesel_kw * DIESEL_CO2
        deg_cost = abs(batt_kw) * DEG_COST_PER_KWH_THROUGHPUT * deg_weight

        total_grid_cost += grid_cost
        total_diesel_cost += diesel_cost
        total_co2 += co2
        total_deg_cost += deg_cost
        total_cost += grid_cost + diesel_cost + deg_cost + co2 * CARBON_PRICE_INR_PER_KG

        log.append(dict(hour=row.hour_of_day, net_demand=net, batt_kw=batt_kw,
                         diesel_kw=diesel_kw, grid_import=grid_import, curtailed_kw=curtailed_kw,
                         soc_frac=soc / BESS_CAP_KWH, tariff=tariff))

    breakdown = dict(grid=total_grid_cost, diesel=total_diesel_cost,
                      degradation=total_deg_cost, carbon=total_co2 * CARBON_PRICE_INR_PER_KG,
                      co2_kg=total_co2)
    return total_cost, breakdown, pd.DataFrame(log)


def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-8] = 0.0
    return ei


def bayesian_optimize(day_df, n_init=15, n_iter=65, seed=0):
    rng = np.random.default_rng(seed)
    dim = BOUNDS.shape[0]

    def unit_to_real(u):
        return BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])

    X_unit = rng.random((n_init, dim))
    X = np.array([unit_to_real(u) for u in X_unit])
    y = np.array([simulate_day(x, day_df)[0] for x in X])

    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(1e-3)
    best_curve = [y.min()]

    for it in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=2, random_state=seed)
        gp.fit(X_unit, y)

        cand_unit = rng.random((2000, dim))
        ei = expected_improvement(cand_unit, gp, y.min())
        next_unit = cand_unit[np.argmax(ei)]
        next_x = unit_to_real(next_unit)
        next_y, _, _ = simulate_day(next_x, day_df)

        X = np.vstack([X, next_x])
        X_unit = np.vstack([X_unit, next_unit])
        y = np.append(y, next_y)
        best_curve.append(y.min())

    best_idx = np.argmin(y)
    return dict(best_theta=X[best_idx], best_cost=y[best_idx],
                convergence=np.array(best_curve), all_X=X, all_y=y)


def pick_regime_day(df, regime):
    daily = df.groupby(df["timestamp_h"] // 24).agg(
        irr=("solar_irradiance_wm2", "mean"), wind=("wind_speed_ms", "mean")
    )
    if regime == "Sunny":
        day_id = daily["irr"].idxmax()
    elif regime == "High Wind":
        day_id = daily["wind"].idxmax()
    elif regime == "Overcast":
        day_id = daily["irr"].idxmin()
    else:
        raise ValueError(regime)
    return df[df["timestamp_h"] // 24 == day_id].reset_index(drop=True)


def run():
    df = pd.read_csv(DATA)
    regimes = ["Sunny", "High Wind", "Overcast"]
    results = {}
    for regime in regimes:
        day_df = pick_regime_day(df, regime)
        res = bayesian_optimize(day_df, n_init=15, n_iter=65, seed=hash(regime) % 1000)
        results[regime] = res
        print(f"{regime}: converged cost = {res['best_cost']:.1f} INR/day, "
              f"theta = {np.round(res['best_theta'], 3).tolist()}")

    # --- Convergence plot: 3 subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"Sunny": "#d99a2b", "High Wind": "#2b6cd9", "Overcast": "#6b6b6b"}
    for ax, regime in zip(axes, regimes):
        curve = results[regime]["convergence"]
        ax.plot(curve, color=colors[regime])
        ax.set_title(f"{regime}\nconverged \u2248 {curve[-1]:.0f} INR", fontsize=9)
        ax.set_xlabel("Evaluation Number")
        ax.set_ylabel("Best Cost Found (INR/day)")
    plt.suptitle("Fig 6.1\u20136.3  Bayesian Optimisation Convergence Across Weather Regimes", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "fig6_convergence.png", dpi=200)
    plt.close()

    # --- Dispatch heatmap for the Sunny day (best theta) ---
    sunny_day = pick_regime_day(df, "Sunny")
    _, breakdown, log = simulate_day(results["Sunny"]["best_theta"], sunny_day)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    mat = np.vstack([log.batt_kw.values, log.diesel_kw.values, log.grid_import.values])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Battery (+dis/-chg, kW)", "Diesel (kW)", "Grid import (kW)"])
    ax.set_xlabel("Hour of day")
    ax.set_title("Fig 6.4  Optimised Hourly Dispatch Heatmap \u2013 Sunny Day", fontweight="bold", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG / "fig6_4_dispatch_heatmap.png", dpi=200)
    plt.close()

    # --- Comparison bar chart across regimes ---
    fig, ax = plt.subplots(figsize=(7, 4))
    costs = [results[r]["best_cost"] for r in regimes]
    bars = ax.bar(regimes, costs, color=[colors[r] for r in regimes])
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 5, f"{c:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Optimised 24h Operating Cost (INR)")
    ax.set_title("Fig 6.5  Optimal Dispatch Cost Comparison Across Weather Regimes", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "fig6_5_regime_comparison.png", dpi=200)
    plt.close()

    out = {}
    for r in regimes:
        theta = results[r]["best_theta"]
        _, breakdown, _ = simulate_day(theta, pick_regime_day(df, r))
        out[r] = {
            "best_cost_inr": round(float(results[r]["best_cost"]), 2),
            "theta": {n: round(float(v), 4) for n, v in zip(PARAM_NAMES, theta)},
            "cost_breakdown_inr": {k: round(float(v), 2) for k, v in breakdown.items()},
        }
    with open(MODELS / "optimizer_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
