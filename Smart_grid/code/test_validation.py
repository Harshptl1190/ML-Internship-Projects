"""
test_validation.py -- Chapter 10 test suite.
Plain assert-based tests (pytest is not installable offline). Run directly:
    python3 test_validation.py
Produces a real pass/fail table -- printed to stdout and dumped to JSON.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_optimizer import (simulate_day, pick_regime_day, BOUNDS, PARAM_NAMES,
                                 BESS_CAP_KWH, SOC_MIN, SOC_MAX, DIESEL_CAP_KW)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "microgrid_timeseries.csv"
MODELS = ROOT / "models"

df = pd.read_csv(DATA)
results = []

def check(test_id, description, condition, detail=""):
    status = "Pass" if condition else "FAIL"
    results.append({"id": test_id, "description": description, "status": status, "detail": detail})
    print(f"[{status}] {test_id}: {description}  {detail}")

# T01: data has no missing values
check("T01", "Generated dataset has zero missing values", df.isnull().sum().sum() == 0,
      f"NaN count = {df.isnull().sum().sum()}")

# T02: PV output never exceeds rated capacity
check("T02", "PV output stays within rated capacity (500 kW)", df.pv_output_kw.max() <= 500.001,
      f"max = {df.pv_output_kw.max():.1f} kW")

# T03: wind output never exceeds rated capacity
check("T03", "Wind output stays within rated capacity (300 kW)", df.wind_output_kw.max() <= 300.001,
      f"max = {df.wind_output_kw.max():.1f} kW")

# T04: tariff values fall in the 4 declared ToU slabs
check("T04", "Grid tariff matches declared ToU slab set", set(df.grid_tariff_inr_per_kwh.unique()) <= {3.75, 8.20, 9.50, 6.10},
      f"unique values = {sorted(df.grid_tariff_inr_per_kwh.unique())}")

# T05: default policy respects battery power bounds during a normal sunny day
sunny = pick_regime_day(df, "Sunny")
mid_theta = BOUNDS.mean(axis=1)
cost, breakdown, log = simulate_day(mid_theta, sunny)
check("T05", "Battery power stays within rated charge/discharge limits", (log.batt_kw.abs() <= 200.001).all(),
      f"max |batt_kw| = {log.batt_kw.abs().max():.1f} kW")

# T06: SOC never leaves [min,max] fraction bounds
soc_ok = log.soc_frac.between(SOC_MIN - 1e-6, SOC_MAX + 1e-6).all()
check("T06", "State of Charge remains within [15%, 95%] bounds", soc_ok,
      f"SOC range = [{log.soc_frac.min():.3f}, {log.soc_frac.max():.3f}]")

# T07: energy balance -- generation (net of curtailment) + storage + diesel + grid meets demand exactly
supply = (sunny.pv_output_kw.values + sunny.wind_output_kw.values - log.curtailed_kw.values
          + log.batt_kw.values + log.diesel_kw.values + log.grid_import.values)
demand = sunny.load_demand_kw.values
imbalance = np.abs(supply - demand).max()
check("T07", "Hourly energy balance holds (generation - curtailment + storage + diesel + grid == demand)",
      imbalance < 0.5, f"max imbalance = {imbalance:.4f} kW")

# T08: unknown regime raises ValueError
try:
    pick_regime_day(df, "Hurricane")
    ok = False
except ValueError:
    ok = True
check("T08", "Unknown weather regime raises ValueError", ok)

# T09: Overcast day converged cost is higher than Sunny/High-Wind converged cost
with open(MODELS / "optimizer_report.json") as f:
    opt = json.load(f)
check("T09", "Overcast day has higher optimised cost than Sunny/High-Wind (physical sanity)",
      opt["Overcast"]["best_cost_inr"] > opt["Sunny"]["best_cost_inr"] > 0
      and opt["Overcast"]["best_cost_inr"] > opt["High Wind"]["best_cost_inr"],
      f"Sunny={opt['Sunny']['best_cost_inr']}, HighWind={opt['High Wind']['best_cost_inr']}, "
      f"Overcast={opt['Overcast']['best_cost_inr']}")

# T10: degradation cost increases with deg_weight parameter (monotonicity)
theta_lo = mid_theta.copy(); theta_lo[6] = 0.0
theta_hi = mid_theta.copy(); theta_hi[6] = 3.0
_, b_lo, _ = simulate_day(theta_lo, sunny)
_, b_hi, _ = simulate_day(theta_hi, sunny)
check("T10", "Degradation cost increases monotonically with deg_weight",
      b_hi["degradation"] >= b_lo["degradation"],
      f"deg_cost(w=0)={b_lo['degradation']:.2f}, deg_cost(w=3)={b_hi['degradation']:.2f}")

# T11: STRESS TEST -- extreme overcast + demand spike must trigger diesel backup
stress_day = sunny.copy()
stress_day["solar_irradiance_wm2"] = 0.0
stress_day["pv_output_kw"] = 0.0
stress_day["wind_output_kw"] = stress_day["wind_output_kw"] * 0.05
stress_day["load_demand_kw"] = stress_day["load_demand_kw"] * 1.6
_, b_stress, log_stress = simulate_day(mid_theta, stress_day)
diesel_engaged = (log_stress.diesel_kw > 0).any()
check("T11", "Diesel backup engages under extreme overcast + 60% demand spike", diesel_engaged,
      f"peak diesel = {log_stress.diesel_kw.max():.1f} kW, CO2 = {b_stress['co2_kg']:.1f} kg")

# T12: diesel dispatch never exceeds rated genset capacity even under stress
check("T12", "Diesel dispatch respects rated genset capacity under stress", (log_stress.diesel_kw <= DIESEL_CAP_KW + 1e-6).all(),
      f"max diesel = {log_stress.diesel_kw.max():.1f} / {DIESEL_CAP_KW} kW")

# T13: grid import is never negative
check("T13", "Grid import is never negative", (log.grid_import >= -1e-6).all())

# T14: reserve_soc parameter prevents full battery depletion
theta_reserve = mid_theta.copy(); theta_reserve[4] = 0.4
_, _, log_r = simulate_day(theta_reserve, sunny)
check("T14", "High reserve_soc setting keeps battery above the reserve floor", log_r.soc_frac.min() >= 0.4 - 0.02,
      f"min SOC observed = {log_r.soc_frac.min():.3f} (reserve = 0.40)")

# T15: best-cost-so-far convergence curve is monotonically non-increasing (BO correctness)
from dispatch_optimizer import bayesian_optimize
res = bayesian_optimize(sunny, n_init=10, n_iter=15, seed=1)
curve = res["convergence"]
check("T15", "BO best-cost-so-far curve is monotonically non-increasing", np.all(np.diff(curve) <= 1e-9),
      f"curve range = [{curve.min():.1f}, {curve.max():.1f}]")

n_pass = sum(1 for r in results if r["status"] == "Pass")
print(f"\n{n_pass}/{len(results)} tests passed.")
with open(MODELS / "test_results.json", "w") as f:
    json.dump(results, f, indent=2)
