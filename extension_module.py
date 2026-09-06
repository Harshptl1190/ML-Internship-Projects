"""extension_module.py -- Chapter 7 figures: degradation sensitivity & cost breakdown."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
MODELS = ROOT / "models"

with open(MODELS / "optimizer_report.json") as f:
    opt = json.load(f)

# --- Fig 7.1: battery cycle-wear cost vs depth-of-discharge (physical model) ---
dod = np.linspace(0.05, 0.95, 100)
cycle_life = 4500 * (0.8 / dod) ** 1.3   # simple inverse power-law DoD-vs-cycle-life relation
repl_cost_per_kwh = 9500
cap_kwh = 800
cost_per_cycle_kwh_throughput = repl_cost_per_kwh / (2 * cycle_life)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dod * 100, cost_per_cycle_kwh_throughput * 100, color="#8a1f1f")
ax.set_xlabel("Depth of Discharge (%)")
ax.set_ylabel("Degradation Cost (INR per 100 kWh throughput)")
ax.set_title("Fig 7.1  BESS Degradation Cost vs Depth-of-Discharge", fontweight="bold", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "fig7_1_degradation_curve.png", dpi=200)
plt.close()

# --- Fig 7.2: stacked cost breakdown across regimes ---
regimes = list(opt.keys())
components = ["grid", "diesel", "degradation", "carbon"]
colors = ["#2b6cd9", "#8a1f1f", "#d99a2b", "#2f8f4e"]
data = {c: [opt[r]["cost_breakdown_inr"][c] for r in regimes] for c in components}

fig, ax = plt.subplots(figsize=(7.5, 4.3))
bottom = np.zeros(len(regimes))
for c, col in zip(components, colors):
    vals = np.array(data[c])
    ax.bar(regimes, vals, bottom=bottom, label=c.capitalize(), color=col)
    bottom += vals
ax.set_ylabel("Cost (INR / day)")
ax.set_title("Fig 7.2  Optimised Cost Breakdown by Component and Weather Regime", fontweight="bold", fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "fig7_2_cost_breakdown.png", dpi=200)
plt.close()

print("Extension figures written.")
print(json.dumps(data, indent=2))
