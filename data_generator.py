"""
data_generator.py
Generates a physically-plausible synthetic dataset for the Smart Microgrid
Renewable Energy Dispatch & Storage Optimizer project.

Since no real SCADA / meteorological feed was available for this training
project, weather and load series are synthesised from standard engineering
models (clear-sky solar model, Weibull-perturbed wind, composite daily load
curve) plus stochastic noise. This is documented explicitly in the report
(Chapter 4) rather than presented as measured field data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

N_DAYS = 120
HOURS = N_DAYS * 24
PV_CAPACITY_KW = 500.0
WIND_CAPACITY_KW = 300.0
DIESEL_CAPACITY_KW = 250.0

def clear_sky_irradiance(hour_of_day, day_of_year):
    """Simple cosine clear-sky model, W/m^2, peaking ~1000 W/m^2 at solar noon,
    modulated by a slow seasonal envelope."""
    solar_time = hour_of_day - 12.0
    daylight = np.clip(np.cos(solar_time * np.pi / 12.0), 0, None)
    seasonal = 0.85 + 0.15 * np.cos(2 * np.pi * (day_of_year - 172) / 365.0)
    return 1000.0 * daylight ** 1.3 * seasonal

def generate():
    idx = np.arange(HOURS)
    hour_of_day = idx % 24
    day_of_year = (idx // 24) % 365
    day_of_week = (idx // 24) % 7

    # ---- Weather --------------------------------------------------
    clear_sky = clear_sky_irradiance(hour_of_day, day_of_year)
    # cloud attenuation: autocorrelated noise so cloudy spells last hours, not seconds
    cloud_noise = np.zeros(HOURS)
    state = 1.0
    for i in range(HOURS):
        state = 0.9 * state + 0.1 * RNG.normal(1.0, 0.35)
        state = np.clip(state, 0.05, 1.15)
        cloud_noise[i] = state
    irradiance = np.clip(clear_sky * cloud_noise, 0, 1100)

    ambient_temp = (
        18 + 9 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)   # seasonal
        + 6 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)        # diurnal
        + RNG.normal(0, 1.2, HOURS)
    )
    humidity = np.clip(
        55 + 20 * np.cos(2 * np.pi * (hour_of_day - 4) / 24.0) + RNG.normal(0, 6, HOURS),
        15, 98,
    )

    wind_state = np.zeros(HOURS)
    w = 6.0
    for i in range(HOURS):
        w = 0.85 * w + 0.15 * RNG.weibull(2.0) * 7.5
        wind_state[i] = w
    wind_speed = np.clip(wind_state, 0, 25)

    # ---- Renewable yield (physical conversion, used as ground truth) ----
    # PV: derated for temperature above 25C (typical -0.4%/C coefficient)
    pv_derate = 1 - 0.004 * np.clip(ambient_temp - 25, 0, None)
    pv_kw = np.clip(PV_CAPACITY_KW * (irradiance / 1000.0) * pv_derate, 0, PV_CAPACITY_KW)

    # Wind turbine power curve: cut-in 3 m/s, rated 12 m/s, cut-out 22 m/s
    wind_kw = np.zeros(HOURS)
    cutin, rated, cutout = 3.0, 12.0, 22.0
    ramp = (wind_speed - cutin) / (rated - cutin)
    wind_kw = np.where(
        (wind_speed >= cutin) & (wind_speed < rated),
        WIND_CAPACITY_KW * np.clip(ramp, 0, 1) ** 3,
        wind_kw,
    )
    wind_kw = np.where((wind_speed >= rated) & (wind_speed < cutout), WIND_CAPACITY_KW, wind_kw)
    wind_kw = np.where(wind_speed >= cutout, 0.0, wind_kw)

    # ---- Load demand: commercial/industrial composite daily curve ----
    base_load = 260 + 40 * np.sin(2 * np.pi * (day_of_year - 200) / 365.0)
    daily_shape = (
        0.55
        + 0.35 * np.exp(-((hour_of_day - 10) ** 2) / (2 * 3.0 ** 2))
        + 0.30 * np.exp(-((hour_of_day - 19) ** 2) / (2 * 2.5 ** 2))
    )
    weekend_factor = np.where(day_of_week >= 5, 0.78, 1.0)
    load_kw = base_load * daily_shape * weekend_factor + RNG.normal(0, 8, HOURS)
    load_kw = np.clip(load_kw, 40, None)

    # ---- Time-of-Use grid tariff (INR/kWh equivalent, generic 3-slab ToU) ----
    tariff = np.select(
        [
            (hour_of_day >= 18) & (hour_of_day < 22),          # evening peak
            (hour_of_day >= 6) & (hour_of_day < 9),             # morning peak
            (hour_of_day >= 0) & (hour_of_day < 6),              # off-peak night
        ],
        [9.50, 8.20, 3.75],
        default=6.10,   # shoulder
    )

    df = pd.DataFrame({
        "timestamp_h": idx,
        "day_of_year": day_of_year,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "solar_irradiance_wm2": irradiance.round(2),
        "wind_speed_ms": wind_speed.round(2),
        "ambient_temp_c": ambient_temp.round(2),
        "humidity_pct": humidity.round(2),
        "pv_output_kw": pv_kw.round(3),
        "wind_output_kw": wind_kw.round(3),
        "load_demand_kw": load_kw.round(3),
        "grid_tariff_inr_per_kwh": tariff,
    })
    df["net_demand_kw"] = (df["load_demand_kw"] - df["pv_output_kw"] - df["wind_output_kw"]).round(3)
    df["pv_ramp_kw"] = df["pv_output_kw"].diff().fillna(0).round(3)
    df["load_ramp_kw"] = df["load_demand_kw"].diff().fillna(0).round(3)

    df.to_csv(OUT / "microgrid_timeseries.csv", index=False)

    specs = pd.DataFrame([
        {"component": "Solar PV array", "parameter": "rated_capacity_kw", "value": PV_CAPACITY_KW},
        {"component": "Wind turbine bank", "parameter": "rated_capacity_kw", "value": WIND_CAPACITY_KW},
        {"component": "Battery Energy Storage (BESS)", "parameter": "usable_capacity_kwh", "value": 800.0},
        {"component": "Battery Energy Storage (BESS)", "parameter": "max_charge_power_kw", "value": 200.0},
        {"component": "Battery Energy Storage (BESS)", "parameter": "max_discharge_power_kw", "value": 200.0},
        {"component": "Battery Energy Storage (BESS)", "parameter": "round_trip_efficiency", "value": 0.92},
        {"component": "Battery Energy Storage (BESS)", "parameter": "min_soc_fraction", "value": 0.15},
        {"component": "Battery Energy Storage (BESS)", "parameter": "max_soc_fraction", "value": 0.95},
        {"component": "Battery Energy Storage (BESS)", "parameter": "cycle_life_at_80pct_dod", "value": 4500},
        {"component": "Battery Energy Storage (BESS)", "parameter": "replacement_cost_inr_per_kwh", "value": 9500},
        {"component": "Diesel/gas backup genset", "parameter": "rated_capacity_kw", "value": DIESEL_CAPACITY_KW},
        {"component": "Diesel/gas backup genset", "parameter": "fuel_cost_inr_per_kwh", "value": 21.5},
        {"component": "Diesel/gas backup genset", "parameter": "co2_emission_kg_per_kwh", "value": 0.82},
    ])
    specs.to_csv(OUT / "microgrid_specs.csv", index=False)
    print("Generated:", len(df), "hourly rows across", N_DAYS, "days")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    return df, specs

if __name__ == "__main__":
    generate()
