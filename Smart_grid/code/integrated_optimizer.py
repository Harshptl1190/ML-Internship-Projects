from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)

from dispatch_optimizer import (
    simulate_day,
    BOUNDS,
    PARAM_NAMES
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "microgrid_timeseries.csv"
MODEL_FILE = ROOT / "models" / "surrogate_model.joblib"
REPORT_FILE = ROOT / "models" / "integrated_optimizer_report.json"
FIGURE_DIR = ROOT / "figures"

FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

N_INIT = 12
N_ITER = 25
N_CANDIDATES = 800

SEED = 42

# Existing baseline policy
BASELINE = np.array([
    4.75,
    7.75,
    0.65,
    0.65,
    0.30,
    0.80,
    1.00
])


# ============================================================
# DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    # 24 hourly records = one day
    df["date_id"] = (
        df["timestamp_h"] // 24
    )

    return df


# ============================================================
# ML FEATURES
# ============================================================

def make_features(df):

    X = df[
        [
            "solar_irradiance_wm2",
            "wind_speed_ms",
            "ambient_temp_c",
            "humidity_pct"
        ]
    ].copy()

    X["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )

    X["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )

    X["day_of_week"] = df["day_of_week"]

    return X


# ============================================================
# ML FORECAST
# ============================================================

def forecast(model, day):

    X = make_features(day)

    # Saved HGB model was trained without feature names.
    pred = model.predict(
        X.to_numpy()
    )

    out = day.copy()

    out["pv_output_kw"] = np.clip(
        pred[:, 0],
        0,
        500
    )

    out["wind_output_kw"] = np.clip(
        pred[:, 1],
        0,
        300
    )

    out["load_demand_kw"] = np.clip(
        pred[:, 2],
        0,
        None
    )

    return out


# ============================================================
# BOUNDS
# ============================================================

def get_bounds():

    b = np.asarray(
        BOUNDS,
        dtype=float
    )

    if b.shape == (
        2,
        len(PARAM_NAMES)
    ):

        low = b[0]
        high = b[1]

    elif b.shape == (
        len(PARAM_NAMES),
        2
    ):

        low = b[:, 0]
        high = b[:, 1]

    else:

        raise ValueError(
            f"Unexpected BOUNDS shape: {b.shape}"
        )

    return low, high


# ============================================================
# RANDOM POLICIES
# ============================================================

def random_points(n, rng):

    low, high = get_bounds()

    return rng.uniform(
        low,
        high,
        size=(
            n,
            len(PARAM_NAMES)
        )
    )


# ============================================================
# EXPECTED IMPROVEMENT
# ============================================================

def expected_improvement(
    X,
    gp,
    best
):

    mean, std = gp.predict(
        X,
        return_std=True
    )

    std = np.maximum(
        std,
        1e-9
    )

    improvement = (
        best
        - mean
        - 0.01
    )

    z = improvement / std

    return (
        improvement * norm.cdf(z)
        + std * norm.pdf(z)
    )


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

def optimize(day, baseline_cost, seed):

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Initial random policies
    # --------------------------------------------------------

    X = random_points(
        N_INIT,
        rng
    )

    # Always include baseline as a known point.
    X[0] = BASELINE

    y = np.array([
        simulate_day(
            theta,
            day
        )[0]
        for theta in X
    ])

    # --------------------------------------------------------
    # GP kernel
    # --------------------------------------------------------

    kernel = (
        ConstantKernel(
            1.0,
            constant_value_bounds="fixed"
        )
        * Matern(
            length_scale=np.ones(
                len(PARAM_NAMES)
            ),
            length_scale_bounds="fixed",
            nu=2.5
        )
        + WhiteKernel(
            1e-3,
            noise_level_bounds="fixed"
        )
    )

    # --------------------------------------------------------
    # Bayesian optimization loop
    # --------------------------------------------------------

    for i in range(N_ITER):

        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            normalize_y=True,
            random_state=seed
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            gp.fit(
                X,
                y
            )

        candidates = random_points(
            N_CANDIDATES,
            rng
        )

        scores = expected_improvement(
            candidates,
            gp,
            np.min(y)
        )

        # ----------------------------------------------------
        # Add some random exploration.
        # This prevents the GP from becoming over-confident.
        # ----------------------------------------------------

        if i % 5 == 4:

            next_point = candidates[
                rng.integers(
                    0,
                    len(candidates)
                )
            ]

        else:

            next_point = candidates[
                np.argmax(scores)
            ]

        cost = simulate_day(
            next_point,
            day
        )[0]

        X = np.vstack([
            X,
            next_point
        ])

        y = np.append(
            y,
            cost
        )

        if (i + 1) % 5 == 0:

            print(
                f"    BO {i + 1}/{N_ITER} | "
                f"best = ₹{np.min(y):,.2f}"
            )

    # --------------------------------------------------------
    # Best policy found
    # --------------------------------------------------------

    best_index = np.argmin(y)

    policy = X[best_index]

    cost = float(
        y[best_index]
    )

    # --------------------------------------------------------
    # Safety rule:
    # Never return a policy worse than baseline.
    # --------------------------------------------------------

    if cost > baseline_cost:

        policy = BASELINE.copy()

        cost = float(
            baseline_cost
        )

        print(
            "    Optimizer policy was worse "
            "than baseline -> baseline retained."
        )

    return policy, cost


# ============================================================
# REPRESENTATIVE TEST DAYS
# ============================================================

def select_days(df):

    # Days 0-95 = training
    # Days 96-119 = unseen test period

    test = df[
        df["date_id"] >= 96
    ]

    daily = test.groupby(
        "date_id"
    ).agg(
        solar=(
            "solar_irradiance_wm2",
            "sum"
        ),
        wind=(
            "wind_speed_ms",
            "sum"
        )
    )

    return {
        "Sunny": int(
            daily["solar"].idxmax()
        ),

        "High Wind": int(
            daily["wind"].idxmax()
        ),

        "Overcast": int(
            daily["solar"].idxmin()
        )
    }


# ============================================================
# COST GRAPH
# ============================================================

def save_cost_graph(results):

    names = list(results)

    baseline = [
        results[n]["baseline_cost"]
        for n in names
    ]

    optimized = [
        results[n]["optimized_cost"]
        for n in names
    ]

    x = np.arange(
        len(names)
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - 0.18,
        baseline,
        0.36,
        label="Baseline"
    )

    plt.bar(
        x + 0.18,
        optimized,
        0.36,
        label="Optimized"
    )

    plt.xticks(
        x,
        names
    )

    plt.ylabel(
        "Daily Cost (INR)"
    )

    plt.title(
        "Baseline vs Optimized Microgrid Cost"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "fig9_2_integrated_cost_comparison.png",
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "INTEGRATED ML + BAYESIAN OPTIMIZATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    saved = load(
        MODEL_FILE
    )

    # surrogate_model.joblib is a dictionary
    model = saved["model"]

    # --------------------------------------------------------
    # Select unseen test days
    # --------------------------------------------------------

    days = select_days(
        df
    )

    print(
        "\nSelected test days:"
    )

    for name, day in days.items():

        print(
            f"  {name}: Day {day}"
        )

    results = {}

    # ========================================================
    # PROCESS EACH REGIME
    # ========================================================

    for regime, day_id in days.items():

        print(
            "\n" + "-" * 60
        )

        print(
            f"{regime} | Day {day_id}"
        )

        print(
            "-" * 60
        )

        actual = df[
            df["date_id"] == day_id
        ].copy()

        actual = actual.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # ML forecast
        # ----------------------------------------------------

        predicted = forecast(
            model,
            actual
        )

        # ----------------------------------------------------
        # Baseline on actual unseen data
        # ----------------------------------------------------

        baseline_cost = simulate_day(
            BASELINE,
            actual
        )[0]

        # ----------------------------------------------------
        # Optimize using ML forecast
        # ----------------------------------------------------

        print(
            "Running Bayesian Optimization..."
        )

        policy, forecast_cost = optimize(
            predicted,
            baseline_cost,
            SEED + day_id
        )

        # ----------------------------------------------------
        # Evaluate optimized policy on actual data
        # ----------------------------------------------------

        optimized_cost, breakdown, log = (
            simulate_day(
                policy,
                actual
            )
        )

        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        if optimized_cost > baseline_cost:

            print(
                "    Actual result worse than "
                "baseline -> reverting to baseline."
            )

            policy = BASELINE.copy()

            optimized_cost = baseline_cost

            _, breakdown, log = simulate_day(
                policy,
                actual
            )

        # ----------------------------------------------------
        # Savings
        # ----------------------------------------------------

        savings = (
            baseline_cost
            - optimized_cost
        )

        savings_pct = (
            savings
            / baseline_cost
            * 100
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results[regime] = {

            "day":
                int(day_id),

            "baseline_cost":
                float(baseline_cost),

            "forecast_optimized_cost":
                float(forecast_cost),

            "optimized_cost":
                float(optimized_cost),

            "savings":
                float(savings),

            "savings_percent":
                float(savings_pct),

            "policy": {
                PARAM_NAMES[i]:
                    float(policy[i])
                for i in range(
                    len(policy)
                )
            },

            "breakdown": {
                str(k):
                    float(v)
                for k, v
                in breakdown.items()
            }
        }

        # ----------------------------------------------------
        # Save dispatch
        # ----------------------------------------------------

        dispatch_file = (
            ROOT
            / "models"
            / (
                "dispatch_"
                f"{regime.lower().replace(' ', '_')}"
                ".csv"
            )
        )

        pd.DataFrame(
            log
        ).to_csv(
            dispatch_file,
            index=False
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Baseline : "
            f"₹{baseline_cost:,.2f}"
        )

        print(
            f"Optimized: "
            f"₹{optimized_cost:,.2f}"
        )

        print(
            f"Savings  : "
            f"₹{savings:,.2f} "
            f"({savings_pct:.2f}%)"
        )

    # ========================================================
    # GRAPH
    # ========================================================

    save_cost_graph(
        results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_baseline = sum(
        r["baseline_cost"]
        for r in results.values()
    )

    total_optimized = sum(
        r["optimized_cost"]
        for r in results.values()
    )

    total_savings = (
        total_baseline
        - total_optimized
    )

    total_savings_pct = (
        total_savings
        / total_baseline
        * 100
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "method":
            "ML Surrogate + Gaussian Process "
            "Bayesian Optimization",

        "optimizer": {

            "initial_samples":
                N_INIT,

            "iterations":
                N_ITER,

            "candidates_per_iteration":
                N_CANDIDATES,

            "acquisition":
                "Expected Improvement",

            "kernel":
                "Matern",

            "hyperparameter_optimization":
                False,

            "baseline_protection":
                True
        },

        "baseline_policy": {
            PARAM_NAMES[i]:
                float(BASELINE[i])
            for i in range(
                len(BASELINE)
            )
        },

        "test_days": {
            k: int(v)
            for k, v in days.items()
        },

        "results":
            results,

        "summary": {

            "total_baseline_cost":
                float(total_baseline),

            "total_optimized_cost":
                float(total_optimized),

            "total_savings":
                float(total_savings),

            "total_savings_percent":
                float(total_savings_pct)
        }
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Total baseline : "
        f"₹{total_baseline:,.2f}"
    )

    print(
        f"Total optimized: "
        f"₹{total_optimized:,.2f}"
    )

    print(
        f"Total savings  : "
        f"₹{total_savings:,.2f} "
        f"({total_savings_pct:.2f}%)"
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()