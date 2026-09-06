"""
surrogate_model.py

Smart Microgrid Renewable Yield & Load Surrogate Model

Version 2.2

Features:
- Chronological train/test split
- Random Forest comparison
- Histogram Gradient Boosting comparison
- PV, wind and load prediction
- Fast feature-importance visualization
- Predicted vs actual visualization
- Saves trained model for integrated optimizer
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "microgrid_timeseries.csv"
FIG = ROOT / "figures"
MODELS = ROOT / "models"

FIG.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "pv_output_kw",
    "wind_output_kw",
    "load_demand_kw"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_cyclical_features(df):
    """
    Add sine/cosine representation of hour of day.
    """

    df = df.copy()

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )

    return df


# ============================================================
# FAST FEATURE IMPORTANCE
# ============================================================

def calculate_fast_feature_importance(
    X_test,
    y_test,
    feature_names
):
    """
    Calculate a fast feature-importance proxy.

    Instead of repeatedly running the model through
    permutation_importance, calculate the absolute
    correlation between each input feature and each target.

    The mean absolute correlation across the three targets
    is used only for visualization.

    This does NOT affect the trained model or its metrics.
    """

    importance = []

    for feature in feature_names:

        correlations = []

        for target in TARGETS:

            corr = X_test[feature].corr(
                y_test[target]
            )

            if pd.isna(corr):
                corr = 0.0

            correlations.append(
                abs(float(corr))
            )

        importance.append(
            np.mean(correlations)
        )

    importance = np.array(
        importance
    )

    total = importance.sum()

    if total > 0:
        importance = (
            importance / total
        )

    return importance


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def run():

    print("=" * 70)
    print("SMART MICROGRID SURROGATE MODEL")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(DATA)

    print(
        f"Total records: {len(df)}"
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = add_cyclical_features(df)

    feat_cols = [
        "solar_irradiance_wm2",
        "wind_speed_ms",
        "ambient_temp_c",
        "humidity_pct",
        "hour_sin",
        "hour_cos",
        "day_of_week"
    ]

    X = df[feat_cols]
    y = df[TARGETS]

    # ========================================================
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # ========================================================

    split_idx = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print("\nChronological split:")

    print(
        f"Training records : "
        f"{len(X_train)}"
    )

    print(
        f"Testing records  : "
        f"{len(X_test)}"
    )

    print(
        f"Training days approximately: "
        f"{len(X_train) / 24:.0f}"
    )

    print(
        f"Testing days approximately : "
        f"{len(X_test) / 24:.0f}"
    )

    # ========================================================
    # SCALE FEATURES
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    y_train_np = y_train.values
    y_test_np = y_test.values

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
    )

    rf.fit(
        X_train_scaled,
        y_train_np
    )

    pred_rf = rf.predict(
        X_test_scaled
    )

    # ========================================================
    # HISTOGRAM GRADIENT BOOSTING
    # ========================================================

    print(
        "Training Histogram "
        "Gradient Boosting..."
    )

    hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            random_state=42
        )
    )

    hgb.fit(
        X_train_scaled,
        y_train_np
    )

    pred_hgb = hgb.predict(
        X_test_scaled
    )

    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    predictions = {

        "RandomForest":
            pred_rf,

        "HistGradientBoosting":
            pred_hgb
    }

    # ========================================================
    # EVALUATION
    # ========================================================

    results = {}

    for backend, pred in predictions.items():

        results[backend] = {}

        for i, target in enumerate(
            TARGETS
        ):

            actual = y_test_np[:, i]

            predicted = pred[:, i]

            mae = mean_absolute_error(
                actual,
                predicted
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )

            r2 = r2_score(
                actual,
                predicted
            )

            results[backend][target] = {

                "MAE":
                    round(
                        float(mae),
                        3
                    ),

                "RMSE":
                    round(
                        float(rmse),
                        3
                    ),

                "R2":
                    round(
                        float(r2),
                        4
                    )
            }

    # ========================================================
    # AVERAGE R²
    # ========================================================

    avg_r2 = {}

    for backend in results:

        avg_r2[backend] = float(
            np.mean(
                [
                    results[backend][target]["R2"]
                    for target in TARGETS
                ]
            )
        )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_backend = max(
        avg_r2,
        key=avg_r2.get
    )

    if best_backend == "RandomForest":

        best_model = rf
        best_predictions = pred_rf

    else:

        best_model = hgb
        best_predictions = pred_hgb

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print("\nAverage R²:")

    for backend, score in avg_r2.items():

        print(
            f"{backend:25s}: "
            f"{score:.4f}"
        )

    print(
        f"\nSelected backend: "
        f"{best_backend}"
    )

    for backend in results:

        print(
            "\n" + "-" * 60
        )

        print(backend)

        print(
            "-" * 60
        )

        for target in TARGETS:

            metrics = results[
                backend
            ][target]

            print(
                f"{target:20s} "
                f"MAE={metrics['MAE']:>8} "
                f"RMSE={metrics['RMSE']:>8} "
                f"R²={metrics['R2']:.4f}"
            )

    # ========================================================
    # FAST FEATURE IMPORTANCE
    # ========================================================

    print(
        "\nGenerating feature importance figure..."
    )

    importance = (
        calculate_fast_feature_importance(
            X_test,
            y_test,
            feat_cols
        )
    )

    order = np.argsort(
        importance
    )[::-1]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ordered_features = [
        feat_cols[i]
        for i in order
    ][::-1]

    ordered_values = (
        importance[order][::-1]
    )

    ax.barh(
        ordered_features,
        ordered_values
    )

    ax.set_xlabel(
        "Relative Importance"
    )

    ax.set_title(
        "Fig 5.1 - Surrogate Model "
        "Feature Importance",
        fontsize=10,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        FIG /
        "fig5_1_feature_importance.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # PREDICTED VS ACTUAL
    # ========================================================

    print(
        "Generating predicted vs actual figure..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4)
    )

    for i, target in enumerate(
        TARGETS
    ):

        actual = y_test_np[:, i]

        predicted = (
            best_predictions[:, i]
        )

        axes[i].scatter(
            actual,
            predicted,
            s=8,
            alpha=0.4
        )

        min_value = min(
            actual.min(),
            predicted.min()
        )

        max_value = max(
            actual.max(),
            predicted.max()
        )

        axes[i].plot(
            [min_value, max_value],
            [min_value, max_value],
            "r--",
            linewidth=1
        )

        axes[i].set_xlabel(
            f"Actual {target}"
        )

        axes[i].set_ylabel(
            f"Predicted {target}"
        )

        r2 = results[
            best_backend
        ][target]["R2"]

        axes[i].set_title(
            f"{target}\n"
            f"R² = {r2:.4f}",
            fontsize=9
        )

    plt.suptitle(
        "Fig 5.2 - Surrogate Model "
        "Predicted vs Actual",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        FIG /
        "fig5_2_pred_vs_actual.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = {

        "model_version":
            "2.2",

        "evaluation_method":
            "chronological_train_test_split",

        "dataset_records":
            int(len(df)),

        "n_train":
            int(len(X_train)),

        "n_test":
            int(len(X_test)),

        "training_days":
            int(len(X_train) / 24),

        "testing_days":
            int(len(X_test) / 24),

        "backend_results":
            results,

        "avg_r2_by_backend":
            {
                key:
                    round(
                        value,
                        4
                    )
                for key, value
                in avg_r2.items()
            },

        "selected_backend":
            best_backend,

        "features":
            feat_cols,

        "targets":
            TARGETS,

        "feature_importance_method":
            "absolute_correlation_proxy",

        "feature_importance":
            {
                feat_cols[i]:
                    round(
                        float(
                            importance[i]
                        ),
                        4
                    )
                for i in order
            }
    }

    report_path = (
        MODELS /
        "surrogate_report.json"
    )

    with open(
        report_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_package = {

        "model":
            best_model,

        "scaler":
            scaler,

        "feat_cols":
            feat_cols,

        "targets":
            TARGETS,

        "backend":
            best_backend,

        "model_version":
            "2.2"
    }

    model_path = (
        MODELS /
        "surrogate_model.joblib"
    )

    joblib.dump(
        model_package,
        model_path
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model saved to:\n"
        f"{model_path}"
    )

    print(
        f"\nReport saved to:\n"
        f"{report_path}"
    )

    print("\nFigures updated:")

    print(
        " - fig5_1_feature_importance.png"
    )

    print(
        " - fig5_2_pred_vs_actual.png"
    )

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()