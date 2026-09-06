from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "microgrid_timeseries.csv"
MODEL_FILE = ROOT / "models" / "surrogate_model.joblib"
REPORT_FILE = ROOT / "models" / "integrated_optimizer_report.json"
MODEL_DIR = ROOT / "models"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Microgrid Optimizer",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-title {
        color: #888;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #333;
        background: #181a20;
        text-align: center;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
    }

    .metric-label {
        color: #999;
        font-size: 13px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin: 12px 0 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data["date_id"] = (data["timestamp_h"] // 24).astype(int)
    data["day"] = data["date_id"] + 1
    return data


@st.cache_data
def load_report():
    if not REPORT_FILE.exists():
        return {}

    with open(REPORT_FILE, "r") as f:
        return json.load(f)


@st.cache_data
def load_dispatch(scenario):
    names = {
        "Sunny": [
            "dispatch_sunny.csv",
            "dispatch_Sunny.csv",
        ],
        "High Wind": [
            "dispatch_high_wind.csv",
            "dispatch_high wind.csv",
            "dispatch_High_Wind.csv",
        ],
        "Overcast": [
            "dispatch_overcast.csv",
            "dispatch_Overcast.csv",
        ],
    }

    for name in names.get(scenario, []):
        path = MODEL_DIR / name
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


df = load_data()
report = load_report()


# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def title(text):
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


def get_results(report):
    results = report.get("results", {})

    if isinstance(results, dict):
        return results

    if isinstance(results, list):
        output = {}

        for i, item in enumerate(results):
            if isinstance(item, dict):
                scenario = item.get(
                    "scenario",
                    item.get("regime", f"Scenario {i + 1}")
                )
                output[scenario] = item

        return output

    return {}


def get_result(results, scenario):
    item = results.get(scenario, {})

    if isinstance(item, dict):
        return item

    return {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ Smart Microgrid Renewable Energy Optimizer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "Renewable generation forecasting, battery dispatch and cost optimisation"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Renewable Yield Heatmap",
        "Dispatch Analysis",
        "Optimisation Results",
        "Live Dispatch Advisor",
    ],
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Microgrid Configuration**

    PV Capacity: 500 kW  
    Wind Capacity: 300 kW  
    Diesel Capacity: 250 kW  
    Battery Storage: 400 kWh

    Data: 120 simulated days
    """
)


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    title("Microgrid Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Records", f"{len(df):,}")

    with c2:
        metric_card("Simulation Days", f"{df['day'].nunique()}")

    with c3:
        metric_card("PV Capacity", "500 kW")

    with c4:
        metric_card("Wind Capacity", "300 kW")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        title("Renewable Generation")

        hourly = (
            df.groupby("hour_of_day")[
                [
                    "pv_output_kw",
                    "wind_output_kw",
                    "load_demand_kw",
                ]
            ]
            .mean()
        )

        st.line_chart(hourly, use_container_width=True)

    with right:

        title("Average Electricity Tariff")

        tariff = (
            df.groupby("hour_of_day")
            ["grid_tariff_inr_per_kwh"]
            .mean()
        )

        st.line_chart(tariff, use_container_width=True)

    title("Dataset Preview")

    cols = [
        "timestamp_h",
        "hour_of_day",
        "solar_irradiance_wm2",
        "wind_speed_ms",
        "pv_output_kw",
        "wind_output_kw",
        "load_demand_kw",
        "grid_tariff_inr_per_kwh",
    ]

    st.dataframe(
        df[cols].head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 2. HEATMAP
# ============================================================

elif page == "Renewable Yield Heatmap":

    title("Renewable Yield Heatmap")

    st.write(
        "Hourly renewable generation across the 120-day simulation."
    )

    tab1, tab2 = st.tabs(["Solar PV", "Wind"])

    with tab1:

        solar = df.pivot_table(
            index="day",
            columns="hour_of_day",
            values="pv_output_kw",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        image = ax.imshow(
            solar,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title("Solar PV Output (kW)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Simulation Day")
        ax.set_xticks(range(24))

        plt.colorbar(
            image,
            ax=ax,
            label="PV Output (kW)",
        )

        st.pyplot(fig)
        plt.close(fig)

    with tab2:

        wind = df.pivot_table(
            index="day",
            columns="hour_of_day",
            values="wind_output_kw",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        image = ax.imshow(
            wind,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title("Wind Output (kW)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Simulation Day")
        ax.set_xticks(range(24))

        plt.colorbar(
            image,
            ax=ax,
            label="Wind Output (kW)",
        )

        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# 3. DISPATCH ANALYSIS
# ============================================================

elif page == "Dispatch Analysis":

    title("Dispatch Analysis")

    st.write(
        "Hourly behaviour of the microgrid under representative operating conditions."
    )

    scenario = st.selectbox(
        "Select Scenario",
        ["Sunny", "High Wind", "Overcast"],
    )

    dispatch = load_dispatch(scenario)

    if dispatch.empty:

        st.error(
            f"Dispatch data for {scenario} could not be found."
        )

    else:

        st.success(
            f"{scenario} scenario dispatch data loaded."
        )

        # ----------------------------------------------------
        # Scenario day
        # ----------------------------------------------------

        results = get_results(report)
        result = get_result(results, scenario)

        selected_day = result.get("day")

        # ----------------------------------------------------
        # Main dispatch chart
        # ----------------------------------------------------

        title("Power Dispatch")

        chart_cols = [
            c for c in [
                "net_demand",
                "grid_import",
                "diesel_kw",
                "batt_kw",
                "curtailed_kw",
            ]
            if c in dispatch.columns
        ]

        if chart_cols:

            st.line_chart(
                dispatch[chart_cols],
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Battery SOC
        # ----------------------------------------------------

        if "soc_frac" in dispatch.columns:

            title("Battery State of Charge")

            soc = dispatch["soc_frac"] * 100

            st.line_chart(
                pd.DataFrame(
                    {"Battery SOC (%)": soc}
                ),
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        grid_energy = (
            dispatch["grid_import"].clip(lower=0).sum()
            if "grid_import" in dispatch
            else 0
        )

        diesel_energy = (
            dispatch["diesel_kw"].clip(lower=0).sum()
            if "diesel_kw" in dispatch
            else 0
        )

        charge_energy = (
            -dispatch.loc[
                dispatch["batt_kw"] < 0,
                "batt_kw"
            ].sum()
            if "batt_kw" in dispatch
            else 0
        )

        discharge_energy = (
            dispatch.loc[
                dispatch["batt_kw"] > 0,
                "batt_kw"
            ].sum()
            if "batt_kw" in dispatch
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Grid Import",
                f"{grid_energy:,.1f} kWh",
            )

        with c2:
            metric_card(
                "Diesel Energy",
                f"{diesel_energy:,.1f} kWh",
            )

        with c3:
            metric_card(
                "Battery Charge",
                f"{charge_energy:,.1f} kWh",
            )

        with c4:
            metric_card(
                "Battery Discharge",
                f"{discharge_energy:,.1f} kWh",
            )

        # ----------------------------------------------------
        # Renewable generation from original dataset
        # ----------------------------------------------------

        if selected_day is not None:

            day_data = df[
                df["day"] == int(selected_day)
            ].copy()

            if not day_data.empty:

                title(
                    f"Renewable Generation — Day {selected_day}"
                )

                renewable = day_data[
                    [
                        "pv_output_kw",
                        "wind_output_kw",
                        "load_demand_kw",
                    ]
                ].copy()

                renewable.index = range(
                    len(renewable)
                )

                st.line_chart(
                    renewable,
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # Dispatch table
        # ----------------------------------------------------

        title("Dispatch Data")

        st.dataframe(
            dispatch,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 4. OPTIMISATION RESULTS
# ============================================================

elif page == "Optimisation Results":

    title("Optimisation Results")

    results = get_results(report)

    if not results:

        st.error(
            "No optimisation results were found."
        )

    else:

        # ----------------------------------------------------
        # Calculate totals
        # ----------------------------------------------------

        baseline_total = 0
        optimized_total = 0

        for result in results.values():

            if not isinstance(result, dict):
                continue

            baseline_total += float(
                result.get("baseline_cost", 0)
            )

            optimized_total += float(
                result.get("optimized_cost", 0)
            )

        total_savings = (
            baseline_total - optimized_total
        )

        savings_pct = (
            total_savings / baseline_total * 100
            if baseline_total
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Baseline Cost",
                f"₹{baseline_total:,.2f}",
            )

        with c2:
            metric_card(
                "Optimised Cost",
                f"₹{optimized_total:,.2f}",
            )

        with c3:
            metric_card(
                "Total Savings",
                f"₹{total_savings:,.2f}",
            )

        with c4:
            metric_card(
                "Savings",
                f"{savings_pct:.2f}%",
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Scenario comparison
        # ----------------------------------------------------

        title("Scenario Cost Comparison")

        rows = []

        for scenario, result in results.items():

            if not isinstance(result, dict):
                continue

            baseline = float(
                result.get("baseline_cost", 0)
            )

            optimized = float(
                result.get("optimized_cost", 0)
            )

            savings = baseline - optimized

            rows.append(
                {
                    "Scenario": scenario,
                    "Baseline Cost": baseline,
                    "Optimised Cost": optimized,
                    "Savings": savings,
                    "Savings %": (
                        savings / baseline * 100
                        if baseline
                        else 0
                    ),
                }
            )

        comparison = pd.DataFrame(rows)

        if not comparison.empty:

            st.bar_chart(
                comparison.set_index("Scenario")[
                    [
                        "Baseline Cost",
                        "Optimised Cost",
                    ]
                ],
                use_container_width=True,
            )

            st.dataframe(
                comparison.style.format(
                    {
                        "Baseline Cost": "₹{:,.2f}",
                        "Optimised Cost": "₹{:,.2f}",
                        "Savings": "₹{:,.2f}",
                        "Savings %": "{:.2f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Scenario details
        # ----------------------------------------------------

        title("Optimised Policy Details")

        for scenario, result in results.items():

            if not isinstance(result, dict):
                continue

            baseline = float(
                result.get("baseline_cost", 0)
            )

            optimized = float(
                result.get("optimized_cost", 0)
            )

            savings = baseline - optimized

            percentage = (
                savings / baseline * 100
                if baseline
                else 0
            )

            with st.expander(
                f"{scenario}  |  Savings: ₹{savings:,.2f} ({percentage:.2f}%)"
            ):

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "Baseline",
                        f"₹{baseline:,.2f}",
                    )

                with c2:
                    metric_card(
                        "Optimised",
                        f"₹{optimized:,.2f}",
                    )

                with c3:
                    metric_card(
                        "Savings",
                        f"₹{savings:,.2f}",
                    )

                policy = result.get(
                    "policy",
                    {}
                )

                if isinstance(policy, dict) and policy:

                    st.markdown("#### Policy Parameters")

                    policy_df = pd.DataFrame(
                        [
                            {
                                "Parameter": key,
                                "Value": value,
                            }
                            for key, value
                            in policy.items()
                        ]
                    )

                    st.dataframe(
                        policy_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                breakdown = result.get(
                    "breakdown",
                    {}
                )

                if isinstance(breakdown, dict) and breakdown:

                    st.markdown("#### Cost Breakdown")

                    breakdown_df = pd.DataFrame(
                        {
                            "Component": list(
                                breakdown.keys()
                            ),
                            "Cost": [
                                float(v)
                                for v
                                in breakdown.values()
                            ],
                        }
                    )

                    st.bar_chart(
                        breakdown_df.set_index(
                            "Component"
                        ),
                        use_container_width=True,
                    )


# ============================================================
# 5. LIVE DISPATCH ADVISOR
# ============================================================

elif page == "Live Dispatch Advisor":

    title("Live Dispatch Advisor")

    st.write(
        "Use the controls below to estimate renewable generation "
        "and load using the trained machine-learning surrogate model."
    )

    try:

        saved = joblib.load(MODEL_FILE)
        model = saved["model"]

        st.markdown("### Operating Conditions")

        c1, c2 = st.columns(2)

        with c1:

            irradiance = st.slider(
                "Solar Irradiance (W/m²)",
                0.0,
                1000.0,
                500.0,
                10.0,
            )

            wind_speed = st.slider(
                "Wind Speed (m/s)",
                0.0,
                20.0,
                7.0,
                0.5,
            )

            temperature = st.slider(
                "Ambient Temperature (°C)",
                10.0,
                45.0,
                28.0,
                1.0,
            )

        with c2:

            humidity = st.slider(
                "Humidity (%)",
                0.0,
                100.0,
                60.0,
                1.0,
            )

            hour = st.slider(
                "Hour of Day",
                0,
                23,
                12,
                1,
            )

            day_of_week = st.slider(
                "Day of Week",
                0,
                6,
                2,
                1,
            )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = pd.DataFrame(
            [
                {
                    "solar_irradiance_wm2": irradiance,
                    "wind_speed_ms": wind_speed,
                    "ambient_temp_c": temperature,
                    "humidity_pct": humidity,
                    "hour_sin": np.sin(
                        2 * np.pi * hour / 24
                    ),
                    "hour_cos": np.cos(
                        2 * np.pi * hour / 24
                    ),
                    "day_of_week": day_of_week,
                }
            ]
        )

        prediction = model.predict(
            features.to_numpy()
        )[0]

        pv = float(
            np.clip(prediction[0], 0, 500)
        )

        wind = float(
            np.clip(prediction[1], 0, 300)
        )

        load = float(
            max(prediction[2], 0)
        )

        renewable = pv + wind
        balance = renewable - load

        st.markdown("---")

        st.markdown("### ML Forecast")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Predicted PV",
                f"{pv:.1f} kW",
            )

        with c2:
            metric_card(
                "Predicted Wind",
                f"{wind:.1f} kW",
            )

        with c3:
            metric_card(
                "Predicted Load",
                f"{load:.1f} kW",
            )

        st.markdown("---")

        if balance >= 0:

            st.success(
                f"Renewable surplus: {balance:.1f} kW"
            )

            st.write(
                "The surplus can be used for battery charging "
                "or renewable curtailment."
            )

        else:

            st.warning(
                f"Renewable deficit: {abs(balance):.1f} kW"
            )

            st.write(
                "Battery discharge, grid import or diesel "
                "support may be required."
            )

        # ----------------------------------------------------
        # Visual
        # ----------------------------------------------------

        title("Current Power Estimate")

        forecast = pd.DataFrame(
            {
                "Solar PV": [pv],
                "Wind": [wind],
                "Load": [load],
            }
        )

        st.bar_chart(
            forecast.T.rename(
                columns={0: "Power (kW)"}
            ),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            "Unable to load the trained surrogate model."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart Microgrid Renewable Energy Dispatch & Storage Optimizer "
    "• ML Forecasting + Bayesian Optimisation"
)