import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import map_utils


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

ENERGY_CSV     = os.path.join(_SRC_DIR, "chicago", "clean_energy.csv")
BENCHMARK_CSV  = os.path.join(_SRC_DIR, "chicago", "clean_benchmark.csv")
COMPLAINTS_CSV = os.path.join(_SRC_DIR, "chicago", "clean_complaints.csv")
    
NUMERIC_COLS = [
    "TOTAL_KWH",
    "KWH_TOTAL_SQFT",
    "TOTAL_THERMS",
    "TOTAL_POPULATION",
    "TOTAL_UNITS",
]


@st.cache_data(show_spinner="Loading environment datasets...")
def _load_and_merge():
    energy     = pd.read_csv(ENERGY_CSV)
    benchmark  = pd.read_csv(BENCHMARK_CSV)
    complaints = pd.read_csv(COMPLAINTS_CSV)

    for col in NUMERIC_COLS:
        energy[col] = (
            energy[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .str.strip()
        )
        energy[col] = pd.to_numeric(energy[col], errors="coerce")
    energy[NUMERIC_COLS] = energy[NUMERIC_COLS].fillna(0)
    energy = energy[energy["TOTAL_POPULATION"] < 1e7]
    energy = energy[energy["TOTAL_UNITS"] < 1e6]
    energy["energy_intensity"] = energy["TOTAL_KWH"] / energy["KWH_TOTAL_SQFT"].replace(0, 1)

    benchmark["size_code"] = benchmark["Cohort_-_Size"].astype("category").cat.codes

    complaints["COMPLAINT_DATE"] = pd.to_datetime(complaints["COMPLAINT_DATE"], errors="coerce")
    complaints = complaints.dropna(subset=["COMPLAINT_DATE"])
    complaints["year"] = complaints["COMPLAINT_DATE"].dt.year

    energy["COMMUNITY_AREA_NAME"]       = energy["COMMUNITY_AREA_NAME"].str.strip().str.upper()
    benchmark["Community_Area_Name"]    = benchmark["Community_Area_Name"].str.strip().str.upper()

    energy_agg = energy.groupby("COMMUNITY_AREA_NAME").agg({
        "TOTAL_KWH": "sum",
        "TOTAL_THERMS": "sum",
        "energy_intensity": "mean",
        "TOTAL_POPULATION": "mean",
        "TOTAL_UNITS": "mean",
    }).reset_index()

    benchmark_agg = benchmark.groupby("Community_Area_Name").agg({
        "Building_ID": "count",
        "size_code": "mean",
    }).reset_index().rename(columns={
        "Community_Area_Name": "COMMUNITY_AREA_NAME",
        "Building_ID": "num_large_buildings",
    })

    complaints["LATITUDE"]  = pd.to_numeric(complaints["LATITUDE"],  errors="coerce")
    complaints["LONGITUDE"] = pd.to_numeric(complaints["LONGITUDE"], errors="coerce")
    complaints_clean = complaints.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

    benchmark_valid = benchmark[["Latitude", "Longitude", "Community_Area_Name"]].copy()
    benchmark_valid["Latitude"]  = pd.to_numeric(benchmark_valid["Latitude"],  errors="coerce")
    benchmark_valid["Longitude"] = pd.to_numeric(benchmark_valid["Longitude"], errors="coerce")
    benchmark_valid = benchmark_valid.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    tree = KDTree(benchmark_valid[["Latitude", "Longitude"]].values)
    _, idx = tree.query(complaints_clean[["LATITUDE", "LONGITUDE"]].values, k=1)
    complaints_clean["COMMUNITY_AREA_NAME"] = benchmark_valid.iloc[idx.flatten()]["Community_Area_Name"].values

    complaints_agg = (
        complaints_clean.groupby("COMMUNITY_AREA_NAME")
        .size().reset_index(name="complaint_count")
    )

    merged = energy_agg.merge(benchmark_agg, on="COMMUNITY_AREA_NAME", how="left")
    merged = merged.merge(complaints_agg, on="COMMUNITY_AREA_NAME", how="left")
    merged["num_large_buildings"] = merged["num_large_buildings"].fillna(0)
    merged["complaint_count"]     = merged["complaint_count"].fillna(0)
    return merged, complaints_clean


@st.cache_data(show_spinner="Training environment model...")
def _train_model(merged_json: str):
    import io
    merged = pd.read_json(io.StringIO(merged_json))
    feats = [f for f in ["energy_intensity", "complaint_count", "num_large_buildings",
                         "TOTAL_POPULATION", "TOTAL_UNITS"] if f in merged.columns]
    work = merged.dropna(subset=feats + ["TOTAL_KWH"])
    if len(work) < 5:
        return None
    X = work[feats]
    y = work["TOTAL_KWH"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = float(np.clip(r2_score(y_test, preds), -1.0, 1.0))
    return {
        "feats":  feats,
        "preds":  preds.tolist(),
        "y_test": y_test.tolist(),
        "mse":    float(mean_squared_error(y_test, preds)),
        "r2":     r2,
        "importances": dict(zip(feats, model.feature_importances_.tolist())),
    }


@st.cache_data(show_spinner=False)
def _build_monthly_series(complaints_json: str, area_title: str):
    complaints_clean = pd.read_json(complaints_json)
    complaints_clean["COMPLAINT_DATE"] = pd.to_datetime(complaints_clean["COMPLAINT_DATE"])
    ca = complaints_clean.copy()
    ca["month"] = ca["COMPLAINT_DATE"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        ca[ca["COMMUNITY_AREA_NAME"] == area_title]
        .groupby("month").size().rename("count").reset_index()
    )
    if monthly.empty:
        return monthly
    full_range = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    monthly = monthly.set_index("month").reindex(full_range, fill_value=0).rename_axis("month").reset_index()
    monthly["Year"]  = monthly["month"].dt.year
    monthly["Month"] = monthly["month"].dt.month
    monthly["count_lag1"]     = monthly["count"].shift(1)
    monthly["count_lag3"]     = monthly["count"].shift(3)
    monthly["count_lag12"]    = monthly["count"].shift(12)
    monthly["count_rolling3"] = monthly["count"].shift(1).rolling(3, min_periods=1).mean()
    return monthly


@st.cache_data(show_spinner="Training complaint forecast model...")
def _run_complaint_forecast(monthly_json: str):
    monthly = pd.read_json(monthly_json)
    feature_cols = ["count_lag1", "count_lag3", "count_lag12", "count_rolling3", "Month", "Year"]
    model_data = monthly.dropna(subset=feature_cols + ["count"])
    if len(model_data) < 10:
        return None
    X = model_data[feature_cols].values
    y = model_data["count"].values
    latest_val = float(model_data["count"].iloc[-1])

    n_splits = min(5, max(2, len(X) // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = {
        "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    }
    best_name, best_r2, best_rmse = None, -np.inf, np.inf
    for name, make_model in candidates.items():
        fold_r2s, fold_rmses = [], []
        for tr_idx, te_idx in tscv.split(X):
            m = make_model()
            m.fit(X[tr_idx], y[tr_idx])
            p = m.predict(X[te_idx])
            fold_r2s.append(r2_score(y[te_idx], p))
            fold_rmses.append(np.sqrt(mean_squared_error(y[te_idx], p)))
        avg_r2 = float(np.clip(np.mean(fold_r2s), -1.0, 1.0))
        if avg_r2 > best_r2:
            best_name = name
            best_r2   = avg_r2
            best_rmse = float(np.mean(fold_rmses))

    model = candidates[best_name]()
    model.fit(X, y)
    prediction = max(0.0, float(model.predict(X[-1].reshape(1, -1))[0]))
    return {
        "prediction": prediction,
        "latest_val": latest_val,
        "best_name":  best_name,
        "best_r2":    best_r2,
        "best_rmse":  best_rmse,
        "n_splits":   n_splits,
    }


def render(city=None, geo=None):
    from city_config import CityConfig
    chicago_geo = geo
    # Only Chicago has environment data wired up
    if isinstance(city, CityConfig) and city.key != "chicago":
        st.info(f"Environment dataset not yet available for {city.name}.")
        return
    st.header("Environment Dashboard")
    st.markdown(
        "Energy consumption, large-building benchmarks, and environmental complaints "
        "joined at the community-area level. Models forecast next-month complaints "
        "and total kWh from structural and complaint features."
    )

    missing = [p for p in (ENERGY_CSV, BENCHMARK_CSV, COMPLAINTS_CSV) if not os.path.exists(p)]
    if missing:
        st.info(
            "Environment datasets not found:\n\n"
            + "\n".join(f"- `{p}`" for p in missing)
            + "\n\nPlace cleaned CSVs under `clean_data/` to enable this tab."
        )
        return

    try:
        merged, complaints_clean = _load_and_merge()
    except Exception as exc:
        st.error(f"Failed to load environment data: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Community areas",  f"{len(merged):,}")
    c2.metric("Total kWh",        f"{merged['TOTAL_KWH'].sum():,.0f}")
    c3.metric("Total therms",     f"{merged['TOTAL_THERMS'].sum():,.0f}")
    c4.metric("Total complaints", f"{int(merged['complaint_count'].sum()):,}")

    st.markdown("---")

    # ── Forecast headline + area selector ─────────────────────────────────────
    st.subheader("Environmental complaint forecast")
    area_options = sorted(merged["COMMUNITY_AREA_NAME"].unique())
    default_idx = area_options.index("ALBANY PARK") if "ALBANY PARK" in area_options else 0
    selected_area = st.selectbox(
        "Community area", area_options, index=default_idx, key="env_forecast_area"
    )

    complaints_json = complaints_clean.to_json()
    monthly = _build_monthly_series(complaints_json, selected_area)
    fc = _run_complaint_forecast(monthly.to_json()) if not monthly.empty else None

    if fc is not None:
        prediction  = fc["prediction"]
        latest_val  = fc["latest_val"]
        best_name   = fc["best_name"]
        best_r2     = fc["best_r2"]
        best_rmse   = fc["best_rmse"]
        n_splits    = fc["n_splits"]

        _today = datetime.now()
        _nm = _today.replace(month=_today.month % 12 + 1,
                             year=_today.year + (_today.month // 12))
        next_month_label = _nm.strftime("%B %Y")

        trend_vs_latest = prediction - latest_val
        arrow   = "▲" if trend_vs_latest >= 0 else "▼"
        chg_dir = "increase" if trend_vs_latest >= 0 else "decrease"

        st.markdown(f"""
<div style="background:rgba(80,192,128,0.1); border-left:4px solid #50c080;
            padding:18px 22px; border-radius:8px; margin:14px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">Complaint Forecast — {next_month_label}</p>
  <p style="margin:6px 0 2px; font-size:2.4rem; font-weight:800;
            color:#ffffff; line-height:1.1;">
    {round(prediction):,}
    <span style="font-size:1.1rem; font-weight:500; color:#80d0a0;">
      &nbsp;ENVIRONMENTAL COMPLAINTS
    </span>
  </p>
  <p style="margin:2px 0 0; font-size:14px; color:#9eaec4;">
    in <strong style="color:#ffffff;">{selected_area}</strong>
    &nbsp;·&nbsp; {arrow} {abs(trend_vs_latest):.0f} from last month
  </p>
</div>
""", unsafe_allow_html=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Forecast", f"{round(prediction):,}")
        col_f2.metric("CV RMSE",  f"±{best_rmse:.1f}")
        col_f3.metric("CV R²",    f"{best_r2:.3f}")
        st.caption(f"Model: **{best_name}** · {n_splits}-fold time-series CV")

        r2_label = (
            "strong" if best_r2 >= 0.6 else
            "moderate" if best_r2 >= 0.3 else
            "weak" if best_r2 >= 0.0 else
            "poor (worse than baseline)"
        )
        st.info(
            f"**What this means:** In **{next_month_label}**, the model expects "
            f"**{round(prediction):,} environmental complaints** in {selected_area} — "
            f"a **{abs(trend_vs_latest):.0f}-complaint {chg_dir}** from last month. "
            f"Typical prediction error: ±{best_rmse:.1f} (CV RMSE). "
            f"Model fit (CV R²): **{best_r2:.3f}** — {r2_label}."
        )
    else:
        st.info(f"Not enough monthly complaint data for {selected_area} to generate a prediction.")

    # ── Choropleth map ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Environment metric map")

    if chicago_geo is None:
        st.warning("Map unavailable — chicago_geo not provided.")
    else:
        mapbox_style = map_utils.mapbox_style_picker(key_prefix="env")
        metric_options = {
            "energy_intensity":    "Mean energy intensity (kWh/sqft)",
            "TOTAL_KWH":           "Total kWh",
            "TOTAL_THERMS":        "Total therms",
            "complaint_count":     "Complaint count",
            "num_large_buildings": "# Large buildings",
        }
        selected_metric = st.selectbox(
            "Metric", list(metric_options),
            format_func=lambda k: metric_options[k],
            key="env_choropleth_metric",
        )

        plot_df = merged.copy()
        # GeoJSON properties.community is uppercase — keep COMMUNITY_AREA_NAME as-is
        plot_df["Community Area Name"] = plot_df["COMMUNITY_AREA_NAME"]

        scale = "Greens" if selected_metric in ("energy_intensity", "TOTAL_KWH", "TOTAL_THERMS") else "Reds"
        fig_map = px.choropleth_map(
            plot_df,
            geojson=chicago_geo,
            locations="Community Area Name",
            featureidkey="properties.community",
            color=selected_metric,
            color_continuous_scale=scale,
            map_style=mapbox_style,
            zoom=9,
            center={"lat": 41.8781, "lon": -87.6298},
            opacity=0.55,
            labels={selected_metric: metric_options[selected_metric]},
        )
        fig_map.update_coloraxes(colorbar_tickformat=".2f")
        st.plotly_chart(fig_map, width="stretch")

    st.markdown("---")

    # ── Top-20 energy intensity ───────────────────────────────────────────────
    st.subheader("Top 20 community areas — mean energy intensity")
    st.caption("Energy intensity = TOTAL_KWH / KWH_TOTAL_SQFT, averaged within community area.")

    top_ei = merged.nlargest(20, "energy_intensity").sort_values("energy_intensity")
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.patch.set_facecolor("#FAFAF8")
    ax1.set_facecolor("#FAFAF8")
    ax1.barh(top_ei["COMMUNITY_AREA_NAME"], top_ei["energy_intensity"],
             color="#1D9E75", alpha=0.9)
    ax1.set_xlabel("Mean energy intensity (kWh / sqft)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── Complaints by area ────────────────────────────────────────────────────
    st.subheader("Top 20 community areas — environmental complaints")

    top_c = merged.nlargest(20, "complaint_count").sort_values("complaint_count")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor("#FAFAF8")
    ax2.set_facecolor("#FAFAF8")
    ax2.barh(top_c["COMMUNITY_AREA_NAME"], top_c["complaint_count"],
             color="#D85A30", alpha=0.9)
    ax2.set_xlabel("Complaint count")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Complaints over time ──────────────────────────────────────────────────
    if "year" in complaints_clean.columns:
        st.subheader("Complaints by year")
        yearly = complaints_clean.groupby("year").size()
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(yearly.index, yearly.values, marker="o", linewidth=2.2, color="#378ADD")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Complaints")
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")

    # ── Model performance ─────────────────────────────────────────────────────
    st.subheader("Random Forest — TOTAL_KWH prediction")

    res = _train_model(merged.to_json())
    if res is None:
        st.info("Not enough data to train model.")
    else:
        mc1, mc2 = st.columns(2)
        mc1.metric("Test R²",  f"{res['r2']:.3f}")
        mc2.metric("Test MSE", f"{res['mse']:,.0f}")

        y_test = np.array(res["y_test"])
        preds  = np.array(res["preds"])
        fig4, ax4 = plt.subplots(figsize=(7, 7))
        ax4.scatter(y_test, preds, alpha=0.6, color="#378ADD", s=30)
        lo = float(min(y_test.min(), preds.min()))
        hi = float(max(y_test.max(), preds.max()))
        ax4.plot([lo, hi], [lo, hi], color="#888780", linestyle="--", linewidth=1.5, label="y = x")
        ax4.set_xlabel("Actual TOTAL_KWH")
        ax4.set_ylabel("Predicted TOTAL_KWH")
        ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Feature importances
        st.subheader("Feature importances")
        importances = pd.Series(res["importances"]).sort_values()
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.barh(importances.index, importances.values, color="#1D9E75", alpha=0.9)
        ax5.set_xlabel("Importance")
        ax5.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with st.expander("Merged community-area dataframe"):
        st.dataframe(merged, use_container_width=True)
