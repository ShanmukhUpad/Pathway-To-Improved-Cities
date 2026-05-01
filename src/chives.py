"""ChiVes (Chicago Healthy and Equitable Index by Space) integration.

Loads the chives-data-public.geojson tract-level dataset, aggregates to
community area, and provides a field-selectable RF+GB prediction view in the
socioeconomics tab.
"""
import json
import os
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

_HERE = Path(__file__).parent
# Chicago-specific dataset lives in src/chicago/
CHIVES_GEOJSON = _HERE / "chicago" / "chives-data-public.geojson"

# Rename map from notebooks/chives_column_rename.ipynb
RENAME_MAP = {
    "comAreaNm":          "Community Area",
    "hrdshpIndx":         "HARDSHIP INDEX",
    "pop18":              "Population",
    "whiteP18":           "White",
    "blackP18":           "Black",
    "hispP18":            "Hispanic",
    "asianP18":           "Asian",
    "mdRent18":           "MEDIAN_RENT",
    "mdHome18":           "MEDIAN_HOME_VALUE",
    "frgnBrn18":          "FOREIGN_BORN",
    "ovrcwdHs18":         "PERCENT OF HOUSING CROWDED",
    "college18":          "BACHELORS_DEGREE",
    "geoid":              "GEOID",
    "CEJI":               "CEJI_SCORE",
    "treeChng":           "TREE_CANOPY_CHANGE",
    "treeCCov17":         "TREE_CANOPY_2017",
    "treeCCov10":         "TREE_CANOPY_2010",
    "HeathDeath_Percent": "HEALTH_DEATH_PERCENT",
    "leadPoisonR":        "LEAD_POISON_RATE",
    "ndvi":               "NDVI",
    "trees_n":            "TREE_COUNT",
    "trees_crown_den":    "TREE_CROWN_DENSITY",
    "socvlnIndx":         "SOCIAL_VULNERABILITY_INDEX",
    "urbanFlood":         "URBAN_FLOOD_RISK",
    "surfcTemp":          "SURFACE_TEMP",
    "trafficVol":         "TRAFFIC_VOLUME",
    "nnetPM25":           "PM25",
    "chAsthmaED":         "CHILD_ASTHMA_ED_RATE",
    "chAsthma17":         "CHILD_ASTHMA_RATE",
    "amIndP18":           "NATIVE_AMERICAN",
    "other18":            "OTHER_RACE",
    "chldren18":          "CHILDREN_PERCENT",
    "senior18":           "SENIOR_PERCENT",
    "zipCode":            "ZIP_CODE",
    "comArea":            "COMMUNITY_AREA_NUMBER",
    "plantDivrs":         "PLANT_DIVERSITY",
    "plantTotl":          "PLANT_TOTAL",
    "plantSpec":          "PLANT_SPECIES",
    "redLined":           "RED_LINED",
    "hhldSize18":         "HOUSEHOLD_SIZE",
    "fmHhld18":           "FAMILY_HOUSEHOLD_PERCENT",
    "popDens18":          "POPULATION_DENSITY",
    "costBurd18":         "COST_BURDEN",
    "hsngSuscp":          "HOUSING_SUSCEPTIBILITY",
    "hsngVuln":           "HOUSING_VULNERABILITY",
    "hsngChng":           "HOUSING_CHANGE",
    "dsplcPresr":         "DISPLACEMENT_PRESSURE",
    "chwAirTemp":         "AIR_TEMP",
    "chwHeatAve":         "HEAT_AVE",
    "chwHeatMax":         "HEAT_MAX",
    "eclipsPM25":         "ECLIPSE_PM25",
    "lungCRt17":          "LUNG_CANCER_RATE",
    "physRt17":           "PHYSICAL_HEALTH_RATE",
    "adAsthma17":         "ADULT_ASTHMA_RATE",
    "cancerRt17":         "CANCER_RATE",
    "hyptRt17":           "HYPERTENSION_RATE",
}

# Curated set of numeric metrics user can pick to visualize/predict.
TARGET_METRICS = [
    "HARDSHIP INDEX", "CEJI_SCORE", "SOCIAL_VULNERABILITY_INDEX",
    "TREE_CANOPY_2017", "TREE_CANOPY_CHANGE", "TREE_COUNT", "NDVI",
    "URBAN_FLOOD_RISK", "SURFACE_TEMP", "AIR_TEMP", "HEAT_AVE", "HEAT_MAX",
    "PM25", "ECLIPSE_PM25", "TRAFFIC_VOLUME",
    "HEALTH_DEATH_PERCENT", "LEAD_POISON_RATE",
    "CHILD_ASTHMA_RATE", "ADULT_ASTHMA_RATE",
    "LUNG_CANCER_RATE", "CANCER_RATE", "HYPERTENSION_RATE", "PHYSICAL_HEALTH_RATE",
    "MEDIAN_RENT", "MEDIAN_HOME_VALUE", "POPULATION_DENSITY",
    "COST_BURDEN", "HOUSING_VULNERABILITY", "DISPLACEMENT_PRESSURE",
    "PERCENT OF HOUSING CROWDED", "BACHELORS_DEGREE", "FOREIGN_BORN",
    "CHILDREN_PERCENT", "SENIOR_PERCENT",
]

# Feature columns used to predict the target (target is excluded at training).
FEATURE_POOL = [
    "TREE_CANOPY_2017", "TREE_CANOPY_CHANGE", "NDVI", "TREE_CROWN_DENSITY",
    "URBAN_FLOOD_RISK", "SURFACE_TEMP", "HEAT_AVE", "PM25",
    "TRAFFIC_VOLUME", "LEAD_POISON_RATE",
    "MEDIAN_RENT", "MEDIAN_HOME_VALUE", "POPULATION_DENSITY", "COST_BURDEN",
    "PERCENT OF HOUSING CROWDED", "BACHELORS_DEGREE", "FOREIGN_BORN",
    "CHILDREN_PERCENT", "SENIOR_PERCENT",
    "White", "Black", "Hispanic", "Asian",
    "HOUSING_VULNERABILITY", "DISPLACEMENT_PRESSURE",
]


@st.cache_data(show_spinner="Loading ChiVes dataset...")
def _load_chives():
    if not CHIVES_GEOJSON.exists():
        return None
    with open(CHIVES_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)
    rows = [feat["properties"] for feat in gj["features"]]
    df = pd.DataFrame(rows)
    df = df.rename(columns=RENAME_MAP)

    # Coerce numeric columns
    for c in df.columns:
        if c in ("Community Area", "GEOID", "ZIP_CODE", "RED_LINED",
                 "DISPLACEMENT_PRESSURE", "COMMUNITY_AREA_NUMBER"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["COMMUNITY_AREA_NUMBER"] = pd.to_numeric(
        df["COMMUNITY_AREA_NUMBER"], errors="coerce"
    )
    df = df.dropna(subset=["COMMUNITY_AREA_NUMBER"])
    df["COMMUNITY_AREA_NUMBER"] = df["COMMUNITY_AREA_NUMBER"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def _aggregate_by_community(df_json: str):
    df = pd.read_json(StringIO(df_json))
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "COMMUNITY_AREA_NUMBER" in numeric_cols:
        numeric_cols.remove("COMMUNITY_AREA_NUMBER")
    name_lookup = (
        df.groupby("COMMUNITY_AREA_NUMBER")["Community Area"]
        .first().to_dict()
    )
    agg = df.groupby("COMMUNITY_AREA_NUMBER")[numeric_cols].mean().reset_index()
    agg["Community Area"] = agg["COMMUNITY_AREA_NUMBER"].map(name_lookup)
    return agg


@st.cache_data(show_spinner="Training ChiVes prediction models...")
def _train_chives_models(agg_json: str, target: str, feature_cols_tuple: tuple):
    agg = pd.read_json(StringIO(agg_json))
    feature_cols = [c for c in feature_cols_tuple if c in agg.columns and c != target]
    work = agg.dropna(subset=feature_cols + [target]).copy()
    if len(work) < 10:
        return None
    X = work[feature_cols]
    y = work[target]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    rf_r2  = float(np.clip(cross_val_score(rf, X, y, cv=kf, scoring="r2").mean(), -1.0, 1.0))
    rf_mae = float(-cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error").mean())
    gb_r2  = float(np.clip(cross_val_score(gb, X, y, cv=kf, scoring="r2").mean(), -1.0, 1.0))
    gb_mae = float(-cross_val_score(gb, X, y, cv=kf, scoring="neg_mean_absolute_error").mean())

    rf.fit(X, y)
    gb.fit(X, y)

    pred_df = work[["COMMUNITY_AREA_NUMBER", "Community Area", target]].copy()
    pred_df["RF_Predicted"] = rf.predict(X).round(3)
    pred_df["GB_Predicted"] = gb.predict(X).round(3)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "pred_df":      pred_df,
        "rf_r2":        float(rf_r2),
        "rf_mae":       float(rf_mae),
        "gb_r2":        float(gb_r2),
        "gb_mae":       float(gb_mae),
        "feature_cols": feature_cols,
        "rf_imp":       importances.to_dict(),
    }


def _format_value(v: float) -> str:
    if abs(v) >= 1_000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def render(chicago_geo=None):
    st.markdown("---")
    st.subheader("ChiVes Equity Index — Tract-level forecast")
    st.caption(
        "**ChiVes** (Chicago Healthy and Equitable Index by Space) joins environmental, "
        "health, climate, and housing metrics across ~800 census tracts. Pick any metric to "
        "visualize and the model will forecast it from the other ChiVes features."
    )

    df = _load_chives()
    if df is None:
        st.info(
            f"`chives-data-public.geojson` not found in `src/`. Place the file there to enable this section."
        )
        return

    df_json = df.to_json()
    agg = _aggregate_by_community(df_json)
    agg_json = agg.to_json()

    available = [m for m in TARGET_METRICS if m in agg.columns and agg[m].notna().sum() >= 10]
    if not available:
        st.warning("No usable ChiVes metrics found.")
        return

    col_metric, col_area = st.columns(2)
    with col_metric:
        target = st.selectbox(
            "Field to visualize and forecast",
            available,
            index=available.index("HARDSHIP INDEX") if "HARDSHIP INDEX" in available else 0,
            key="chives_target_metric",
        )
    with col_area:
        area_options = sorted(agg["Community Area"].dropna().unique().tolist())
        default_idx_a = area_options.index("NORTH LAWNDALE") if "NORTH LAWNDALE" in area_options else 0
        selected_area = st.selectbox(
            "Community area", area_options,
            index=default_idx_a, key="chives_area",
        )

    feature_cols_tuple = tuple(c for c in FEATURE_POOL if c != target)
    res = _train_chives_models(agg_json, target, feature_cols_tuple)

    if res is None:
        st.warning(f"Not enough data to train a model for `{target}`.")
        return

    pred_df = res["pred_df"]
    row = pred_df[pred_df["Community Area"] == selected_area]

    if not row.empty:
        actual_v = float(row[target].iloc[0])
        rf_pred  = float(row["RF_Predicted"].iloc[0])
        city_avg = float(pred_df[target].mean())
        delta    = rf_pred - city_avg
        arrow    = "▲" if delta >= 0 else "▼"
        direction = "above" if delta >= 0 else "below"

        st.markdown(f"""
<div style="background:rgba(155,93,229,0.1); border-left:4px solid #9b5de5;
            padding:18px 22px; border-radius:8px; margin:14px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">ChiVes Forecast — {target.replace('_', ' ')}</p>
  <p style="margin:6px 0 2px; font-size:2.4rem; font-weight:800;
            color:#ffffff; line-height:1.1;">
    {_format_value(rf_pred)}
    <span style="font-size:1.1rem; font-weight:500; color:#c0a0ee;">
      &nbsp;{target.replace('_', ' ')}
    </span>
  </p>
  <p style="margin:2px 0 0; font-size:14px; color:#9eaec4;">
    in <strong style="color:#ffffff;">{selected_area}</strong>
    &nbsp;·&nbsp; {arrow} {_format_value(abs(delta))} vs city average
  </p>
</div>
""", unsafe_allow_html=True)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("RF prediction", _format_value(rf_pred))
        cc2.metric("Actual",        _format_value(actual_v))
        cc3.metric("City average",  _format_value(city_avg))
        cc4.metric("CV R² (RF)",    f"{res['rf_r2']:.3f}")
        st.caption(
            f"RF MAE: {res['rf_mae']:.3f} · GB R²: {res['gb_r2']:.3f} · GB MAE: {res['gb_mae']:.3f} · "
            f"{len(res['feature_cols'])} features · 5-fold CV"
        )
        st.info(
            f"**What this means:** The Random Forest predicts **{_format_value(rf_pred)}** for "
            f"{target.replace('_', ' ').title()} in **{selected_area}** — "
            f"**{_format_value(abs(delta))} {direction}** the city average of {_format_value(city_avg)}. "
            f"Actual measured value: {_format_value(actual_v)}. "
            f"Model fit (CV R²): {res['rf_r2']:.3f}."
        )
    else:
        st.info(f"No ChiVes data for {selected_area}.")

    # ── Choropleth: actual vs predicted ──────────────────────────────────────
    if chicago_geo is not None:
        merged_geo = pred_df.copy()
        merged_geo["area_num"] = merged_geo["COMMUNITY_AREA_NUMBER"]

        col_act, col_pred = st.columns(2)
        with col_act:
            fig_actual = px.choropleth_map(
                merged_geo, geojson=chicago_geo,
                locations="area_num", featureidkey="properties.area_num_1",
                color=target, color_continuous_scale="Viridis",
                map_style="open-street-map", zoom=9,
                center={"lat": 41.85, "lon": -87.68}, opacity=0.7,
                hover_name="Community Area",
                hover_data={target: ":.2f", "RF_Predicted": ":.2f", "area_num": False},
                title=f"Actual {target.replace('_', ' ').title()}",
            )
            fig_actual.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=420)
            st.plotly_chart(fig_actual, width="stretch")

        with col_pred:
            fig_pred = px.choropleth_map(
                merged_geo, geojson=chicago_geo,
                locations="area_num", featureidkey="properties.area_num_1",
                color="RF_Predicted", color_continuous_scale="Viridis",
                map_style="open-street-map", zoom=9,
                center={"lat": 41.85, "lon": -87.68}, opacity=0.7,
                hover_name="Community Area",
                hover_data={target: ":.2f", "RF_Predicted": ":.2f", "area_num": False},
                title="RF Predicted",
            )
            fig_pred.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=420)
            st.plotly_chart(fig_pred, width="stretch")

    # ── Feature importances ──────────────────────────────────────────────────
    with st.expander("Top features driving this prediction"):
        imp = pd.Series(res["rf_imp"]).sort_values(ascending=True).tail(15)
        fig_imp = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=imp.values, color_continuous_scale="Purples",
        )
        fig_imp.update_layout(coloraxis_showscale=False, height=420, margin={"t": 10})
        st.plotly_chart(fig_imp, width="stretch")
