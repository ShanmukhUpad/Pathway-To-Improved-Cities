import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import file_loader
import ml_predictor
import map_utils
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

_SRC = os.path.dirname(os.path.abspath(__file__))
CRASH_CSV_LATEST = os.path.join(_SRC, "traffic_crashes_latest.csv")
CRASH_CSV_LEGACY = os.path.join(_SRC, "Traffic_Crashes_-_Crashes_20260309.csv")


def _resolve_crash_csv() -> str | None:
    """Return the best available crash CSV path, or None if neither exists."""
    if os.path.exists(CRASH_CSV_LATEST):
        return CRASH_CSV_LATEST
    if os.path.exists(CRASH_CSV_LEGACY):
        return CRASH_CSV_LEGACY
    return None


DAY_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ── Model file paths (must be in the same src/ directory as this file) ─────────
MODEL_FILES = {
    "accident":    os.path.join(_SRC, "accident_occurrence_model.joblib"),
    "hit_and_run": os.path.join(_SRC, "gbc_hit_and_run_model.joblib"),
}

# ── Known features per model (read from feature_names_in_ at load time) ────────
ACC_FEATURES = ['POSTED_SPEED_LIMIT', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
                'TRAFFICWAY_TYPE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH']

HR_FEATURES  = ['POSTED_SPEED_LIMIT', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
                'TRAFFICWAY_TYPE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
                'IS_WEEKEND', 'IS_RUSH_HOUR']


# ──────────────────────────────────────────────
# Data loading & cleaning
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Loading crash data...")
def load_crash_data():
    path = _resolve_crash_csv()
    if path is None:
        raise FileNotFoundError("No crash CSV found.")
    df = pd.read_csv(path, low_memory=False)

    df['CRASH_DATE']        = pd.to_datetime(df['CRASH_DATE'])
    df['CRASH_HOUR']        = df['CRASH_DATE'].dt.hour
    df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek
    df['CRASH_MONTH']       = df['CRASH_DATE'].dt.month

    # ── df1: road/environment conditions ────────────────────────────────
    df1 = df[[
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
        'ALIGNMENT', 'TRAFFICWAY_TYPE', 'LANE_CNT', 'POSTED_SPEED_LIMIT',
        'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'INTERSECTION_RELATED_I',
        'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
        'FIRST_CRASH_TYPE'
    ]].copy()

    df1.dropna(inplace=True)
    df1['LANE_CNT'] = pd.to_numeric(df1['LANE_CNT'], errors='coerce')
    df1.dropna(subset=['LANE_CNT'], inplace=True)
    df1['LANE_CNT'] = df1['LANE_CNT'].astype(int)

    cat_cols_1 = [
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
        'ALIGNMENT', 'TRAFFICWAY_TYPE', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
        'FIRST_CRASH_TYPE'
    ]
    for col in cat_cols_1:
        df1 = df1[df1[col].str.upper().str.strip() != 'UNKNOWN']
        df1 = df1[df1[col].str.strip() != '']

    df1 = df1[df1['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df1 = df1[(df1['POSTED_SPEED_LIMIT'] > 0) & (df1['POSTED_SPEED_LIMIT'] <= 100)]
    df1 = df1[(df1['LANE_CNT'] > 0) & (df1['LANE_CNT'] <= 20)]
    df1 = df1[df1['CRASH_HOUR'].between(0, 23)]
    df1 = df1[df1['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df1 = df1[df1['CRASH_MONTH'].between(1, 12)]
    df1.reset_index(drop=True, inplace=True)

    # ── df2: severity / damage ───────────────────────────────────────────
    df2 = df[[
        'FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
        'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE',
        'INTERSECTION_RELATED_I', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'DAMAGE', 'NUM_UNITS',
        'HIT_AND_RUN_I'
    ]].copy()

    df2.dropna(inplace=True)

    cat_cols_2 = [
        'FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
        'ROADWAY_SURFACE_COND', 'TRAFFICWAY_TYPE'
    ]
    for col in cat_cols_2:
        df2 = df2[df2[col].str.upper().str.strip() != 'UNKNOWN']
        df2 = df2[df2[col].str.strip() != '']

    df2 = df2[df2['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['HIT_AND_RUN_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['DAMAGE'].str.upper().str.strip().isin(['$500 OR LESS', '$501 - $1,500', 'OVER $1,500'])]
    df2 = df2[(df2['POSTED_SPEED_LIMIT'] > 0) & (df2['POSTED_SPEED_LIMIT'] <= 100)]
    df2 = df2[(df2['NUM_UNITS'] > 0) & (df2['NUM_UNITS'] <= 50)]
    df2 = df2[df2['CRASH_HOUR'].between(0, 23)]
    df2 = df2[df2['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df2.reset_index(drop=True, inplace=True)

    return df1, df2


def _split_and_clean(df):
    """Same cleaning as load_crash_data() but accepts a raw DataFrame directly."""
    df = df.copy()
    df['CRASH_DATE']        = pd.to_datetime(df['CRASH_DATE'])
    df['CRASH_HOUR']        = df['CRASH_DATE'].dt.hour
    df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek
    df['CRASH_MONTH']       = df['CRASH_DATE'].dt.month

    df1 = df[[
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
        'ALIGNMENT', 'TRAFFICWAY_TYPE', 'LANE_CNT', 'POSTED_SPEED_LIMIT',
        'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'INTERSECTION_RELATED_I',
        'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', 'FIRST_CRASH_TYPE'
    ]].copy().dropna()
    df1['LANE_CNT'] = pd.to_numeric(df1['LANE_CNT'], errors='coerce')
    df1.dropna(subset=['LANE_CNT'], inplace=True)
    df1['LANE_CNT'] = df1['LANE_CNT'].astype(int)
    for col in ['WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
                'ALIGNMENT', 'TRAFFICWAY_TYPE', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'FIRST_CRASH_TYPE']:
        df1 = df1[df1[col].str.upper().str.strip() != 'UNKNOWN']
        df1 = df1[df1[col].str.strip() != '']
    df1 = df1[df1['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df1 = df1[(df1['POSTED_SPEED_LIMIT'] > 0) & (df1['POSTED_SPEED_LIMIT'] <= 100)]
    df1 = df1[(df1['LANE_CNT'] > 0) & (df1['LANE_CNT'] <= 20)]
    df1 = df1[df1['CRASH_HOUR'].between(0, 23)]
    df1 = df1[df1['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df1 = df1[df1['CRASH_MONTH'].between(1, 12)]
    df1.reset_index(drop=True, inplace=True)

    df2 = df[[
        'FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
        'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE',
        'INTERSECTION_RELATED_I', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'DAMAGE',
        'NUM_UNITS', 'HIT_AND_RUN_I'
    ]].copy().dropna()
    for col in ['FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
                'ROADWAY_SURFACE_COND', 'TRAFFICWAY_TYPE']:
        df2 = df2[df2[col].str.upper().str.strip() != 'UNKNOWN']
        df2 = df2[df2[col].str.strip() != '']
    df2 = df2[df2['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['HIT_AND_RUN_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['DAMAGE'].str.upper().str.strip().isin(['$500 OR LESS', '$501 - $1,500', 'OVER $1,500'])]
    df2 = df2[(df2['POSTED_SPEED_LIMIT'] > 0) & (df2['POSTED_SPEED_LIMIT'] <= 100)]
    df2 = df2[(df2['NUM_UNITS'] > 0) & (df2['NUM_UNITS'] <= 50)]
    df2 = df2[df2['CRASH_HOUR'].between(0, 23)]
    df2 = df2[df2['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df2.reset_index(drop=True, inplace=True)
    return df1, df2


# ──────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load both pre-trained models from disk. Cached so disk is only hit once."""
    models = {}
    for key, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception as exc:
                st.warning(f"Could not load {os.path.basename(path)}: {exc}")
                models[key] = None
        else:
            models[key] = None
    return models


def _prepare_X(raw_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Given a dataframe and a list of feature names, return a numeric-only
    DataFrame ready for model.predict().

    - Engineers IS_WEEKEND and IS_RUSH_HOUR if needed.
    - Label-encodes any remaining string columns using the data itself
      (consistent with how both models were originally trained).
    - Fills NaNs with -1.
    """
    df = raw_df.copy()

    # Time-derived features
    if 'CRASH_DATE' in df.columns and 'CRASH_HOUR' not in df.columns:
        df['CRASH_DATE']        = pd.to_datetime(df['CRASH_DATE'])
        df['CRASH_HOUR']        = df['CRASH_DATE'].dt.hour
        df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek
        df['CRASH_MONTH']       = df['CRASH_DATE'].dt.month

    if 'IS_WEEKEND' in features:
        df['IS_WEEKEND'] = df['CRASH_DAY_OF_WEEK'].isin([5, 6]).astype(int)
    if 'IS_RUSH_HOUR' in features:
        df['IS_RUSH_HOUR'] = (
            df['CRASH_HOUR'].between(6, 9) | df['CRASH_HOUR'].between(15, 19)
        ).astype(int)

    Xf = df[features].copy()

    # Label-encode any string columns (same strategy used during training)
    for col in Xf.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        Xf[col] = le.fit_transform(Xf[col].astype(str))

    for col in Xf.columns:
        Xf[col] = pd.to_numeric(Xf[col], errors='coerce')
    Xf.fillna(-1, inplace=True)

    return Xf


def _render_model_metrics(model, X_eval: pd.DataFrame, y_true: pd.Series,
                           title: str, class_labels: list):
    """Render accuracy / precision / recall / F1, confusion matrix, ROC, and
    feature importances for a binary classifier."""
    st.markdown(f"#### {title}")

    try:
        y_pred = model.predict(X_eval)
    except Exception as exc:
        st.warning(f"Prediction failed: {exc}")
        return

    acc  = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.1%}")
    c2.metric("Precision", f"{prec:.1%}")
    c3.metric("Recall",    f"{recall:.1%}")
    c4.metric("F1 Score",  f"{f1:.3f}")

    col_cm, col_roc = st.columns(2)

    # Confusion matrix
    with col_cm:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            x=class_labels, y=class_labels,
            color_continuous_scale='OrRd',
            text_auto=True,
            labels=dict(x='Predicted', y='Actual'),
            title='Confusion Matrix',
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    with col_roc:
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_eval)[:, 1]
                auc   = roc_auc_score(y_true, proba)
                fpr, tpr, _ = roc_curve(y_true, proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                             name=f'ROC (AUC={auc:.3f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                             name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(
                    title=f'ROC Curve (AUC = {auc:.3f})',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception:
                st.info("ROC curve unavailable.")

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature':    X_eval.columns.tolist(),
            'Importance': model.feature_importances_,
        }).sort_values('Importance', ascending=False)
        fig_imp = px.bar(
            imp, x='Importance', y='Feature', orientation='h',
            title='Feature Importances',
            color='Importance', color_continuous_scale='Blues',
        )
        fig_imp.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ──────────────────────────────────────────────
# Main render function
# ──────────────────────────────────────────────

def render(chicago_geo=None):
    st.header("Transportation Dashboard")
    st.markdown(
        "Traffic crash patterns across Chicago — road conditions, timing, "
        "crash types, and damage severity."
    )

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="crash")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(
            domain="transportation",
            local_csv=None,
            label="Upload a crash dataset"
        )

    try:
        df1, df2 = load_crash_data()
    except FileNotFoundError:
        st.info("No local crash data found. Fetching the latest data from the Chicago Data Portal…")
        try:
            import data_fetcher
            data_fetcher.fetch_crashes(force=True)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(
                f"Auto-fetch failed: {exc}\n\n"
                "You can also download the file manually from the "
                "[Chicago Data Portal — Traffic Crashes]"
                "(https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if) "
                "and place it in the `src/` folder."
            )
        return

    # ── Section 1: Temporal patterns ────────────────────────────────────
    st.subheader("Crash Timing")
    col_h, col_d, col_m = st.columns(3)

    with col_h:
        hourly = df1.groupby('CRASH_HOUR').size().reset_index(name='Crashes')
        fig_h = px.bar(
            hourly, x='CRASH_HOUR', y='Crashes',
            labels={'CRASH_HOUR': 'Hour of Day', 'Crashes': 'Number of Crashes'},
            title='Crashes by Hour of Day',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_h.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_h, width="stretch")

    with col_d:
        daily = df1.groupby('CRASH_DAY_OF_WEEK').size().reset_index(name='Crashes')
        daily['Day'] = daily['CRASH_DAY_OF_WEEK'].map(DAY_LABELS)
        fig_d = px.bar(
            daily, x='Day', y='Crashes',
            labels={'Day': 'Day of Week', 'Crashes': 'Number of Crashes'},
            title='Crashes by Day of Week',
            color='Crashes', color_continuous_scale='Oranges',
            category_orders={'Day': list(DAY_LABELS.values())}
        )
        fig_d.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_d, width="stretch")

    with col_m:
        monthly = df1.groupby('CRASH_MONTH').size().reset_index(name='Crashes')
        monthly['Month'] = monthly['CRASH_MONTH'].map(MONTH_LABELS)
        fig_m = px.bar(
            monthly, x='Month', y='Crashes',
            labels={'Month': 'Month', 'Crashes': 'Number of Crashes'},
            title='Crashes by Month',
            color='Crashes', color_continuous_scale='Oranges',
            category_orders={'Month': list(MONTH_LABELS.values())}
        )
        fig_m.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_m, width="stretch")

    peak_hour  = hourly.loc[hourly["Crashes"].idxmax(), "CRASH_HOUR"]
    peak_day   = daily.loc[daily["Crashes"].idxmax(), "Day"]
    peak_month = monthly.loc[monthly["Crashes"].idxmax(), "Month"]
    st.info(
        f"**When crashes happen most:** Peak hour is **{peak_hour}:00** "
        f"({'evening rush' if 15 <= peak_hour <= 19 else 'morning rush' if 6 <= peak_hour <= 9 else 'overnight' if peak_hour < 6 else 'midday'}), "
        f"peak day is **{peak_day}**, and peak month is **{peak_month}**. "
        "Targeted enforcement and road safety campaigns during these windows could meaningfully reduce crash frequency."
    )

    # ── Crash Density Heatmap ────────────────────────────────────────────
    st.divider()
    st.subheader("Crash Location Density")
    path = _resolve_crash_csv()
    if path and chicago_geo:
        try:
            raw_coords = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False)
            raw_coords["LATITUDE"]  = pd.to_numeric(raw_coords["LATITUDE"],  errors="coerce")
            raw_coords["LONGITUDE"] = pd.to_numeric(raw_coords["LONGITUDE"], errors="coerce")
            coords = raw_coords.dropna(subset=["LATITUDE", "LONGITUDE"])
            coords = coords[(coords["LATITUDE"] > 41.6) & (coords["LATITUDE"] < 42.1)]

            if not coords.empty:
                geo_url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
                gdf_ca  = gpd.read_file(geo_url)
                gdf_ca["area_num_1"] = gdf_ca["area_num_1"].astype(int)

                crash_pts = gpd.GeoDataFrame(
                    coords,
                    geometry=gpd.points_from_xy(coords["LONGITUDE"], coords["LATITUDE"]),
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(crash_pts, gdf_ca[["area_num_1", "community", "geometry"]],
                                   how="inner", predicate="within")
                crash_counts = joined.groupby(["area_num_1", "community"]).size().reset_index(name="Crash Count")
                crash_counts["area_num_1"] = crash_counts["area_num_1"].astype(str)

                fig_density = px.choropleth_map(
                    crash_counts, geojson=chicago_geo,
                    locations="area_num_1", featureidkey="properties.area_num_1",
                    color="Crash Count", color_continuous_scale="YlOrRd",
                    map_style=mapbox_style,
                    zoom=9.5, center={"lat": 41.8358, "lon": -87.6877},
                    hover_name="community",
                    hover_data={"Crash Count": True, "area_num_1": False},
                    title="Crashes by Community Area",
                    opacity=0.7,
                )
                fig_density.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
                st.plotly_chart(fig_density, width="stretch")

                top3      = crash_counts.nlargest(3, "Crash Count")
                top_names = ", ".join(
                    f"**{r['community']}** ({r['Crash Count']:,})"
                    for _, r in top3.iterrows()
                )
                st.info(
                    f"**Crash hotspots by community area:** The highest crash counts are in {top_names}. "
                    "These areas should be prioritized for safety interventions such as traffic calming, "
                    "signal improvements, or targeted enforcement."
                )
        except Exception as exc:
            st.warning(f"Could not load crash location data: {exc}")

    st.divider()

    # ── Section 2: Road & environment conditions ─────────────────────────
    st.subheader("Road & Environment Conditions")

    condition_options = {
        'Weather Condition':         'WEATHER_CONDITION',
        'Lighting Condition':        'LIGHTING_CONDITION',
        'Roadway Surface Condition': 'ROADWAY_SURFACE_COND',
        'Road Defect':               'ROAD_DEFECT',
        'Traffic Control Device':    'TRAFFIC_CONTROL_DEVICE',
        'Alignment':                 'ALIGNMENT',
    }
    selected_condition = st.selectbox(
        "Breakdown by condition",
        list(condition_options.keys()),
        key="infra_condition"
    )
    col = condition_options[selected_condition]
    cond_counts = df1[col].value_counts().reset_index()
    cond_counts.columns = [selected_condition, 'Crashes']

    fig_cond = px.bar(
        cond_counts, x='Crashes', y=selected_condition,
        orientation='h',
        title=f'Crash Count by {selected_condition}',
        color='Crashes', color_continuous_scale='Oranges'
    )
    fig_cond.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cond, width="stretch")

    top_cond       = cond_counts.iloc[0][selected_condition]
    top_cond_count = cond_counts.iloc[0]["Crashes"]
    total_crashes  = cond_counts["Crashes"].sum()
    top_cond_pct   = top_cond_count / total_crashes * 100
    st.info(
        f"**{selected_condition} insight:** **{top_cond}** accounts for "
        f"**{top_cond_pct:.1f}% of all crashes** ({top_cond_count:,} incidents). "
        "This is the highest-risk condition category. Infrastructure improvements or signage "
        "targeting this condition would have the greatest safety impact."
    )

    st.divider()

    # ── Section 3: Crash type breakdown ──────────────────────────────────
    st.subheader("Crash Type Breakdown")
    col_ct, col_tw = st.columns(2)

    with col_ct:
        ct_counts = df1['FIRST_CRASH_TYPE'].value_counts().head(12).reset_index()
        ct_counts.columns = ['Crash Type', 'Count']
        fig_ct = px.bar(
            ct_counts, x='Count', y='Crash Type', orientation='h',
            title='Top Crash Types',
            color='Count', color_continuous_scale='Oranges'
        )
        fig_ct.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_ct, width="stretch")

    with col_tw:
        tw_counts = df2['TRAFFICWAY_TYPE'].value_counts().head(12).reset_index()
        tw_counts.columns = ['Trafficway Type', 'Count']
        fig_tw = px.bar(
            tw_counts, x='Count', y='Trafficway Type', orientation='h',
            title='Crashes by Trafficway Type',
            color='Count', color_continuous_scale='Oranges'
        )
        fig_tw.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_tw, width="stretch")

    top_crash_type = ct_counts.iloc[0]["Crash Type"]
    top_crash_pct  = ct_counts.iloc[0]["Count"] / ct_counts["Count"].sum() * 100
    st.info(
        f"**Most common crash type: {top_crash_type}** ({top_crash_pct:.1f}% of crashes). "
        "Understanding dominant crash types informs the design of intersections, signage, and driver education programs."
    )

    st.divider()

    # ── Section 4: Damage severity ────────────────────────────────────────
    st.subheader("Damage Severity")
    col_dmg, col_hr = st.columns(2)

    with col_dmg:
        damage_order = ['$500 OR LESS', '$501 - $1,500', 'OVER $1,500']
        dmg_counts = (
            df2['DAMAGE'].str.upper().str.strip()
            .value_counts()
            .reindex(damage_order, fill_value=0)
            .reset_index()
        )
        dmg_counts.columns = ['Damage Level', 'Crashes']
        fig_dmg = px.pie(
            dmg_counts, names='Damage Level', values='Crashes',
            title='Crash Distribution by Damage Level',
            color_discrete_sequence=px.colors.sequential.Oranges[2:]
        )
        st.plotly_chart(fig_dmg, width="stretch")

    with col_hr:
        hr_counts = (
            df2['HIT_AND_RUN_I'].str.upper().str.strip()
            .map({'Y': 'Hit and Run', 'N': 'Not Hit and Run'})
            .value_counts()
            .reset_index()
        )
        hr_counts.columns = ['Type', 'Crashes']
        fig_hr = px.pie(
            hr_counts, names='Type', values='Crashes',
            title='Hit and Run vs. Not Hit and Run',
            color_discrete_sequence=['#fd8d3c', '#fdbe85']
        )
        st.plotly_chart(fig_hr, width="stretch")

    over_1500_pct = dmg_counts.loc[dmg_counts["Damage Level"] == "OVER $1,500", "Crashes"].sum() / dmg_counts["Crashes"].sum() * 100
    hr_pct        = df2["HIT_AND_RUN_I"].str.upper().str.strip().eq("Y").mean() * 100
    st.info(
        f"**Damage & hit-and-run:** {over_1500_pct:.1f}% of crashes result in damage over $1,500, "
        f"and **{hr_pct:.1f}% are hit-and-run** incidents. "
        + ("A high hit-and-run rate points to enforcement gaps. Increased camera coverage or penalties may deter this behavior."
           if hr_pct > 15 else
           "Hit-and-run rates are within a typical range, but continued monitoring is recommended.")
    )

    st.divider()

    # ── Section 5: Speed limit & lane count distributions ─────────────────
    st.subheader("Road Characteristics")
    col_sp, col_ln = st.columns(2)

    with col_sp:
        speed_counts = df1['POSTED_SPEED_LIMIT'].value_counts().sort_index().reset_index()
        speed_counts.columns = ['Speed Limit (mph)', 'Crashes']
        fig_sp = px.bar(
            speed_counts, x='Speed Limit (mph)', y='Crashes',
            title='Crashes by Posted Speed Limit',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_sp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sp, width="stretch")

    with col_ln:
        lane_counts = df1['LANE_CNT'].value_counts().sort_index().reset_index()
        lane_counts.columns = ['Lane Count', 'Crashes']
        fig_ln = px.bar(
            lane_counts, x='Lane Count', y='Crashes',
            title='Crashes by Number of Lanes',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_ln.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ln, width="stretch")

    st.divider()

    # ── Section 6: Intersection vs. non-intersection ──────────────────────
    st.subheader("Intersection-Related Crashes")
    col_int, col_units = st.columns(2)

    with col_int:
        int_counts = (
            df2['INTERSECTION_RELATED_I'].str.upper().str.strip()
            .map({'Y': 'Intersection-Related', 'N': 'Not Intersection-Related'})
            .value_counts()
            .reset_index()
        )
        int_counts.columns = ['Type', 'Crashes']
        fig_int = px.pie(
            int_counts, names='Type', values='Crashes',
            title='Intersection vs. Non-Intersection Crashes',
            color_discrete_sequence=['#e6550d', '#fdae6b']
        )
        st.plotly_chart(fig_int, width="stretch")

    with col_units:
        unit_counts = df2['NUM_UNITS'].value_counts().sort_index().reset_index()
        unit_counts.columns = ['Units Involved', 'Crashes']
        fig_units = px.bar(
            unit_counts.head(15), x='Units Involved', y='Crashes',
            title='Crashes by Number of Units Involved',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_units.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_units, width="stretch")

    # ── Section 7: Scatterplots ────────────────────────────────────────────
    st.divider()
    st.subheader("Speed and Lane Analysis Scatterplots")
    scatter_sample = df1.sample(min(5000, len(df1)), random_state=42) if len(df1) > 5000 else df1
    col_sc1, col_sc2 = st.columns(2)

    with col_sc1:
        fig_sc1 = px.scatter(
            scatter_sample, x="POSTED_SPEED_LIMIT", y="CRASH_HOUR",
            color="LIGHTING_CONDITION",
            title="Speed Limit vs Crash Hour by Lighting",
            labels={"POSTED_SPEED_LIMIT": "Posted Speed Limit (mph)", "CRASH_HOUR": "Hour of Day"},
            opacity=0.4,
        )
        fig_sc1.update_layout(margin={"t": 30}, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig_sc1, width="stretch")

    with col_sc2:
        fig_sc2 = px.scatter(
            scatter_sample, x="LANE_CNT", y="POSTED_SPEED_LIMIT",
            color="ROADWAY_SURFACE_COND",
            title="Lane Count vs Speed Limit by Surface Condition",
            labels={"LANE_CNT": "Number of Lanes", "POSTED_SPEED_LIMIT": "Speed Limit (mph)"},
            opacity=0.4,
        )
        fig_sc2.update_layout(margin={"t": 30}, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig_sc2, width="stretch")

    st.info(
        "**Scatter analysis:** These plots reveal relationships between road design and crash timing. "
        "Clusters at specific speed-hour combinations highlight when certain road types are most dangerous."
    )

    # ── Section 8: Moran's I Spatial Autocorrelation ──────────────────────
    st.divider()
    try:
        if path:
            raw_for_moran = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False)
            raw_for_moran["LATITUDE"]  = pd.to_numeric(raw_for_moran["LATITUDE"],  errors="coerce")
            raw_for_moran["LONGITUDE"] = pd.to_numeric(raw_for_moran["LONGITUDE"], errors="coerce")
            raw_for_moran = raw_for_moran.dropna(subset=["LATITUDE", "LONGITUDE"])
            raw_for_moran = raw_for_moran[
                (raw_for_moran["LATITUDE"] > 41.6) & (raw_for_moran["LATITUDE"] < 42.1)
            ]

            if len(raw_for_moran) > 100:
                geo_url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
                gdf_ca  = gpd.read_file(geo_url)
                gdf_ca["area_num_1"] = gdf_ca["area_num_1"].astype(int)

                crash_points = gpd.GeoDataFrame(
                    raw_for_moran,
                    geometry=gpd.points_from_xy(raw_for_moran["LONGITUDE"], raw_for_moran["LATITUDE"]),
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(
                    crash_points,
                    gdf_ca[["area_num_1", "community", "geometry"]],
                    how="inner", predicate="within",
                )
                crash_by_ca = joined.groupby("area_num_1").size().reset_index(name="crash_count")
                gdf_merged  = gdf_ca.merge(crash_by_ca, on="area_num_1", how="inner")
                gdf_merged["area_num_str"] = gdf_merged["area_num_1"].astype(str)

                if len(gdf_merged) >= 10 and chicago_geo:
                    map_utils.render_moran_analysis(
                        gdf=gdf_merged,
                        value_col="crash_count",
                        name_col="community",
                        id_col="area_num_str",
                        geojson=chicago_geo,
                        featureidkey="properties.area_num_1",
                        key_prefix="crash_moran",
                        map_style=mapbox_style,
                    )
    except Exception as exc:
        st.warning(f"Could not compute spatial autocorrelation: {exc}")

    # ── Section 9: ML Predictor (generic, uses df2) ───────────────────────
    st.divider()
    _CRASH_DEFAULT_FEATURES = [
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
        'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE', 'INTERSECTION_RELATED_I',
        'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'NUM_UNITS', 'HIT_AND_RUN_I',
        'FIRST_CRASH_TYPE', 'CRASH_TYPE',
    ]
    ml_predictor.render_predictor(
        df2,
        key_prefix="crash",
        default_target="DAMAGE",
        default_features=_CRASH_DEFAULT_FEATURES,
    )

    # ── Section 10: Pre-trained models ────────────────────────────────────
    st.divider()
    st.subheader("🧠 Pre-trained Models")
    st.markdown(
        "The models below were trained on the full Chicago crash dataset. "
        "The **accident occurrence model** (Random Forest) predicts whether a crash "
        "is likely to occur given road conditions. The **hit-and-run model** (Gradient "
        "Boosting) predicts whether a crash will be a hit-and-run."
    )

    models = load_models()
    model_acc = models.get("accident")
    model_hr  = models.get("hit_and_run")

    # Show a status banner for each model
    st.markdown("**Model status:**")
    s1, s2 = st.columns(2)
    s1.success("✅ Accident occurrence model loaded" if model_acc else "❌ accident_occurrence_model.joblib not found in src/")
    s2.success("✅ Hit-and-run model loaded"          if model_hr  else "❌ gbc_hit_and_run_model.joblib not found in src/")

    tab_eval, tab_predict = st.tabs(["📊 Model Evaluation", "🔮 Make a Prediction"])

    # ── Tab 1: Evaluation on live data ────────────────────────────────────
    with tab_eval:
        st.markdown(
            "Evaluate both models against the currently loaded crash dataset. "
            "Metrics are computed on the full cleaned dataset (not a held-out split)."
        )

        # ── Accident occurrence model evaluation ──────────────────────────
        if model_acc is not None:
            st.markdown("---")
            try:
                # The accident model predicts whether a crash occurred (binary).
                # We proxy 'occurrence' using NUM_UNITS > 0 as the positive label,
                # but since all rows in df2 ARE crashes, we evaluate using CRASH_TYPE
                # (INJURY/TOW = 1, NO INJURY/DRIVE AWAY = 0) as a meaningful proxy target.
                acc_proxy = df2['CRASH_TYPE'].str.upper().str.strip()
                y_acc = (acc_proxy != 'NO INJURY / DRIVE AWAY').astype(int)

                X_acc = _prepare_X(df2, ACC_FEATURES)
                valid = y_acc.notna()
                X_acc = X_acc[valid].reset_index(drop=True)
                y_acc = y_acc[valid].reset_index(drop=True)

                _render_model_metrics(
                    model_acc, X_acc, y_acc,
                    title="Accident Occurrence Model (Random Forest)",
                    class_labels=["No Injury/Drive Away", "Injury or Tow"]
                )
            except Exception as exc:
                st.warning(f"Could not evaluate accident model: {exc}")
        else:
            st.info("Place `accident_occurrence_model.joblib` in the `src/` folder to enable evaluation.")

        # ── Hit-and-run model evaluation ──────────────────────────────────
        if model_hr is not None:
            st.markdown("---")
            try:
                y_hr  = df2['HIT_AND_RUN_I'].str.upper().str.strip().map({'Y': 1, 'N': 0})
                X_hr  = _prepare_X(df2, HR_FEATURES)
                valid = y_hr.notna()
                X_hr  = X_hr[valid].reset_index(drop=True)
                y_hr  = y_hr[valid].reset_index(drop=True)

                _render_model_metrics(
                    model_hr, X_hr, y_hr,
                    title="Hit-and-Run Model (Gradient Boosting)",
                    class_labels=["Not Hit-and-Run", "Hit-and-Run"]
                )
            except Exception as exc:
                st.warning(f"Could not evaluate hit-and-run model: {exc}")
        else:
            st.info("Place `gbc_hit_and_run_model.joblib` in the `src/` folder to enable evaluation.")

    # ── Tab 2: Single-record prediction ───────────────────────────────────
    with tab_predict:
        st.markdown("Fill in road conditions to get a prediction from each model.")

        # Shared inputs (used by both models)
        st.markdown("#### Conditions")
        c1, c2, c3 = st.columns(3)
        with c1:
            weather   = st.selectbox("Weather",           sorted(df2['WEATHER_CONDITION'].unique()),  key="pm_weather")
            lighting  = st.selectbox("Lighting",          sorted(df2['LIGHTING_CONDITION'].unique()), key="pm_lighting")
        with c2:
            trafficway = st.selectbox("Trafficway Type",  sorted(df2['TRAFFICWAY_TYPE'].unique()),    key="pm_trafficway")
            speed      = st.slider("Posted Speed Limit (mph)", 5, 100, 30, step=5,                   key="pm_speed")
        with c3:
            hour  = st.slider("Crash Hour",      0, 23, 12,  key="pm_hour")
            dow   = st.selectbox("Day of Week",  list(DAY_LABELS.items()),
                                 format_func=lambda x: x[1], key="pm_dow")
            month = st.selectbox("Month",        list(MONTH_LABELS.items()),
                                 format_func=lambda x: x[1], key="pm_month")

        is_weekend   = int(dow[0] in [5, 6])
        is_rush_hour = int(6 <= hour <= 9 or 15 <= hour <= 19)

        if st.button("Run Predictions", type="primary"):
            base_row = {
                'POSTED_SPEED_LIMIT': speed,
                'WEATHER_CONDITION':  weather,
                'LIGHTING_CONDITION': lighting,
                'TRAFFICWAY_TYPE':    trafficway,
                'CRASH_HOUR':         hour,
                'CRASH_DAY_OF_WEEK':  dow[0],
                'CRASH_MONTH':        month[0],
                'IS_WEEKEND':         is_weekend,
                'IS_RUSH_HOUR':       is_rush_hour,
            }

            pred_col1, pred_col2 = st.columns(2)

            # Accident occurrence prediction
            with pred_col1:
                if model_acc is not None:
                    try:
                        X_single_acc = _prepare_X(pd.DataFrame([base_row]), ACC_FEATURES)
                        pred_acc     = model_acc.predict(X_single_acc)[0]
                        label_acc    = "🚨 Likely Injury / Tow" if pred_acc == 1 else "✅ Likely No Injury / Drive Away"
                        st.metric("Accident Occurrence Model", label_acc)
                        if hasattr(model_acc, 'predict_proba'):
                            proba_acc = model_acc.predict_proba(X_single_acc)[0]
                            fig_p = px.bar(
                                x=["No Injury/Drive Away", "Injury or Tow"],
                                y=proba_acc,
                                labels={'x': 'Outcome', 'y': 'Probability'},
                                title='Prediction Probabilities',
                                color=proba_acc,
                                color_continuous_scale='Oranges',
                            )
                            fig_p.update_layout(coloraxis_showscale=False, showlegend=False)
                            st.plotly_chart(fig_p, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Accident model prediction failed: {exc}")
                else:
                    st.info("Accident model not loaded.")

            # Hit-and-run prediction
            with pred_col2:
                if model_hr is not None:
                    try:
                        X_single_hr = _prepare_X(pd.DataFrame([base_row]), HR_FEATURES)
                        pred_hr     = model_hr.predict(X_single_hr)[0]
                        label_hr    = "⚠️ Likely Hit-and-Run" if pred_hr == 1 else "✅ Likely Not a Hit-and-Run"
                        st.metric("Hit-and-Run Model", label_hr)
                        if hasattr(model_hr, 'predict_proba'):
                            proba_hr = model_hr.predict_proba(X_single_hr)[0]
                            fig_q = px.bar(
                                x=["Not Hit-and-Run", "Hit-and-Run"],
                                y=proba_hr,
                                labels={'x': 'Outcome', 'y': 'Probability'},
                                title='Prediction Probabilities',
                                color=proba_hr,
                                color_continuous_scale='Reds',
                            )
                            fig_q.update_layout(coloraxis_showscale=False, showlegend=False)
                            st.plotly_chart(fig_q, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Hit-and-run model prediction failed: {exc}")
                else:
                    st.info("Hit-and-run model not loaded.")