import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import file_loader
import ml_predictor

_SRC = os.path.dirname(os.path.abspath(__file__))
# Prefer the auto-fetched file; fall back to the bundled snapshot
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

# ──────────────────────────────────────────────
# Data loading & cleaning  (notebook-faithful)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Loading crash data...")
def load_crash_data():
    path = _resolve_crash_csv()
    if path is None:
        raise FileNotFoundError("No crash CSV found.")
    df = pd.read_csv(path, low_memory=False)

    # Parse date and extract time fields (notebook cell 4)
    df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'])
    df['CRASH_HOUR']        = df['CRASH_DATE'].dt.hour
    df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['CRASH_MONTH']       = df['CRASH_DATE'].dt.month

    # ── df1: road/environment conditions (notebook cell 5) ──────────────
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

    # ── df2: severity / damage (notebook cell 6) ────────────────────────
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


# ──────────────────────────────────────────────
# Main render function — called from dashboard.py
# ──────────────────────────────────────────────

def _split_and_clean(df):
    """Same cleaning as load_crash_data() but accepts a raw DataFrame directly."""
    df = df.copy()
    df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'])
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


def render(chicago_geo=None):
    st.header("Transportation Dashboard")
    st.markdown(
        "Traffic crash patterns across Chicago — road conditions, timing, "
        "crash types, and damage severity."
    )

    with st.expander("Upload a supplemental dataset"):
        uploaded_df, _ = file_loader.uploader(
            domain="transportation",
            local_csv=None,
            label="Upload a crash dataset"
        )

    if uploaded_df is not None:
        df1, df2 = _split_and_clean(uploaded_df)
    else:
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
        st.plotly_chart(fig_h, use_container_width=True)

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
        st.plotly_chart(fig_d, use_container_width=True)

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
        st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # ── Section 2: Road & environment conditions ─────────────────────────
    st.subheader("Road & Environment Conditions")

    condition_options = {
        'Weather Condition':        'WEATHER_CONDITION',
        'Lighting Condition':       'LIGHTING_CONDITION',
        'Roadway Surface Condition':'ROADWAY_SURFACE_COND',
        'Road Defect':              'ROAD_DEFECT',
        'Traffic Control Device':   'TRAFFIC_CONTROL_DEVICE',
        'Alignment':                'ALIGNMENT',
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
    st.plotly_chart(fig_cond, use_container_width=True)

    st.divider()

    # ── Section 3: Crash type breakdown ─────────────────────────────────
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
        fig_ct.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    with col_tw:
        tw_counts = df2['TRAFFICWAY_TYPE'].value_counts().head(12).reset_index()
        tw_counts.columns = ['Trafficway Type', 'Count']
        fig_tw = px.bar(
            tw_counts, x='Count', y='Trafficway Type', orientation='h',
            title='Crashes by Trafficway Type',
            color='Count', color_continuous_scale='Oranges'
        )
        fig_tw.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tw, use_container_width=True)

    st.divider()

    # ── Section 4: Damage severity ───────────────────────────────────────
    st.subheader("Damage Severity")
    col_dmg, col_hr = st.columns(2)

    with col_dmg:
        damage_order = ['$500 OR LESS', '$501 - $1,500', 'OVER $1,500']
        dmg_counts = (
            df2['DAMAGE']
            .str.upper().str.strip()
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
        st.plotly_chart(fig_dmg, use_container_width=True)

    with col_hr:
        hr_counts = (
            df2['HIT_AND_RUN_I']
            .str.upper().str.strip()
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
        st.plotly_chart(fig_hr, use_container_width=True)

    st.divider()

    # ── Section 5: Speed limit & lane count distributions ────────────────
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
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_ln:
        lane_counts = df1['LANE_CNT'].value_counts().sort_index().reset_index()
        lane_counts.columns = ['Lane Count', 'Crashes']
        fig_ln = px.bar(
            lane_counts, x='Lane Count', y='Crashes',
            title='Crashes by Number of Lanes',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_ln.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ln, use_container_width=True)

    st.divider()

    # ── Section 6: Intersection vs. non-intersection ─────────────────────
    st.subheader("Intersection-Related Crashes")
    col_int, col_units = st.columns(2)

    with col_int:
        int_counts = (
            df2['INTERSECTION_RELATED_I']
            .str.upper().str.strip()
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
        st.plotly_chart(fig_int, use_container_width=True)

    with col_units:
        unit_counts = df2['NUM_UNITS'].value_counts().sort_index().reset_index()
        unit_counts.columns = ['Units Involved', 'Crashes']
        fig_units = px.bar(
            unit_counts.head(15), x='Units Involved', y='Crashes',
            title='Crashes by Number of Units Involved',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_units.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_units, use_container_width=True)

    # ── Section 7: ML Predictions ─────────────────────────────────────────────
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