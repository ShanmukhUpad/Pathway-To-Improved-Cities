import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import requests
import folium
from streamlit_folium import st_folium
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Running energy forecast...")
def _forecast_area_energy(area_name: str, fc_series_json: str):
    fc_series = pd.read_json(fc_series_json, typ="series").values.astype(float)
    months = np.arange(1, 13)
    lag1 = np.concatenate([[np.nan], fc_series[:-1]])
    lag2 = np.concatenate([[np.nan, np.nan], fc_series[:-2]])
    fc_df = pd.DataFrame({"month": months, "kwh": fc_series, "lag1": lag1, "lag2": lag2})
    fc_df = fc_df.dropna()
    if len(fc_df) < 5:
        return None
    X = fc_df[["month", "lag1", "lag2"]].values
    y = fc_df["kwh"].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    pred_kwh = max(0.0, float(model.predict([[13, fc_series[-1], fc_series[-2]]])[0]))
    return {"pred_kwh": pred_kwh, "latest_kwh": float(fc_series[-1])}

# ── Palette & constants ───────────────────────────────────────────────────────
PALETTE = {
    "Residential": "#378ADD",
    "Commercial":  "#1D9E75",
    "Industrial":  "#D85A30",
}
MARKERS = {"Residential": "o", "Commercial": "s", "Industrial": "^"}

MONTHLY_COLS = [
    "KWH JANUARY 2010", "KWH FEBRUARY 2010", "KWH MARCH 2010",
    "KWH APRIL 2010",   "KWH MAY 2010",      "KWH JUNE 2010",
    "KWH JULY 2010",    "KWH AUGUST 2010",   "KWH SEPTEMBER 2010",
    "KWH OCTOBER 2010", "KWH NOVEMBER 2010", "KWH DECEMBER 2010",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


import os as _os
_SRC_DIR = _os.path.dirname(_os.path.abspath(__file__))


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(csv_path: str | None = None):
    if csv_path is None:
        # Fallback: look alongside this file (old single-city layout)
        csv_path = _os.path.join(_SRC_DIR, "energy-usage-2010-1.csv")
    return pd.read_csv(csv_path)


# ── helper: load GeoJSON ──────────────────────────────────────────────────────
@st.cache_data
def load_chicago_geojson():
    primary = (
        "https://data.cityofchicago.org/api/geospatial/cauq-8yn6"
        "?method=export&type=GeoJSON"
    )
    try:
        r = requests.get(primary, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        mirror = (
            "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes"
            "/master/data/chicago-community-areas.geojson"
        )
        r = requests.get(mirror, timeout=10)
        r.raise_for_status()
        return r.json()


# ── helper: aggregate energy by community area ────────────────────────────────
def aggregate_by_area(dff):
    agg = (
        dff.dropna(subset=["COMMUNITY AREA NAME", "KWH MEAN 2010", "THERM MEAN 2010"])
        .groupby("COMMUNITY AREA NAME")
        .agg(
            kwh_mean    = ("KWH MEAN 2010",   "mean"),
            therm_mean  = ("THERM MEAN 2010", "mean"),
            total_kwh   = ("TOTAL KWH",       "sum"),
            total_therms= ("TOTAL THERMS",     "sum"),
            buildings   = ("KWH MEAN 2010",   "count"),
        )
        .reset_index()
    )
    agg["total_energy_kwh"] = agg["total_kwh"] + agg["total_therms"] * 29.3
    return agg


def render(city=None):
    from city_config import CityConfig
    energy_csv = None
    if isinstance(city, CityConfig) and city.energy_path and _os.path.exists(city.energy_path):
        energy_csv = city.energy_path
    elif not isinstance(city, CityConfig) or not city.has_energy_layer:
        st.info("Energy infrastructure layer not yet available for this city.")
        return
    df = load_data(energy_csv)

    # ── Header ────────────────────────────────────────────────────────────────
    st.subheader("Chicago Energy Usage — 2010")
    st.markdown(
        "Electricity and gas consumption across Chicago census blocks by building type and community area."
    )

    # ── Inline filters (not sidebar — avoids polluting other tabs) ────────────
    fc1, fc2 = st.columns(2)
    all_types = sorted(df["BUILDING TYPE"].dropna().unique())
    selected_types = fc1.multiselect(
        "Building type", options=all_types, default=all_types, key="energy_building_type",
    )
    all_areas = sorted(df["COMMUNITY AREA NAME"].dropna().unique())
    selected_areas = fc2.multiselect(
        "Community area", options=all_areas, default=all_areas, key="energy_community_area",
    )

    mask = (
        df["BUILDING TYPE"].isin(selected_types) &
        df["COMMUNITY AREA NAME"].isin(selected_areas)
    )
    dff = df[mask].copy()

    mc1, mc2 = st.columns(2)
    mc1.metric("Rows selected", f"{len(dff):,}")
    mc2.metric("Community areas", f"{dff['COMMUNITY AREA NAME'].nunique()}")
    st.markdown("---")

    # ── Energy forecast headline ──────────────────────────────────────────────
    st.subheader("Energy usage forecast")
    all_areas_fc = sorted(df["COMMUNITY AREA NAME"].dropna().unique())
    fc_area = st.selectbox("Forecast area", all_areas_fc, key="infra_fc_area")

    area_df = df[df["COMMUNITY AREA NAME"] == fc_area]
    area_monthly = area_df[MONTHLY_COLS].mean()

    if not area_monthly.isnull().all():
        fc_result = _forecast_area_energy(fc_area, pd.Series(area_monthly.values.astype(float)).to_json())
        if fc_result:
            pred_kwh   = fc_result["pred_kwh"]
            latest_kwh = fc_result["latest_kwh"]
            delta  = pred_kwh - latest_kwh
            arrow  = "▲" if delta >= 0 else "▼"
            chg_dir = "increase" if delta >= 0 else "decrease"
            st.markdown(f"""
<div style="background:rgba(55,138,221,0.1);border-left:4px solid #378ADD;
            padding:18px 22px;border-radius:8px;margin:14px 0;">
  <p style="margin:0;font-size:11px;color:#9eaec4;text-transform:uppercase;
            letter-spacing:.08em;">Energy Forecast — January 2011</p>
  <p style="margin:6px 0 2px;font-size:2.4rem;font-weight:800;color:#fff;line-height:1.1;">
    {pred_kwh:,.0f}
    <span style="font-size:1.1rem;font-weight:500;color:#88bbee;">&nbsp;kWh (mean per block)</span>
  </p>
  <p style="margin:2px 0 0;font-size:14px;color:#9eaec4;">
    in <strong style="color:#fff;">{fc_area}</strong>
    &nbsp;·&nbsp; {arrow} {abs(delta):,.0f} kWh from December 2010
  </p>
</div>""", unsafe_allow_html=True)
            fcc1, fcc2 = st.columns(2)
            fcc1.metric("Forecast (Jan 2011)", f"{pred_kwh:,.0f} kWh")
            fcc2.metric("Dec 2010 actual", f"{latest_kwh:,.0f} kWh")
    st.markdown("---")

    def no_data():
        st.warning("No data matches the current filters. Adjust filter selections.")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 1 — Scatter: KWH Mean vs Building Age
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Electricity use vs. building age")
    st.caption("Mean kWh per account vs. average building age, coloured by building type. OLS trend line per group.")

    df_scatter = dff.dropna(subset=["KWH MEAN 2010", "AVERAGE BUILDING AGE", "BUILDING TYPE"]).copy()
    df_scatter["KWH MEAN 2010"] = pd.to_numeric(df_scatter["KWH MEAN 2010"], errors="coerce")
    df_scatter = df_scatter.dropna(subset=["KWH MEAN 2010"])
    kwh_cap    = df_scatter["KWH MEAN 2010"].quantile(0.99) if len(df_scatter) else 1
    df_scatter = df_scatter[df_scatter["KWH MEAN 2010"] <= kwh_cap]

    if df_scatter.empty:
        no_data()
    else:
        fig1, ax1 = plt.subplots(figsize=(11, 5))
        fig1.patch.set_facecolor("#FAFAF8")
        ax1.set_facecolor("#FAFAF8")

        for btype, grp in df_scatter.groupby("BUILDING TYPE"):
            color = PALETTE.get(btype, "#888780")
            ax1.scatter(
                grp["AVERAGE BUILDING AGE"], grp["KWH MEAN 2010"],
                c=color, alpha=0.35, s=12, linewidths=0,
                label=btype, rasterized=True,
            )

        for btype, grp in df_scatter.groupby("BUILDING TYPE"):
            color = PALETTE.get(btype, "#888780")
            x, y  = grp["AVERAGE BUILDING AGE"].values, grp["KWH MEAN 2010"].values
            if len(x) > 1:
                m, b      = np.polyfit(x, y, 1)
                x_range   = np.linspace(x.min(), x.max(), 200)
                ax1.plot(x_range, m * x_range + b, color=color, linewidth=2.2, alpha=0.9)

        ax1.set_xlabel("Average building age (years)", fontsize=11, color="#3d3d3a")
        ax1.set_ylabel("Mean electricity use (kWh)", fontsize=11, color="#3d3d3a")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.spines[["left", "bottom"]].set_color("#c8c6bc")
        ax1.grid(axis="y", color="#e4e2d8", linewidth=0.6, linestyle="--")
        ax1.set_axisbelow(True)
        ax1.tick_params(colors="#73726c", labelsize=9)

        handles = [
            mlines.Line2D([], [], color=PALETTE[bt], marker=MARKERS[bt],
                          linestyle="-", markersize=6, linewidth=2, label=bt)
            for bt in selected_types if bt in PALETTE
        ]
        ax1.legend(handles=handles, title="Building type", fontsize=9,
                   title_fontsize=9, framealpha=0.85, edgecolor="#d3d1c7")
        ax1.annotate(
            f"Top 1% outliers removed (>{kwh_cap:,.0f} kWh)  ·  n = {len(df_scatter):,}",
            xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8, color="#888780",
        )

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 2 — Line: Monthly KWH by Building Type
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Mean monthly electricity use by building type")
    st.caption("Average kWh per census block for each month of 2010, by building type.")

    df_line = dff.dropna(subset=MONTHLY_COLS + ["BUILDING TYPE"])

    if df_line.empty:
        no_data()
    else:
        monthly_means = (
            df_line.groupby("BUILDING TYPE")[MONTHLY_COLS]
            .mean()
            .T
            .set_index(pd.Index(MONTHS, name="Month"))
        )
        monthly_means = monthly_means[
            [c for c in monthly_means.columns if c in selected_types and c in PALETTE]
        ]

        fig2, ax = plt.subplots(figsize=(11, 6))

        for btype in monthly_means.columns:
            ax.plot(
                monthly_means.index,
                monthly_means[btype],
                color=PALETTE[btype],
                marker=MARKERS[btype],
                linewidth=2.4,
                markersize=7,
                label=btype,
            )

        ax.set_xlabel("Month", fontsize=12, labelpad=8)
        ax.set_ylabel("Mean electricity use (kWh)", fontsize=12, labelpad=8)
        ax.set_title("Mean monthly electricity use by building type — 2010",
                     fontsize=14, fontweight="medium", pad=14)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(title="Building type", fontsize=10, title_fontsize=10)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 3 — Grouped bar: Top 20 Community Areas
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Mean electricity & gas use — top 20 community areas")
    st.caption(
        "Top 20 areas ranked by total energy (kWh + therms converted to kWh equivalent). "
        "Bars show mean kWh and mean therms per census block."
    )

    df_bar = dff.dropna(subset=["COMMUNITY AREA NAME", "KWH MEAN 2010", "THERM MEAN 2010"])

    if df_bar.empty:
        no_data()
    else:
        area = df_bar.groupby("COMMUNITY AREA NAME").agg(
            kwh_mean   =("KWH MEAN 2010",   "mean"),
            therm_mean =("THERM MEAN 2010", "mean"),
            total_kwh  =("KWH MEAN 2010",   "sum"),
            total_therm=("THERM MEAN 2010", "sum"),
        ).reset_index()
        area["total_energy"] = area["total_kwh"] + area["total_therm"] * 29.3
        top20 = area.nlargest(20, "total_energy").sort_values("total_energy", ascending=True)

        x     = np.arange(len(top20))
        width = 0.38

        fig3, ax3 = plt.subplots(figsize=(11, 8))
        fig3.patch.set_facecolor("#FAFAF8")
        ax3.set_facecolor("#FAFAF8")

        ax3.barh(x + width / 2, top20["kwh_mean"],   width,
                 label="Mean kWh",    color="#378ADD", alpha=0.88, edgecolor="none")
        ax3.barh(x - width / 2, top20["therm_mean"], width,
                 label="Mean therms", color="#1D9E75", alpha=0.88, edgecolor="none")

        ax3.set_yticks(x)
        ax3.set_yticklabels(top20["COMMUNITY AREA NAME"], fontsize=9)
        ax3.set_xlabel("Mean energy use per census block", fontsize=11, color="#3d3d3a")
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.spines[["left", "bottom"]].set_color("#c8c6bc")
        ax3.grid(axis="x", color="#e4e2d8", linewidth=0.6, linestyle="--", alpha=0.7)
        ax3.set_axisbelow(True)
        ax3.tick_params(colors="#73726c", labelsize=9)
        ax3.legend(title="Energy type", fontsize=9, title_fontsize=9,
                   framealpha=0.85, edgecolor="#d3d1c7")

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 4 — Interactive choropleth map
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Energy consumption map — Chicago community areas")
    st.caption(
        "Choropleth map of mean electricity use per neighbourhood. "
        "Hover over any area for a detailed tooltip. "
        "Colors rescale dynamically as you zoom — only visible neighbourhoods "
        "are used to compute the relative colour scale."
    )

    col_metric, col_scheme, col_zoom = st.columns([2, 2, 1])
    with col_metric:
        metric = st.selectbox(
            "Energy metric",
            options=["kwh_mean", "therm_mean", "total_kwh", "total_energy_kwh"],
            format_func=lambda k: {
                "kwh_mean":         "Mean kWh per block",
                "therm_mean":       "Mean therms per block",
                "total_kwh":        "Total kWh",
                "total_energy_kwh": "Total energy (kWh equiv.)",
            }[k],
            key="energy_map_metric",
        )
    with col_scheme:
        scheme = st.selectbox(
            "Colour scheme",
            ["YlOrRd", "YlGnBu", "RdPu", "BuPu", "OrRd"],
            key="energy_map_scheme",
        )
    with col_zoom:
        map_height = st.slider("Map height", 400, 800, 600, 50, key="energy_map_height")

    metric_labels = {
        "kwh_mean":         "Mean kWh",
        "therm_mean":       "Mean therms",
        "total_kwh":        "Total kWh",
        "total_energy_kwh": "Total energy (kWh equiv.)",
    }

    agg = aggregate_by_area(dff)

    if agg.empty:
        no_data()
        return

    try:
        geo = load_chicago_geojson()

        for feat in geo["features"]:
            props = feat["properties"]
            name_key = next(
                (k for k in ["community", "COMMUNITY", "name", "NAME"]
                 if k in props), None
            )
            if name_key:
                props["community"] = props[name_key].title()

        m = folium.Map(
            location=[41.8781, -87.6298],
            zoom_start=11,
            tiles="CartoDB positron",
            prefer_canvas=True,
        )

        folium.Choropleth(
            geo_data=geo,
            name="Energy usage",
            data=agg,
            columns=["COMMUNITY AREA NAME", metric],
            key_on="feature.properties.community",
            fill_color=scheme,
            fill_opacity=0.75,
            line_opacity=0.4,
            line_color="#555",
            nan_fill_color="#e0e0e0",
            nan_fill_opacity=0.4,
            legend_name=metric_labels[metric],
            highlight=True,
        ).add_to(m)

        area_lookup = agg.set_index("COMMUNITY AREA NAME").to_dict("index")

        for feat in geo["features"]:
            area_name = feat["properties"].get("community", "")
            data_row  = area_lookup.get(area_name, {})

            kwh_m   = data_row.get("kwh_mean",         None)
            thm_m   = data_row.get("therm_mean",        None)
            tot_kwh = data_row.get("total_kwh",         None)
            tot_ene = data_row.get("total_energy_kwh",  None)
            bldgs   = data_row.get("buildings",         None)

            def fmt(v, decimals=0):
                if v is None:
                    return "N/A"
                return f"{v:,.{decimals}f}"

            tooltip_html = f"""
            <div style="font-family:sans-serif;font-size:13px;min-width:200px;">
                <b style="font-size:14px;">{area_name}</b><br/>
                <hr style="margin:4px 0;border-color:#ddd;"/>
                <table style="width:100%;border-collapse:collapse;">
                  <tr><td style="color:#555;">Mean kWh/block</td>
                      <td style="text-align:right;font-weight:500;">{fmt(kwh_m)}</td></tr>
                  <tr><td style="color:#555;">Mean therms/block</td>
                      <td style="text-align:right;font-weight:500;">{fmt(thm_m)}</td></tr>
                  <tr><td style="color:#555;">Total kWh</td>
                      <td style="text-align:right;font-weight:500;">{fmt(tot_kwh)}</td></tr>
                  <tr><td style="color:#555;">Total energy (kWh eq.)</td>
                      <td style="text-align:right;font-weight:500;">{fmt(tot_ene)}</td></tr>
                  <tr><td style="color:#555;">Census blocks</td>
                      <td style="text-align:right;font-weight:500;">{fmt(bldgs)}</td></tr>
                </table>
            </div>
            """

            folium.GeoJson(
                feat,
                style_function=lambda x: {
                    "fillOpacity": 0,
                    "weight": 0,
                },
                tooltip=folium.Tooltip(
                    tooltip_html,
                    sticky=True,
                    style=(
                        "background-color:white;"
                        "border:1px solid #ccc;"
                        "border-radius:6px;"
                        "padding:8px;"
                        "box-shadow:2px 2px 6px rgba(0,0,0,0.15);"
                    ),
                ),
            ).add_to(m)

        for feat in geo["features"]:
            area_name = feat["properties"].get("community", "")
            row = area_lookup.get(area_name, {})
            feat["properties"]["_energy_value"] = row.get(metric)

        folium.LayerControl().add_to(m)

        st_folium(
            m,
            width="100%",
            height=map_height,
            returned_objects=["last_object_clicked_tooltip", "bounds"],
        )

        st.markdown("#### Visible area summary")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Areas with data",    f"{len(agg):,}")
        mc2.metric("Mean kWh (filtered)", f"{agg['kwh_mean'].mean():,.0f}")
        mc3.metric("Mean therms (filtered)", f"{agg['therm_mean'].mean():,.0f}")
        mc4.metric("Total energy (kWh eq.)", f"{agg['total_energy_kwh'].sum():,.0f}")

    except Exception as e:
        st.error(
            f"Map could not load — GeoJSON fetch failed: {e}\n\n"
            "Check your internet connection or try again. "
            "The other charts above are unaffected."
        )

    st.markdown("---")
    st.caption("Data: Chicago Energy Usage 2010 · EDA dashboard")
