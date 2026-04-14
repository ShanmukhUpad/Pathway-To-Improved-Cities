import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Energy Usage 2010",
    page_icon="⚡",
    layout="wide",
)

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

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("energy-usage-2010-1.csv")
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("Filters")
st.sidebar.markdown("Applied to all charts below.")

all_types = sorted(df["BUILDING TYPE"].dropna().unique())
selected_types = st.sidebar.multiselect(
    "Building type",
    options=all_types,
    default=all_types,
)

all_areas = sorted(df["COMMUNITY AREA NAME"].dropna().unique())
selected_areas = st.sidebar.multiselect(
    "Community area",
    options=all_areas,
    default=all_areas,
)

# Apply filters
mask = (
    df["BUILDING TYPE"].isin(selected_types) &
    df["COMMUNITY AREA NAME"].isin(selected_areas)
)
dff = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Rows selected", f"{len(dff):,}")
st.sidebar.metric("Community areas", f"{dff['COMMUNITY AREA NAME'].nunique()}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Chicago Energy Usage — 2010 EDA")
st.markdown(
    "Exploratory analysis of electricity and gas consumption across "
    "Chicago census blocks. Use the sidebar to filter by building type and community area."
)
st.markdown("---")

# ── Helper: empty state ───────────────────────────────────────────────────────
def no_data():
    st.warning("No data matches the current filters. Adjust the sidebar selections.")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Scatter: KWH Mean vs Building Age
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Electricity use vs. building age")
st.caption("Mean kWh per account vs. average building age, coloured by building type. OLS trend line per group.")

df_scatter = dff.dropna(subset=["KWH MEAN 2010", "AVERAGE BUILDING AGE", "BUILDING TYPE"])
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

# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Line: Monthly KWH by Building Type
# ─────────────────────────────────────────────────────────────────────────────
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
    # Keep only building types that are both selected and in the palette
    monthly_means = monthly_means[
        [c for c in monthly_means.columns if c in selected_types and c in PALETTE]
    ]

    palette  = PALETTE
    markers  = MARKERS

    fig2, ax = plt.subplots(figsize=(11, 6))

    for btype in monthly_means.columns:
        ax.plot(
            monthly_means.index,
            monthly_means[btype],
            color=palette[btype],
            marker=markers[btype],
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

# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Grouped bar: Top 20 Community Areas
# ─────────────────────────────────────────────────────────────────────────────
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: Chicago Energy Usage 2010 · EDA dashboard")