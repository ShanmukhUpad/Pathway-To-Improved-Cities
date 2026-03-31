import geopandas as gpd
import file_loader
import pandas as pd
import folium
import json
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

_HERE = Path(__file__).parent
GEOJSON_PATH = _HERE / "chicago-community-areas.geojson"
GEOJSON_URL  = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
CSV_PATH     = _HERE / "censusChicago.csv"


def _load_geojson() -> gpd.GeoDataFrame:
    if GEOJSON_PATH.exists():
        return gpd.read_file(GEOJSON_PATH)
    return gpd.read_file(GEOJSON_URL)


@st.cache_data
def load_and_train():
    gdf = _load_geojson()
    df  = pd.read_csv(CSV_PATH)
    gdf.columns = gdf.columns.str.strip()
    df.columns  = df.columns.str.strip()

    gdf["area_num_1"] = gdf["area_num_1"].fillna(0).astype(int)

    df["Community Area Number"] = pd.to_numeric(
        df["Community Area Number"], errors="coerce"
    )
    df = df.dropna(subset=["Community Area Number", "HARDSHIP INDEX"])
    df["Community Area Number"] = df["Community Area Number"].astype(int)

    if len(df) == 0:
        raise ValueError(
            "Census CSV loaded 0 usable rows after cleaning. "
            f"Check that '{CSV_PATH}' exists and has numeric 'Community Area Number' values."
        )

    FEATURE_COLS = [
        "PERCENT OF HOUSING CROWDED",
        "PERCENT HOUSEHOLDS BELOW POVERTY",
        "PERCENT AGED 16+ UNEMPLOYED",
        "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
        "PERCENT AGED UNDER 18 OR OVER 64",
        "PER CAPITA INCOME",
    ]
    SHORT_NAMES = [
        "Housing Crowded", "Below Poverty", "Unemployed 16+",
        "No HS Diploma", "Under 18/Over 64", "Per Capita Income"
    ]

    X = df[FEATURE_COLS]
    y = df["HARDSHIP INDEX"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    rf_r2  = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    rf_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    gb_r2  = cross_val_score(gb, X, y, cv=kf, scoring="r2").mean()
    gb_mae = -cross_val_score(gb, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()

    rf.fit(X, y)
    gb.fit(X, y)

    df = df.copy()
    df["RF_Predicted"] = rf.predict(X).round(1)
    df["GB_Predicted"] = gb.predict(X).round(1)
    merged = gdf.merge(df, left_on="area_num_1", right_on="Community Area Number")

    scatter = [
        {
            "name":    row["COMMUNITY AREA NAME"],
            "actual":  float(row["HARDSHIP INDEX"]),
            "rf_pred": float(row["RF_Predicted"]),
            "gb_pred": float(row["GB_Predicted"]),
            "poverty": float(row["PERCENT HOUSEHOLDS BELOW POVERTY"]),
            "income":  int(row["PER CAPITA INCOME"]),
        }
        for _, row in df.iterrows()
    ]

    return {
        "merged":        merged,
        "df":            df,
        "metrics":       {"rf": {"r2": round(rf_r2, 3), "mae": round(rf_mae, 2)},
                          "gb": {"r2": round(gb_r2, 3), "mae": round(gb_mae, 2)}},
        "feature_names": SHORT_NAMES,
        "rf_importances": [round(x, 4) for x in rf.feature_importances_.tolist()],
        "gb_importances": [round(x, 4) for x in gb.feature_importances_.tolist()],
        "scatter":        scatter,
    }


@st.cache_data
def load_income_regression():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["PER CAPITA INCOME", "COMMUNITY AREA NAME"])

    FEATURE_COLS = [
        "PERCENT OF HOUSING CROWDED",
        "PERCENT HOUSEHOLDS BELOW POVERTY",
        "PERCENT AGED 16+ UNEMPLOYED",
        "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
        "PERCENT AGED UNDER 18 OR OVER 64",
        "HARDSHIP INDEX",
    ]
    SHORT_NAMES = [
        "Housing Crowded", "Below Poverty", "Unemployed 16+",
        "No HS Diploma", "Under 18 / Over 64", "Hardship Index",
    ]

    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values
    y = df["PER CAPITA INCOME"].values

    models = {
        "Random Forest":       RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        "Linear Regression":   LinearRegression(),
        "Ridge Regression":    Ridge(alpha=1.0),
    }

    loo = LeaveOneOut()
    model_maes = {}
    for name, model in models.items():
        scores = -cross_val_score(model, X, y, cv=loo, scoring="neg_mean_absolute_error")
        model_maes[name] = round(float(scores.mean()), 0)

    # Fit RF on full data for importance + residuals
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    residuals = y - preds

    scatter_df = pd.DataFrame({
        "neighborhood":   df["COMMUNITY AREA NAME"].values,
        "income":         y,
        "no_hs_diploma":  df["PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA"].values,
        "predicted":      preds,
        "residual":       residuals,
    })

    return {
        "model_maes":      model_maes,
        "feature_names":   SHORT_NAMES,
        "rf_importances":  rf.feature_importances_.tolist(),
        "scatter_df":      scatter_df,
    }


def render():
    st.set_page_config(page_title="Chicago Hardship ML", layout="wide")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(
            domain="socioeconomics",
            local_csv=None,
            label="Upload a socioeconomic dataset",
        )

    if not CSV_PATH.exists():
        st.error(
            f"Missing data file: `censusChicago.csv`\n\n"
            "Download it from the Chicago Data Portal:\n"
            "**Census Data — Selected Socioeconomic Indicators in Chicago, 2008–2012**\n\n"
            "Save it as `censusChicago.csv` inside the `src/` folder, then refresh the page."
        )
        return

    try:
        data = load_and_train()
    except Exception as exc:
        st.error(f"Failed to load socioeconomic data: {exc}")
        return

    merged  = data["merged"]
    metrics = data["metrics"]

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
            html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #0e1117; color: #e8e8e8; }
            h1, h2, h3 { font-family: 'DM Serif Display', serif; }
            .metric-card {
                background: #1a1f2e; border: 1px solid #2a3044;
                border-radius: 12px; padding: 20px 24px; text-align: center;
            }
            .metric-card .label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #8892a4; margin-bottom: 4px; }
            .metric-card .value { font-size: 2rem; font-weight: 600; color: #e8e8e8; line-height: 1; }
            .metric-card .sub   { font-size: 12px; color: #5d6a84; margin-top: 4px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("# Chicago Community Hardship — ML Dashboard")
    st.markdown("Random Forest & Gradient Boosting models trained on 77 Chicago community areas.")

    cols = st.columns(4)
    cards = [
        ("Random Forest R²",      metrics["rf"]["r2"],  "5-fold cross-val"),
        ("Random Forest MAE",     metrics["rf"]["mae"], "avg pts off"),
        ("Gradient Boost R²",     metrics["gb"]["r2"],  "5-fold cross-val"),
        ("Gradient Boost MAE",    metrics["gb"]["mae"], "avg pts off"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️  Choropleth Map",
        "📊  Model Diagnostics",
        "🔍  Feature Importance",
        "📈  Income Regression",
    ])

    #  TAB 1: Map 
    with tab1:
        model_choice = st.radio(
            "Colour map by:",
            ["Actual Hardship Index", "RF Predicted", "GB Predicted"],
            horizontal=True,
        )
        col_map = {
            "Actual Hardship Index": "HARDSHIP INDEX",
            "RF Predicted":          "RF_Predicted",
            "GB Predicted":          "GB_Predicted",
        }
        chosen_col = col_map[model_choice]

        m = folium.Map(location=[41.85, -87.68], zoom_start=10, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=merged,
            data=merged,
            columns=["area_num_1", chosen_col],
            key_on="feature.properties.area_num_1",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name=model_choice,
            highlight=True,
        ).add_to(m)

        folium.GeoJson(
            merged,
            style_function=lambda _: {
                "fillColor": "transparent", "color": "black",
                "weight": 1, "fillOpacity": 0,
            },
            highlight_function=lambda _: {
                "weight": 3, "color": "#4f8ef7", "fillOpacity": 0.15,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["community", "HARDSHIP INDEX", "RF_Predicted", "GB_Predicted",
                        "PER CAPITA INCOME", "PERCENT HOUSEHOLDS BELOW POVERTY"],
                aliases=["Community:", "Actual Hardship:", "RF Predicted:", "GB Predicted:",
                         "Per Capita Income:", "% Below Poverty:"],
                localize=True, sticky=False, labels=True,
                style="background-color:white;border:1px solid #ccc;border-radius:4px;padding:6px;",
            ),
        ).add_to(m)

        html(m._repr_html_(), height=620)

        st.markdown("<div class='section-title'>Predicted Hardship Index — Side by Side</div>", unsafe_allow_html=True)
        st.caption("Left: Random Forest · Right: Gradient Boosting. Hover any area to compare values.")

        left_col, right_col = st.columns(2)

        for col, pred_col, label, pin_color in [
            (left_col,  "RF_Predicted", "Random Forest Predicted", "#4f8ef7"),
            (right_col, "GB_Predicted", "Gradient Boosting Predicted", "#f7934f"),
        ]:
            m_pred = folium.Map(location=[41.85, -87.68], zoom_start=10, tiles="CartoDB positron")

            folium.Choropleth(
                geo_data=merged,
                data=merged,
                columns=["area_num_1", pred_col],
                key_on="feature.properties.area_num_1",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name=label,
                highlight=True,
            ).add_to(m_pred)

            folium.GeoJson(
                merged,
                style_function=lambda _: {
                    "fillColor": "transparent", "color": "black",
                    "weight": 1, "fillOpacity": 0,
                },
                highlight_function=lambda _, c=pin_color: {
                    "weight": 3, "color": c, "fillOpacity": 0.2,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["community", "HARDSHIP INDEX", pred_col],
                    aliases=["Community:", "Actual Hardship:", f"{label}:"],
                    localize=True, sticky=False, labels=True,
                    style="background-color:white;border:1px solid #ccc;border-radius:4px;padding:6px;",
                ),
            ).add_to(m_pred)

            with col:
                st.markdown(f"**{label}**")
                html(m_pred._repr_html_(), height=480)

    with tab2:
        scatter_json = json.dumps(data["scatter"])
        chart_html = f"""<!DOCTYPE html><html>
        <head><style>
          body {{ margin:0; background:#0e1117; font-family:'DM Sans',sans-serif; color:#e8e8e8; }}
          .wrap {{ display:flex; gap:20px; padding:16px; }}
          .panel {{ flex:1; background:#1a1f2e; border:1px solid #2a3044; border-radius:12px; padding:20px; }}
          h3 {{ margin:0 0 14px; font-size:13px; font-weight:600; letter-spacing:.05em; color:#9eaec4; text-transform:uppercase; }}
          canvas {{ display:block; }}
          .tip {{
            position:fixed; background:#1a1f2e; border:1px solid #3a4460;
            border-radius:8px; padding:10px 14px; font-size:12px; pointer-events:none;
            display:none; color:#e8e8e8; line-height:1.7; box-shadow:0 4px 20px #0008; z-index:9;
          }}
        </style></head>
        <body>
        <div id="tip" class="tip"></div>
        <div class="wrap">
          <div class="panel"><h3>Random Forest — Actual vs Predicted</h3><canvas id="c1" width="430" height="390"></canvas></div>
          <div class="panel"><h3>Gradient Boosting — Actual vs Predicted</h3><canvas id="c2" width="430" height="390"></canvas></div>
        </div>
        <script>
        const raw = {scatter_json};
        const tip = document.getElementById('tip');

        function draw(canvasId, predKey, color) {{
          const cv = document.getElementById(canvasId);
          const ctx = cv.getContext('2d');
          const p = {{l:50,r:20,t:20,b:46}};
          const W = cv.width - p.l - p.r, H = cv.height - p.t - p.b;

          ctx.fillStyle='#0e1117'; ctx.fillRect(0,0,cv.width,cv.height);

          ctx.strokeStyle='#2a3044'; ctx.lineWidth=1;
          for(let i=0;i<=5;i++) {{
            const x=p.l+W*i/5, y=p.t+H*i/5;
            ctx.beginPath(); ctx.moveTo(x,p.t); ctx.lineTo(x,p.t+H); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(p.l,y); ctx.lineTo(p.l+W,y); ctx.stroke();
          }}

          // diagonal reference line
          ctx.strokeStyle='#3a4460'; ctx.lineWidth=1.5; ctx.setLineDash([6,4]);
          ctx.beginPath(); ctx.moveTo(p.l,p.t+H); ctx.lineTo(p.l+W,p.t); ctx.stroke();
          ctx.setLineDash([]);

          // axis tick labels
          ctx.fillStyle='#5d6a84'; ctx.font='11px sans-serif'; ctx.textAlign='center';
          for(let i=0;i<=5;i++) {{
            const v=i*20;
            ctx.fillText(v, p.l+W*i/5, p.t+H+16);
            ctx.save(); ctx.translate(p.l-16, p.t+H-H*i/5);
            ctx.rotate(-Math.PI/2); ctx.fillText(v,0,0); ctx.restore();
          }}
          ctx.fillStyle='#8892a4'; ctx.font='12px sans-serif';
          ctx.fillText('Actual Hardship Index', p.l+W/2, p.t+H+36);
          ctx.save(); ctx.translate(14,p.t+H/2); ctx.rotate(-Math.PI/2);
          ctx.fillText('Predicted',0,0); ctx.restore();

          // dots
          const pts = raw.map(d => ({{
            x: p.l + (d.actual/100)*W,
            y: p.t + H - (d[predKey]/100)*H,
            d
          }}));
          pts.forEach(pt => {{
            ctx.beginPath(); ctx.arc(pt.x,pt.y,5.5,0,Math.PI*2);
            ctx.fillStyle=color+'bb'; ctx.fill();
            ctx.strokeStyle='#0e1117'; ctx.lineWidth=1; ctx.stroke();
          }});

          cv.onmousemove = e => {{
            const r=cv.getBoundingClientRect();
            const mx=e.clientX-r.left, my=e.clientY-r.top;
            let best=null, minD=1e9;
            pts.forEach(pt => {{
              const d=Math.hypot(pt.x-mx,pt.y-my);
              if(d<minD){{minD=d;best=pt;}}
            }});
            if(minD<16) {{
              tip.style.display='block';
              tip.style.left=(e.clientX+14)+'px';
              tip.style.top=(e.clientY-10)+'px';
              const err=(best.d.actual-best.d[predKey]).toFixed(1);
              tip.innerHTML=`<b>${{best.d.name}}</b><br>Actual: ${{best.d.actual}}<br>Predicted: ${{best.d[predKey]}}<br>Error: ${{err}}`;
            }} else tip.style.display='none';
          }};
          cv.onmouseleave=()=>tip.style.display='none';
        }}

        draw('c1','rf_pred','#4f8ef7');
        draw('c2','gb_pred','#f7934f');
        </script></body></html>"""
        html(chart_html, height=450)

    # ── TAB 3: Feature Importance ─────────────────────────────────
    with tab3:
        importance_html = f"""<!DOCTYPE html><html>
        <head><style>
          body {{ margin:0; background:#0e1117; font-family:'DM Sans',sans-serif; color:#e8e8e8; padding:16px; }}
          .wrap {{ display:flex; gap:20px; }}
          .panel {{ flex:1; background:#1a1f2e; border:1px solid #2a3044; border-radius:12px; padding:24px; }}
          h3 {{ margin:0 0 20px; font-size:13px; font-weight:600; letter-spacing:.06em; color:#9eaec4; text-transform:uppercase; }}
          .row {{ margin-bottom:18px; }}
          .lbl {{ font-size:13px; color:#c8d0de; margin-bottom:6px; display:flex; justify-content:space-between; }}
          .track {{ background:#2a3044; border-radius:6px; height:11px; overflow:hidden; }}
          .fill  {{ height:100%; border-radius:6px; width:0%; transition:width .7s cubic-bezier(.4,0,.2,1); }}
          .note  {{ margin-top:20px; font-size:12px; color:#5d6a84; line-height:1.8; border-top:1px solid #2a3044; padding-top:14px; }}
        </style></head>
        <body>
        <div class="wrap">
          <div class="panel" id="rfp"><h3>Random Forest — Feature Importance</h3></div>
          <div class="panel" id="gbp"><h3>Gradient Boosting — Feature Importance</h3></div>
        </div>
        <script>
        const names = {json.dumps(data['feature_names'])};
        const rfImp = {json.dumps(data['rf_importances'])};
        const gbImp = {json.dumps(data['gb_importances'])};

        function renderBars(panelId, imps, grad) {{
          const panel = document.getElementById(panelId);
          const pairs = names.map((n,i)=>[n,imps[i]]).sort((a,b)=>b[1]-a[1]);
          pairs.forEach(([name,val])=>{{
            const pct=(val*100).toFixed(1);
            const div=document.createElement('div');
            div.className='row';
            div.innerHTML=`
              <div class="lbl"><span>${{name}}</span><span>${{pct}}%</span></div>
              <div class="track"><div class="fill" style="background:${{grad}}" data-w="${{pct}}"></div></div>`;
            panel.appendChild(div);
          }});
          setTimeout(()=>panel.querySelectorAll('.fill').forEach(el=>el.style.width=el.dataset.w+'%'), 80);
          panel.insertAdjacentHTML('beforeend',
            `<div class="note">Importance = how much each feature reduced prediction error across all trees. A near-100% score for Per Capita Income means both models lean heavily on income to predict hardship.</div>`);
        }}

        renderBars('rfp', rfImp, 'linear-gradient(90deg,#3a6fd8,#4f8ef7)');
        renderBars('gbp', gbImp, 'linear-gradient(90deg,#d86b3a,#f7934f)');
        </script></body></html>"""
        html(importance_html, height=440)

    # ── TAB 4: Income Regression ──────────────────────────────────────
    with tab4:
        st.markdown("### Chicago Community Areas — Per Capita Income Regression")
        st.caption(
            "Random Forest trained on socioeconomic indicators to predict per capita income. "
            "Leave-one-out cross-validation measures how well each model generalises to unseen neighborhoods."
        )

        inc = load_income_regression()
        sdf = inc["scatter_df"]

        # ── Panel 1: Model comparison (LOO-CV MAE) ────────────────────
        st.markdown(
            "**Model comparison — leave-one-out CV error**  \n"
            "<sup>How accurately each model predicts income when tested on neighborhoods it has never seen. "
            "Lower bar = fewer dollars of average error.</sup>",
            unsafe_allow_html=True,
        )
        mae_df = pd.DataFrame(
            {"Model": list(inc["model_maes"].keys()), "Mean Absolute Error ($)": list(inc["model_maes"].values())}
        ).sort_values("Mean Absolute Error ($)")

        fig_mae = px.bar(
            mae_df, x="Mean Absolute Error ($)", y="Model",
            orientation="h",
            color="Mean Absolute Error ($)",
            color_continuous_scale="Blues_r",
            text="Mean Absolute Error ($)",
        )
        fig_mae.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_mae.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin={"t": 10},
        )
        st.plotly_chart(fig_mae, use_container_width=True)

        best_model = mae_df.iloc[0]["Model"]
        best_mae   = mae_df.iloc[0]["Mean Absolute Error ($)"]
        worst_model= mae_df.iloc[-1]["Model"]
        worst_mae  = mae_df.iloc[-1]["Mean Absolute Error ($)"]
        st.info(
            f"**Model comparison insight:** **{best_model}** performs best with a LOO-CV MAE of "
            f"**${best_mae:,.0f}**, meaning when predicting the income of a neighborhood it has never seen, "
            f"it is off by ${best_mae:,.0f} on average. "
            f"{worst_model} has the highest error (${worst_mae:,.0f}). "
            "Lower error models should be used when targeting resource allocation to specific neighborhoods."
        )

        st.divider()

        col_imp, col_scatter = st.columns(2)

        # ── Panel 2: Feature importance ───────────────────────────────
        with col_imp:
            st.markdown(
                "**Feature importance (Random Forest)**  \n"
                "<sup>% no HS diploma and % unemployed together account for the majority of what the model learns.</sup>",
                unsafe_allow_html=True,
            )
            imp_df = pd.DataFrame({
                "Feature":    inc["feature_names"],
                "Importance": inc["rf_importances"],
            }).sort_values("Importance")

            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                text="Importance",
            )
            fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_imp.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
                margin={"t": 10},
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            imp_sorted   = sorted(zip(inc["feature_names"], inc["rf_importances"]), key=lambda x: -x[1])
            top_feat, top_imp = imp_sorted[0]
            sec_feat, sec_imp = imp_sorted[1] if len(imp_sorted) > 1 else ("", 0)
            st.info(
                f"**Feature importance insight:** **{top_feat}** is the dominant predictor of per capita income "
                f"(importance: {top_imp:.3f}), meaning it has the single greatest influence on what the model "
                f"learns. **{sec_feat}** is second ({sec_imp:.3f}). Together they account for "
                f"**{(top_imp + sec_imp)*100:.1f}%** of model decisions. "
                f"Policy interventions targeting {top_feat.lower()}, such as workforce development or "
                "adult education programs, are likely to have the highest income-related impact."
            )

        # ── Panel 3: Income vs. top predictor scatter ─────────────────
        with col_scatter:
            st.markdown(
                "**Income vs. top predictor (hover any dot to see the neighborhood)**  \n"
                "<sup>% of residents aged 25+ without a high school diploma is the single strongest predictor "
                "of per capita income.</sup>",
                unsafe_allow_html=True,
            )
            fig_scatter = px.scatter(
                sdf,
                x="no_hs_diploma",
                y=sdf["income"] / 1000,
                hover_name="neighborhood",
                hover_data={"income": ":$,.0f", "no_hs_diploma": ":.1f"},
                labels={
                    "no_hs_diploma": "% aged 25+ without HS diploma",
                    "y": "Per capita income ($K)",
                },
                trendline="ols",
                color_discrete_sequence=["#4f8ef7"],
            )
            fig_scatter.update_traces(marker_size=8, selector={"mode": "markers"})
            fig_scatter.update_layout(margin={"t": 10})
            st.plotly_chart(fig_scatter, use_container_width=True)

            corr = sdf[["no_hs_diploma", "income"]].corr().iloc[0, 1]
            low_edu_high_inc = sdf.nlargest(1, "income").iloc[0]
            st.info(
                f"**Scatter insight:** The correlation between % no HS diploma and per capita income is "
                f"**r = {corr:.2f}**, a {'strong' if abs(corr) > 0.6 else 'moderate'} negative relationship. "
                f"As the share of residents without a diploma rises, income falls sharply. "
                f"**{low_edu_high_inc['neighborhood']}** is a notable outlier with high income despite having "
                f"{low_edu_high_inc['no_hs_diploma']:.1f}% without a diploma, suggesting other wealth drivers "
                "like investment income or housing equity are at play."
            )

        st.divider()

        # ── Panel 4: Residuals ────────────────────────────────────────
        st.markdown(
            "**Residuals: where the model is surprised**  \n"
            "<sup>Residual = actual income − predicted income. "
            "Blue = richer than expected, red = poorer than expected.</sup>",
            unsafe_allow_html=True,
        )
        sdf_sorted = sdf.sort_values("residual").reset_index(drop=True)
        fig_resid = px.bar(
            sdf_sorted,
            x="neighborhood",
            y="residual",
            color="residual",
            color_continuous_scale=["#d73027", "#fee08b", "#4575b4"],
            color_continuous_midpoint=0,
            labels={"neighborhood": "Neighborhoods (sorted by residual, low → high)", "residual": "Residual ($)"},
            hover_data={"income": ":$,.0f", "predicted": ":$,.0f", "residual": ":$,.0f"},
        )
        fig_resid.update_layout(
            xaxis={"tickangle": -45, "tickfont": {"size": 9}},
            coloraxis_showscale=False,
            margin={"t": 10},
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        most_over  = sdf_sorted.iloc[-1]
        most_under = sdf_sorted.iloc[0]
        st.info(
            f"**Residuals insight:** **{most_over['neighborhood']}** is the most over-performing neighborhood. "
            f"Its actual income is **${most_over['residual']:+,.0f}** above what its socioeconomic profile predicts. "
            f"**{most_under['neighborhood']}** is the most under-performing, "
            f"**${abs(most_under['residual']):,.0f} poorer** than expected given its indicators. "
            "Large negative residuals (red bars) are priority candidates for economic development programs, "
            "as they suggest structural barriers not captured by standard socioeconomic variables."
        )

