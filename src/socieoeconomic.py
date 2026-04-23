import geopandas as gpd
import file_loader
import pandas as pd
import json
import os
import streamlit as st
import plotly.express as px
from streamlit.components.v1 import html
import map_utils
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from city_config import CityConfig
import warnings
warnings.filterwarnings('ignore')


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
    "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
]


def _name_col(df: pd.DataFrame) -> str:
    """Pick the column holding area names (Chicago uses 'COMMUNITY AREA NAME')."""
    for c in ("COMMUNITY AREA NAME", "NAME", "Name"):
        if c in df.columns:
            return c
    return df.columns[0]


@st.cache_data
def _load_and_train(city_key: str, csv_path: str, id_col: str, geo_json_str: str | None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if id_col not in df.columns:
        raise ValueError(f"Census CSV missing id column `{id_col}`. Have: {list(df.columns)[:8]}")

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    needed = [id_col] + [c for c in FEATURE_COLS if c in df.columns] + (["HARDSHIP INDEX"] if "HARDSHIP INDEX" in df.columns else [])
    df = df.dropna(subset=needed)
    df[id_col] = df[id_col].astype(int)

    if df.empty:
        raise ValueError("Census CSV cleaned to 0 rows.")

    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    if "HARDSHIP INDEX" not in df.columns or len(feat_present) < 3:
        # Not enough columns to train
        return {
            "df": df, "merged": None, "metrics": None,
            "feature_names": [], "rf_importances": [], "gb_importances": [],
            "scatter": [], "name_col": _name_col(df),
        }

    X = df[feat_present]
    y = df["HARDSHIP INDEX"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    rf_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    rf_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    gb_r2 = cross_val_score(gb, X, y, cv=kf, scoring="r2").mean()
    gb_mae = -cross_val_score(gb, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()

    rf.fit(X, y); gb.fit(X, y)
    df = df.copy()
    df["RF_Predicted"] = rf.predict(X).round(1)
    df["GB_Predicted"] = gb.predict(X).round(1)

    merged = None
    if geo_json_str:
        try:
            gdf = gpd.read_file(geo_json_str, driver="GeoJSON")
            # try to coerce join key
            for cand in ("area_num_1", id_col, "GEOID", "GEOID10"):
                if cand in gdf.columns:
                    try:
                        gdf[cand] = gdf[cand].astype(int)
                        merged = gdf.merge(df, left_on=cand, right_on=id_col)
                        break
                    except (ValueError, TypeError):
                        continue
        except Exception:
            merged = None

    name_col = _name_col(df)
    short_names = [SHORT_NAMES[FEATURE_COLS.index(c)] for c in feat_present]
    scatter = [
        {
            "name":    str(row[name_col]),
            "actual":  float(row["HARDSHIP INDEX"]),
            "rf_pred": float(row["RF_Predicted"]),
            "gb_pred": float(row["GB_Predicted"]),
            "poverty": float(row.get("PERCENT HOUSEHOLDS BELOW POVERTY", 0)),
            "income":  int(row.get("PER CAPITA INCOME", 0)),
        }
        for _, row in df.iterrows()
    ]

    return {
        "df": df, "merged": merged,
        "metrics": {"rf": {"r2": round(rf_r2, 3), "mae": round(rf_mae, 2)},
                    "gb": {"r2": round(gb_r2, 3), "mae": round(gb_mae, 2)}},
        "feature_names": short_names,
        "rf_importances": [round(x, 4) for x in rf.feature_importances_.tolist()],
        "gb_importances": [round(x, 4) for x in gb.feature_importances_.tolist()],
        "scatter": scatter,
        "name_col": name_col,
    }


def render(city: CityConfig, geo: dict | None = None):
    mapbox_style = map_utils.mapbox_style_picker(key_prefix=f"socio_{city.key}")

<<<<<<< Updated upstream
    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(
            domain="socioeconomics",
            local_csv=None,
            label="Upload a socioeconomic dataset",
        )

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="socio")
=======
    csv_path = city.census_path
    if not os.path.exists(csv_path):
        st.warning(f"No census CSV at `{csv_path}`.")
        return
>>>>>>> Stashed changes

    geo_str = json.dumps(geo) if geo and geo.get("features") else None
    try:
        data = _load_and_train(city.key, csv_path, city.census_id_col, geo_str)
    except Exception as exc:
        st.error(f"Census load failed: {exc}")
        return

    st.markdown(f"# {city.name} — Socioeconomic Hardship")

    if data["metrics"] is None:
        st.info("Model skipped — census file missing required feature columns.")
    else:
        m = data["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("RF — R²", f"{m['rf']['r2']:.3f}", help=f"MAE {m['rf']['mae']}")
        c2.metric("GB — R²", f"{m['gb']['r2']:.3f}", help=f"MAE {m['gb']['mae']}")

    tab_map, tab_diag, tab_imp = st.tabs([
        "Choropleth Map", "Model Diagnostics", "Feature Importance",
    ])

    # ── Map ─────────────────────────────────────────────────────────
    with tab_map:
        merged = data["merged"]
        if merged is None or len(merged) == 0:
            st.info(f"No boundary geometry available for {city.name} — choropleth skipped.")
        else:
            choice = st.radio(
                "Color map by:",
                ["Actual Hardship Index", "RF Predicted", "GB Predicted"],
                horizontal=True, key=f"socio_choice_{city.key}",
            )
            col_map = {
                "Actual Hardship Index": "HARDSHIP INDEX",
                "RF Predicted":          "RF_Predicted",
                "GB Predicted":          "GB_Predicted",
            }
            chosen = col_map[choice]
            geojson_dict = merged.__geo_interface__
<<<<<<< Updated upstream

            fig_main = px.choropleth_map(
                merged, geojson=geojson_dict,
                locations="area_num_1", featureidkey="properties.area_num_1",
                color=chosen_col, color_continuous_scale="YlOrRd",
                map_style=mapbox_style,
                zoom=9, center={"lat": 41.85, "lon": -87.68},
                opacity=0.7,
                hover_name="community",
                hover_data={"HARDSHIP INDEX": True, "RF_Predicted": True, "GB_Predicted": True,
                           "PER CAPITA INCOME": True, "PERCENT HOUSEHOLDS BELOW POVERTY": True},
                title=model_choice,
            )
            fig_main.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=620)
            st.plotly_chart(fig_main, width="stretch")

            st.markdown("**Predicted Hardship Index - Side by Side**")
            st.caption("Left: Random Forest. Right: Gradient Boosting. Hover any area to compare values.")

            left_col, right_col = st.columns(2)
            for col, pred_col, label in [
                (left_col,  "RF_Predicted", "Random Forest Predicted"),
                (right_col, "GB_Predicted", "Gradient Boosting Predicted"),
            ]:
                with col:
                    fig_pred = px.choropleth_map(
                        merged, geojson=geojson_dict,
                        locations="area_num_1", featureidkey="properties.area_num_1",
                        color=pred_col, color_continuous_scale="YlOrRd",
                        map_style=mapbox_style,
                        zoom=9, center={"lat": 41.85, "lon": -87.68},
                        opacity=0.7,
                        hover_name="community",
                        hover_data={"HARDSHIP INDEX": True, pred_col: True},
                        title=label,
                    )
                    fig_pred.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=480)
                    st.plotly_chart(fig_pred, width="stretch")
=======
            fig = px.choropleth_map(
                merged, geojson=geojson_dict,
                locations=city.census_id_col,
                featureidkey=f"properties.{city.census_id_col}",
                color=chosen, color_continuous_scale="YlOrRd",
                map_style=mapbox_style,
                zoom=city.zoom, center={"lat": city.center[0], "lon": city.center[1]},
                opacity=0.7,
                hover_name=data["name_col"] if data["name_col"] in merged.columns else None,
                title=choice,
            )
            fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=620)
            st.plotly_chart(fig, width="stretch")

    # ── Diagnostics ─────────────────────────────────────────────────
    with tab_diag:
        if not data["scatter"]:
            st.info("No model diagnostics — feature columns missing.")
>>>>>>> Stashed changes
        else:
            scatter_json = json.dumps(data["scatter"])
            chart_html = f"""<!DOCTYPE html><html>
<head><style>
  body {{ margin:0; background:#0e1117; font-family:'DM Sans',sans-serif; color:#e8e8e8; }}
  .wrap {{ display:flex; gap:20px; padding:16px; }}
  .panel {{ flex:1; background:#1a1f2e; border:1px solid #2a3044; border-radius:12px; padding:20px; }}
  h3 {{ margin:0 0 14px; font-size:13px; font-weight:600; letter-spacing:.05em; color:#9eaec4; text-transform:uppercase; }}
  canvas {{ display:block; }}
  .tip {{ position:fixed; background:#1a1f2e; border:1px solid #3a4460; border-radius:8px;
          padding:10px 14px; font-size:12px; pointer-events:none; display:none;
          color:#e8e8e8; line-height:1.7; box-shadow:0 4px 20px #0008; z-index:9; }}
</style></head><body>
<div id="tip" class="tip"></div>
<div class="wrap">
  <div class="panel"><h3>Random Forest — Actual vs Predicted</h3><canvas id="c1" width="430" height="390"></canvas></div>
  <div class="panel"><h3>Gradient Boosting — Actual vs Predicted</h3><canvas id="c2" width="430" height="390"></canvas></div>
</div>
<script>
const raw = {scatter_json};
const tip = document.getElementById('tip');
function draw(canvasId, predKey, color) {{
  const cv = document.getElementById(canvasId), ctx = cv.getContext('2d');
  const p = {{l:50,r:20,t:20,b:46}};
  const W = cv.width-p.l-p.r, H = cv.height-p.t-p.b;
  ctx.fillStyle='#0e1117'; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#2a3044';
  for(let i=0;i<=5;i++){{const x=p.l+W*i/5,y=p.t+H*i/5;
    ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,p.t+H);ctx.stroke();
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+W,y);ctx.stroke();}}
  ctx.strokeStyle='#3a4460'; ctx.setLineDash([6,4]);
  ctx.beginPath();ctx.moveTo(p.l,p.t+H);ctx.lineTo(p.l+W,p.t);ctx.stroke();
  ctx.setLineDash([]);
  const pts = raw.map(d => ({{x:p.l+(d.actual/100)*W, y:p.t+H-(d[predKey]/100)*H, d}}));
  pts.forEach(pt=>{{ctx.beginPath();ctx.arc(pt.x,pt.y,5.5,0,Math.PI*2);
    ctx.fillStyle=color+'bb';ctx.fill();ctx.strokeStyle='#0e1117';ctx.stroke();}});
  cv.onmousemove=e=>{{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    let best=null,minD=1e9;
    pts.forEach(pt=>{{const d=Math.hypot(pt.x-mx,pt.y-my);if(d<minD){{minD=d;best=pt;}}}});
    if(minD<16){{tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';
      tip.innerHTML=`<b>${{best.d.name}}</b><br>Actual: ${{best.d.actual}}<br>Predicted: ${{best.d[predKey]}}`;}}
    else tip.style.display='none';}};
  cv.onmouseleave=()=>tip.style.display='none';
}}
draw('c1','rf_pred','#4f8ef7');
draw('c2','gb_pred','#f7934f');
</script></body></html>"""
            html(chart_html, height=450)

    # ── Feature importance ─────────────────────────────────────────
    with tab_imp:
        if not data["feature_names"]:
            st.info("No feature importance — model not trained.")
        else:
            imp_html = f"""<!DOCTYPE html><html>
<head><style>
  body {{ margin:0; background:#0e1117; font-family:'DM Sans',sans-serif; color:#e8e8e8; padding:16px; }}
  .wrap {{ display:flex; gap:20px; }}
  .panel {{ flex:1; background:#1a1f2e; border:1px solid #2a3044; border-radius:12px; padding:24px; }}
  h3 {{ margin:0 0 20px; font-size:13px; font-weight:600; letter-spacing:.06em; color:#9eaec4; text-transform:uppercase; }}
  .row {{ margin-bottom:18px; }}
  .lbl {{ font-size:13px; color:#c8d0de; margin-bottom:6px; display:flex; justify-content:space-between; }}
  .track {{ background:#2a3044; border-radius:6px; height:11px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:6px; width:0%; transition:width .7s; }}
</style></head><body>
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
    const div=document.createElement('div'); div.className='row';
    div.innerHTML=`<div class="lbl"><span>${{name}}</span><span>${{pct}}%</span></div>
                   <div class="track"><div class="fill" style="background:${{grad}}" data-w="${{pct}}"></div></div>`;
    panel.appendChild(div);
  }});
  setTimeout(()=>panel.querySelectorAll('.fill').forEach(el=>el.style.width=el.dataset.w+'%'),80);
}}
renderBars('rfp', rfImp, 'linear-gradient(90deg,#3a6fd8,#4f8ef7)');
renderBars('gbp', gbImp, 'linear-gradient(90deg,#d86b3a,#f7934f)');
</script></body></html>"""
            html(imp_html, height=440)

    # ── Scatter ────────────────────────────────────────────────────
    df_sc = data["df"]
    if "HARDSHIP INDEX" in df_sc.columns and "PERCENT HOUSEHOLDS BELOW POVERTY" in df_sc.columns:
        st.divider()
        st.subheader("Socioeconomic Scatterplots")
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_pov = px.scatter(
                df_sc, x="PERCENT HOUSEHOLDS BELOW POVERTY", y="HARDSHIP INDEX",
                hover_name=data["name_col"],
                color="PER CAPITA INCOME" if "PER CAPITA INCOME" in df_sc.columns else None,
                color_continuous_scale="RdYlGn_r",
                title="Poverty vs Hardship",
            )
            fig_pov.update_layout(margin={"t": 30})
            st.plotly_chart(fig_pov, width="stretch")
        with sc2:
            if "PER CAPITA INCOME" in df_sc.columns:
                fig_inc = px.scatter(
                    df_sc, x="PER CAPITA INCOME", y="HARDSHIP INDEX",
                    hover_name=data["name_col"],
                    color="PERCENT AGED 16+ UNEMPLOYED" if "PERCENT AGED 16+ UNEMPLOYED" in df_sc.columns else None,
                    color_continuous_scale="YlOrRd",
                    title="Income vs Hardship",
                )
                fig_inc.update_layout(margin={"t": 30})
                st.plotly_chart(fig_inc, width="stretch")

        # Moran's I — only if merged geometry exists
        merged = data["merged"]
        if merged is not None and len(merged) >= 10:
            st.divider()
            try:
                merged = merged.copy()
                merged["_id_str"] = merged[city.census_id_col].astype(str)
                map_utils.render_moran_analysis(
                    gdf=merged,
                    value_col="HARDSHIP INDEX",
                    name_col=data["name_col"] if data["name_col"] in merged.columns else city.census_id_col,
                    id_col="_id_str",
                    geojson=geo,
                    featureidkey=f"properties.{city.census_id_col}",
                    key_prefix=f"socio_moran_{city.key}",
                    map_style=mapbox_style,
                )
            except Exception as exc:
                st.warning(f"Moran's I unavailable: {exc}")
