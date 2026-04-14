"""
socieoeconomic.py — New York City
Same ML pipeline as Chicago; fetches ACS 5-Year census data on first load.
Requires CENSUS_API_KEY in .env (free: https://api.census.gov/data/key_signup.html)
"""
import os
import sys
import json
import time
import requests
import numpy as np
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from dotenv import load_dotenv
import file_loader
import map_utils

# ── City-specific constants ───────────────────────────────────────────────────
CITY_NAME      = "Austin"
CENSUS_STATE   = "48"
CENSUS_COUNTY  = "453"   # Travis County
MAP_CENTER     = {"lat": 30.2672, "lon": -97.7431}
MAP_ZOOM       = 10
# TIGER tract GeoJSON is fetched and cached locally alongside census.csv
# ─────────────────────────────────────────────────────────────────────────────

_HERE     = Path(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = _HERE / "census.csv"
GEO_PATH  = _HERE / "census_tracts.geojson"   # downloaded on first load

load_dotenv(_HERE.parent / ".env")

FEATURE_COLS  = [
    "PERCENT OF HOUSING CROWDED",
    "PERCENT HOUSEHOLDS BELOW POVERTY",
    "PERCENT AGED 16+ UNEMPLOYED",
    "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
    "PERCENT AGED UNDER 18 OR OVER 64",
    "PER CAPITA INCOME",
]
FEATURE_SHORT = ["Housing Crowded", "Below Poverty", "Unemployed 16+",
                 "No HS Diploma", "Under 18/Over 64", "Per Capita Income"]
HARDSHIP_COL  = "HARDSHIP INDEX"
AREA_COL      = "GEOID"
NAME_COL      = "NAME"


# ── Census data fetch ─────────────────────────────────────────────────────────

def _fetch_acs_csv() -> bool:
    """Fetch ACS 5-Year tract data and save to CSV_PATH. Returns True on success."""
    api_key = os.environ.get("CENSUS_API_KEY", "")
    acs_vars = [
        "NAME", "DP04_0078PE", "DP03_0119PE", "DP03_0009PE",
        "DP02_0060PE", "DP02_0061PE", "DP05_0019PE", "DP05_0024PE", "DP03_0088E",
    ]
    all_frames = []
    for cnty in CENSUS_COUNTY.split(","):
        cnty = cnty.strip()
        params = {"get": ",".join(acs_vars), "for": "tract:*", "in": f"state:{CENSUS_STATE} county:{cnty}"}
        if api_key:
            params["key"] = api_key
        try:
            resp = requests.get("https://api.census.gov/data/2022/acs/acs5/profile", params=params, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            all_frames.append(pd.DataFrame(raw[1:], columns=raw[0]))
            time.sleep(0.2)
        except Exception as exc:
            st.error(f"Census API error (county {cnty}): {exc}")
            return False

    df = pd.concat(all_frames, ignore_index=True)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    numeric_vars = [v for v in acs_vars if v != "NAME"]
    for v in numeric_vars:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce").where(lambda s: s > -999_999)

    df["PERCENT OF HOUSING CROWDED"]              = df.get("DP04_0078PE")
    df["PERCENT HOUSEHOLDS BELOW POVERTY"]        = df.get("DP03_0119PE")
    df["PERCENT AGED 16+ UNEMPLOYED"]             = df.get("DP03_0009PE")
    df["PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA"] = (
        df.get("DP02_0060PE", pd.Series(0.0, index=df.index)).fillna(0)
        + df.get("DP02_0061PE", pd.Series(0.0, index=df.index)).fillna(0)
    )
    df["PERCENT AGED UNDER 18 OR OVER 64"] = (
        df.get("DP05_0019PE", pd.Series(0.0, index=df.index)).fillna(0)
        + df.get("DP05_0024PE", pd.Series(0.0, index=df.index)).fillna(0)
    )
    df["PER CAPITA INCOME"] = df.get("DP03_0088E")

    # Hardship index — z-score composite scaled 0-100
    inputs = [("PERCENT OF HOUSING CROWDED", 1), ("PERCENT HOUSEHOLDS BELOW POVERTY", 1),
              ("PERCENT AGED 16+ UNEMPLOYED", 1), ("PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA", 1),
              ("PERCENT AGED UNDER 18 OR OVER 64", 1), ("PER CAPITA INCOME", -1)]
    z_cols = []
    for col, d in inputs:
        s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
        m, std = s.mean(), s.std()
        z_cols.append(((s - m) / std * d) if pd.notna(m) and std > 0 else pd.Series(0.0, index=df.index))
    composite = pd.concat(z_cols, axis=1).mean(axis=1)
    mn, mx = composite.min(), composite.max()
    df[HARDSHIP_COL] = ((composite - mn) / (mx - mn) * 100).round(1) if mx > mn else 50.0

    out_cols = ["GEOID", "NAME"] + FEATURE_COLS + [HARDSHIP_COL]
    out_df = df[[c for c in out_cols if c in df.columns]].dropna(subset=[HARDSHIP_COL])
    out_df.to_csv(CSV_PATH, index=False)
    return True


def _fetch_tiger_geojson() -> bool:
    """Download TIGER/Line census tract boundaries and save to GEO_PATH."""
    counties = [c.strip().zfill(3) for c in CENSUS_COUNTY.split(",")]
    counties_q = ",".join(f"'{c}'" for c in counties)
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query"
    all_features = []
    offset = 0
    batch = 1_000
    while True:
        params = {"where": f"STATEFP='{CENSUS_STATE}' AND COUNTYFP IN ({counties_q})",
                  "outFields": "GEOID,NAMELSAD", "outSR": "4326", "f": "geojson",
                  "resultRecordCount": batch, "resultOffset": offset}
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            st.error(f"TIGER API error: {exc}")
            return False
        features = resp.json().get("features", [])
        all_features.extend(features)
        if len(features) < batch:
            break
        offset += batch
        time.sleep(0.3)
    import json as _json
    with open(GEO_PATH, "w") as fh:
        _json.dump({"type": "FeatureCollection", "features": all_features}, fh)
    return True


# ── ML training ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Training socioeconomic models...")
def load_and_train(csv_path_str: str, geo_path_str: str):
    gdf = gpd.read_file(geo_path_str)
    df  = pd.read_csv(csv_path_str)
    gdf.columns = gdf.columns.str.strip()
    df.columns  = df.columns.str.strip()
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.strip()
    df["GEOID"]  = df["GEOID"].astype(str).str.strip()
    df = df.dropna(subset=["GEOID", HARDSHIP_COL])

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = pd.to_numeric(df[HARDSHIP_COL], errors="coerce")
    valid = y.notna()
    X, y = X[valid], y[valid]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    rf_r2  = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    rf_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    gb_r2  = cross_val_score(gb, X, y, cv=kf, scoring="r2").mean()
    gb_mae = -cross_val_score(gb, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()

    rf.fit(X, y); gb.fit(X, y)
    df2 = df.copy()
    df2.loc[valid, "RF_Predicted"] = rf.predict(X).round(1)
    df2.loc[valid, "GB_Predicted"] = gb.predict(X).round(1)

    merged = gdf.merge(df2, on="GEOID", how="inner")
    short_names = [FEATURE_SHORT[FEATURE_COLS.index(c)] if c in FEATURE_COLS else c for c in available_features]

    scatter = [
        {"name": str(row.get("NAME", row["GEOID"])), "actual": float(row[HARDSHIP_COL]),
         "rf_pred": float(row.get("RF_Predicted", 0)), "gb_pred": float(row.get("GB_Predicted", 0)),
         "poverty": float(row.get("PERCENT HOUSEHOLDS BELOW POVERTY", 0)),
         "income": int(row.get("PER CAPITA INCOME", 0)) if pd.notna(row.get("PER CAPITA INCOME")) else 0}
        for _, row in df2[valid].iterrows()
    ]
    return {
        "merged": merged, "df": df2, "metrics": {
            "rf": {"r2": round(rf_r2, 3), "mae": round(rf_mae, 2)},
            "gb": {"r2": round(gb_r2, 3), "mae": round(gb_mae, 2)},
        },
        "feature_names": short_names,
        "rf_importances": [round(x, 4) for x in rf.feature_importances_.tolist()],
        "gb_importances": [round(x, 4) for x in gb.feature_importances_.tolist()],
        "scatter": scatter,
    }


# ── Chart helpers (XSS-safe DOM manipulation) ─────────────────────────────────

def _scatter_chart_html(scatter_json: str) -> str:
    return f"""<!DOCTYPE html><html><head><style>
  body{{margin:0;background:#0e1117;font-family:'DM Sans',sans-serif;color:#e8e8e8;}}
  .wrap{{display:flex;gap:20px;padding:16px;}}
  .panel{{flex:1;background:#1a1f2e;border:1px solid #2a3044;border-radius:12px;padding:20px;}}
  h3{{margin:0 0 14px;font-size:13px;font-weight:600;letter-spacing:.05em;color:#9eaec4;text-transform:uppercase;}}
  canvas{{display:block;}}
  #tip{{position:fixed;background:#1a1f2e;border:1px solid #3a4460;border-radius:8px;padding:10px 14px;
        font-size:12px;pointer-events:none;display:none;color:#e8e8e8;line-height:1.7;
        box-shadow:0 4px 20px #0008;z-index:9;}}
</style></head><body>
<div id="tip"></div>
<div class="wrap">
  <div class="panel"><h3>Random Forest — Actual vs Predicted</h3><canvas id="c1" width="430" height="390"></canvas></div>
  <div class="panel"><h3>Gradient Boosting — Actual vs Predicted</h3><canvas id="c2" width="430" height="390"></canvas></div>
</div>
<script>
const raw={scatter_json};const tip=document.getElementById('tip');
function setTip(pt,pk){{while(tip.firstChild)tip.removeChild(tip.firstChild);
  const b=document.createElement('b');b.textContent=pt.d.name;tip.appendChild(b);
  tip.appendChild(document.createElement('br'));
  ['Actual: '+pt.d.actual,'Predicted: '+pt.d[pk],'Error: '+(pt.d.actual-pt.d[pk]).toFixed(1)].forEach(t=>{{
    tip.appendChild(document.createTextNode(t));tip.appendChild(document.createElement('br'));
  }});}}
function draw(cid,pk,col){{const cv=document.getElementById(cid),ctx=cv.getContext('2d');
  const p={{l:50,r:20,t:20,b:46}},W=cv.width-p.l-p.r,H=cv.height-p.t-p.b;
  ctx.fillStyle='#0e1117';ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#2a3044';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){{const x=p.l+W*i/5,y=p.t+H*i/5;
    ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,p.t+H);ctx.stroke();
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+W,y);ctx.stroke();}}
  ctx.strokeStyle='#3a4460';ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
  ctx.beginPath();ctx.moveTo(p.l,p.t+H);ctx.lineTo(p.l+W,p.t);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#5d6a84';ctx.font='11px sans-serif';ctx.textAlign='center';
  for(let i=0;i<=5;i++){{const v=i*20;ctx.fillText(v,p.l+W*i/5,p.t+H+16);
    ctx.save();ctx.translate(p.l-16,p.t+H-H*i/5);ctx.rotate(-Math.PI/2);ctx.fillText(v,0,0);ctx.restore();}}
  const pts=raw.map(d=>({{x:p.l+(d.actual/100)*W,y:p.t+H-(d[pk]/100)*H,d}}));
  pts.forEach(pt=>{{ctx.beginPath();ctx.arc(pt.x,pt.y,5.5,0,Math.PI*2);
    ctx.fillStyle=col+'bb';ctx.fill();ctx.strokeStyle='#0e1117';ctx.lineWidth=1;ctx.stroke();}});
  cv.onmousemove=e=>{{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    let best=null,minD=1e9;pts.forEach(pt=>{{const d=Math.hypot(pt.x-mx,pt.y-my);if(d<minD){{minD=d;best=pt;}}}});
    if(minD<16){{tip.style.display='block';tip.style.left=(e.clientX+14)+'px';
      tip.style.top=(e.clientY-10)+'px';setTip(best,pk);}}else tip.style.display='none';}};
  cv.onmouseleave=()=>tip.style.display='none';}}
draw('c1','rf_pred','#4f8ef7');draw('c2','gb_pred','#f7934f');
</script></body></html>"""


def _importance_chart_html(names_json: str, rf_json: str, gb_json: str) -> str:
    return f"""<!DOCTYPE html><html><head><style>
  body{{margin:0;background:#0e1117;font-family:'DM Sans',sans-serif;color:#e8e8e8;padding:16px;}}
  .wrap{{display:flex;gap:20px;}}.panel{{flex:1;background:#1a1f2e;border:1px solid #2a3044;
    border-radius:12px;padding:24px;}}
  h3{{margin:0 0 20px;font-size:13px;font-weight:600;letter-spacing:.06em;color:#9eaec4;text-transform:uppercase;}}
  .row{{margin-bottom:18px;}}.lbl{{font-size:13px;color:#c8d0de;margin-bottom:6px;
    display:flex;justify-content:space-between;}}
  .track{{background:#2a3044;border-radius:6px;height:11px;overflow:hidden;}}
  .fill{{height:100%;border-radius:6px;width:0%;transition:width .7s cubic-bezier(.4,0,.2,1);}}
  .note{{margin-top:20px;font-size:12px;color:#5d6a84;line-height:1.8;
    border-top:1px solid #2a3044;padding-top:14px;}}
</style></head><body>
<div class="wrap">
  <div class="panel" id="rfp"><h3>Random Forest — Feature Importance</h3></div>
  <div class="panel" id="gbp"><h3>Gradient Boosting — Feature Importance</h3></div>
</div>
<script>
const names={names_json};const rfImp={rf_json};const gbImp={gb_json};
function renderBars(panelId,imps,g1,g2){{
  const panel=document.getElementById(panelId);
  const pairs=names.map((n,i)=>[n,imps[i]]).sort((a,b)=>b[1]-a[1]);
  pairs.forEach(([name,val])=>{{const pct=(val*100).toFixed(1);
    const row=document.createElement('div');row.className='row';
    const lbl=document.createElement('div');lbl.className='lbl';
    const ns=document.createElement('span');ns.textContent=name;
    const ps=document.createElement('span');ps.textContent=pct+'%';
    lbl.appendChild(ns);lbl.appendChild(ps);
    const track=document.createElement('div');track.className='track';
    const fill=document.createElement('div');fill.className='fill';
    fill.style.background='linear-gradient(90deg,'+g1+','+g2+')';fill.dataset.w=pct;
    track.appendChild(fill);row.appendChild(lbl);row.appendChild(track);panel.appendChild(row);}});
  setTimeout(()=>panel.querySelectorAll('.fill').forEach(el=>el.style.width=el.dataset.w+'%'),80);
  const note=document.createElement('div');note.className='note';
  note.textContent='Importance = how much each feature reduced prediction error across all trees.';
  panel.appendChild(note);}}
renderBars('rfp',rfImp,'#3a6fd8','#4f8ef7');renderBars('gbp',gbImp,'#d86b3a','#f7934f');
</script></body></html>"""


# ── Main render ───────────────────────────────────────────────────────────────

def render(chicago_geo=None, area_map=None):
    st.markdown(f"# {CITY_NAME} Community Hardship — ML Dashboard")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="socioeconomics", local_csv=None, label="Upload a socioeconomic dataset")

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="socio")

    # Auto-fetch census data and TIGER GeoJSON if not cached
    if not CSV_PATH.exists() or not GEO_PATH.exists():
        with st.spinner(f"Fetching census data for {CITY_NAME} from the US Census Bureau ACS API…"):
            if not os.environ.get("CENSUS_API_KEY"):
                st.warning(
                    "Set `CENSUS_API_KEY` in your `.env` file for best results.  \n"
                    "Free key: https://api.census.gov/data/key_signup.html  \n"
                    "Proceeding without a key (rate-limited)…"
                )
            if not CSV_PATH.exists():
                if not _fetch_acs_csv():
                    st.error("Failed to fetch census data.")
                    return
            if not GEO_PATH.exists():
                with st.spinner("Downloading TIGER tract boundaries…"):
                    if not _fetch_tiger_geojson():
                        st.error("Failed to download tract boundaries.")
                        return

    data = None
    try:
        data = load_and_train(str(CSV_PATH), str(GEO_PATH))
    except Exception as exc:
        st.error(f"Failed to load socioeconomic data: {exc}")

    tab1, tab2, tab3 = st.tabs(["Choropleth Map", "Model Diagnostics", "Feature Importance"])

    with tab1:
        if data is not None:
            merged = data["merged"]
            model_choice = st.radio("Colour map by:",
                                    ["Actual Hardship Index", "RF Predicted", "GB Predicted"], horizontal=True)
            col_map = {"Actual Hardship Index": HARDSHIP_COL,
                       "RF Predicted": "RF_Predicted", "GB Predicted": "GB_Predicted"}
            chosen_col = col_map[model_choice]
            geojson_dict = merged.__geo_interface__
            fig_main = px.choropleth_map(
                merged, geojson=geojson_dict, locations="GEOID",
                featureidkey="properties.GEOID",
                color=chosen_col, color_continuous_scale="YlOrRd",
                map_style=mapbox_style, zoom=MAP_ZOOM, center=MAP_CENTER, opacity=0.7,
                hover_name="NAME",
                hover_data={HARDSHIP_COL: True, "RF_Predicted": True, "GB_Predicted": True},
                title=model_choice,
            )
            fig_main.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=620)
            st.plotly_chart(fig_main, width="stretch")

            st.markdown("**Predicted Hardship Index — Side by Side**")
            lc, rc = st.columns(2)
            for col, pred_col, label in [(lc, "RF_Predicted", "Random Forest Predicted"),
                                          (rc, "GB_Predicted", "Gradient Boosting Predicted")]:
                with col:
                    fig_p = px.choropleth_map(
                        merged, geojson=geojson_dict, locations="GEOID",
                        featureidkey="properties.GEOID",
                        color=pred_col, color_continuous_scale="YlOrRd",
                        map_style=mapbox_style, zoom=MAP_ZOOM, center=MAP_CENTER, opacity=0.7,
                        hover_name="NAME", title=label,
                    )
                    fig_p.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=480)
                    st.plotly_chart(fig_p, width="stretch")
        else:
            st.info(f"Load census data for {CITY_NAME} to view this tab.")

    with tab2:
        if data is not None:
            html(_scatter_chart_html(json.dumps(data["scatter"])), height=450)
        else:
            st.info(f"Load census data for {CITY_NAME} to view this tab.")

    with tab3:
        if data is not None:
            html(_importance_chart_html(json.dumps(data["feature_names"]),
                                        json.dumps(data["rf_importances"]),
                                        json.dumps(data["gb_importances"])), height=440)
        else:
            st.info(f"Load census data for {CITY_NAME} to view this tab.")

    if data is not None:
        st.divider()
        st.subheader("Socioeconomic Scatterplots")
        df_sc = data["df"]
        n_col = NAME_COL if NAME_COL in df_sc.columns else AREA_COL
        sc1, sc2 = st.columns(2)
        with sc1:
            pov = "PERCENT HOUSEHOLDS BELOW POVERTY"
            inc = "PER CAPITA INCOME"
            if pov in df_sc.columns:
                fig_p = px.scatter(df_sc, x=pov, y=HARDSHIP_COL, hover_name=n_col,
                                   color=inc if inc in df_sc.columns else None,
                                   color_continuous_scale="RdYlGn_r",
                                   title="Poverty Rate vs Hardship Index")
                fig_p.update_layout(margin={"t": 30})
                st.plotly_chart(fig_p, width="stretch")
        with sc2:
            inc = "PER CAPITA INCOME"
            une = "PERCENT AGED 16+ UNEMPLOYED"
            if inc in df_sc.columns:
                fig_i = px.scatter(df_sc, x=inc, y=HARDSHIP_COL, hover_name=n_col,
                                   color=une if une in df_sc.columns else None,
                                   color_continuous_scale="YlOrRd",
                                   title="Per Capita Income vs Hardship Index")
                fig_i.update_layout(margin={"t": 30})
                st.plotly_chart(fig_i, width="stretch")

        st.divider()
        try:
            merged2 = data["merged"].copy()
            merged2["_id_str"] = merged2["GEOID"].astype(str)
            map_utils.render_moran_analysis(
                gdf=merged2, value_col=HARDSHIP_COL, name_col="NAME",
                id_col="_id_str", geojson=merged2.__geo_interface__,
                featureidkey="properties.GEOID", key_prefix="socio_moran", map_style=mapbox_style,
            )
        except Exception as exc:
            st.warning(f"Could not compute spatial autocorrelation: {exc}")
