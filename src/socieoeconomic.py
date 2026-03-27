import geopandas as gpd
import pandas as pd
import folium
import streamlit as st
from streamlit.components.v1 import html

def render():
    gdf = gpd.read_file("chicago-community-areas.geojson")
    df = pd.read_csv("censusChicago.csv")
    gdf.columns = gdf.columns.str.strip()
    df.columns = df.columns.str.strip()

    gdf["area_num_1"] = gdf["area_num_1"].fillna(0).astype(int)
    df["Community Area Number"] = df["Community Area Number"].fillna(0).astype(int)

    merged = gdf.merge(
        df,
        left_on="area_num_1",
        right_on="Community Area Number"
    )

    m = folium.Map(location=[41.85, -87.68], zoom_start=10, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=["area_num_1", "HARDSHIP INDEX"],
        key_on="feature.properties.area_num_1",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Hardship Index",
        highlight=True,
    ).add_to(m)

    tooltip = folium.GeoJsonTooltip(
        fields=["community", "HARDSHIP INDEX", "PER CAPITA INCOME", "PERCENT HOUSEHOLDS BELOW POVERTY"],
        aliases=["Community:", "Hardship Index:", "Per Capita Income:", "% Households Below Poverty:"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: white;
            border: 1px solid black;
            border-radius: 4px;
            box-shadow: 3px;
        """
    )

    folium.GeoJson(
        merged,
        style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0,
        },
        highlight_function=lambda feature: {
            "weight": 3,
            "color": "blue",
            "fillOpacity": 0.15,
        },
        tooltip=tooltip,
    ).add_to(m)

    html(m._repr_html_(), height=650)

