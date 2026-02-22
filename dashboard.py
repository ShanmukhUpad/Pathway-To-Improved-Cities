# dashboard.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json
import plotly.express as px

st.title("Pathway to Improved Cities - Crime Dashboard")

# --- 1. Load data ---
pivot = pd.read_csv("crime_monthly_pivot.csv")

# Convert Numeric Community Area Codes to Names
community_area_names = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
    29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
    33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
    37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
    41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
    53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
    57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
    61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
    65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
    69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
    73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
    76: "O'Hare", 77: "Edgewater"
}

pivot['Community Area Name'] = pivot['Community Area'].map(community_area_names)

# --- Create lag features for all crimes globally ---
lag_crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']]
for crime in lag_crime_cols:
    pivot[f'{crime}_lag1'] = pivot.groupby('Community Area')[crime].shift(1)
    pivot[f'{crime}_lag3'] = pivot.groupby('Community Area')[crime].shift(3)

# Community Area Dropdowns
community_areas = sorted(pivot['Community Area Name'].unique())  # sorted alphabetically
crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name'] 
              and not c.endswith('_lag1') and not c.endswith('_lag3')]

selected_area = st.selectbox("Select Community Area", community_areas)
selected_crime = st.selectbox("Select Crime Type", crime_cols)

# Subset Data
area_data = pivot[pivot['Community Area Name'] == selected_area].sort_values(['Year','Month'])

# Show historical trend
st.subheader(f"Historical {selected_crime} counts")
st.line_chart(area_data[selected_crime].values)

# Predict next month
feature_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name', selected_crime]]

# Drop rows with NaN (from lag features)
model_data = area_data.dropna(subset=feature_cols + [selected_crime])

if len(model_data) > 0:
    X = model_data[feature_cols]
    y = model_data[selected_crime]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict next month: use last row's features as input
    next_month_features = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(next_month_features)[0]
    
    st.subheader(f"Predicted {selected_crime} for next month")
    st.write(round(prediction))
else:
    st.write("Not enough data to predict yet.")

# --- Interactive map ---
st.subheader("Chicago Crime Map")

crime_map_type = st.selectbox("Select Crime Type for Map", crime_cols, key="map_select")
map_data = pivot.groupby('Community Area Name')[crime_map_type].sum().reset_index()

# Load GeoJSON for Chicago community areas
with open(r"C:\Users\moose\.cache\kagglehub\datasets\doyouevendata\chicago-community-areas-geojson\versions\1\chicago-community-areas.geojson") as f:
    chicago_geo = json.load(f)

st.json(chicago_geo['features'][0])

# --- Interactive Map Section ---
st.subheader("Chicago Crime Map")

# List of crime columns (exclude non-crime columns)
crime_cols = [col for col in pivot.columns if col not in ['Community Area', 'Year', 'Month', 'Community Area Name']]

# Load GeoJSON for Chicago community areas
geojson_path = r"C:\Users\moose\.cache\kagglehub\datasets\doyouevendata\chicago-community-areas-geojson\versions\1\chicago-community-areas.geojson"
with open(geojson_path) as f:
    chicago_geo = json.load(f)

# Map numeric Community Area IDs → names
area_map = {int(f['properties']['area_num_1']): f['properties']['community'] for f in chicago_geo['features']}
pivot['Community Area Name'] = pivot['Community Area'].map(area_map)

# Dropdown to select crime type
crime_map_type = st.selectbox("Select Crime Type for Map", crime_cols, key="crime_map_select")

# Aggregate crime counts by community area
map_data = pivot.groupby('Community Area Name')[crime_map_type].sum().reset_index()

# Create the choropleth map
fig = px.choropleth_mapbox(
    map_data,
    geojson=chicago_geo,
    locations='Community Area Name',
    featureidkey="properties.community",  # points to the 'community' property in GeoJSON
    color=crime_map_type,
    color_continuous_scale="Reds",
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 41.8781, "lon": -87.6298},
    opacity=0.6,
    labels={crime_map_type: "Crime Count"}
)

st.plotly_chart(fig, use_container_width=True)

# --- Prediction Map Section ---
st.subheader("Predicted Crime Map")

# Dropdown to select crime type for prediction
crime_pred_type = st.selectbox("Select Crime Type to Predict", crime_cols, key="crime_map_pred_select")

# Normalize crime name to uppercase to match lag columns
crime_pred_type_upper = crime_pred_type.upper()

# Select lag columns for the chosen crime
lag_cols = [f'{crime_pred_type_upper}_lag1', f'{crime_pred_type_upper}_lag3']
X = pivot[lag_cols].dropna()
y = pivot[crime_pred_type_upper][X.index]

if X.empty or y.empty:
    st.warning(f"No lag features found for {crime_pred_type}. Cannot generate predictions.")
else:
    # Train Random Forest on lag features for this crime
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict the next month for each community area
    # Get the latest month for each community area
    latest_month = pivot.groupby('Community Area Name').tail(1).copy()

    # Prepare X_pred from lag columns
    X_pred = latest_month[lag_cols]

    # Predict
    latest_month['Predicted'] = model.predict(X_pred)

    # Aggregate predicted counts by community area (though one row per area)
    pred_map_data = latest_month[['Community Area Name', 'Predicted']]

    # Create choropleth map of predictions
    fig_pred = px.choropleth_mapbox(
        pred_map_data,
        geojson=chicago_geo,
        locations='Community Area Name',
        featureidkey="properties.community",
        color='Predicted',
        color_continuous_scale="Reds",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 41.8781, "lon": -87.6298},
        opacity=0.6,
        labels={'Predicted': f'Predicted {crime_pred_type} Count'}
    )

    st.plotly_chart(fig_pred, use_container_width=True)