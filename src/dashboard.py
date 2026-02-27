import requests
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# --- 2. Load GeoJSON remotely (no hardcoded path) ---
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

chicago_geo = load_geojson()

# Remap Community Area Names using GeoJSON properties
area_map = {
    int(f['properties']['area_num_1']): f['properties']['community']
    for f in chicago_geo['features']
}
pivot['Community Area Name'] = pivot['Community Area'].map(area_map)

# --- 3. Create lag features for all crimes globally ---
lag_crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']]
for crime in lag_crime_cols:
    pivot[f'{crime}_lag1'] = pivot.groupby('Community Area')[crime].shift(1)
    pivot[f'{crime}_lag3'] = pivot.groupby('Community Area')[crime].shift(3)

# --- 4. Community Area dropdowns ---
community_areas = sorted(pivot['Community Area Name'].dropna().unique())
crime_cols = [
    c for c in pivot.columns
    if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']
    and not c.endswith('_lag1')
    and not c.endswith('_lag3')
]

selected_area = st.selectbox("Select Community Area", community_areas)
selected_crime = st.selectbox("Select Crime Type", crime_cols)

# --- 5. Subset data and show historical trend ---
area_data = pivot[pivot['Community Area Name'] == selected_area].sort_values(['Year', 'Month'])

st.subheader(f"Historical {selected_crime} counts")
st.line_chart(area_data[selected_crime].values)

# --- 6. Predict next month for selected area/crime ---
feature_cols = [
    c for c in pivot.columns
    if c not in ['Community Area', 'Year', 'Month', 'Community Area Name', selected_crime]
]

model_data = area_data.dropna(subset=feature_cols + [selected_crime])

if len(model_data) > 0:
    X = model_data[feature_cols]
    y = model_data[selected_crime]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    next_month_features = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(next_month_features)[0]

    st.subheader(f"Predicted {selected_crime} counts for next month")
    st.write(round(prediction))
else:
    st.write("Not enough data to predict yet.")

# --- 7. Interactive Crime Map ---
st.subheader("Chicago Crime Map")

crime_map_type = st.selectbox("Select Crime Type for Map", crime_cols, key="crime_map_select")

map_data = pivot.groupby('Community Area Name')[crime_map_type].sum().reset_index()

fig = px.choropleth_mapbox(
    map_data,
    geojson=chicago_geo,
    locations='Community Area Name',
    featureidkey="properties.community",
    color=crime_map_type,
    color_continuous_scale="Reds",
    mapbox_style="open-street-map",
    zoom=9,
    center={"lat": 41.8781, "lon": -87.6298},
    opacity=0.5,
    labels={crime_map_type: "Crime Count"}
)

# Scale bar truncated to 2 decimal places
fig.update_coloraxes(colorbar_tickformat='.2f')

st.plotly_chart(fig, use_container_width=True)

# --- 8. Predicted Crime Map ---
st.subheader("Predicted Crime Map")

crime_pred_type = st.selectbox("Select Crime Type to Predict", crime_cols, key="crime_map_pred_select")
crime_pred_type_upper = crime_pred_type.upper()

lag_cols = [f'{crime_pred_type_upper}_lag1', f'{crime_pred_type_upper}_lag3']

# Check that lag columns exist
missing_lags = [col for col in lag_cols if col not in pivot.columns]
if missing_lags:
    st.warning(f"No lag features found for '{crime_pred_type}'. Cannot generate predictions.")
else:
    pred_data = pivot.dropna(subset=lag_cols + [crime_pred_type_upper])
    X_train = pred_data[lag_cols]
    y_train = pred_data[crime_pred_type_upper]

    if X_train.empty or y_train.empty:
        st.warning(f"Not enough data to predict for '{crime_pred_type}'.")
    else:
        pred_model = RandomForestRegressor(n_estimators=100, random_state=42)
        pred_model.fit(X_train, y_train)

        latest_month = pivot.groupby('Community Area Name').tail(1).copy()
        X_pred = latest_month[lag_cols].fillna(0)

        # Truncate predictions to 2 decimal places
        latest_month['Predicted'] = pred_model.predict(X_pred).round(2)
        pred_map_data = latest_month[['Community Area Name', 'Predicted']]

        fig_pred = px.choropleth_mapbox(
            pred_map_data,
            geojson=chicago_geo,
            locations='Community Area Name',
            featureidkey="properties.community",
            color='Predicted',
            color_continuous_scale="Reds",
            mapbox_style="open-street-map",
            zoom=9,
            center={"lat": 41.8781, "lon": -87.6298},
            opacity=0.5,
            labels={'Predicted': f'Predicted {crime_pred_type} Count'}
        )

        # Scale bar truncated to 2 decimal places
        fig_pred.update_coloraxes(colorbar_tickformat='.2f')

        st.plotly_chart(fig_pred, use_container_width=True)