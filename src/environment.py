import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

## Load Data
energy = pd.read_csv('clean_data/clean_energy.csv')
benchmark = pd.read_csv('clean_data/clean_benchmark.csv')
complaints = pd.read_csv('clean_data/clean_complaints.csv')

## Feature Engineering
# energy data
numeric_cols = [
    'TOTAL_KWH',
    'KWH_TOTAL_SQFT',
    'TOTAL_THERMS',
    'TOTAL_POPULATION',
    'TOTAL_UNITS'
]

for col in numeric_cols:
    energy[col] = (
        energy[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace(r'[^0-9.\-]', '', regex=True)  # remove garbage characters
        .str.strip()
    )
    energy[col] = pd.to_numeric(energy[col], errors='coerce')

energy[numeric_cols] = energy[numeric_cols].fillna(0)

energy = energy[energy['TOTAL_POPULATION'] < 1e7]
energy = energy[energy['TOTAL_UNITS'] < 1e6]

energy['energy_intensity'] = energy['TOTAL_KWH'] / energy['KWH_TOTAL_SQFT'].replace(0, 1)

# benchmark data
benchmark['size_code'] = benchmark['Cohort_-_Size'].astype('category').cat.codes

# complaints data
complaints['COMPLAINT_DATE'] = pd.to_datetime(
    complaints['COMPLAINT_DATE'],
    errors='coerce'
)

complaints = complaints.dropna(subset=['COMPLAINT_DATE'])
complaints['year'] = complaints['COMPLAINT_DATE'].dt.year

## Merge datasets
energy['COMMUNITY_AREA_NAME'] = energy['COMMUNITY_AREA_NAME'].str.strip().str.upper()
benchmark['Community_Area_Name'] = benchmark['Community_Area_Name'].str.strip().str.upper()

# aggregate energy
energy_agg = energy.groupby('COMMUNITY_AREA_NAME').agg({
    'TOTAL_KWH': 'sum',
    'TOTAL_THERMS': 'sum',
    'energy_intensity': 'mean',
    'TOTAL_POPULATION': 'mean',
    'TOTAL_UNITS': 'mean'
}).reset_index()

# aggregate benchmark
benchmark_agg = benchmark.groupby('Community_Area_Name').agg({
    'Building_ID': 'count',
    'size_code': 'mean'
}).reset_index()

benchmark_agg.rename(columns={
    'Community_Area_Name': 'COMMUNITY_AREA_NAME',
    'Building_ID': 'num_large_buildings'
}, inplace=True)


complaints_clean = complaints.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

# process complaints
complaints['LATITUDE'] = pd.to_numeric(complaints['LATITUDE'], errors='coerce')
complaints['LONGITUDE'] = pd.to_numeric(complaints['LONGITUDE'], errors='coerce')
complaints_clean = complaints.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

benchmark_valid = benchmark[['Latitude', 'Longitude', 'Community_Area_Name']].copy()
benchmark_valid['Latitude'] = pd.to_numeric(benchmark_valid['Latitude'], errors='coerce')
benchmark_valid['Longitude'] = pd.to_numeric(benchmark_valid['Longitude'], errors='coerce')
benchmark_valid = benchmark_valid.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)

# check
print("NaNs in complaints coords:", complaints_clean[['LATITUDE', 'LONGITUDE']].isna().sum())
print("NaNs in benchmark coords:", benchmark_valid[['Latitude', 'Longitude']].isna().sum())

## KDTree
complaints_coords = complaints_clean[['LATITUDE', 'LONGITUDE']].values
benchmark_coords = benchmark_valid[['Latitude', 'Longitude']].values

complaints_coords = complaints_coords[~np.isnan(complaints_coords).any(axis=1)]
benchmark_coords = benchmark_coords[~np.isnan(benchmark_coords).any(axis=1)]

# build tree
tree = KDTree(benchmark_coords)

dist, idx = tree.query(complaints_coords, k=1)

complaints_clean = complaints_clean.iloc[:len(idx)].copy()
complaints_clean['COMMUNITY_AREA_NAME'] = benchmark_valid.iloc[idx.flatten()]['Community_Area_Name'].values

complaints_agg = complaints_clean.groupby('COMMUNITY_AREA_NAME').size().reset_index(name='complaint_count')

# final merge
merged = energy_agg.merge(
    benchmark_agg,
    on='COMMUNITY_AREA_NAME',
    how='left'
)

merged = merged.merge(
    complaints_agg,
    on='COMMUNITY_AREA_NAME',
    how='left'
)

merged['num_large_buildings'] = merged['num_large_buildings'].fillna(0)
merged['complaint_count'] = merged['complaint_count'].fillna(0)

## Model
X = merged[[
    'energy_intensity',
    'complaint_count',
    'num_large_buildings',
    'TOTAL_POPULATION',
    'TOTAL_UNITS'
]]

y = merged['TOTAL_KWH']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

# evaluation
mse = mean_squared_error(y_test, preds)
print("MSE:", mse)
