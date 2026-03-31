# import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# load datasets
def load_data():
    bike = pd.read_csv("transport_access_clean_data/bike_routes_clean.csv")
    bus = pd.read_csv("transport_access_clean_data/bus_stops_clean.csv")
    divvy = pd.read_csv("transport_access_clean_data/divvy_bicycle_clean.csv")
    
    #print("Bike columns: ", bike.columns)
    #print("Bus columns: ", bus.columns)
    #print("Divvy columns: ", divvy.columns)
    bike.columns = bike.columns.str.strip().str.lower()
    bus.columns = bus.columns.str.strip().str.lower()
    divvy.columns = divvy.columns.str.strip().str.lower()
    
    return bike, bus, divvy


# bike routes feature engineering
def process_bike_routes(bike):
    print("Processing bike routes...")

    # count bike routes per street
    bike_counts = bike.groupby("st_name").size().reset_index(name="num_bike_routes")

    # create contraflow flag (1 if exists, 0 otherwise)
    bike["contraflow_flag"] = bike["contraflow"].notnull().astype(int)

    # count contraflow routes per street
    contraflow_counts = bike.groupby("st_name")["contraflow_flag"].sum().reset_index()

    # merge features
    bike_features = pd.merge(
        bike_counts,
        contraflow_counts,
        on="st_name",
        how="left"
    )

    return bike_features


# bus stops feature engineering
def process_bus_stops(bus):
    print("Processing bus stops...")

    # count number of routes per stop
    bus["num_routes"] = bus["routes"].apply(
        lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
    )

    # aggregate per ward
    bus_per_ward = bus.groupby("ward").agg({
        "stop_id": "count",
        "num_routes": "mean"
    }).rename(columns={
        "stop_id": "num_stops",
        "num_routes": "avg_routes_per_stop"
    }).reset_index()

    return bus_per_ward


# divvy bike stations feature engineering
def process_divvy(divvy):
    print("Processing Divvy stations...")

    # keep only active stations
    divvy = divvy[divvy["status"].str.lower() == "in service"]

    # utilization ratio
    divvy["utilization_ratio"] = divvy["docks_in_service"] / divvy["total_docks"]

    return divvy


# assign bike stations to wards (approximation)
def assign_stations_to_wards(divvy, bus):
    print("Assigning Divvy stations to wards...")

    # assign ward by using nearest bus stop 
    def find_nearest_ward(lat, lon):
        distances = ((bus["latitude"] - lat)**2 + (bus["longitude"] - lon)**2)
        nearest_idx = distances.idxmin()
        return bus.loc[nearest_idx, "ward"]

    divvy["ward"] = divvy.apply(
        lambda row: find_nearest_ward(row["latitude"], row["longitude"]), axis=1
    )

    return divvy


# aggregate divvy by ward
def aggregate_divvy_by_ward(divvy):
    print("Aggregating Divvy stations by ward...")

    divvy_per_ward = divvy.groupby("ward").agg({
        "id": "count",
        "total_docks": "mean"
    }).rename(columns={
        "id": "num_stations",
        "total_docks": "avg_docks"
    }).reset_index()

    return divvy_per_ward


# create bike route features by ward (approximation) 
def create_bike_route_features(bike):
    print("Creating bike route features...")

    # total routes and contraflow routes
    total_routes = len(bike)
    total_contraflow = bike["contraflow"].notnull().sum()

    bike_summary = pd.DataFrame({
        "num_bike_routes": [total_routes],
        "num_contraflow": [total_contraflow]
    })

    return bike_summary


# merge datasets
def merge_datasets(bus_per_ward, divvy_per_ward):
    print("Merging datasets...")

    df = pd.merge(bus_per_ward, divvy_per_ward, on="ward", how="left")

    # fill missing values
    df.fillna(0, inplace=True)

    return df


# create target variables
def create_targets(df):
    print("Creating target variables...")

    # accessibility score
    df["accessibility_score"] = df["num_stops"] + df["avg_routes_per_stop"]

    # binary target -> "has strong bike infrastructure"
    df["has_bike_infra"] = (df["num_stations"] > df["num_stations"].median()).astype(int)

    return df

# ==========================================================================================
# train regression model: predict accessibility score
def train_regression_model(df):
    print("Training regression model...")

    features = ["num_stations", "avg_docks"]
    target = "accessibility_score"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Regression Results:")
    print("R2 Score:", r2_score(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

    return model


# ======================================================================================
# train classification model: predict bike infrastructure presence
def train_classification_model(df):
    print("Training classification model...")

    features = ["num_stops", "avg_routes_per_stop"]
    target = "has_bike_infra"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Classification Accuracy:", accuracy_score(y_test, predictions))

    return model


# identify underserved areas
def find_underserved_areas(df):
    print("Identifying underserved wards...")

    underserved = df[
        (df["num_stops"] < df["num_stops"].median()) &
        (df["num_stations"] < df["num_stations"].median())
    ]

    print("Underserved wards:")
    print(underserved[["ward", "num_stops", "num_stations"]])

    return underserved


# main 
def main():
    bike, bus, divvy = load_data()

    bike_features = process_bike_routes(bike)
    bus_per_ward = process_bus_stops(bus)

    divvy = process_divvy(divvy)
    divvy = assign_stations_to_wards(divvy, bus)
    divvy_per_ward = aggregate_divvy_by_ward(divvy)

    df = merge_datasets(bus_per_ward, divvy_per_ward)
    df = create_targets(df)

    print("\nFinal Dataset Preview:")
    print(df.head())

    # train models
    train_regression_model(df)
    train_classification_model(df)

    # find underserved areas
    find_underserved_areas(df)

if __name__ == "__main__":
    main()