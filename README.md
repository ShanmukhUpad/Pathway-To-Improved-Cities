# Pathway to Improved Cities

## Overview
This project analyzes urban datasets from cities to predict trends in crime, environmental issues, and traffic incidents. The goal is to identify potential problems and provide insights for city planning.

## Goals
- Aggregated urban datasets for modeling.  
- Predictive models to forecast trends in urban data.   
- Interactive dashboard for exploring predictions and visualizations.  

## Methodology
- **Data Analysis:** Collect data from open data portal websites, inspect the data, and wrangle the data.  
- **Data Engineering:** Create lag features, rolling averages, and population-normalized metrics.  
- **Modeling:** Develop baseline and tree-based models (Random Forest, Gradient Boosting) to predict future trends; evaluate with RMSE, MAE, and R² using time-aware splits.
- **Insights:** Analyze residuals, determine feature importance, and identify problems.  
- **Dashboard:** Build a Streamlit dashboard to explore predictions and have cool vizualizations of spatial and temporal trends.

## Languages / Tools / Frameworks
- **Language:** Python  
- **Libraries:** Pandas, GeoPandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, Folium, Streamlit

## How to Install Dependencies and Run the Dashboard

### 1. Set up the virtual environment (In terminal)
```
bash setup.sh
```
For Mac/Linux:
```
source venv/bin/activate
```
For Windows:
```
venv\Scripts\activate
```
### 2: Running the dashboard: 
```Streamlit run src/dashboard.py``` (or if you are already in src then just ```Streamlit run dashboard.py```)

## Timeline:
- Week 6 (Complete): Setup repository, create a dashboard, get an example of a dataset visualization for current and predicted
- Week 7: Assign people into different domains, each person picks datasets to work with, more datasets are added to dashboard
- Week 8: Train models' random forest algorithms to be more accurate, create lag features for each dataset. Integrate all the domains into dashboard, with a tab to select each one.

SPRING BREAK

- Week 9: Implement gradient boosting for the models, and maybe also add the RMSE, MAE, and R^2
- Week 10: Make new types of visualizations (scatterplots, KDE, etc) and also add a feature to search for datasets via city
- Week 11: Implement File Upload System, where any user can upload their own dataset, if it has certain columns, then they can choose what domain the dataset belongs to, and it will be added to the dashboard.
- Week 12: Implement computer vision features such as building a CNN to recognize image data to then visualize problem areas within a city.
