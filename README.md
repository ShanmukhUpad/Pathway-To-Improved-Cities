# Pathway to Improved Cities

## Overview
This project analyzes urban datasets from cities to predict trends in crime, environmental issues, and traffic incidents. The goal is to identify potential problems and provide insights for city planning.

## Methodology
- **Data Analysis:** Collect data from open data portal websites, inspect the data, and wrangle the data.  
- **Data Engineering:** Create lag features, rolling averages, and population-normalized metrics.  
- **Modeling:** Develop baseline and tree-based models (Random Forest, Gradient Boosting) to predict future trends; evaluate with RMSE, MAE, and R² using time-aware splits.
- **Insights:** Analyze residuals, determine feature importance, and identify problems.  
- **Dashboard:** Build a Streamlit app to explore predictions and have cool vizualizations of spatial and temporal trends.
