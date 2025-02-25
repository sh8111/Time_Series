# Sales Demand Forecasting using Time Series and Machine Learning Models

## Overview
This project focuses on forecasting product sales on a **monthly** and **weekly** basis using a variety of time series and machine learning techniques. The objective is to accurately predict product demand to assist in inventory planning and resource allocation. Models implemented include **SARIMAX**, **ARIMA**, **Prophet**, **LSTM**, and **XGBoost**, ensuring a robust comparison across statistical, machine learning, and deep learning approaches.

## Project Structure
- **Monthly_Forecast.ipynb:** Develops models to predict monthly sales trends and demand.
- **Weekly_Forecast.ipynb:** Focuses on weekly sales forecasting with detailed analysis and model comparisons.
- **Data:** Contains the datasets used for both monthly and weekly sales forecasting.
- **Outputs:** Visualizations and evaluation metrics generated from the models.

## Methodology
### Data Exploration & Cleaning  
- Handled missing values and outliers using techniques like Winsorization and Z-score analysis.  
- Converted date columns into appropriate datetime formats and sorted the data chronologically.  
- Conducted exploratory data analysis (EDA) to identify trends, seasonality, and stationarity using plots and statistical tests.

### Feature Engineering  
- Created time-based features (month, week, year) to enhance model performance.  
- Applied differencing and decomposition techniques to handle seasonality and trends.

### Modeling Approaches  
- **SARIMAX & ARIMA:** Statistical models used for capturing autoregressive and moving average components.  
- **Prophet:** Deployed for capturing trend and seasonal components with holiday effects.  
- **LSTM:** Implemented a deep learning model to capture long-term dependencies in time series data.  
- **XGBoost:** Used for leveraging machine learning capabilities to predict based on lagged features and engineered variables.  
- **Bayesian Optimization:** Utilized for hyperparameter tuning, improving overall model performance.

### Evaluation Metrics  
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- **Mean Absolute Percentage Error (MAPE)**  

## Results  
- The **LSTM model** achieved the best performance in capturing complex temporal patterns.  
- **Prophet** and **SARIMAX** performed well for simpler trends and seasonality.  
- **XGBoost** showed competitive results with quick training times and effective feature utilization.

## Visualizations  
The notebooks include detailed visualizations:  
- Time series plots for actual vs. predicted values.  
- Residual analysis plots for error diagnostics.  
- Seasonal decomposition graphs showcasing trend, seasonality, and residual components.

## Requirements  
- Python 3.x  
- Libraries: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `prophet`, `tensorflow`, `xgboost`, `scikit-learn`, `pmdarima`, `scipy`  

## How to Run  
1. Clone the repository.  
2. Install the required libraries using `pip install -r requirements.txt`.  
3. Run `Monthly_Forecast.ipynb` or `Weekly_Forecast.ipynb` in Jupyter Notebook.  
4. Analyze the outputs and compare model performances.  

## Conclusion  
This project provides a comprehensive analysis of various forecasting methods, highlighting the strengths of each approach in different scenarios. The results demonstrate the importance of selecting appropriate models based on data characteristics and prediction requirements.

## Author  
Sumanth Bharadwaj Hachalli Karanam

