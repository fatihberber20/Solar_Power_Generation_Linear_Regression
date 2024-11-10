Multiple Linear Regression Analysis on Solar Energy Dataset

This repository contains an implementation of Multiple Linear Regression (MLR) on a solar energy dataset to predict energy output based on environmental and positional factors. The analysis explores the relationship between factors such as altitude, solar radiation, temperature, and power levels with energy output. Using Python libraries, the project performs regression modeling, feature selection, and visualization to identify the most influential predictors.

Project Overview
This project demonstrates the application of Multiple Linear Regression to analyze a dataset with 1620 records and 8 variables related to solar energy output. The dataset, Pv.xlsx, contains information on environmental factors such as altitude, solar duration, temperature, and global solar radiation, along with the power level and axis orientation, which influence the target variable Result (energy output).

Key steps include:

Data Correlation Analysis
Descriptive Statistical Summaries
Data Normalization and Scaling
Feature Selection using Backward Elimination
Model Evaluation with performance metrics like MAE, MSE, RMSE, R², and MAPE
Dataset Description
The dataset columns are as follows:

ALT: Altitude (m)
LAT: Latitude (geographical coordinate)
SSD: Solar radiation duration (hours/day)
T: Average temperature (°C)
GSR: Daily global solar radiation (kWh/m²)
Power: Power level setting
Axis: Axis number (orientation)
Result: Target variable representing the energy output
Code Structure
Correlation Analysis: Generates a heatmap to show relationships between variables.
Descriptive Statistics: Provides statistical summaries of independent and dependent variables.
Data Normalization: Scales features and target variables using MinMaxScaler.
Multiple Linear Regression: Fits a regression model on the training set.
Backward Elimination: Selects optimal features for regression using the backward elimination technique.
Performance Metrics Calculation: Computes MAE, MSE, RMSE, R², and MAPE to evaluate model accuracy.
Evaluation Metrics
MAE (Mean Absolute Error): Average magnitude of absolute errors.
MSE (Mean Squared Error): Average of squared differences between actual and predicted values.
MedAE (Median Absolute Error): Median of absolute errors, showing error spread.
R² (Coefficient of Determination): Proportion of variance explained by the model.
RMSE (Root Mean Squared Error): Provides error magnitude in original units.
MAPE (Mean Absolute Percentage Error): Prediction error as a percentage.

Installation and Usage
Clone the repository:
git clone https://github.com/username/repo-name.git

Install required libraries:
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
Place Pv.xlsx in the repository folder.

Run the code:
python mlr_analysis.py

Dependencies
numpy
pandas
scikit-learn
statsmodels
matplotlib
seaborn

Acknowledgements
This project is built using open-source libraries and a solar energy dataset. Special thanks to the data providers and the open-source community for tools and resources.
