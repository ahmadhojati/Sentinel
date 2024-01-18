# Sentinel
Snow density estimation using sentinel data

# Overview
This Python code is designed for modeling and forecasting snow density using machine learning techniques. It utilizes data from Sentinel satellites, SNOTEL (Snow Telemetry) stations, Landsat 8, and climate class information. The main components of the code include data preprocessing, feature engineering, model training, and evaluation.  

# Prerequisites
Make sure you have the following libraries installed:  

pandas  
matplotlib  
scikit-learn  
skforecast  
numpy  
tensorflow  
Signal_processing (custom module)  
# Code Structure   
## Import Libraries:   
Import necessary Python libraries, including pandas, matplotlib, scikit-learn, skforecast, numpy, and TensorFlow.  

## Functions Definitions:   
Define various functions used in the code, such as rounding coordinates, adding day of water year to the data, calculating R-squared, and others.  

## Initial Values:  
Set initial values such as process resolution, minimum snow density, maximum snow density, and model training parameters.  

## Read Full Data Frame:  
Load the data from a CSV file into a pandas DataFrame.  

## Apply Required Filters:  
Apply filters to the DataFrame based on snow density, snow depth, and resolution.  

## Add Orbit Information:  
Extract the hour from the "Time" column and create an "Orbit" column based on the hour.  

## Find Coordinates with Sufficient Measurements:  
Identify coordinates with at least 200 measurements and filter the DataFrame accordingly.  

## Data Normalization:  
Normalize the data using min-max normalization.  

## Modeling and Forecasting:  
Utilize machine learning models, such as a neural network, to train and evaluate the forecasting performance.  

## Feature Importance:  
Evaluate the importance of features in the trained model.  

# Usage  
To use this code, follow these steps:  

Install the required libraries mentioned in the "Prerequisites" section.  
Ensure that the custom module Signal_processing is available.  
Adjust the initial values and parameters based on your specific requirements.  
Run the code step by step.  
# Important Notes  
The code assumes the existence of a CSV file containing the input data.  
Model training parameters, such as the number of epochs and learning rate, can be adjusted as needed.  
The code includes functions for calculating R-squared, feature importance, and normalizing data.  
# Output  
The code generates various outputs, including model evaluation metrics, feature importance plots, and saved model files.  
