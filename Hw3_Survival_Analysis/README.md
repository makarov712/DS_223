# Survival Analysis and Customer Lifetime Value (CLV) Prediction

## Overview
This project implements Accelerated Failure Time (AFT) models to evaluate churn risk for telecommunication subscribers and estimates their Customer Lifetime Value (CLV). 
Three parametric distributions are tested and compared:
- Weibull
- Log-Normal
- Log-Logistic

The best-fitting model is used to predict individual survival probabilities over time, which are then discounted to calculate each customer's CLV.

## Setup
Install dependencies:

    pip install -r requirements.txt

## Run
Launch the Jupyter Notebook:

    jupyter notebook Survival_Analysis_Telco.ipynb

## Output
- Inline survival curve plots - visual comparison of the three AFT models
- CLV calculations - appended to the dataset for segment analysis
- Markdown Report - final written analysis on churn risk factors, segment value, and retention budget recommendations.

## Project Structure
- Survival_Analysis_Telco.ipynb
- telco.csv
- requirements.txt
- README.md

## Results
- Best-fitting AFT Model: Log-Normal (based on AIC comparison)
- Key churn drivers: [Add 1-2 significant features here, e.g., internet service, marital status]
- Highest CLV Segment: [Add your most valuable segment from the final printout here]
- Annual Retention Budget Estimate: $[Add your budget estimate here]

The Log-Normal model provided the best fit for the data and was utilized to segment the most valuable customers for targeted retention strategies.