# Survival Analysis and Customer Lifetime Value (CLV) Prediction

## Overview
This project implements Accelerated Failure Time (AFT) models to evaluate churn risk for telecommunication subscribers and estimates their Customer Lifetime Value (CLV). 
Three parametric distributions were tested and compared:
- Weibull
- Log-Normal
- Log-Logistic

The Log-Normal model was selected as the best fit and used to predict individual survival probabilities over a 12-month period, which were then discounted to calculate each customer's CLV.

## Setup
Install dependencies:

    pip install -r requirements.txt

## Run
Launch the Jupyter Notebook:

    jupyter notebook Survival_Analysis_Telco.ipynb

## Output
- **Survival Curve Plots:** Visual comparison showing the probability of survival over time.
- **CLV & Risk Analysis:** Individual CLV scores and risk flags for customers likely to churn within 1 year.
- **Segment Exploration:** Breakdown of value across service tiers and demographics.

## Project Structure
- `Survival_Analysis_Telco.ipynb`: Full analysis and reporting.
- `telco.csv`: Subscriber dataset.
- `requirements.txt`: Python library dependencies.
- `README.md`: Project documentation.

## Results & Business Insights
- **Best-fitting Model:** **Log-Normal AFT**, identified by the lowest AIC score (2954.02).
- **Significant Churn Drivers:** Having internet service is associated with higher churn risk (lower survival time). Conversely, longer tenure at a current address and being subscribed to premium tiers like "Total service," "E-service," or "Plus service" significantly prolong subscriber retention.
- **High-Value Segment:** We identified a core group of **250 high-value, low-risk customers** in the highest CLV quartile (Q4). Their average CLV is **$6,511**. A large majority of these highly valuable customers belong to the "Plus service" (123 customers) and "E-service" (66 customers) categories.
- **Retention Budget Estimate:** A key finding of this analysis is that **0 at-risk customers** fall into the highest value brackets (Q3 & Q4). Therefore, the estimated immediate budget needed to retain our most valuable cohort over the next 12 months is **$0.00**.

**Recommendation:** Since our highest-value customers (Q3 and Q4) already exhibit extremely high 12-month survival probabilities (over 97%), heavy retention spending is not currently required for this group. Instead, business efforts should focus on upselling lower-tier (Q1 and Q2) customers who *are* at risk, or acquiring new customers that match the demographic profile of our "Plus service" and "E-service" loyalists.
