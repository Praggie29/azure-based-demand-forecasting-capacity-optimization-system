# Azure Demand Forecasting & Capacity Optimization System

## Project Overview
This project focuses on building a predictive system to accurately forecast Azure Compute and Storage demand. The goal is to support the Azure Supply Chain team in making informed capacity provisioning decisions, reducing CAPEX waste, and optimizing infrastructure investment.

## Features & Outcomes
* **Improved Accuracy:** High-precision forecasting for Azure service demand.
* **Capacity Planning:** Optimized provisioning across various regions.
* **Cost Savings:** Contributing to potential annual savings of ~$120M per 1% gain in accuracy.
* **Actionable Insights:** Integrated intelligence for the supply chain team.

## Dataset Description
The dataset includes historical Azure usage data with the following dimensions:
* **Timestamp:** Temporal data for time-series analysis.
* **Region & Service Type:** Regional (e.g., eastus, westus) and service-specific (Compute, Storage) dimensions.
* **Usage Units:** The target variable for forecasting.
* **External Variables:** Includes economic indicators like `energy_price_index`.

## Project Milestones

### Milestone 1: Data Collection & Preparation (Completed)
* **Data Cleaning:** Handled missing values in `usage_units` using median imputation.
* **Validation:** Removed duplicate records and verified data types.
* **Formatting:** Converted `timestamp` to datetime objects and sorted records chronologically

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy

