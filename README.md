# Retail Sales Analysis & High-Sale Prediction

A compact data analysis and machine learning project to explore retail sales patterns and predict **high-selling product categories** using **XGBoost**.

## Overview
This project analyzes a retail sales dataset to:
- Visualize sales trends by product category and age.
- Build a predictive model to classify **high-sale** product categories.
- Identify key factors driving high sales using feature importance.

## Dataset
- `retail_sales_dataset.csv`: Contains transaction data including:
  - Customer: `Age`, `Gender`
  - Transaction: `Date`, `Quantity`, `Price per Unit`, `Total Amount`
  - Product: `Product Category`

## Key Steps
1. **Data Preprocessing**:
   - Encode `Gender` (0/1)
   - Extract `Month` & `Day` from `Date`
   - Drop irrelevant columns (`Transaction ID`, `Date`)

2. **Exploratory Analysis**:
   - Total sales by product category
   - Average spending by age

3. **Target Creation**:
   - `High_Sale` = 1 if category's average sale ≥ overall mean

4. **Modeling**:
   - Train/Test split (80/20)
   - XGBoost Classifier
   - Evaluation: Accuracy, Precision, Recall, F1

5. **Feature Importance**:
   - Top predictors: **Age**, **Day**, **Month**

## Results
- **Electronics, Clothing, Beauty** are top-selling categories.
- Spending peaks at **age 35–40**.
- Model achieves **~57% accuracy**, strong on high-sale detection (**F1 = 0.69**).
- **Gender** has minimal impact.

## Visualizations
- Bar plot: Total sales by category
- Line plot: Avg. spending by age
- Feature importance plot (XGBoost)

## Requirements
```txt
pandas
numpy
matplotlib
scikit-learn
xgboost
mlxtend
