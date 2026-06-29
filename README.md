# 💳 Credit Card Fraud Detection 

![Machine Learning Logo](https://img.shields.io/badge/MachineLearning%20-FF0000?logo=MachineLearning&logoColor=white)
![Level: Intermediate](https://img.shields.io/badge/Level-Intermediate-brightgreen)


## Overview

This project uses Machine Learning to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud datasets, preprocessing and sampling techniques were used to improve model performance.

## Problem Statement

Credit card fraud causes significant financial losses worldwide. The goal of this project is to build a machine learning model that can accurately identify fraudulent transactions while minimizing false positives.

## Dataset

- Source: Kaggle Credit Card Fraud Detection Dataset
- Total Transactions: 284,807
- Fraud Cases: 492
- Features: V1-V28, Time, Amount

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
├── models/
├── project/
├── reports/
├── README.md
```

## Data Preprocessing

- Checked missing values
- Analyzed class distribution
- Applied undersampling to balance the dataset
- Prepared training and testing datasets

## Model Training

Model Used:

- Logistic Regression

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

## Results

The model successfully identifies fraudulent transactions and achieves strong performance on the balanced dataset.

## Future Improvements

- Implement SMOTE
- Compare multiple ML models
- Deploy using Streamlit
- Build real-time prediction system

## Author

Prayag Sapariya
