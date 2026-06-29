# 💳 Credit Card Fraud Detection 

![Machine Learning Logo](https://img.shields.io/badge/MachineLearning%20-FF0000?logo=MachineLearning&logoColor=white)
![Level: Intermediate](https://img.shields.io/badge/Level-Intermediate-brightgreen)

### 🚀 Live Demo
https://credit-card-fraud-detection-3lylg3xdjbpurrd2bmgm9d.streamlit.app/

### 📂 GitHub Repository
https://github.com/prayagsapariya07/credit-card-fraud-detection


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

## Screenshots

<img width="1917" height="891" alt="P1" src="https://github.com/user-attachments/assets/c8ad0ad0-a49e-40c6-9485-271277f27b1b" />
<img width="1442" height="527" alt="P2" src="https://github.com/user-attachments/assets/58c402c5-48f2-405b-8d41-22ee0b4ca7e4" />
<img width="1462" height="585" alt="P3" src="https://github.com/user-attachments/assets/fe5573b3-bc92-4173-96ef-ceee0eb91b49" />


Prediction

Result

## Results

The model successfully identifies fraudulent transactions and achieves strong performance on the balanced dataset.

## Future Improvements

- Implement SMOTE
- Compare multiple ML models
- Deploy using Streamlit
- Build real-time prediction system

## Author

Prayag Sapariya
