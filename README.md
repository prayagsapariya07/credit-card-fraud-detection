# 💳 Credit Card Fraud Detection using Machine Learning

##  Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Due to the highly imbalanced nature of the dataset, special care is taken to ensure accurate fraud detection while minimizing false negatives.

---

##  Objectives

* Detect fraudulent transactions with high recall
* Handle imbalanced dataset effectively
* Build a reliable and interpretable ML model

---

##  Dataset

Dataset provided by ULB Machine Learning Group (Kaggle)

* Total Transactions: 284,807
* Fraud Cases: 492
* Features: V1–V28 (PCA transformed), Time, Amount

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

##  Workflow

1. Data Cleaning
2. Handling Missing Values
3. Exploratory Data Analysis (EDA)
4. Handling Imbalanced Data (Undersampling)
5. Model Training (Logistic Regression)
6. Model Evaluation

---

##  Model Used

Logistic Regression (Baseline Model)

---

##  Performance Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97%   |
| Precision | 1.00  |
| Recall    | 0.95  |
| F1-score  | 0.97  |

---

## 📊 Confusion Matrix

|               | Predicted Normal | Predicted Fraud |
| ------------- | ---------------- | --------------- |
| Actual Normal | 17               | 0               |
| Actual Fraud  | 1                | 18              |

---

##  Key Insight

Minimizing **False Negatives** is critical in fraud detection, as missing a fraudulent transaction can lead to financial loss.

---

##  Future Improvements

* Implement SMOTE for better data balancing
* Use advanced models (Random Forest, XGBoost)
* Deploy using Streamlit
* Real-time fraud detection system

---

##  Project Structure

(Explain your folders here)

---

##  Conclusion

The model demonstrates strong performance in detecting fraud with high recall and precision. Further improvements can make it production-ready.

---
