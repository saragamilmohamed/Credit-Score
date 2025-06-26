

# 💳 Credit Score Prediction

This project aims to build robust machine learning and deep learning models to predict **creditworthiness** of individuals based on financial and behavioral data. The goal is to help financial institutions assess loan risk and automate credit scoring decisions.

The project is implemented in two phases:

1. **Credit Score ML.ipynb** — Traditional machine learning pipeline using Scikit-learn and XGBoost
2. **Credit Score DL.ipynb** — Deep learning approach using TensorFlow/Keras

---

## 📁 Project Structure

```
.
├── Credit Score ML.ipynb        # ML pipeline: preprocessing, modeling, evaluation
├── Credit Score DL.ipynb        # Deep learning pipeline with Keras
├── data/
│   └── credit_score_dataset.csv # Input dataset (assumed path)
├── README.md                    # Project documentation
```

---

## 📘 Notebook 1: `Credit Score ML.ipynb`

A complete machine learning pipeline including:

### 🔍 Exploratory Data Analysis (EDA)

* Visual analysis of feature distributions, outliers, and correlations
* Understanding of target class imbalance

### ⚙️ Preprocessing & Feature Engineering

* One-hot encoding for categorical variables
* Feature scaling using `StandardScaler`
* Splitting into train/test sets (80/20)

### 🤖 Machine Learning Model

* **Decision Tree**


### 📊 Model Evaluation

* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* ROC-AUC Curve
* Accuracy and overfitting analysis

---

## 📘 Notebook 2: `Credit Score DL.ipynb`

A deep learning approach using a dense feedforward neural network:

### ⚙️ Preprocessing

* Data loading and cleaning (same dataset)
* Label encoding and scaling
* Train/Validation split (with stratification)

### 🧠 Deep Learning Model (Keras)

* 3 Dense hidden layers with ReLU activation
* Dropout layers for regularization
* Binary output with sigmoid activation

### 📉 Training & Validation

* Monitored loss and accuracy curves
* Used `EarlyStopping` to prevent overfitting

### 📊 Evaluation

* Model accuracy on validation set
* Prediction examples
* Confusion matrix for final performance

---

## 📈 Results Summary

| Model               | Accuracy  | 
| ------------------- | --------- | 
| Decision Tree       | \~69%     | 
| Deep Learning       | \~70%     |

---

## 🚀 Getting Started

### 🧩 Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

## 🧠 Key Takeaways

* A clean pipeline for both classical ML and deep learning
* Model performance metrics and visualizations included
* Useful for real-world loan/credit scoring applications


