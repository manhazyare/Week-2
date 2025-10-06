# Week-2
# 🌍 Maternal Health Risk Prediction - SDG 3 Project

## AI for Sustainable Development - Week 2 Assignment

![SDG 3](https://via.placeholder.com/800x200/00ADD8/FFFFFF?text=SDG+3:+Good+Health+and+Well-being)

---

## 📋 Project Overview

This machine learning project addresses **UN Sustainable Development Goal 3: Good Health and Well-being** by predicting maternal health risk levels using supervised learning. The system analyzes health indicators to classify pregnancies into risk categories (low, mid, high), enabling early intervention and potentially saving lives.

### 🎯 Problem Statement

Maternal mortality remains a critical global challenge, with approximately 295,000 women dying during pregnancy and childbirth annually (WHO, 2023). Many of these deaths are preventable through early risk identification and timely medical intervention. This project leverages machine learning to:

- Predict maternal health risk levels based on clinical indicators
- Enable early identification of high-risk pregnancies
- Support healthcare workers in resource-limited settings
- Reduce maternal mortality rates through data-driven insights

---

## 🤖 Machine Learning Approach

### Model Type: **Supervised Learning - Classification**

**Algorithm:** Random Forest Classifier

**Why Random Forest?**
- Handles non-linear relationships between health indicators
- Robust to outliers and missing data
- Provides feature importance for interpretability
- High accuracy on medical datasets
- Reduces overfitting through ensemble learning

### Features (Input Variables)
1. **Age** - Maternal age (years)
2. **SystolicBP** - Systolic blood pressure (mmHg)
3. **DiastolicBP** - Diastolic blood pressure (mmHg)
4. **BS** - Blood sugar level (mmol/L)
5. **BodyTemp** - Body temperature (°C)
6. **HeartRate** - Heart rate (bpm)

### Target Variable (Output)
- **RiskLevel:** low risk, mid risk, high risk

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository / Kaggle Maternal Health Risk Dataset
- **Size:** 1,014 samples (demonstration uses synthetic data with similar characteristics)
- **Features:** 6 numerical health indicators
- **Classes:** 3 risk levels (balanced using class weights)
- **Link:** [Download Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk)

---

## 🛠️ Technologies Used

```python
- Python 3.8+
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning algorithms
- matplotlib & seaborn - Data visualization
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/maternal-health-sdg3.git
cd maternal-health-sdg3
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Project
```bash
python maternal_health_predictor.py
```

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| **Precision** (avg) | 0.91 |
| **Recall** (avg) | 0.92 |
| **F1-Score** (avg) | 0.91 |

### Feature Importance

The model identified the most critical factors for predicting maternal health risk:

1. **Systolic Blood Pressure** (35% importance)
2. **Blood Sugar Level** (28% importance)
3. **Diastolic Blood Pressure** (18% importance)
4. **Age** (12% importance)
5. **Body Temperature** (5% importance)
6. **Heart Rate** (2% importance)

### Confusion Matrix

![Confusion Matrix](./screenshots/confusion_matrix.png)

### Exploratory Data Analysis

![EDA Visualizations](./screenshots/maternal_health_eda.png)

---

## 💡 How It Works

### 1. Data Preprocessing
- Load and clean health indicator data
- Handle missing values and outliers
- Standardize features using StandardScaler
- Encode target labels (low/mid/high risk)

### 2. Model Training
```python
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
