# 💳 Credit Card Fraud Detection System

## 🚀 Overview

The Credit Card Fraud Detection System is an end-to-end Machine Learning project designed to identify fraudulent credit card transactions accurately and efficiently. Fraud detection is one of the most important challenges in banking and fintech industries due to the increasing number of online transactions and cyber fraud cases.

This project uses advanced Machine Learning techniques to analyze transaction patterns and classify whether a transaction is legitimate or fraudulent.

The system includes:

* Data Preprocessing & Feature Scaling
* Imbalanced Data Handling using SMOTE
* Model Training using XGBoost
* Automated Evaluation & Visualization
* HTML Report Generation
* FastAPI-based Prediction API

---

## 🎯 Problem Statement

Credit card fraud detection is a binary classification problem where:

* **0** → Legitimate Transaction
* **1** → Fraudulent Transaction

### Key Challenges:

* Highly imbalanced dataset
* Need high Recall (detect fraud cases)
* Need good Precision (reduce false alerts)
* Real-time fraud detection required

---

## 🧠 Solution Approach

### 🔄 Pipeline

Raw Data → Preprocessing → SMOTE → Model Training → Evaluation → Prediction → Visualization → API

---

## ⚙️ Techniques Used

### Data Processing

* StandardScaler
* Train-Test Split
* Data Cleaning

### Imbalance Handling

* SMOTE (Synthetic Minority Oversampling Technique)

### Machine Learning Model

* XGBoost Classifier

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## 🏗️ Project Structure

```bash id="d7s92a"
Credit-Card-Fraud-Detection/
│
├── data/                 # Dataset (creditcard.csv)
├── src/                  # Core ML pipeline
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── api/                  # FastAPI App
│   └── app.py
│
├── models/               # Saved Model
├── images/               # Generated Graphs
├── outputs/              # Predictions
├── reports/              # HTML Report
│
├── main.py               # Main Pipeline Runner
├── requirements.txt
└── README.md
```

---

## 📊 Results & Outputs

### ✔ Generated Visualizations

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

### ✔ Files Generated

```bash id="u3x0ew"
images/
models/model.pkl
outputs/predictions.csv
reports/report.html
```

---

## ⚡ Installation & Setup

### 1. Clone Repository

```bash id="wxtjkt"
git clone https://github.com/your-username/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system
```

### 2. Create Virtual Environment

```bash id="jlwmj2"
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash id="ng8m4v"
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash id="9v30g9"
python main.py
```

### Output:

* Model Trained Successfully
* Graphs saved in `/images`
* Predictions saved in `/outputs`
* Report generated in `/reports`

---

## 🌐 Run API (Optional)

```bash id="lglyry"
uvicorn api.app:app --reload
```

Open in browser:

```bash id="u4j3e8"
http://127.0.0.1:8000/docs
```

---

## 🧪 Dataset

* Source: Public Credit Card Transactions Dataset
* Total Records: 284,807
* Fraud Cases: 492
* Features: 31 Columns

---

## 💡 Key Highlights

✔ Handles highly imbalanced data
✔ Uses XGBoost Classifier
✔ Generates automated visual reports
✔ Deployable via FastAPI
✔ Clean modular architecture
✔ Real-world fintech use case

---

## 📈 Future Improvements

* SHAP Explainability
* Real-time Streaming with Kafka
* Threshold Optimization
* Dashboard using Next.js
* Cloud Deployment

---

## 🎤 Interview Talking Points

* Handled class imbalance using SMOTE
* Optimized fraud detection using XGBoost
* Focused on Recall to minimize fraud loss
* Built end-to-end ML pipeline
* Created API deployment using FastAPI

---

## 👨‍💻 Author

Deepali Verma
