import os
import joblib
import pandas as pd

from src.preprocess import load_data, preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_pr_curve,
    plot_roc_curve
)
from src.utils import generate_html_report

# =========================
# CREATE REQUIRED FOLDERS
# =========================
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("🚀 Starting Fraud Detection Pipeline...")

# =========================
# LOAD DATA
# =========================
df = load_data("data/creditcard.csv")
print("✅ Data Loaded:", df.shape)

# =========================
# CLASS DISTRIBUTION IMAGE
# =========================
plot_class_distribution(df["Class"])

# =========================
# PREPROCESS DATA
# =========================
X, y, scaler = preprocess_data(df)

# =========================
# TRAIN MODEL
# =========================
model, X_test, y_test = train_model(X, y)
print("✅ Model Trained")

# =========================
# EVALUATE MODEL
# =========================
y_pred, cm = evaluate_model(model, X_test, y_test)

# =========================
# PROBABILITIES FOR CURVES
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# VISUALIZATIONS
# =========================
plot_confusion_matrix(cm)
plot_pr_curve(y_test, y_prob)
plot_roc_curve(y_test, y_prob)

print("📊 All graphs generated")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/model.pkl")
print("💾 Model saved")

# =========================
# SAVE SAMPLE PREDICTIONS
# =========================
output_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

output_df.to_csv("outputs/predictions.csv", index=False)
print("📁 Predictions saved")

# =========================
# GENERATE HTML REPORT
# =========================
generate_html_report()
print("📄 Report generated")

print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("👉 Check /images, /models, /reports, /outputs")