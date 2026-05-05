import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc

os.makedirs("images", exist_ok=True)

# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.close()

# =========================
# CLASS DISTRIBUTION
# =========================
def plot_class_distribution(y):
    plt.figure()
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.savefig("images/class_distribution.png")
    plt.close()

# =========================
# PRECISION-RECALL CURVE
# =========================
def plot_pr_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("images/pr_curve.png")
    plt.close()

# =========================
# ROC CURVE
# =========================
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.savefig("images/roc_curve.png")
    plt.close()