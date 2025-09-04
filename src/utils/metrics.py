# src/utils/metrics.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, classification_report
import json

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pr_curve(y_true, y_probs, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, color="purple", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_classification_report(y_true, y_pred, save_path=None):
    report = classification_report(y_true, y_pred, output_dict=True)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)
    return report
