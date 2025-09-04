# src/models/evaluate.py
import os, json, joblib, pandas as pd , numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

# Config - edit if needed
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "Model_XGB_RSV_bal.pkl"))
TEST_CSV  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "processed_churn.csv"))
OUT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
FIG_DIR   = os.path.join(OUT_DIR, "figures")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
PRED_CSV = os.path.join(OUT_DIR, "predictions.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# Load
model = joblib.load(MODEL_PATH)
df_test = pd.read_csv(TEST_CSV)
y_true = df_test["Churn"] if "Churn" in df_test.columns else None
X_test = df_test.drop(columns=["Churn"], errors="ignore")

# Predict
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:,1]
else:
    # fallback
    y_score = model.decision_function(X_test)
    y_proba = (y_score - y_score.min())/(y_score.max()-y_score.min()+1e-9)
y_pred = (y_proba >= 0.5).astype(int)

# Save predictions for PowerBI
pred_df = X_test.copy()
pred_df["churn_proba"] = y_proba
pred_df["churn_pred"] = y_pred
if y_true is not None:
    pred_df["Churn_true"] = y_true.values
pred_df.to_csv(PRED_CSV, index=False)
print("Saved predictions ->", PRED_CSV)

# Metrics
results = {}
if y_true is not None:
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["precision"] = float(precision_score(y_true, y_pred))
    results["recall"] = float(recall_score(y_true, y_pred))
    results["f1"] = float(f1_score(y_true, y_pred))
    try:
        results["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        results["roc_auc"] = None
    results["classification_report"] = classification_report(y_true, y_pred, output_dict=True)

with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("Saved metrics ->", RESULTS_JSON)

# Confusion matrix plot
if y_true is not None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0,1], ["No Churn","Churn"])
    plt.yticks([0,1], ["No Churn","Churn"])
    for (i,j),val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center", color="white", fontsize=14)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(FIG_DIR,"confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ROC curve
if y_true is not None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {results.get('roc_auc', 0):.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR,"roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Precision-Recall curve
if y_true is not None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(FIG_DIR,"pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("Saved figures to:", FIG_DIR)