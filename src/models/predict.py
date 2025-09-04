import os, joblib, pandas as pd
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(_file_), "..", "..", "models", "final_pipeline.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)

pipe = joblib.load(MODEL_PATH)
print("Loaded pipeline:", MODEL_PATH)

# read input CSV (processed features expected)
df = pd.read_csv(os.path.join("data","processed","telco_processed_test.csv"))

X = df.drop("Churn", axis=1, errors="ignore")
y = df["Churn"] if "Churn" in df.columns else None

pred = pipe.predict(X)
proba = pipe.predict_proba(X)[:,1]

out = X.copy()
out["predicted_churn"] = pred
out["churn_proba"] = proba
out.to_csv("reports/predictions.csv", index=False)
print("Saved predictions -> reports/predictions.csv")

if y is not None:
    print("Accuracy:", accuracy_score(y, pred))
    print(classification_report(y, pred))