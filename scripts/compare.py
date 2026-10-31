import pandas as pd
import joblib

from config import TEST_CSV, MODEL_DIR
from evaluate import get_metrics

# display name
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

test = pd.read_csv(TEST_CSV)
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

rows = []
for name, filename in MODELS.items():
    model = joblib.load(MODEL_DIR / filename)
    scores = get_metrics(model, X_test, y_test)
    rows.append({"ML Model Name": name, **scores})

results = pd.DataFrame(rows).round(4)
results.to_csv(MODEL_DIR / "results.csv", index=False)

# markdown version
cols = list(results.columns)
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for _, r in results.iterrows():
    print("| " + " | ".join(str(r[c]) for c in cols) + " |")