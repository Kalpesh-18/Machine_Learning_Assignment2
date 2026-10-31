import sys
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# the helper lives in scripts/, so make it importable
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))
from evaluate import get_metrics

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.title("Dry Bean Classifier")

uploaded = st.file_uploader("Upload test CSV", type="csv")
choice = st.selectbox("Choose a model", list(MODELS.keys()))

if uploaded is not None:
    data = pd.read_csv(uploaded)
    if "Class" not in data.columns:
        st.error("The CSV needs a 'Class' column.")
    else:
        X = data.drop(columns=["Class"])
        y = data["Class"]
        model = joblib.load(MODEL_DIR / MODELS[choice])
        scores = get_metrics(model, X, y)
        st.subheader(f"Results for {choice}")
        st.table(pd.DataFrame(scores, index=["score"]).round(4))
        
        preds = model.predict(X)

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y, preds, ax=ax, xticks_rotation=45, colorbar=False
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Classification report")
        report = classification_report(y, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))