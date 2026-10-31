import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "scripts"))
from evaluate import get_metrics

MODEL_DIR = BASE_DIR / "model"

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")


@st.cache_resource
def load_model(filename):
    # cached so switching models doesn't reload the file every time
    return joblib.load(MODEL_DIR / filename)


# ---------- intro (rewrite this in your own words) ----------
st.title("Dry Bean Classifier")
st.write(
    "This app scores five classification models on the UCI Dry Bean dataset. "
    "Each bean is described by 16 shape and size measurements taken from images, "
    "and the task is to name its type: Barbunya, Bombay, Cali, Dermason, "
    "Horoz, Seker or Sira."
)
st.write(
    "Upload a test CSV in the sidebar, pick a model, and the saved model is "
    "scored on your file. Nothing is retrained."
)

# ---------- sidebar ----------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
choice = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))

results_path = MODEL_DIR / "results.csv"
if results_path.exists():
    st.sidebar.download_button(
        label="Download results.csv",
        data=results_path.read_bytes(),
        file_name="results.csv",
        mime="text/csv",
    )
    st.sidebar.caption("Metrics of all five models on the original test split.")

# ---------- wait for a file ----------
if uploaded is None:
    st.info("Upload a CSV in the sidebar to get started. The file needs all 16 feature columns plus a 'Class' column.")
    st.stop()

data = pd.read_csv(uploaded)

if "Class" not in data.columns:
    st.error("The CSV needs a 'Class' column.")
    st.stop()

X = data.drop(columns=["Class"])
y = data["Class"]

# check the feature columns against what the models were trained on
expected = list(load_model(MODELS[choice]).feature_names_in_)
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error(f"Missing feature columns: {', '.join(missing)}")
    st.stop()
X = X[expected]

st.write(f"Loaded **{len(data)}** rows.")

tab_one, tab_all = st.tabs(["Selected model", "All models"])

# ---------- tab 1: one model in detail ----------
with tab_one:
    model = load_model(MODELS[choice])
    try:
        scores = get_metrics(model, X, y)
    except ValueError as err:
        st.error(f"Could not compute the metrics: {err}")
        st.stop()

    st.subheader(f"Results for {choice}")
    st.table(pd.DataFrame(scores, index=["score"]).round(4))

    preds = model.predict(X)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y, preds, ax=ax, xticks_rotation=45, colorbar=False
        )
        plt.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

# ---------- tab 2: every model on the same file ----------
with tab_all:
    st.subheader("All five models on your uploaded file")
    rows = []
    for name, filename in MODELS.items():
        try:
            row = get_metrics(load_model(filename), X, y)
        except ValueError as err:
            st.error(f"{name}: {err}")
            st.stop()
        rows.append({"ML Model Name": name, **row})

    comparison = pd.DataFrame(rows).round(4)
    st.dataframe(comparison, hide_index=True)

    best = comparison.loc[comparison["Accuracy"].idxmax(), "ML Model Name"]
    st.caption(f"Highest accuracy on this file: {best}")