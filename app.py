import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Benchmark", layout="wide")
st.title("Machine Learning Model Evaluation Dashboard")
st.caption("M.Tech AI/ML - Assignment 2 Deployment")

st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

models_mapping = {
    "Logistic Regression": ("model/logistic_regression.joblib", True),
    "Decision Tree": ("model/decision_tree.joblib", False),
    "kNN": ("model/knn.joblib", True),
    "Naive Bayes": ("model/naive_bayes.joblib", True),
    "Random Forest (Ensemble)": ("model/random_forest_ensemble.joblib", False),
    "Gradient Boosting (Ensemble)": ("model/gradient_boosting_ensemble.joblib", False)
}

st.sidebar.header("2. Select Model")
selected_model_name = st.sidebar.selectbox("Choose Classifier", list(models_mapping.keys()))

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    if "target" not in df_test.columns:
        st.error("Uploaded CSV missing required 'target' column.")
    else:
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]

        model_path, requires_scaling = models_mapping[selected_model_name]

        if not os.path.exists(model_path) or not os.path.exists("model/scaler.joblib"):
            st.error("Model artifacts not found. Run `train_models.py` to generate `.joblib` files.")
        else:
            model = joblib.load(model_path)
            scaler = joblib.load("model/scaler.joblib")

            X_eval = scaler.transform(X_test) if requires_scaling else X_test.values
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

            st.subheader(f"Performance Metrics: {selected_model_name}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            c2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
            c3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            c4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            c5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            c6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                st.pyplot(fig)

            with right_col:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))
else:
    st.info("Upload `test_data.csv` via the sidebar to run dynamic inference.")