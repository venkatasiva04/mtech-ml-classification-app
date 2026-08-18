import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

def run_pipeline():
    # 1. Create model output folder
    os.makedirs("model", exist_ok=True)

    # 2. Load UCI Breast Cancer dataset (30 features, 569 instances)
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # 3. Stratified 80-20 train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # Save test partition as required by assignment specs
    test_df.to_csv("test_data.csv", index=False)
    print("✓ Saved test_data.csv to root directory.")

    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_test, y_test = test_df.drop('target', axis=1), test_df['target']

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "model/scaler.joblib")

    # 5. Model Definitions
    models = {
        "logistic_regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "decision_tree": (DecisionTreeClassifier(random_state=42), False),
        "knn": (KNeighborsClassifier(n_neighbors=5), True),
        "naive_bayes": (GaussianNB(), True),
        "random_forest_ensemble": (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "gradient_boosting_ensemble": (GradientBoostingClassifier(random_state=42), False)
    }

    metrics = []

    # 6. Fit, Export, and Evaluate
    for name, (model, scale_required) in models.items():
        X_tr = X_train_scaled if scale_required else X_train.values
        X_te = X_test_scaled if scale_required else X_test.values

        model.fit(X_tr, y_train)
        joblib.dump(model, f"model/{name}.joblib")

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics.append({
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
        })

    print("✓ Trained and exported all 6 classification models into model/")
    print("\nModel Results:")
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__ == "__main__":
    run_pipeline()