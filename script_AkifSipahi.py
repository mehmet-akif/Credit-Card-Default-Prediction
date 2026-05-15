import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


def load_data(excel_path: str):
    df = pd.read_excel(excel_path, sheet_name="Data", header=1)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    target_col = "default payment next month"
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: '{target_col}'")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def safe_proba(model, X):
    """Return positive-class probabilities if available, else None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "default of credit card clients.xls")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Couldn't find the dataset file here:\n{data_path}\n\n"
            "Fix: Put 'default of credit card clients.xls' in the SAME folder as this script."
        )

    X, y = load_data(data_path)

    print(f"Shape: {(X.shape[0], X.shape[1] + 1)}")  
    print("Target distribution:")
    print(y.value_counts(normalize=True).rename("proportion"))



    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )


    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25))
        ]),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
    }

 
    best_name, best_f1 = None, -1.0

    print("\n==== VALIDATION RESULTS (used to select best model) ====")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        print(f"{name:18s}  F1={val_f1:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name

    print(f"\nBest model (by Validation F1): {best_name}")

 
    rows = []
    print("\n==== TEST RESULTS (comparison across models) ====")

    for name, model in models.items():
        # Train on ALL non-test data
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)

        proba = safe_proba(model, X_test)
        auc = roc_auc_score(y_test, proba) if proba is not None else np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc
        })

    results = pd.DataFrame(rows).sort_values(by="F1", ascending=False)
    print(results.to_string(index=False))

    best_model = models[best_name]
    best_model.fit(X_train_full, y_train_full)
    y_best = best_model.predict(X_test)

    print("\nConfusion Matrix (best model selected via validation):")
    print(confusion_matrix(y_test, y_best))


    print("\n==== FEATURE IMPORTANCE ====")

    rf = models["RandomForest"]
    rf.fit(X_train_full, y_train_full)
    rf_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 15 (RandomForest):")
    print(rf_imp.head(15).to_string(index=False))


    lr = models["LogisticRegression"]
    lr.fit(X_train_full, y_train_full)
    lr_coef = lr.named_steps["clf"].coef_.ravel()
    lr_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(lr_coef)
    }).sort_values(by="importance", ascending=False)

    print("\nTop 15 (Logistic Regression |coef|):")
    print(lr_imp.head(15).to_string(index=False))

    top15 = rf_imp.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top15["feature"], top15["importance"])
    plt.title("Top 15 Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    results.to_csv("model_results_test.csv", index=False)
    rf_imp.head(50).to_csv("feature_importance_rf_top50.csv", index=False)
    lr_imp.head(50).to_csv("feature_importance_lr_top50.csv", index=False)
    print("\nSaved: model_results_test.csv, feature_importance_rf_top50.csv, feature_importance_lr_top50.csv")
    print("DONE ✅")


if __name__ == "__main__":
    main()