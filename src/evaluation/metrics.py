import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        # Probabilidades
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob)
        })
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    return df_results


def plot_roc_models(models, X_test, y_test):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.predict(X_test).flatten()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.savefig(f"../imagenes/roc_auc_modelos.png", bbox_inches="tight")
    plt.show()


def confusion_matrices(models, X_test, y_test):
    matrices = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        matrices[name] = confusion_matrix(y_test, y_pred)
    return matrices