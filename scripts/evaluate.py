from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)


def get_metrics(model, X, y):
    """Return the six assignment metrics for a fitted model."""
    preds = model.predict(X)
    proba = model.predict_proba(X)

    return {
        "Accuracy": accuracy_score(y, preds),
        # multi-class AUC: one-vs-rest, averaged over the classes
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y, preds, average="macro"),
        "Recall": recall_score(y, preds, average="macro"),
        "F1": f1_score(y, preds, average="macro"),
        "MCC": matthews_corrcoef(y, preds),
    }