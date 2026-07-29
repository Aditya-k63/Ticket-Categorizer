import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, log_loss,
)


def evaluate_model(model, X_test_vec, y_test, classes):
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")
    per_class_f1 = f1_score(y_test, pred, average=None)
    per_class_precision = precision_score(y_test, pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    try:
        proba = model.predict_proba(X_test_vec)
        ll = log_loss(y_test, proba)
    except Exception:
        ll = None

    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")
    if ll is not None:
        print(f"  Log Loss      : {ll:.4f}")

    print(f"\n  {'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7}  {'Support':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7}  {'-'*8}")
    for i, cls in enumerate(classes):
        support = (y_test == cls).sum()
        print(f"  {cls:<12} {per_class_precision[i]:>7.3f} {per_class_recall[i]:>7.3f} "
              f"{per_class_f1[i]:>7.3f}  {support:>8}")

    print(f"\n  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                         columns=[f"Pred {c}" for c in classes])
    for line in cm_df.to_string().split("\n"):
        print(f"    {line}")
