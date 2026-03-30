# src/evaluate_model.py
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model_path: Path, X_test, y_test, labels=(1, 2, 3, 4, 5)):
    """
    Klasik (joblib) modeller için:
    - accuracy
    - classification_report (precision/recall/f1 + macro/weighted)
    - confusion_matrix
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path.resolve()}")

    pipe = joblib.load(model_path)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(labels))

    return {
        "model_path": str(model_path.as_posix()),
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
