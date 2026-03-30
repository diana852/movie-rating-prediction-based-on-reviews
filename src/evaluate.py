from __future__ import annotations

import json
import joblib
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .config import Config


def main() -> None:
    cfg = Config()
    model_path = cfg.model_dir / "tfidf_logreg.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}. Önce train.py çalıştır.")

    df = pd.read_csv(cfg.data_path)[[cfg.text_col, cfg.label_col]].dropna()
    X = df[cfg.text_col].astype(str).tolist()
    y = df[cfg.label_col].astype(int).tolist()

    # Train scriptiyle aynı split (aynı random_state ve stratify)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = joblib.load(model_path)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    report = classification_report(y_test, y_pred, digits=4)

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "accuracy": acc,
        "confusion_matrix_labels": [1, 2, 3, 4, 5],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    out_path = cfg.outputs_dir / "eval_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Değerlendirme tamamlandı.")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (labels 1..5):\n", cm)
    print("\nClassification Report:\n", report)
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    main()
