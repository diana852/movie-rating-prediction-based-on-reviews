from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from .config import Config

from .sentiment_features import VaderSentimentTransformer  # noqa: F401



LABELS = [1, 2, 3, 4, 5]


def top_confusions(cm: np.ndarray, labels: list[int], top_k: int = 8):
    """
    En çok yapılan hataları (gerçek->tahmin) sayıya göre listeler.
    Diagonal (doğru tahminler) hariç tutulur.
    """
    items = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                items.append((cnt, true_lab, pred_lab))

    items.sort(reverse=True, key=lambda x: x[0])
    return items[:top_k]


def run(model_path: str):
    cfg = Config()

    # aynı split garantisi
    df = pd.read_csv(cfg.data_path)[[cfg.text_col, cfg.label_col]].dropna()
    X = df[cfg.text_col].astype(str).tolist()
    y = df[cfg.label_col].astype(int).tolist()

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    print(f"\n=== Model: {model_path} ===")
    print("Confusion Matrix (rows=true, cols=pred) labels=1..5:\n", cm)

    # sınıf bazlı metrikler (precision/recall/f1)
    report_dict = classification_report(
        y_test, y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    # sadece sınıflar + accuracy/macro avg/weighted avg
    print("\nClassification Report (özet):")
    print(report_df.loc[[str(l) for l in LABELS] + ["accuracy", "macro avg", "weighted avg"],
                        ["precision", "recall", "f1-score", "support"]])

    # en çok karışanlar
    conf = top_confusions(cm, LABELS, top_k=10)
    print("\nEn çok karışan sınıflar (true -> pred):")
    for cnt, t, p in conf:
        print(f"  {t} -> {p}: {cnt}")

    # kaydet
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model_path": model_path,
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "top_confusions": [{"count": c, "true": t, "pred": p} for c, t, p in conf],
    }
    safe_name = model_path.replace("\\", "_").replace("/", "_").replace(":", "")
    out_path = cfg.outputs_dir / f"confusion_{safe_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {out_path}")


def main():
    cfg = Config()
    # E2 ve E3'ü analiz etmek için varsayılan yollar
    e2 = str(cfg.model_dir / "E2_tfidf20000_uni.joblib")
    e3 = str(cfg.model_dir / "E3_tfidf_sentiment.joblib")

    # Hangileri varsa onları çalıştır
    for p in [e2, e3]:
        try:
            run(p)
        except FileNotFoundError:
            print(f"\n[Uyarı] Model bulunamadı, atlandı: {p}")


if __name__ == "__main__":
    main()
