import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer
)
from datasets import Dataset
import pandas as pd


def load_test_data():
    """
    CSV: data/raw/Movie Review And Rating.csv
    """
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "Movie Review And Rating.csv"

    df = pd.read_csv(csv_path)

    X = df["Review"].astype(str).tolist()
    y = (df["Rating"].astype(int) - 1).tolist()  # 1–5 → 0–4

    return X, y


def main():
    print("[INFO] DistilBERT evaluation başlıyor...")

    model_dir = Path("models/distilbert_rating")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_dir}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    X, y = load_test_data()

    ds = Dataset.from_dict({"text": X, "label": y})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    trainer = Trainer(model=model)

    preds = trainer.predict(ds)

    y_true = np.array(y)
    y_pred = np.argmax(preds.predictions, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=["1", "2", "3", "4", "5"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    results = {
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "distilbert_evaluation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- EKRANA BAS ----
    print("\n=== DistilBERT Evaluation ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(
        f"Macro   P/R/F1: "
        f"{report['macro avg']['precision']:.4f} / "
        f"{report['macro avg']['recall']:.4f} / "
        f"{report['macro avg']['f1-score']:.4f}"
    )
    print(
        f"Weighted P/R/F1: "
        f"{report['weighted avg']['precision']:.4f} / "
        f"{report['weighted avg']['recall']:.4f} / "
        f"{report['weighted avg']['f1-score']:.4f}"
    )

    print("\nConfusion Matrix (rows=true, cols=pred) labels=1..5:")
    labels = [1, 2, 3, 4, 5]
    print("     " + " ".join([f"{l:>4}" for l in labels]))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>3} " + " ".join([f"{v:>4}" for v in row]))

    print(f"\n[OK] Kaydedildi: {out_path.resolve()}")


if __name__ == "__main__":
    main()
