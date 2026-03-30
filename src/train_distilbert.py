from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from .config import Config

MODEL_NAME = "distilbert-base-uncased"  # İngilizce yorumlar için


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    cfg = Config()
    set_seed(cfg.random_state)

    df = pd.read_csv(cfg.data_path)[[cfg.text_col, cfg.label_col]].dropna()
    texts = df[cfg.text_col].astype(str).tolist()

    # Rating 1..5 -> label 0..4
    labels = (df[cfg.label_col].astype(int) - 1).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
    )

    out_dir = cfg.model_dir / "distilbert_rating"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,

        logging_steps=50,
        report_to="none",

        fp16=False,  # GPU varsa True yapabilirsin
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Best model + tokenizer kaydet
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = trainer.evaluate()
    print("\nDistilBERT Evaluation:", metrics)
    print(f"Model kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
