from __future__ import annotations

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=str)
    args = parser.parse_args()

    cfg = Config()
    model_dir = cfg.model_dir / "distilbert_rating"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()

    pred_label = int(probs.argmax())  # 0..4
    pred_rating = pred_label + 1      # 1..5

    print(f"Predicted rating: {pred_rating}")
    print("Probabilities:")
    for i, p in enumerate(probs, start=1):
        print(f"  {i}: {p:.4f}")


if __name__ == "__main__":
    main()
