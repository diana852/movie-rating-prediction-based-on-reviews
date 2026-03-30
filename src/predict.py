from __future__ import annotations

import argparse
import joblib
from .config import Config

from .sentiment_features import VaderSentimentTransformer  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Tahmin edilecek yorum metni")

    # İstersen model seç
    parser.add_argument(
        "--model",
        type=str,
        default="E3",  # default E3: sentiment'li
        choices=["E0", "E1", "E2", "E3", "E4", "E5"],
        help="Kullanılacak model: E0/E1/E2/E3"
    )
    args = parser.parse_args()

    cfg = Config()

    model_map = {
        "E0": cfg.model_dir / "E0_baseline.joblib",
        "E1": cfg.model_dir / "E1_tfidf20000.joblib",
        "E2": cfg.model_dir / "E2_tfidf20000_uni.joblib",
        "E3": cfg.model_dir / "E3_tfidf_sentiment.joblib",
        "E4": cfg.model_dir / "E4_word_char_tfidf.joblib",
        "E5": cfg.model_dir / "E5_word_char_tfidf_balanced.joblib",

    }
    model_path = model_map[args.model]

    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}. Önce ilgili experiment/train çalıştır.")

    pipe = joblib.load(model_path)

    pred = pipe.predict([args.text])[0]

    print(f"Model: {args.model} ({model_path.name})")
    print(f"Tahmin edilen rating: {pred}")

    # predict_proba her zaman olmayabilir ama LogisticRegression'da var
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([args.text])[0]
        labels = pipe.classes_.tolist()
        print("Sınıf olasılıkları:")
        for lab, p in zip(labels, probs):
            print(f"  {lab}: {p:.4f}")


if __name__ == "__main__":
    main()
