# src/run_all_evaluations.py
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluate_model import evaluate


MODELS = {
    "E0": "models/E0_baseline.joblib",
    "E1": "models/E1_tfidf20000.joblib",
    "E2": "models/E2_tfidf20000_uni.joblib",
    "E3": "models/E3_tfidf_sentiment.joblib",
    "E4": "models/E4_word_char_tfidf.joblib",
    "E5": "models/E5_word_char_tfidf_balanced.joblib",
}


def load_data(test_size: float = 0.30, random_state: int = 42):
    project_root = Path(__file__).resolve().parents[1]  # .../movie_rating
    csv_path = project_root / "data" / "raw" / "Movie Review And Rating.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

    print(f"[INFO] CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Review" not in df.columns or "Rating" not in df.columns:
        raise ValueError(
            f"CSV kolonları beklenenden farklı. Bulunan kolonlar: {list(df.columns)} "
            f"(Beklenen: 'Review', 'Rating')"
        )

    X = df["Review"].astype(str).tolist()
    y = df["Rating"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def pretty_print_cm(cm, labels=(1, 2, 3, 4, 5)):
    # cm: list[list[int]]
    header = "     " + " ".join([f"{l:>4}" for l in labels])
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>3} " + " ".join([f"{v:>4}" for v in row]))


def main():
    _, X_test, _, y_test = load_data()

    results = {}

    for exp_name, rel_path in MODELS.items():
        model_path = Path(rel_path)

        print(f"\n=== {exp_name} ===")
        print(f"Model: {model_path}")

        res = evaluate(model_path, X_test, y_test, labels=(1, 2, 3, 4, 5))
        results[exp_name] = res

        macro = res["classification_report"]["macro avg"]
        weighted = res["classification_report"]["weighted avg"]

        print(f"Accuracy: {res['accuracy']:.4f}")
        print(
            f"Macro   P/R/F1: "
            f"{macro['precision']:.4f} / {macro['recall']:.4f} / {macro['f1-score']:.4f}"
        )
        print(
            f"Weighted P/R/F1: "
            f"{weighted['precision']:.4f} / {weighted['recall']:.4f} / {weighted['f1-score']:.4f}"
        )

        # ✅ Confusion matrix ekrana bas
        print("\nConfusion Matrix (rows=true, cols=pred) labels=1..5:")
        pretty_print_cm(res["confusion_matrix"], labels=(1, 2, 3, 4, 5))

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "all_model_evaluations.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] Kaydedildi: {out_path.resolve()}")


if __name__ == "__main__":
    main()
