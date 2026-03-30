from __future__ import annotations

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .config import Config
from .text_preprocess import clean_text


def load_split(cfg: Config):
    df = pd.read_csv(cfg.data_path)[[cfg.text_col, cfg.label_col]].dropna()
    X = df[cfg.text_col].astype(str).tolist()
    y = df[cfg.label_col].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def make_tfidf(cfg: Config, max_features=None, ngram_range=(1, 1), min_df=2):
    return TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def make_model(cfg: Config):
    return LogisticRegression(max_iter=cfg.max_iter)


def run_one(name: str, pipe: Pipeline, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return {
        "experiment": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }, pipe


def main():
    cfg = Config()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(cfg)

    experiments = []

    # E0: Baseline (senin şu anki)
    pipe0 = Pipeline([("tfidf", make_tfidf(cfg, max_features=5000, ngram_range=(1, 2), min_df=2)),
                      ("clf", make_model(cfg))])
    res0, model0 = run_one("E0_baseline_tfidf5000_uni_bi", pipe0, X_train, y_train, X_test, y_test)
    experiments.append(res0)
    joblib.dump(model0, cfg.model_dir / "E0_baseline.joblib")

    # E1: TF-IDF max_features arttır (örn 20000)
    pipe1 = Pipeline([("tfidf", make_tfidf(cfg, max_features=20000, ngram_range=(1, 2), min_df=2)),
                      ("clf", make_model(cfg))])
    res1, model1 = run_one("E1_tfidf20000_uni_bi", pipe1, X_train, y_train, X_test, y_test)
    experiments.append(res1)
    joblib.dump(model1, cfg.model_dir / "E1_tfidf20000.joblib")

    # E2: Sadece unigram (kontrol deneyi)
    pipe2 = Pipeline([("tfidf", make_tfidf(cfg, max_features=20000, ngram_range=(1, 1), min_df=2)),
                      ("clf", make_model(cfg))])
    res2, model2 = run_one("E2_tfidf20000_unigram", pipe2, X_train, y_train, X_test, y_test)
    experiments.append(res2)
    joblib.dump(model2, cfg.model_dir / "E2_tfidf20000_uni.joblib")

    out_df = pd.DataFrame(experiments).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    out_path = cfg.outputs_dir / "exp_results.csv"
    out_df.to_csv(out_path, index=False)

    print("Experiments bitti. Sonuçlar:")
    print(out_df)
    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
