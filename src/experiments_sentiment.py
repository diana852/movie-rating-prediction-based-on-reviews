from __future__ import annotations

import pandas as pd
import numpy as np
import joblib

from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

from .config import Config
from .text_preprocess import clean_text
from .sentiment_features import VaderSentimentTransformer




def main():
    cfg = Config()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    # -------- Data --------
    df = pd.read_csv(cfg.data_path)[[cfg.text_col, cfg.label_col]].dropna()
    X = df[cfg.text_col].astype(str).tolist()
    y = df[cfg.label_col].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # -------- Feature Union --------
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=20000,      # E2'den gelen en iyi ayar
        ngram_range=(1, 1),
        min_df=2,
    )

    sentiment = VaderSentimentTransformer()

    features = FeatureUnion([
        ("tfidf", tfidf),
        ("sentiment", sentiment),
    ])

    pipe = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    # -------- Train & Eval --------
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    # -------- Save --------
    joblib.dump(pipe, cfg.model_dir / "E3_tfidf_sentiment.joblib")

    result = {
        "experiment": "E3_tfidf_unigram_20k_plus_sentiment",
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    out_path = cfg.outputs_dir / "exp_sentiment_result.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)

    print("Sentiment experiment bitti")
    print(result)
    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
