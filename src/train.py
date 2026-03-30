from __future__ import annotations

import json
from dataclasses import asdict
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import Config
from .text_preprocess import clean_text


def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    # zorunlu kolon kontrolü
    for col in (cfg.text_col, cfg.label_col):
        if col not in df.columns:
            raise ValueError(f"CSV içinde beklenen kolon yok: {col}. Mevcut kolonlar: {list(df.columns)}")
    df = df[[cfg.text_col, cfg.label_col]].dropna()
    return df


def build_pipeline(cfg: Config) -> Pipeline:
    # Logistic Regression multiclass: auto (ovr/multinomial solver'a göre)
    clf = LogisticRegression(max_iter=cfg.max_iter, n_jobs=None)

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text,
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                min_df=cfg.min_df,
            )),
            ("clf", clf),
        ]
    )
    return pipe


def main() -> None:
    cfg = Config()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(cfg)

    X = df[cfg.text_col].astype(str).tolist()
    y = df[cfg.label_col].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)

    # Kaydet
    model_path = cfg.model_dir / "tfidf_logreg.joblib"
    joblib.dump(pipe, model_path)

    cfg_dict = asdict(cfg)

    # Path objelerini string'e çevir (WindowsPath JSON'a gitmez)
    for k, v in list(cfg_dict.items()):
        if hasattr(v, "__fspath__"):  # Path-like
            cfg_dict[k] = str(v)

    # split bilgisi + config kaydı (tekrar üretilebilirlik)
    meta = {
        "config": cfg_dict,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_counts": {int(k): int(v) for k, v in df[cfg.label_col].value_counts().sort_index().to_dict().items()},
        "model_path": str(model_path),
    }

    (cfg.outputs_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Eğitim tamamlandı.")
    print(f"Model kaydedildi: {model_path}")
    print(f"Meta: {cfg.outputs_dir / 'train_meta.json'}")


if __name__ == "__main__":
    main()

