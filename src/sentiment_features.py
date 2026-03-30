from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment import SentimentIntensityAnalyzer


class VaderSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scores = [self.analyzer.polarity_scores(text)["compound"] for text in X]
        return np.array(scores).reshape(-1, 1)
