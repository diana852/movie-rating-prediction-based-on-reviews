from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "data" / "raw" / "Movie Review And Rating.csv"
    model_dir: Path = project_root / "models"
    outputs_dir: Path = project_root / "outputs"

    # Data columns
    text_col: str = "Review"
    label_col: str = "Rating"

    # Split
    test_size: float = 0.30
    random_state: int = 42

    # TF-IDF (temel model)
    max_features: int | None = 5000
    ngram_range: tuple[int, int] = (1, 2)   # unigram + bigram (temel için güvenli)
    min_df: int = 2

    # Logistic Regression
    max_iter: int = 2000
