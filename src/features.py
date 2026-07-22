import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(train_texts: list[str], max_features: int = 5000) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer on the training texts ONLY.

    ngram_range=(1,2): unigrams + bigrams — captures phrases like
    "breaking news" as a single feature, not just "breaking" and "news"
    separately.

    max_features caps the vocabulary size to the top N terms by
    frequency, to avoid overfitting on a beginner-scale dataset.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(train_texts)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: list[str]):
    """
    Transforms new texts into TF-IDF vectors using an ALREADY-FITTED
    vectorizer. Never call .fit() again here — that would learn a new
    vocabulary and break consistency with the trained model.
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Saves a fitted vectorizer to disk with joblib."""
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Loads a previously fitted vectorizer from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    # Quick manual smoke test
    sample_texts = [
        "breaking news government announcement today",
        "shocking scandal exposed by insider sources",
        "government announcement today confirmed by officials",
    ]

    vec = fit_vectorizer(sample_texts, max_features=50)
    matrix = transform_texts(vec, sample_texts)

    print("Vocabulary size:", len(vec.vocabulary_))
    print("Matrix shape:", matrix.shape)

    save_vectorizer(vec, "models/vectorizer_test.pkl")
    loaded_vec = load_vectorizer("models/vectorizer_test.pkl")
    print("Reloaded vocab size matches:", len(loaded_vec.vocabulary_) == len(vec.vocabulary_))