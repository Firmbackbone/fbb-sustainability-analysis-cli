# keyword_extractor.py
"""
Keyword Extractor – works for regression *and* classification, anonymises text,
language‑aware stop‑words, and caps keyword count.

Example:
    python keyword_extractor.py \
        --text-data data/pages.ndjson \
        --metrics   data/metrics.csv \
        --output    data/keyword_imp.csv \
        --n-keywords 25 --model-type auto
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
from pathlib import Path
from typing import Final

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils_io import load_dataframe, save_dataframe
from utils_nlp import (
    detect_lang,
    keyword_is_entity,
    remove_person_org,
    stopwords_for,
)

MAX_KEYWORDS: Final = 50

# Additional stopwords for sustainability domain - words that appear commonly but don't add meaning
DOMAIN_STOPWORDS = {
    "button",
    "click",
    "search",
    "menu",
    "page",
    "contact",
    "website",
    "cookie",
    "navigate",
    "navigation",
    "footer",
    "header",
    "javascript",
    "login",
    "register",
    "email",
    "phone",
    "address",
    "location",
    "sitemap",
    "toggle",
    "submit",
    "send",
}


###############################################################################
# GPU Detection and Setup
###############################################################################
def setup_gpu():
    """Detect and setup GPU if available."""
    try:
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            import torch

            if torch.backends.mps.is_available():
                torch.device("mps")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                logging.info("Using Apple Silicon GPU (MPS)")
                return True

        # Check for NVIDIA GPU
        try:
            import torch

            if torch.cuda.is_available():
                torch.device("cuda")
                logging.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                return True
        except (ImportError, AttributeError):
            pass

        # Check for TensorFlow compatible GPU
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Using TensorFlow GPU: {len(gpus)} device(s) available")
                return True
        except (ImportError, AttributeError):
            pass

        # Check for CuML (Rapids) for GPU-accelerated ML
        try:
            import cuml

            logging.info("Using RAPIDS cuML for GPU-accelerated machine learning")
            return True
        except ImportError:
            pass

        logging.info("No GPU detected, running on CPU")
        return False
    except Exception as e:
        logging.warning(f"Error while setting up GPU: {e}")
        logging.info("Falling back to CPU")
        return False


###############################################################################
# Ensure NLTK tokeniser
###############################################################################
for pkg in ("punkt", "wordnet", "omw-1.4", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

TOKENIZER: Final = nltk.tokenize.RegexpTokenizer(r"[A-Za-z]+")
LEMMATIZER: Final = WordNetLemmatizer()


def is_meaningful_keyword(keyword: str, lang: str, min_length: int = 3) -> bool:
    """Check if a keyword is meaningful and not a stopword."""
    # Check for common URL patterns and HTML artifacts
    if keyword in DOMAIN_STOPWORDS:
        return False

    # Check for length (filter out very short words)
    if len(keyword) < min_length:
        return False

    # Check for purely numeric content
    if keyword.isdigit():
        return False

    # Check for code-like tokens
    if any(c in keyword for c in "{}[]()<>/\\"):
        return False

    # Get additional stopwords from NLTK for the given language
    try:
        if lang == "nl":
            nltk_stops = set(nltk_stopwords.words("dutch"))
        else:
            nltk_stops = set(nltk_stopwords.words("english"))
    except:
        nltk_stops = set()  # Fallback if language not available

    # Combined stopwords
    stops = stopwords_for(lang).union(nltk_stops)

    return keyword.lower() not in stops


def preprocess(text: str) -> str:
    """Use the same tokenization as index_generator."""
    lang = detect_lang(text)
    stops = stopwords_for(lang)
    text = remove_person_org(text, lang)

    # Language-specific preprocessing
    if lang == "nl":
        # For Dutch text, handle common compound words and special characters
        # Dutch compound words are often not recognized correctly by lemmatizers
        text = text.replace("-", " ").replace("_", " ")

        # Convert some common Dutch prefixes to improve lemmatization
        for prefix in ["ge", "be", "ver", "ont", "her"]:
            text = text.replace(f" {prefix}", f" {prefix}_")

    # Filter for meaningful tokens
    tokens = [
        LEMMATIZER.lemmatize(t.lower())
        for t in TOKENIZER.tokenize(text)
        if t.lower() not in stops
        and len(t) > 2
        and is_meaningful_keyword(t.lower(), lang)
    ]

    # For Dutch, add compound words consisting of consecutive tokens
    # since Dutch uses many compound words
    if lang == "nl" and len(tokens) > 1:
        compound_tokens = []
        for i in range(len(tokens) - 1):
            if (
                len(tokens[i]) > 3 and len(tokens[i + 1]) > 3
            ):  # Only meaningful compounds
                compound = tokens[i] + tokens[i + 1]
                if len(compound) < 25:  # Avoid excessively long compounds
                    compound_tokens.append(compound)
        tokens.extend(compound_tokens)

    return " ".join(tokens)


###############################################################################
# Extractor class
###############################################################################
class KeywordExtractor:
    def __init__(
        self,
        n_keywords: int,
        test_size: float | None = 0.2,
        random_state: int = 42,
        min_importance: float = 0.01,  # Minimum importance threshold
        ngram_range: tuple = (1, 2),  # Include bigrams for multi-word keywords
    ) -> None:
        self.n_keywords = min(n_keywords, MAX_KEYWORDS)
        self.test_size = test_size
        self.random_state = random_state
        self.min_importance = min_importance
        self.ngram_range = ngram_range

        # Use TF-IDF vectorizer with n-grams to capture multi-word keywords
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 6000 to catch more potential keywords
            min_df=2,  # A keyword must appear in at least 2 documents
            max_df=0.9,  # Exclude terms that appear in >90% of documents
            ngram_range=ngram_range,
            use_idf=True,
            sublinear_tf=True,  # Apply sublinear TF scaling (1 + log(TF))
            smooth_idf=True,  # Add 1 to document frequencies to avoid division by zero
        )

        # Check if GPU is available
        self.use_gpu = setup_gpu()
        self.device = None

        if self.use_gpu:
            try:
                import torch

                # Try CUDA first
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logging.info("KeywordExtractor using CUDA device")
                # Then try Apple Silicon
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = torch.device("mps")
                    logging.info("KeywordExtractor using MPS device")
            except ImportError:
                logging.info("PyTorch not available for KeywordExtractor")

        # Count vectorizer for additional term frequency information
        self.count_vectorizer = CountVectorizer(
            max_features=10000, min_df=2, max_df=0.9, ngram_range=ngram_range
        )

    # ------------------------------------------------------------------ #
    def _detect_problem_type(self, y: pd.Series) -> str:
        # If the target variable has continuous values, treat it as regression
        if y.dtype.kind in {"i", "f"}:
            return "regression"
        return "classification"

    def _build_model(self, problem: str):
        # If GPU is available and cuML is installed, use GPU-accelerated models
        if self.use_gpu:
            try:
                if problem == "regression":
                    from cuml.ensemble import RandomForestRegressor as CumlRFRegressor

                    logging.info("Using GPU-accelerated RandomForestRegressor")
                    return CumlRFRegressor(
                        n_estimators=400, random_state=self.random_state
                    )
                else:
                    from cuml.ensemble import RandomForestClassifier as CumlRFClassifier

                    logging.info("Using GPU-accelerated RandomForestClassifier")
                    return CumlRFClassifier(
                        n_estimators=400, random_state=self.random_state
                    )
            except ImportError:
                logging.info("cuML not available, using scikit-learn models")

        # Fallback to CPU models
        if problem == "regression":
            return RandomForestRegressor(
                n_estimators=400, n_jobs=-1, random_state=self.random_state
            )
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=self.random_state
        )

    def _is_relevant_keyword(self, keyword: str, importance: float, lang: str) -> bool:
        """Determine if a keyword is relevant based on additional criteria."""
        # Skip if importance below threshold
        if importance < self.min_importance:
            return False

        # Skip very short keywords (unless they're significant acronyms)
        if len(keyword) <= 2 and not keyword.isupper():
            return False

        # Skip HTML/CSS/JS artifacts and common website elements
        if keyword in DOMAIN_STOPWORDS:
            return False

        # Skip if it's a person or organization entity
        if keyword_is_entity(keyword, lang):
            return False

        # Additional filtering for language-specific stopwords
        if not is_meaningful_keyword(keyword, lang):
            return False

        return True

    # ------------------------------------------------------------------ #
    def run(
        self,
        df_text: pd.DataFrame,
        df_metrics: pd.DataFrame,
        metric_col: str | None = None,
    ) -> pd.DataFrame:
        # Extract base domain from URLs
        df_text["base_domain"] = df_text["domain"].apply(
            lambda x: x.replace("https://", "").replace("http://", "").split("/")[0]
        )
        df_metrics["base_domain"] = df_metrics["domain"].apply(
            lambda x: x.replace("https://", "").replace("http://", "").split("/")[0]
        )

        # merge
        df = pd.merge(df_text, df_metrics, on="base_domain")
        if metric_col is None:
            metric_col = df_metrics.select_dtypes(include=[np.number]).columns.tolist()[
                0
            ]

        tqdm.pandas(desc="Cleaning text")
        df["clean"] = df["text"].progress_apply(preprocess)

        # Detect majority language for filtering
        lang_majority = df["clean"].str[:500].map(detect_lang).value_counts().idxmax()
        logging.info(f"Majority language detected: {lang_majority}")

        # Vectorise with TF-IDF
        X = self.vectorizer.fit_transform(df["clean"])

        # Also get raw counts
        X_counts = self.count_vectorizer.fit_transform(df["clean"])

        y = df[metric_col]

        problem = self._detect_problem_type(y)
        model = self._build_model(problem)

        if self.test_size:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            if problem == "regression":
                logging.info("R² = %.3f", r2_score(y_te, y_pred))
            else:
                logging.info("\n%s", classification_report(y_te, y_pred))
        else:
            model.fit(X, y)

        # Feature importance and raw counts combined for more robust ranking
        imp = model.feature_importances_
        feats = self.vectorizer.get_feature_names_out()
        count_feats = self.count_vectorizer.get_feature_names_out()

        # Get document frequency (in how many documents a term appears)
        doc_freq = np.asarray(X_counts.sum(axis=0)).flatten()

        # Create term frequency dict for count features
        term_freq = {}
        for i, term in enumerate(count_feats):
            term_freq[term] = doc_freq[i]

        # Combine all data
        keyword_data = []
        for i, (feat, importance) in enumerate(zip(feats, imp)):
            # Get term frequency if available
            freq = term_freq.get(feat, 0)

            # Add additional relevance check
            if self._is_relevant_keyword(feat, importance, lang_majority):
                # Calculate combined score (importance weighted by frequency)
                combined_score = importance * (1 + np.log1p(freq))
                keyword_data.append(
                    {
                        "keyword": feat,
                        "importance": importance,
                        "frequency": freq,
                        "combined_score": combined_score,
                    }
                )

        # Convert to dataframe and sort by combined score
        df_imp = pd.DataFrame(keyword_data).sort_values(
            "combined_score", ascending=False
        )

        # Take top keywords
        df_imp = df_imp.head(self.n_keywords * 2)  # Get more for additional filtering

        # Final cleanup - remove any problematic keywords that got through
        df_imp = df_imp[
            ~df_imp["keyword"].apply(
                lambda kw: keyword_is_entity(kw, lang_majority)
                or kw in DOMAIN_STOPWORDS
                or not is_meaningful_keyword(kw, lang_majority)
            )
        ]

        # Return only the requested number of keywords
        return df_imp.head(self.n_keywords).reset_index(drop=True)


###############################################################################
# CLI layer
###############################################################################
def _build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract impactful keywords.")
    p.add_argument("--text-data", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric-column")
    p.add_argument("--n-keywords", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--min-importance",
        type=float,
        default=0.01,
        help="Minimum importance threshold for keywords",
    )
    p.add_argument(
        "--ngrams",
        type=int,
        default=2,
        help="Maximum ngram size (1=unigrams only, 2=unigrams+bigrams)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli()

    if args.n_keywords > MAX_KEYWORDS:
        logging.info("n-keywords capped from %d → %d", args.n_keywords, MAX_KEYWORDS)

    df_text = load_dataframe(args.text_data)
    df_metrics = load_dataframe(args.metrics)

    extractor = KeywordExtractor(
        args.n_keywords,
        args.test_size,
        min_importance=args.min_importance,
        ngram_range=(1, args.ngrams),
    )
    df_out = extractor.run(df_text, df_metrics, args.metric_column)
    save_dataframe(df_out, args.output)
    logging.info("Saved → %s (%d rows)", Path(args.output).resolve(), len(df_out))


if __name__ == "__main__":
    main()
