#keyword_extractor.py
"""
Keyword Extractor – works for regression *and* classification, anonymises text,
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
from pathlib import Path
from typing import Final, List
import re

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from bs4 import BeautifulSoup
import string
from nltk.stem import WordNetLemmatizer

from utils_io import load_dataframe, save_dataframe
from utils_nlp import (
    detect_lang,
    keyword_is_entity,
    remove_person_org,
    stopwords_for,
)

MAX_KEYWORDS: Final = 50

###############################################################################
# Ensure NLTK tokeniser
###############################################################################
for pkg in ("punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

TOKENIZER: Final = nltk.tokenize.RegexpTokenizer(r"[A-Za-z]+")
LEMMATIZER: Final = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """Use the same tokenization as index_generator."""
    lang = detect_lang(text)
    stops = stopwords_for(lang)
    text = remove_person_org(text, lang)
    tokens = [
        LEMMATIZER.lemmatize(t.lower())
        for t in TOKENIZER.tokenize(text)
        if t.lower() not in stops and len(t) > 2
    ]
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
    ) -> None:
        self.n_keywords = min(n_keywords, MAX_KEYWORDS)
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            min_df=1,
            max_df=0.95
        )

    # ------------------------------------------------------------------ #
    def _detect_problem_type(self, y: pd.Series) -> str:
        # If the target variable has continuous values, treat it as regression
        if y.dtype.kind in {"i", "f"}:
            return "regression"
        return "classification"

    def _build_model(self, problem: str):
        if problem == "regression":
            return RandomForestRegressor(
                n_estimators=400, n_jobs=-1, random_state=self.random_state
            )
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=self.random_state
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        df_text: pd.DataFrame,
        df_metrics: pd.DataFrame,
        metric_col: str | None = None,
    ) -> pd.DataFrame:
        # Extract base domain from URLs
        df_text['base_domain'] = df_text['domain'].apply(lambda x: x.replace('https://', '').replace('http://', '').split('/')[0])
        df_metrics['base_domain'] = df_metrics['domain'].apply(lambda x: x.replace('https://', '').replace('http://', '').split('/')[0])
        
        # merge
        df = pd.merge(df_text, df_metrics, on="base_domain")
        if metric_col is None:
            metric_col = (
                df_metrics.select_dtypes(include=[np.number]).columns.tolist()[0]
            )

        tqdm.pandas(desc="Cleaning text")
        df["clean"] = df["text"].progress_apply(preprocess)

        # Vectorise
        X = self.vectorizer.fit_transform(df["clean"])
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

        # Importance
        imp = model.feature_importances_
        feats = self.vectorizer.get_feature_names_out()
        df_imp = (
            pd.DataFrame({"keyword": feats, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(self.n_keywords)
        )

        # Drop keywords that are entities
        lang_majority = (
            df["clean"].str[:500].map(detect_lang).value_counts().idxmax()
        )
        df_imp = df_imp[
            ~df_imp["keyword"].apply(
                lambda kw: keyword_is_entity(kw, lang_majority)
            )
        ]
        return df_imp.reset_index(drop=True)


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
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_cli()

    if args.n_keywords > MAX_KEYWORDS:
        logging.info("n-keywords capped from %d → %d", args.n_keywords, MAX_KEYWORDS)

    df_text = load_dataframe(args.text_data)
    df_metrics = load_dataframe(args.metrics)

    extractor = KeywordExtractor(args.n_keywords, args.test_size)
    df_out = extractor.run(df_text, df_metrics, args.metric_column)
    save_dataframe(df_out, args.output)
    logging.info("Saved → %s (%d rows)", Path(args.output).resolve(), len(df_out))


if __name__ == "__main__":
    main()
