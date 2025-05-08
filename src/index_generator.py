#index_generator.py
"""
Index Generator – now language‑aware and NER‑anonymised.

Usage:
    python index_generator.py --data data/pages.ndjson \
                              --keywords data/keywords.csv \
                              --output data/scores.csv
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Final, List

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

from utils_io import load_dataframe, save_dataframe
from utils_nlp import detect_lang, keyword_is_entity, remove_person_org

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# NLTK resources (download just once)
###############################################################################
for pkg in ("punkt", "wordnet", "vader_lexicon", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

TOKENIZER: Final = RegexpTokenizer(r"[A-Za-z]+")
LEMMATIZER: Final = WordNetLemmatizer()


def _tokenise(text: str) -> List[str]:
    return [LEMMATIZER.lemmatize(t.lower()) for t in TOKENIZER.tokenize(text)]


###############################################################################
# Index generator
###############################################################################
class IndexGenerator:
    def __init__(
        self,
        w_simple: float,
        w_advanced: float,
        w_sent: float,
    ) -> None:
        self.w_simple = w_simple
        self.w_advanced = w_advanced
        self.w_sent = w_sent
        self.sia = SentimentIntensityAnalyzer()
        self._syn_cache: dict[str, set[str]] = defaultdict(set)
        logger.info(f"Initialized IndexGenerator with weights: simple={w_simple}, advanced={w_advanced}, sentiment={w_sent}")

    # --------------------------------------------------------------------- #
    def _synonyms(self, word: str) -> set[str]:
        """English synonyms via WordNet (cached)."""
        if word in self._syn_cache:
            return self._syn_cache[word]
        syns: set[str] = {word}
        for syn in nltk.corpus.wordnet.synsets(word, pos="n"):
            syns.update(lemma.name().replace("_", " ") for lemma in syn.lemmas())
        self._syn_cache[word] = syns
        return syns

    def score_one(self, text: str, keywords_en: list[str]) -> tuple[int, float, float]:
        """Return (simple, advanced, sentiment) scores for a single doc."""
        tokens = set(_tokenise(text))
        
        # simple
        simple = sum(1 for kw in keywords_en if kw in tokens)
        if simple > 0:
            logger.debug(f"Found {simple} simple matches")
        
        # advanced (+0.5 per synonym)
        advanced = 0.0
        matched_keywords = set()
        
        for kw in keywords_en:
            if kw in tokens:
                advanced += 1
                matched_keywords.add(kw)
                logger.debug(f"Found direct match for keyword: {kw}")
            for syn in self._synonyms(kw):
                if syn != kw and syn in tokens:
                    advanced += 0.5
                    matched_keywords.add(kw)
                    logger.debug(f"Found synonym match: {syn} for keyword: {kw}")
        
        # Calculate sentiment only for matched portions
        sentiment = 0.0
        if matched_keywords:
            # Split text into sentences
            sentences = nltk.sent_tokenize(text)
            matched_sentences = []
            
            for sentence in sentences:
                sentence_tokens = set(_tokenise(sentence))
                # Check if sentence contains any matched keywords
                if any(kw in sentence_tokens for kw in matched_keywords):
                    matched_sentences.append(sentence)
            
            if matched_sentences:
                logger.debug(f"Found {len(matched_sentences)} sentences with keyword matches")
                # Calculate average sentiment of matched sentences
                sentiments = [self.sia.polarity_scores(s)["compound"] for s in matched_sentences]
                sentiment = sum(sentiments) / len(sentiments)
                logger.debug(f"Average sentiment score: {sentiment}")
        
        return simple, advanced, sentiment

    # --------------------------------------------------------------------- #
    # Public
    # --------------------------------------------------------------------- #
    def run(self, df_docs: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
        results = []
        logger.info(f"Starting processing of {len(df_docs)} documents")
        logger.info(f"Number of keywords before filtering: {len(keywords)}")
        
        # pre‑filter keywords that are PERSON / ORG in *either* language
        keywords = [
            kw
            for kw in keywords
            if not keyword_is_entity(kw, "en") and not keyword_is_entity(kw, "nl")
        ]
        logger.info(f"Number of keywords after filtering: {len(keywords)}")
        
        keywords_en = [kw.lower() for kw in keywords]  # WordNet only EN
        logger.info(f"Keywords to be used: {keywords_en[:5]}...")  # Log first 5 keywords

        for _, row in tqdm(df_docs.iterrows(), total=len(df_docs), desc="Scoring"):
            raw_text: str = str(row["text"])
            domain: str = row["domain"]
            lang = detect_lang(raw_text)
            text = remove_person_org(raw_text, lang)  # anonymise

            logger.debug(f"Processing domain: {domain}, language: {lang}")

            simple, advanced, sentiment = (
                self.score_one(text, keywords_en)
                if lang == "en"
                else (0, 0, 0)  # skip for non‑EN
            )
            
            # Only include sentiment in final score if there are matches
            has_matches = simple > 0 or advanced > 0
            final_score = (
                self.w_simple * simple
                + self.w_advanced * advanced
                + (self.w_sent * (sentiment + 1) if has_matches else 0)
            )
            
            if simple > 0 or advanced > 0:
                logger.info(f"Domain {domain} - Simple: {simple}, Advanced: {advanced}, Sentiment: {sentiment}, Final: {final_score}")
            
            results.append(
                dict(
                    domain=domain,
                    simple_match_score=simple,
                    advanced_match_score=advanced,
                    sentiment_score=sentiment,
                    final_score=final_score,
                )
            )
        return pd.DataFrame(results)


###############################################################################
# CLI
###############################################################################
def _build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sustainability index.")
    p.add_argument("--data", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--simple-weight", type=float, default=0.3)
    p.add_argument("--advanced-weight", type=float, default=0.4)
    p.add_argument("--sentiment-weight", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args = _build_cli()

    df_docs = load_dataframe(args.data)
    df_kw = load_dataframe(args.keywords)
    kw_list = (
        df_kw["keyword"].tolist()
        if "keyword" in df_kw.columns
        else df_kw.iloc[:, 0].tolist()
    )

    gen = IndexGenerator(args.simple_weight, args.advanced_weight, args.sentiment_weight)
    df_out = gen.run(df_docs, kw_list)
    save_dataframe(df_out, args.output)
    logger.info("Saved → %s (%d rows)", Path(args.output).resolve(), len(df_out))


if __name__ == "__main__":
    main()
