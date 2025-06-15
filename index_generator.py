# index_generator.py
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
import os
import platform
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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
                logger.info("Using Apple Silicon GPU (MPS)")
                return True

        # Check for NVIDIA GPU
        try:
            import torch

            if torch.cuda.is_available():
                torch.device("cuda")
                logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
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
                logger.info(f"Using TensorFlow GPU: {len(gpus)} device(s) available")
                return True
        except (ImportError, AttributeError):
            pass

        logger.info("No GPU detected, running on CPU")
        return False
    except Exception as e:
        logger.warning(f"Error while setting up GPU: {e}")
        logger.info("Falling back to CPU")
        return False


# Try to setup GPU
USE_GPU = setup_gpu()

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
    """Tokenize text while preserving multi-word phrases."""
    # First, split into words
    words = [LEMMATIZER.lemmatize(t.lower()) for t in TOKENIZER.tokenize(text)]

    # Create a list to store both individual words and potential phrases
    tokens = []

    # Add individual words
    tokens.extend(words)

    # Add potential phrases (2-3 word combinations)
    for i in range(len(words) - 1):
        # Add 2-word phrases
        phrase = f"{words[i]} {words[i + 1]}"
        tokens.append(phrase)

        # Add 3-word phrases if possible
        if i < len(words) - 2:
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            tokens.append(phrase)

    return tokens


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
        logger.info(
            f"Initialized IndexGenerator with weights: simple={w_simple}, advanced={w_advanced}, sentiment={w_sent}"
        )
        self.device = None

        # Set up device for processing if GPU is available
        if USE_GPU:
            try:
                import torch

                # Try CUDA first
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info("Using CUDA device for processing")
                # Then try Apple Silicon
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = torch.device("mps")
                    logger.info("Using MPS device for processing")
                else:
                    logger.info("PyTorch available but no GPU detected")
            except ImportError:
                logger.info("PyTorch not available, using CPU only")

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

    def _clean_keyword(self, keyword: str) -> str:
        """Remove quotes from keywords if present."""
        return keyword.strip("\"'") if keyword else keyword

    def score_one(self, text: str, keywords_en: list[str]) -> tuple[int, float, float]:
        """Return (simple, advanced, sentiment) scores for a single doc."""
        # Clean keywords by removing quotes
        cleaned_keywords = [self._clean_keyword(kw) for kw in keywords_en]

        # Process with GPU if available (for future processing)
        # This currently uses CPU processing but is structured for GPU acceleration
        if self.device:
            return self._score_one_gpu(text, cleaned_keywords)
        else:
            return self._score_one_cpu(text, cleaned_keywords)

    def _score_one_cpu(
        self, text: str, keywords_en: list[str]
    ) -> tuple[int, float, float]:
        """CPU implementation of scoring."""
        tokens = set(_tokenise(text))

        # Detect language for language-specific adjustments
        lang = detect_lang(text[:1000] if len(text) > 1000 else text)

        # simple - now handles multi-word keywords
        simple = sum(1 for kw in keywords_en if kw.lower() in tokens)
        if simple > 0:
            logger.debug(f"Found {simple} simple matches")

        # advanced (+0.5 per synonym)
        advanced = 0.0
        matched_keywords = set()

        for kw in keywords_en:
            kw_lower = kw.lower()
            if kw_lower in tokens:
                advanced += 1
                matched_keywords.add(kw)
                logger.debug(f"Found direct match for keyword: {kw}")
            # Only check synonyms for single-word keywords
            elif " " not in kw:
                for syn in self._synonyms(kw):
                    if syn != kw and syn.lower() in tokens:
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
                logger.debug(
                    f"Found {len(matched_sentences)} sentences with keyword matches"
                )

                # Calculate sentiment with language-specific adjustments
                if lang == "nl":
                    # For Dutch: analyze English translation if possible, or analyze Dutch directly
                    try:
                        # Here we would typically use a translation service, but for now we'll just adjust the weights
                        # Dutch sentiment scores tend to be closer to neutral, so we'll scale them a bit
                        sentiments = [
                            self.sia.polarity_scores(s)["compound"]
                            * 1.2  # Scale Dutch sentiment scores
                            for s in matched_sentences
                        ]
                        sentiment = sum(sentiments) / len(sentiments)
                        sentiment = max(min(sentiment, 1.0), -1.0)  # Clamp to [-1, 1]
                        logger.debug(f"Dutch sentiment adjusted score: {sentiment}")
                    except Exception as e:
                        logger.warning(f"Error in Dutch sentiment adjustment: {e}")
                        # Fallback to regular sentiment
                        sentiments = [
                            self.sia.polarity_scores(s)["compound"]
                            for s in matched_sentences
                        ]
                        sentiment = sum(sentiments) / len(sentiments)
                        logger.debug(f"Fallback sentiment score: {sentiment}")
                else:
                    # For English and other languages, use standard VADER sentiment
                    sentiments = [
                        self.sia.polarity_scores(s)["compound"]
                        for s in matched_sentences
                    ]
                    sentiment = sum(sentiments) / len(sentiments)
                    logger.debug(f"Standard sentiment score: {sentiment}")

        return simple, advanced, sentiment

    def _score_one_gpu(
        self, text: str, keywords_en: list[str]
    ) -> tuple[int, float, float]:
        """GPU implementation of scoring - currently similar to CPU but structured for future optimizations."""
        # For now, this uses the same implementation as CPU
        # In the future, this could use batched operations, tensor-based processing
        # or other GPU-accelerated NLP libraries
        return self._score_one_cpu(text, keywords_en)

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
        logger.info(
            f"Keywords to be used: {keywords_en[:5]}..."
        )  # Log first 5 keywords

        for _, row in tqdm(df_docs.iterrows(), total=len(df_docs), desc="Scoring"):
            raw_text: str = str(row["text"])
            domain: str = row["domain"]
            lang = detect_lang(raw_text)
            text = remove_person_org(raw_text, lang)  # anonymise

            logger.debug(f"Processing domain: {domain}, language: {lang}")

            # Process all documents the same way, regardless of language
            simple, advanced, sentiment = self.score_one(text, keywords_en)

            # Only include sentiment in final score if there are matches
            has_matches = simple > 0 or advanced > 0
            final_score = (
                self.w_simple * simple
                + self.w_advanced * advanced
                + (self.w_sent * (sentiment + 1) if has_matches else 0)
            )

            if simple > 0 or advanced > 0:
                logger.info(
                    f"Domain {domain} - Lang: {lang}, Simple: {simple}, Advanced: {advanced}, Sentiment: {sentiment}, Final: {final_score}"
                )

            results.append(
                dict(
                    domain=domain,
                    language=lang,
                    simple_match_score=simple,
                    advanced_match_score=advanced,
                    sentiment_score=sentiment,
                    final_score=final_score,
                    raw_score=final_score,  # Keep the original score for reference
                )
            )

        # Create DataFrame with all results
        df_results = pd.DataFrame(results)

        # Normalize final_score to be between 0 and 1
        max_score = df_results["final_score"].max()
        if max_score > 0:  # Avoid division by zero
            df_results["final_score"] = df_results["final_score"] / max_score
            logger.info(f"Normalized all scores by dividing by max score: {max_score}")

        return df_results


###############################################################################
# CLI
###############################################################################
def _build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sustainability index.")
    p.add_argument("--data", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--simple-weight", type=float, default=0.4)
    p.add_argument("--advanced-weight", type=float, default=0.4)
    p.add_argument("--sentiment-weight", type=float, default=0.2)
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

    gen = IndexGenerator(
        args.simple_weight, args.advanced_weight, args.sentiment_weight
    )
    df_out = gen.run(df_docs, kw_list)
    save_dataframe(df_out, args.output)
    logger.info("Saved → %s (%d rows)", Path(args.output).resolve(), len(df_out))


if __name__ == "__main__":
    main()
