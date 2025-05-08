# utils_nlp.py
"""Language detection, NER‑based anonymisation and stop‑word utilities."""
from __future__ import annotations

import functools
import logging
from typing import Final, Set

import langdetect
import nltk
import spacy
from spacy.language import Language

# --------------------------------------------------------------------------- #
# Download NLTK stop‑words only once
# --------------------------------------------------------------------------- #
for pkg in ("stopwords",):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

_STOPWORDS = {
    "en": set(nltk.corpus.stopwords.words("english")),
    "nl": set(nltk.corpus.stopwords.words("dutch")),
}

# --------------------------------------------------------------------------- #
# spaCy model loader (cached)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=2)
def _load_spacy(lang_code: str) -> Language:
    model = "en_core_web_trf" if lang_code == "en" else "nl_core_news_sm"
    try:
        return spacy.load(model)
    except OSError as err:
        logging.error(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            model,
            model,
        )
        raise err


def detect_lang(text: str) -> str:
    """Return 'en' or 'nl' (fallback → 'en')."""
    try:
        return langdetect.detect(text[:1000])
    except langdetect.lang_detect_exception.LangDetectException:
        return "en"


def remove_person_org(text: str, lang_code: str) -> str:
    """Strip PERSON + ORG entity tokens from a string."""
    nlp = _load_spacy(lang_code)
    doc = nlp(text)
    tokens = [
        tok.text
        for tok in doc
        if tok.ent_type_ not in {"PERSON", "ORG"}  # drop names & companies
    ]
    return " ".join(tokens)


def keyword_is_entity(keyword: str, lang_code: str) -> bool:
    nlp = _load_spacy(lang_code)
    doc = nlp(keyword)
    return any(ent.label_ in {"PERSON", "ORG"} for ent in doc.ents)


def stopwords_for(lang_code: str) -> Set[str]:
    return _STOPWORDS.get(lang_code, _STOPWORDS["en"])
