import sys

import nltk
import spacy


def test_nltk():
    print("Testing NLTK...")
    try:
        # Try both local and system paths
        nltk_paths = ["./models/nltk", nltk.data.path[0]]
        nltk.data.path = nltk_paths

        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
        nltk.data.find("sentiment/vader_lexicon")
        nltk.data.find("corpora/omw-1.4")
        print("✓ NLTK data found successfully")
        return True
    except LookupError as e:
        print(f"✗ NLTK data not found: {e}")
        return False


def test_spacy():
    print("\nTesting spaCy...")
    try:
        # Add our local models directory to Python path
        if "./models/spacy" not in sys.path:
            sys.path.insert(0, "./models/spacy")

        # Try to load English model
        nlp_en = spacy.load("en_core_web_trf")
        print("✓ English spaCy model loaded successfully")

        # Try to load Dutch model
        try:
            nlp_nl = spacy.load("nl_core_news_sm")
            print("✓ Dutch spaCy model loaded successfully")
        except Exception as e:
            print(f"✗ Dutch spaCy model not found: {e}")
            print(
                "Please download the Dutch model using: python -m spacy download nl_core_news_sm"
            )
            return False

        return True
    except Exception as e:
        print(f"✗ spaCy models not found: {e}")
        return False


if __name__ == "__main__":
    # Test both
    nltk_ok = test_nltk()
    spacy_ok = test_spacy()

    if nltk_ok and spacy_ok:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
